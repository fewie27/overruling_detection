from __future__ import annotations

import os
import time
from typing import Any, Optional

from src.api_connectors.base import (
    BatchStatus,
    BatchSubmitResult,
    ConnectorResponse,
    LLMConnector,
)
from src.config.models import parse_reasoning_effort, resolve_model_name, warn_if_missing_pricing
from src.models import PreparedLLMRequest

_DEFAULT_MAX_TOKENS = 8192
_RETRIES_ON_HTTP_500 = 3

# Map the CLI reasoning-effort levels to Anthropic extended-thinking budgets
# (budget_tokens; Anthropic requires a minimum of 1024). "none" disables thinking.
_THINKING_BUDGET_BY_EFFORT: dict[str, int] = {
    "minimal": 1024,
    "low": 2048,
    "medium": 4096,
    "high": 8192,
    "xhigh": 16384,
}


def _thinking_budget_tokens(reasoning_effort: str | None) -> int | None:
    """Return the extended-thinking budget for a reasoning effort, or None to disable."""
    if reasoning_effort is None or reasoning_effort == "none":
        return None
    return _THINKING_BUDGET_BY_EFFORT.get(reasoning_effort)


def _read_max_tokens() -> int:
    raw = os.getenv("ANTHROPIC_MAX_TOKENS", "").strip()
    if not raw:
        return _DEFAULT_MAX_TOKENS
    try:
        n = int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid ANTHROPIC_MAX_TOKENS={raw!r}; expected an integer") from exc
    if n < 1:
        raise ValueError("ANTHROPIC_MAX_TOKENS must be >= 1")
    return n


def _read_optional_temperature() -> Optional[float]:
    raw = os.getenv("ANTHROPIC_TEMPERATURE", "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid ANTHROPIC_TEMPERATURE={raw!r}; expected a number") from exc


def _extract_content_and_reasoning(response: Any) -> tuple[str, str | None]:
    """Split a Claude message into (answer_text, reasoning_text).

    ``text`` blocks form the answer; ``thinking`` blocks form the reasoning. If the
    message only has thinking blocks (no text), the thinking is used as the answer
    fallback so downstream parsing still has content to work with."""
    text_chunks: list[str] = []
    thinking_chunks: list[str] = []
    for block in getattr(response, "content", None) or []:
        btype = getattr(block, "type", None)
        if btype == "text":
            t = getattr(block, "text", None)
            if isinstance(t, str) and t.strip():
                text_chunks.append(t.strip())
        elif btype == "thinking":
            t = getattr(block, "thinking", None)
            if isinstance(t, str) and t.strip():
                thinking_chunks.append(t.strip())
    reasoning = "\n".join(thinking_chunks).strip() or None
    if text_chunks:
        content = "\n".join(text_chunks).strip()
    else:
        content = reasoning or ""
    return (content or "{}"), reasoning


class ClaudeConnector(LLMConnector):
    def __init__(
        self,
        model: str | None = None,
        *,
        reasoning_effort: str | None = None,
    ) -> None:
        self._model = resolve_model_name("claude", model)
        warn_if_missing_pricing("claude", self._model)
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY is required when using the Claude connector")
        self.max_tokens = _read_max_tokens()
        self.temperature = _read_optional_temperature()
        self.reasoning_effort = parse_reasoning_effort(reasoning_effort)
        self.thinking_budget_tokens = _thinking_budget_tokens(self.reasoning_effort)
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise ImportError(
                "Claude connector requires `anthropic`. Install it with: pip install anthropic"
            ) from exc
        base_url = os.getenv("ANTHROPIC_BASE_URL", "").strip() or None
        self.client = Anthropic(api_key=self.api_key, base_url=base_url)

    @property
    def label(self) -> str:
        return "claude"

    @property
    def model(self) -> str:
        return self._model

    def extract_overruled_cases(
        self, *, system_prompt: str, user_prompt: str
    ) -> ConnectorResponse:
        kwargs = self._message_params(system_prompt=system_prompt, user_prompt=user_prompt)

        from anthropic import APIStatusError

        response = None
        for attempt in range(_RETRIES_ON_HTTP_500 + 1):
            try:
                response = self.client.messages.create(**kwargs)
                break
            except APIStatusError as exc:
                if getattr(exc, "status_code", None) != 500 or attempt >= _RETRIES_ON_HTTP_500:
                    raise
                wait_s = 1.0 + attempt
                print(
                    f"[Claude] HTTP 500 (attempt {attempt + 1}/{_RETRIES_ON_HTTP_500 + 1}), "
                    f"retrying in {wait_s:.0f}s... {exc}",
                    flush=True,
                )
                time.sleep(wait_s)
        assert response is not None

        content, reasoning = _extract_content_and_reasoning(response)
        return ConnectorResponse(
            raw_content=content,
            raw_response=response,
            reasoning_content=reasoning,
        )

    def _message_params(self, *, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        budget = self.thinking_budget_tokens
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        if budget is not None:
            # Extended thinking: max_tokens must exceed budget_tokens and the answer
            # still needs room, so reserve the configured max_tokens on top of the
            # thinking budget. Anthropic also requires temperature to remain default
            # (1) when thinking is enabled, so temperature is intentionally omitted.
            params["max_tokens"] = budget + self.max_tokens
            params["thinking"] = {"type": "enabled", "budget_tokens": budget}
        elif self.temperature is not None:
            params["temperature"] = self.temperature
        return params

    def submit_batch(self, requests: list[PreparedLLMRequest]) -> BatchSubmitResult:
        batch_requests = [
            {
                "custom_id": req.request_id,
                "params": self._message_params(system_prompt=req.system_prompt, user_prompt=req.user_prompt),
            }
            for req in requests
        ]
        batch = self.client.beta.messages.batches.create(requests=batch_requests)
        return BatchSubmitResult(batch_id=batch.id)

    def get_batch_status(self, batch_id: str) -> BatchStatus:
        batch = self.client.beta.messages.batches.retrieve(batch_id)
        ps = str(getattr(batch, "processing_status", "unknown"))
        is_terminal = ps == "ended"
        return BatchStatus(
            batch_id=batch_id,
            status=ps,
            is_terminal=is_terminal,
            is_success=is_terminal,
        )

    def collect_batch_results(self, batch_id: str) -> dict[str, ConnectorResponse]:
        results: dict[str, ConnectorResponse] = {}
        stream = self.client.beta.messages.batches.results(batch_id)
        try:
            for row in stream:
                request_id = str(getattr(row, "custom_id", "") or "")
                if not request_id:
                    continue
                res = getattr(row, "result", None)
                if res is None:
                    continue
                rtype = getattr(res, "type", None)
                if rtype != "succeeded":
                    continue
                msg = getattr(res, "message", None)
                if msg is None:
                    continue
                content, reasoning = _extract_content_and_reasoning(msg)
                results[request_id] = ConnectorResponse(
                    raw_content=content,
                    raw_response=msg,
                    reasoning_content=reasoning,
                )
        finally:
            stream.close()
        return results
