from __future__ import annotations

import os
import time
from typing import Any, Optional

from openai import APIStatusError, OpenAI

from src.api_connectors.base import ConnectorResponse, LLMConnector
from src.config.models import parse_reasoning_effort, resolve_model_name, warn_if_missing_pricing

_DEFAULT_BASE_URL = "https://api.deepseek.com"
_DEFAULT_MAX_TOKENS = 16384
_RETRIES_ON_HTTP_500 = 3


def _read_base_url() -> str:
    raw = os.getenv("DEEPSEEK_BASE_URL", "").strip()
    return (raw or _DEFAULT_BASE_URL).rstrip("/")


def _read_max_tokens() -> int:
    raw = os.getenv("DEEPSEEK_MAX_TOKENS", "").strip()
    if not raw:
        return _DEFAULT_MAX_TOKENS
    try:
        n = int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid DEEPSEEK_MAX_TOKENS={raw!r}; expected an integer") from exc
    if n < 1:
        raise ValueError("DEEPSEEK_MAX_TOKENS must be >= 1")
    return n


def _read_optional_temperature() -> Optional[float]:
    raw = os.getenv("DEEPSEEK_TEMPERATURE", "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid DEEPSEEK_TEMPERATURE={raw!r}; expected a number") from exc


def _read_http_timeout_seconds() -> float | None:
    raw = os.getenv("DEEPSEEK_HTTP_TIMEOUT_SECONDS", "").strip()
    if not raw:
        return 900.0
    try:
        sec = float(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid DEEPSEEK_HTTP_TIMEOUT_SECONDS={raw!r}; expected a number") from exc
    if sec <= 0:
        return None
    return sec


def _thinking_and_reasoning_kwargs(reasoning_effort: str | None) -> dict[str, Any]:
    """Map CLI reasoning effort to DeepSeek thinking / reasoning_effort API fields."""
    if reasoning_effort is None:
        return {}
    if reasoning_effort == "none":
        return {"thinking": {"type": "disabled"}}
    ds_effort = "max" if reasoning_effort == "xhigh" else "high"
    return {"thinking": {"type": "enabled"}, "reasoning_effort": ds_effort}


def _chat_completion_content(response: Any) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return "{}"
    message = getattr(choices[0], "message", None)
    if message is None:
        return "{}"
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return content.strip()
    return "{}"


def _chat_completion_reasoning(response: Any) -> str | None:
    """Extract the reasoner's chain-of-thought (``message.reasoning_content``)."""
    choices = getattr(response, "choices", None) or []
    if not choices:
        return None
    message = getattr(choices[0], "message", None)
    if message is None:
        return None
    reasoning = getattr(message, "reasoning_content", None)
    if reasoning is None and isinstance(message, dict):
        reasoning = message.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning.strip()
    return None


class DeepSeekConnector(LLMConnector):
    """DeepSeek API via the OpenAI-compatible Chat Completions endpoint."""

    def __init__(
        self,
        model: str | None = None,
        *,
        reasoning_effort: str | None = None,
    ) -> None:
        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY is not set")
        self._model = resolve_model_name("deepseek", model)
        warn_if_missing_pricing("deepseek", self._model)
        self.reasoning_effort = parse_reasoning_effort(reasoning_effort)
        self.temperature = _read_optional_temperature()
        self.max_tokens = _read_max_tokens()
        timeout = _read_http_timeout_seconds()
        client_kw: dict[str, Any] = {
            "api_key": api_key,
            "base_url": _read_base_url(),
        }
        if timeout is not None:
            client_kw["timeout"] = timeout
        self.client = OpenAI(**client_kw)

    @property
    def label(self) -> str:
        return "deepseek"

    @property
    def model(self) -> str:
        return self._model

    def extract_overruled_cases(
        self, *, system_prompt: str, user_prompt: str
    ) -> ConnectorResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": self.max_tokens,
        }
        kwargs.update(_thinking_and_reasoning_kwargs(self.reasoning_effort))
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature

        response = None
        for attempt in range(_RETRIES_ON_HTTP_500 + 1):
            try:
                response = self.client.chat.completions.create(**kwargs)
                break
            except APIStatusError as exc:
                if exc.status_code != 500 or attempt >= _RETRIES_ON_HTTP_500:
                    raise
                wait_s = 1.0 + attempt
                print(
                    f"[DeepSeek] HTTP 500 (attempt {attempt + 1}/{_RETRIES_ON_HTTP_500 + 1}), "
                    f"retrying in {wait_s:.0f}s... {exc}",
                    flush=True,
                )
                time.sleep(wait_s)
        assert response is not None

        finish_reason = None
        if getattr(response, "choices", None):
            finish_reason = getattr(response.choices[0], "finish_reason", None)
        if finish_reason == "content_filter":
            raise RuntimeError(
                "DeepSeek returned finish_reason=content_filter. "
                "The request or completion was stopped by the content policy."
            )

        content = _chat_completion_content(response)
        reasoning = _chat_completion_reasoning(response)
        return ConnectorResponse(
            raw_content=content,
            raw_response=response,
            reasoning_content=reasoning,
        )
