from __future__ import annotations

import os
import time
import json
import tempfile
from typing import Any, Literal, Optional

from openai import APIStatusError, OpenAI

from src.api_connectors.base import BatchStatus, BatchSubmitResult, ConnectorResponse, LLMConnector
from src.config.models import parse_reasoning_effort, resolve_model_name, warn_if_missing_pricing
from src.models import PreparedLLMRequest

ReasoningEffortLevel = Literal["none", "minimal", "low", "medium", "high", "xhigh"]
# Reasoning (esp. high/xhigh) + answer share this budget; GPT-5.4 allows up to 128k.
_MAX_OUTPUT_TOKENS = 65536
_RETRIES_ON_HTTP_500 = 3
_OPENAI_BATCH_COMPLETION_WINDOW = "24h"
_TERMINAL_BATCH_STATES = frozenset({"completed", "failed", "cancelled", "expired"})


def _merge_batch_usage_into_body(
    body: dict[str, Any], response_wrapper: dict[str, Any] | None
) -> dict[str, Any]:
    """Batch JSONL often puts token ``usage`` on ``response``, not inside ``response.body``."""
    if body.get("usage") is not None:
        return body
    if isinstance(response_wrapper, dict) and response_wrapper.get("usage") is not None:
        return {**body, "usage": response_wrapper["usage"]}
    return body


def _batch_collect_failure_message(batch_id: str, batch: Any) -> str:
    """Human-readable detail when neither output nor error JSONL yields any rows."""
    parts = [
        f"Batch {batch_id!r} produced no result rows from output_file_id or error_file_id.",
        "If status is completed but output_file_id is null, all requests may have failed (see error_file_id).",
        "If both file ids are null, the batch may still be finalizing or files expired.",
    ]
    status = getattr(batch, "status", None)
    if status is not None:
        parts.append(f"status={status!r}.")
    parts.append(f"output_file_id={getattr(batch, 'output_file_id', None)!r}.")
    parts.append(f"error_file_id={getattr(batch, 'error_file_id', None)!r}.")
    rc = getattr(batch, "request_counts", None)
    if rc is not None:
        try:
            dumped = rc.model_dump() if hasattr(rc, "model_dump") else rc
            parts.append(f"request_counts={dumped!r}.")
        except Exception:
            parts.append(f"request_counts={rc!r}.")
    return " ".join(parts)


def _read_optional_temperature() -> Optional[float]:
    raw = os.getenv("OPENAI_TEMPERATURE", "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid OPENAI_TEMPERATURE={raw!r}; expected a number") from exc


def _read_http_timeout_seconds() -> float | None:
    """Avoid hanging forever on a stuck connection; reasoning calls may need a high value."""
    raw = os.getenv("OPENAI_HTTP_TIMEOUT_SECONDS", "").strip()
    if not raw:
        return 900.0
    try:
        sec = float(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid OPENAI_HTTP_TIMEOUT_SECONDS={raw!r}; expected a number") from exc
    if sec <= 0:
        return None
    return sec


def _incomplete_details_dict(response: Any) -> dict[str, Any] | None:
    inc = getattr(response, "incomplete_details", None)
    if inc is None:
        return None
    if isinstance(inc, dict):
        return inc
    if hasattr(inc, "model_dump"):
        dumped = inc.model_dump()
        return dumped if isinstance(dumped, dict) else None
    return None


class OpenAIConnector(LLMConnector):
    def __init__(
        self,
        model: str | None = None,
        *,
        reasoning_effort: str | None = None,
    ) -> None:
        self._model = resolve_model_name("openai", model)
        warn_if_missing_pricing("openai", self._model)
        self.reasoning_effort = parse_reasoning_effort(reasoning_effort)
        self.temperature = _read_optional_temperature()
        self.max_output_tokens = _MAX_OUTPUT_TOKENS
        timeout = _read_http_timeout_seconds()
        client_kw: dict[str, Any] = {"api_key": os.getenv("OPENAI_API_KEY")}
        if timeout is not None:
            client_kw["timeout"] = timeout
        self.client = OpenAI(**client_kw)

    @property
    def label(self) -> str:
        return "openai"

    @property
    def model(self) -> str:
        return self._model

    def extract_overruled_cases(
        self, *, system_prompt: str, user_prompt: str
    ) -> ConnectorResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "instructions": system_prompt,
            "input": [{"role": "user", "content": user_prompt}],
            "text": {"format": {"type": "json_object"}},
            "max_output_tokens": self.max_output_tokens,
        }
        if self.reasoning_effort is not None:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature

        response = None
        for attempt in range(_RETRIES_ON_HTTP_500 + 1):
            try:
                response = self.client.responses.create(**kwargs)
                break
            except APIStatusError as exc:
                if exc.status_code != 500 or attempt >= _RETRIES_ON_HTTP_500:
                    raise
                wait_s = 1.0 + attempt
                print(
                    f"[OpenAI] HTTP 500 (attempt {attempt + 1}/{_RETRIES_ON_HTTP_500 + 1}), "
                    f"retrying in {wait_s:.0f}s... {exc}",
                    flush=True,
                )
                time.sleep(wait_s)
        assert response is not None
        inc = _incomplete_details_dict(response)
        if inc and inc.get("reason") == "content_filter":
            raise RuntimeError(
                "OpenAI returned status=incomplete with incomplete_details: content_filter. "
                "The request or completion was stopped by the content policy, so you may see 0 token usage and truncated/invalid output. "
                "For legal-opinion benchmarks (e.g. capital punishment), try another model or provider, shorten or redact the opinion excerpt, or use an account/policy suitable for that research use."
            )

        content = (response.output_text or "").strip() or "{}"
        return ConnectorResponse(raw_content=content, raw_response=response)

    def submit_batch(self, requests: list[PreparedLLMRequest]) -> BatchSubmitResult:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".jsonl", delete=True) as f:
            for req in requests:
                payload = self._request_payload(system_prompt=req.system_prompt, user_prompt=req.user_prompt)
                line = {
                    "custom_id": req.request_id,
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": payload,
                }
                f.write(json.dumps(line, ensure_ascii=True) + "\n")
            f.flush()
            file_obj = self.client.files.create(file=open(f.name, "rb"), purpose="batch")

        batch = self.client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/responses",
            completion_window=_OPENAI_BATCH_COMPLETION_WINDOW,
        )
        return BatchSubmitResult(batch_id=batch.id)

    def get_batch_status(self, batch_id: str) -> BatchStatus:
        batch = self.client.batches.retrieve(batch_id)
        status = str(getattr(batch, "status", "unknown"))
        is_terminal = status in _TERMINAL_BATCH_STATES
        return BatchStatus(
            batch_id=batch_id,
            status=status,
            is_terminal=is_terminal,
            is_success=(status == "completed"),
        )

    def collect_batch_results(self, batch_id: str) -> dict[str, ConnectorResponse]:
        batch = self.client.batches.retrieve(batch_id)
        results: dict[str, ConnectorResponse] = {}

        output_file_id = getattr(batch, "output_file_id", None)
        if output_file_id:
            content_text = self.client.files.content(output_file_id).text
            if content_text:
                for raw_line in content_text.splitlines():
                    line = raw_line.strip()
                    if not line:
                        continue
                    payload = json.loads(line)
                    request_id = str(payload.get("custom_id") or "")
                    if not request_id:
                        continue

                    response_wrapper = payload.get("response")
                    rw = response_wrapper if isinstance(response_wrapper, dict) else None
                    body_raw = (rw or {}).get("body") if rw else None
                    body = body_raw if isinstance(body_raw, dict) else {}
                    body = _merge_batch_usage_into_body(body, rw)
                    content = _production_output_text_from_batch_body(body)
                    results[request_id] = ConnectorResponse(raw_content=content, raw_response=body)

        error_file_id = getattr(batch, "error_file_id", None)
        if error_file_id:
            err_text = self.client.files.content(error_file_id).text
            if err_text:
                for raw_line in err_text.splitlines():
                    line = raw_line.strip()
                    if not line:
                        continue
                    payload = json.loads(line)
                    request_id = str(payload.get("custom_id") or "")
                    if not request_id or request_id in results:
                        continue
                    response_obj = payload.get("response")
                    if isinstance(response_obj, dict):
                        body_inner = response_obj.get("body")
                        if isinstance(body_inner, dict) and body_inner:
                            body = _merge_batch_usage_into_body(body_inner, response_obj)
                            content = _production_output_text_from_batch_body(body)
                            results[request_id] = ConnectorResponse(raw_content=content, raw_response=body)
                            continue
                    if payload.get("error") is not None:
                        results[request_id] = ConnectorResponse(
                            raw_content="{}",
                            raw_response=payload,
                        )

        if not results:
            raise RuntimeError(_batch_collect_failure_message(batch_id, batch))
        return results

    def _request_payload(self, *, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "instructions": system_prompt,
            "input": [{"role": "user", "content": user_prompt}],
            "text": {"format": {"type": "json_object"}},
            "max_output_tokens": self.max_output_tokens,
        }
        if self.reasoning_effort is not None:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        return kwargs


def _text_from_response_content_part(part: dict[str, Any]) -> str | None:
    """Map Responses API message content parts to a string (text, JSON, or parsed blob)."""
    t = part.get("text")
    if isinstance(t, str) and t.strip():
        return t.strip()
    parsed = part.get("parsed")
    if isinstance(parsed, (dict, list)):
        try:
            return json.dumps(parsed, ensure_ascii=True)
        except (TypeError, ValueError):
            return None
    return None


def _production_output_text_from_batch_body(body: dict[str, Any]) -> str:
    output_text = body.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    chunks: list[str] = []
    output_items = body.get("output")
    if isinstance(output_items, list):
        for item in output_items:
            if not isinstance(item, dict):
                continue
            content_items = item.get("content")
            if not isinstance(content_items, list):
                continue
            for content_item in content_items:
                if not isinstance(content_item, dict):
                    continue
                piece = _text_from_response_content_part(content_item)
                if piece:
                    chunks.append(piece)

    if chunks:
        return "\n".join(chunks)
    return "{}"
