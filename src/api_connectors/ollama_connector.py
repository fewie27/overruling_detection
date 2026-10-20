from __future__ import annotations

import os
import re
import time
from typing import Any, Optional

import requests

from src.api_connectors.base import BatchSubmitResult, ConnectorResponse, LLMConnector
from src.config.models import resolve_model_name
from src.models import PreparedLLMRequest

_DEFAULT_BASE_URL = "http://127.0.0.1:11434"
_DEFAULT_NUM_PREDICT = 8192
_DEFAULT_TIMEOUT_S = 900
_RETRIES_ON_HTTP_500 = 3
_THINK_BLOCK = re.compile(r"<think>\s*.*?(?:</think>|$)", re.DOTALL | re.IGNORECASE)


def _read_base_url() -> str:
    raw = os.getenv("OLLAMA_BASE_URL", "").strip()
    if raw:
        return raw.rstrip("/")
    host = os.getenv("OLLAMA_HOST", "").strip()
    if host:
        if not host.startswith(("http://", "https://")):
            host = f"http://{host}"
        return host.rstrip("/")
    return _DEFAULT_BASE_URL


def _read_num_predict() -> int:
    raw = os.getenv("OLLAMA_NUM_PREDICT", "").strip()
    if not raw:
        return _DEFAULT_NUM_PREDICT
    try:
        n = int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid OLLAMA_NUM_PREDICT={raw!r}; expected an integer") from exc
    if n < 1:
        raise ValueError("OLLAMA_NUM_PREDICT must be >= 1")
    return n


def _read_optional_temperature() -> Optional[float]:
    raw = os.getenv("OLLAMA_TEMPERATURE", "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid OLLAMA_TEMPERATURE={raw!r}; expected a number") from exc


def _read_timeout_s() -> float:
    raw = os.getenv("OLLAMA_TIMEOUT", "").strip()
    if not raw:
        return float(_DEFAULT_TIMEOUT_S)
    try:
        t = float(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid OLLAMA_TIMEOUT={raw!r}; expected a number") from exc
    if t <= 0:
        raise ValueError("OLLAMA_TIMEOUT must be > 0")
    return t


def _read_format_json() -> bool:
    """Whether to force Ollama ``format: json``.

    Default is off so R1-style / finetuned models can emit ``<think>…</think>``
    before the answer JSON. The shared JSON parser strips those tags. Set
    ``OLLAMA_FORMAT_JSON=1`` only if you want constrained JSON-only decoding.
    """
    raw = os.getenv("OLLAMA_FORMAT_JSON", "").strip().lower()
    if not raw:
        return False
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    raise ValueError(f"Invalid OLLAMA_FORMAT_JSON={raw!r}; expected true/false")


def _split_think_content(text: str) -> tuple[str, str | None]:
    """Split ``<think>…</think>`` from the answer body (same convention as finetune export)."""
    s = (text or "").strip()
    if not s:
        return "", None
    matches = list(_THINK_BLOCK.finditer(s))
    if not matches:
        return s, None
    reasoning_parts: list[str] = []
    for m in matches:
        inner = m.group(0)
        inner = re.sub(r"^<think>\s*", "", inner, count=1, flags=re.IGNORECASE)
        inner = re.sub(r"\s*</think>\s*$", "", inner, count=1, flags=re.IGNORECASE)
        if inner.strip():
            reasoning_parts.append(inner.strip())
    answer = _THINK_BLOCK.sub("", s).strip()
    reasoning = "\n".join(reasoning_parts).strip() or None
    return answer, reasoning


class OllamaConnector(LLMConnector):
    """LLM backend via a local or remote Ollama HTTP API (`/api/chat`)."""

    def __init__(self, model: str | None = None) -> None:
        self._model = resolve_model_name("ollama", model)
        self.base_url = _read_base_url()
        self.num_predict = _read_num_predict()
        self.temperature = _read_optional_temperature()
        self.timeout_s = _read_timeout_s()
        self.format_json = _read_format_json()

    @property
    def label(self) -> str:
        return "ollama"

    @property
    def model(self) -> str:
        return self._model

    def extract_overruled_cases(
        self, *, system_prompt: str, user_prompt: str
    ) -> ConnectorResponse:
        url = f"{self.base_url}/api/chat"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
        }
        if self.format_json:
            payload["format"] = "json"
        options: dict[str, Any] = {"num_predict": self.num_predict}
        if self.temperature is not None:
            options["temperature"] = self.temperature
        payload["options"] = options

        data: dict[str, Any]
        for attempt in range(_RETRIES_ON_HTTP_500 + 1):
            resp = requests.post(url, json=payload, timeout=self.timeout_s)
            if resp.status_code == 500 and attempt < _RETRIES_ON_HTTP_500:
                wait_s = 1.0 + attempt
                print(
                    f"[Ollama] HTTP 500 (attempt {attempt + 1}/{_RETRIES_ON_HTTP_500 + 1}), "
                    f"retrying in {wait_s:.0f}s...",
                    flush=True,
                )
                time.sleep(wait_s)
                continue
            resp.raise_for_status()
            parsed = resp.json()
            if not isinstance(parsed, dict):
                raise RuntimeError(f"Ollama returned non-object JSON: {type(parsed)}")
            data = parsed
            break

        message = data.get("message") or {}
        content = ""
        if isinstance(message, dict):
            raw = message.get("content")
            if isinstance(raw, str):
                content = raw.strip()

        answer, reasoning = _split_think_content(content)
        return ConnectorResponse(
            raw_content=answer or content,
            raw_response=data,
            reasoning_content=reasoning,
        )

    def submit_batch(self, requests: list[PreparedLLMRequest]) -> BatchSubmitResult:
        _ = requests
        raise NotImplementedError(
            "Batch mode is not supported for the Ollama connector. Omit --batch or use openai, gemini, or claude."
        )
