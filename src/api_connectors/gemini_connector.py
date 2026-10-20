from __future__ import annotations

import os
from typing import Any

from src.api_connectors.base import BatchStatus, BatchSubmitResult, ConnectorResponse, LLMConnector
from src.config.models import resolve_model_name, warn_if_missing_pricing
from src.models import PreparedLLMRequest


def _extract_text_from_gemini_response(response: Any) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    candidates = getattr(response, "candidates", None) or []
    for c in candidates:
        content = getattr(c, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            part_text = getattr(part, "text", None)
            if isinstance(part_text, str) and part_text.strip():
                return part_text.strip()
    return "{}"


class GeminiConnector(LLMConnector):
    def __init__(self, model: str | None = None) -> None:
        self._model = resolve_model_name("gemini", model)
        warn_if_missing_pricing("gemini", self._model)
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required when using the Gemini connector")
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError(
                "Gemini connector requires `google-genai`. Install it with: pip install google-genai"
            ) from exc
        self._genai = genai
        self.client = genai.Client(api_key=self.api_key)

    @property
    def label(self) -> str:
        return "gemini"

    @property
    def model(self) -> str:
        return self._model

    def extract_overruled_cases(
        self, *, system_prompt: str, user_prompt: str
    ) -> ConnectorResponse:
        prompt = f"{system_prompt}\n\n{user_prompt}"
        response = self.client.models.generate_content(model=self.model, contents=prompt)
        content = _extract_text_from_gemini_response(response)
        return ConnectorResponse(raw_content=content, raw_response=response)

    def submit_batch(self, requests: list[PreparedLLMRequest]) -> BatchSubmitResult:
        inlined_requests = []
        for req in requests:
            prompt = f"{req.system_prompt}\n\n{req.user_prompt}"
            inlined_requests.append(
                {
                    "key": req.request_id,
                    "request": {
                        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                        "config": {"response_mime_type": "application/json"},
                    },
                }
            )
        batch_job = self.client.batches.create(model=self.model, src={"inlined_requests": inlined_requests})
        return BatchSubmitResult(batch_id=batch_job.name)

    def get_batch_status(self, batch_id: str) -> BatchStatus:
        batch_job = self.client.batches.get(name=batch_id)
        state = getattr(getattr(batch_job, "state", None), "name", "JOB_STATE_UNSPECIFIED")
        terminal_states = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}
        return BatchStatus(
            batch_id=batch_id,
            status=str(state),
            is_terminal=state in terminal_states,
            is_success=(state == "JOB_STATE_SUCCEEDED"),
        )

    def collect_batch_results(self, batch_id: str) -> dict[str, ConnectorResponse]:
        batch_job = self.client.batches.get(name=batch_id)
        inlined_responses = getattr(getattr(batch_job, "dest", None), "inlined_responses", None) or []
        results: dict[str, ConnectorResponse] = {}
        for idx, inline_response in enumerate(inlined_responses, start=1):
            request_id = getattr(inline_response, "key", None) or f"row-{idx}"
            response_obj = getattr(inline_response, "response", None)
            content = _extract_text_from_gemini_response(response_obj) if response_obj is not None else "{}"
            rid = str(request_id)
            results[rid] = ConnectorResponse(raw_content=content, raw_response=response_obj)
        return results
