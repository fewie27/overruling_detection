"""``directly_overruled`` / ``overruled_cases`` list-of-strings objects (prompt_a … prompt_e).

Parses the raw object and emits each citation classified as ``"overruled"`` (these prompts
do not classify, so the benchmark-equivalent default class is used).
"""

from __future__ import annotations

import json
from typing import Any

from src.models import ClassifiedCitation, ExtractionResult
from src.p03a_evaluation.response_parsers.base import LlmResponseParser
from src.p03a_evaluation.response_parsers.json_lenient import load_json_lenient


def _list_str_field(payload: dict[str, Any], key: str) -> list[str]:
    vals = payload.get(key, [])
    if not isinstance(vals, list):
        raise ValueError(f"JSON field {key!r} must be a list")
    return [str(v).strip() for v in vals if str(v).strip()]


def _explanation_from_payload(payload: dict[str, Any]) -> str | None:
    for key in ("explanation", "explaination"):
        val = payload.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    exps = payload.get("explanations")
    if isinstance(exps, list):
        parts = [str(x).strip() for x in exps if str(x).strip()]
        if parts:
            return "\n".join(parts)
    reasoning = payload.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning.strip()
    return None


def _cite_strings_from_payload(payload: dict[str, Any]) -> list[str]:
    merged: list[str] = []
    if "directly_overruled" in payload:
        merged.extend(_list_str_field(payload, "directly_overruled"))
    legacy = payload.get("overruled_cases", [])
    if isinstance(legacy, list):
        merged.extend(str(v).strip() for v in legacy if str(v).strip())
    return merged


class P1OverruleParser(LlmResponseParser):
    def parse(self, raw_content: str) -> ExtractionResult:
        content = (raw_content or "").strip() or "{}"
        try:
            data = load_json_lenient(content)
        except (json.JSONDecodeError, ValueError):
            return ExtractionResult(classified_citations=[], raw_content=content)
        if not isinstance(data, dict):
            return ExtractionResult(classified_citations=[], raw_content=content)
        try:
            cites = _cite_strings_from_payload(data)
        except ValueError:
            cites = []
        classified = [ClassifiedCitation.make(c, "overruled") for c in cites if c]
        return ExtractionResult(
            classified_citations=classified,
            raw_content=content,
            explanation=_explanation_from_payload(data),
        )
