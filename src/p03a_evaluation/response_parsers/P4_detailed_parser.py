"""``overruled_cases`` as objects with ``case`` + ``classification`` (prompt_g / prompt_h JSON schema)."""

from __future__ import annotations

import json
from typing import Any

from src.models import ClassifiedCitation, ExtractionResult
from src.p03a_evaluation.citation_treatment_labels import to_benchmark_class
from src.p03a_evaluation.response_parsers.base import LlmResponseParser
from src.p03a_evaluation.response_parsers.json_lenient import load_json_lenient


def _explanation_from_payload(payload: dict[str, Any]) -> str | None:
    lq = payload.get("legal_question")
    if isinstance(lq, str) and lq.strip():
        return lq.strip()
    return None


def _classified_from_payload(payload: dict[str, Any]) -> list[ClassifiedCitation]:
    raw_list = payload.get("overruled_cases", []) or []
    if not isinstance(raw_list, list):
        raise ValueError("JSON field 'overruled_cases' must be a list")
    out: list[ClassifiedCitation] = []
    for item in raw_list:
        if not isinstance(item, dict):
            continue
        case = item.get("case")
        cls = item.get("classification")
        if not isinstance(case, str) or not case.strip():
            continue
        if not isinstance(cls, str) or not cls.strip():
            continue
        out.append(ClassifiedCitation.make(case, to_benchmark_class(cls)))
    return out


class P4DetailedParser(LlmResponseParser):
    def parse(self, raw_content: str) -> ExtractionResult:
        content = (raw_content or "").strip() or "{}"
        try:
            data = load_json_lenient(content)
        except (json.JSONDecodeError, ValueError):
            return ExtractionResult(classified_citations=[], raw_content=content)
        if not isinstance(data, dict):
            return ExtractionResult(classified_citations=[], raw_content=content)
        try:
            classified = _classified_from_payload(data)
        except ValueError:
            classified = []
        return ExtractionResult(
            classified_citations=classified,
            raw_content=content,
            explanation=_explanation_from_payload(data),
        )
