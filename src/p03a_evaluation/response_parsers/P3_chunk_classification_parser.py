"""One JSON object per call: ``classification`` + ``cited_citation`` (chunk prompts)."""

from __future__ import annotations

import json
from typing import Any

from src.models import ClassifiedCitation, ExtractionResult
from src.p03a_evaluation.citation_treatment_labels import (
    is_valid_citation_treatment_label,
    to_benchmark_class,
)
from src.p03a_evaluation.response_parsers.base import LlmResponseParser
from src.p03a_evaluation.response_parsers.json_lenient import load_json_lenient


def _explanation_from_payload(payload: dict[str, Any]) -> str | None:
    for key in ("explanation", "explaination"):
        val = payload.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    reasoning = payload.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning.strip()
    return None


class P3ChunkClassificationParser(LlmResponseParser):
    def parse(self, raw_content: str) -> ExtractionResult:
        text = (raw_content or "").strip() or "{}"
        try:
            data = load_json_lenient(text)
        except (json.JSONDecodeError, ValueError):
            return ExtractionResult(classified_citations=[], raw_content=text)
        if not isinstance(data, dict):
            return ExtractionResult(classified_citations=[], raw_content=text)

        explanation = _explanation_from_payload(data)
        cls_raw = data.get("classification")
        cite_raw = data.get("cited_citation") or data.get("citation") or data.get("case_citation") or ""
        cite_s = str(cite_raw).strip()

        classified: list[ClassifiedCitation] = []
        if isinstance(cls_raw, str) and cls_raw.strip() and cite_s:
            key = cls_raw.strip().lower()
            if is_valid_citation_treatment_label(key):
                classified.append(ClassifiedCitation.make(cite_s, to_benchmark_class(key)))

        return ExtractionResult(
            classified_citations=classified,
            raw_content=text,
            explanation=explanation,
        )
