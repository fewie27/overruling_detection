"""Full-opinion citation list: ``{\"citations\": [{\"cite\", \"class\"}, ...] }``.

Every classified citation is forwarded with its label mapped onto the benchmark vocabulary,
so a generic-class benchmark can match the model's labels (e.g. ``direct_overruling`` ->
``overruled``).
"""

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


def _class_from_item(item: dict[str, Any]) -> str:
    for k in ("class", "classification"):
        v = item.get(k)
        if v is not None and str(v).strip():
            return str(v).strip().lower()
    return ""


def _cite_from_item(item: dict[str, Any]) -> str:
    for k in ("cite", "cited_citation", "citation", "case_citation"):
        v = item.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return ""


class P2ClassificationParser(LlmResponseParser):
    def parse(self, raw_content: str) -> ExtractionResult:
        text = (raw_content or "").strip() or "{}"
        try:
            data = load_json_lenient(text)
        except (json.JSONDecodeError, ValueError):
            return ExtractionResult(classified_citations=[], raw_content=text)

        items: list[Any]
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict) and isinstance(data.get("citations"), list):
            items = data["citations"]
        else:
            return ExtractionResult(classified_citations=[], raw_content=text)

        classified: list[ClassifiedCitation] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            label = _class_from_item(it)
            cite = _cite_from_item(it)
            if not label or not cite or not is_valid_citation_treatment_label(label):
                continue
            classified.append(ClassifiedCitation.make(cite, to_benchmark_class(label)))

        return ExtractionResult(classified_citations=classified, raw_content=text)
