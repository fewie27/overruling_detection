from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.p01_preparation._01_retrieval.case_providers.base import CaseProvider
from src.models import CaseDocument


class BenchmarkCleanedTextProvider(CaseProvider):
    def __init__(self, benchmark_entry: dict[str, Any]) -> None:
        self.benchmark_entry = benchmark_entry

    @property
    def label(self) -> str:
        return "benchmark_opinion"

    def get_case_by_citation(
        self,
        citation: str,
        *,
        case_law_caption_matcher: Callable[[str], bool] | None = None,
        require_full_casebody: bool = False,
    ) -> CaseDocument:
        cleaned_text = str(self.benchmark_entry.get("cleaned_text_opinion") or "").strip()
        if not cleaned_text:
            raise RuntimeError("Benchmark JSON entry has no non-empty 'cleaned_text_opinion'")

        name = str(self.benchmark_entry.get("case_name") or "").strip()
        cite = str(self.benchmark_entry.get("citation") or "").strip() or citation

        payload = {
            "name": name or cite,
            "name_abbreviation": name or cite,
            "casebody": {"opinions": [{"text": cleaned_text}]},
        }
        return CaseDocument(citation=cite, source_url="benchmark_json", raw_data=payload)
