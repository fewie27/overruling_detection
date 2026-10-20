from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.models.classified_citation import ClassifiedCitation


@dataclass
class CaseTraceMetadata:
    """Case identity fields carried through retrieval, LLM batch jobs, and extract output."""

    case_law_id: int | None = None
    decision_date: str | None = None
    volume: str | None = None
    reporter: str | None = None
    page: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> CaseTraceMetadata:
        if not data:
            return cls()
        return cls(
            case_law_id=data.get("case_law_id"),
            decision_date=data.get("decision_date"),
            volume=data.get("volume"),
            reporter=data.get("reporter"),
            page=data.get("page"),
        )


@dataclass
class PipelineInputRow:
    """Unified pipeline input for evaluate (benchmark) and production (volume) runs."""

    row_no: int
    case_citation: str
    expected_classifications: list[ClassifiedCitation]
    display_case_name: str
    lookup_citation: str
    trace: CaseTraceMetadata = field(default_factory=CaseTraceMetadata)
    case_name_override: str | None = None
    opinion_text_override: str | None = None
    raw_entry: dict[str, Any] | None = None
