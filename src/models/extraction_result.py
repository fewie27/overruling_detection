from dataclasses import dataclass, field

from src.models.classified_citation import ClassifiedCitation, dedupe_classified


@dataclass
class ExtractionResult:
    """Normalized LLM output: a list of citations with their treatment-class label."""

    classified_citations: list[ClassifiedCitation]
    raw_content: str
    explanation: str | None = None

    def __post_init__(self) -> None:
        self.classified_citations = dedupe_classified(self.classified_citations)

    @property
    def cite_strings(self) -> list[str]:
        """Convenience: just the citation strings (de-duplicated, preserves order)."""
        out: list[str] = []
        seen: set[str] = set()
        for c in self.classified_citations:
            if c.cite and c.cite not in seen:
                seen.add(c.cite)
                out.append(c.cite)
        return out
