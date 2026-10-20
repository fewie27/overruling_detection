from dataclasses import dataclass
from typing import Any


@dataclass
class CaseDocument:
    """Retrieved case from a provider. Optional overrides support benchmark JSON metadata."""

    citation: str
    source_url: str
    raw_data: dict[str, Any]
    case_name_override: str | None = None
    opinion_text_override: str | None = None
    retrieval_provider_label: str | None = None
    # "majority_only" = intentionally majority/lead only (e.g. CourtListener);
    # "full_casebody" = full CAP casebody opinions available.
    opinion_scope: str | None = None

    @property
    def case_name(self) -> str:
        if self.case_name_override and self.case_name_override.strip():
            return self.case_name_override.strip()
        return self.raw_data.get("name") or self.raw_data.get("name_abbreviation") or self.citation

    def _casebody_opinion_texts(self) -> list[str]:
        opinions = self.raw_data.get("casebody", {}).get("opinions", [])
        if not isinstance(opinions, list):
            return []
        out: list[str] = []
        for op in opinions:
            if not isinstance(op, dict):
                continue
            text = op.get("text", "")
            if isinstance(text, str) and text.strip():
                out.append(text)
        return out

    @property
    def majority_opinion(self) -> str:
        if self.opinion_text_override is not None and self.opinion_text_override.strip():
            return self.opinion_text_override
        texts = self._casebody_opinion_texts()
        return texts[0] if texts else ""

    @property
    def full_opinion_text(self) -> str:
        """All casebody opinion texts joined (majority + concurrences/dissents when present)."""
        if self.opinion_text_override is not None and self.opinion_text_override.strip():
            return self.opinion_text_override
        return "\n\n".join(self._casebody_opinion_texts())

    def prompt_opinion_text(self, *, full: bool = False) -> str:
        return self.full_opinion_text if full else self.majority_opinion
