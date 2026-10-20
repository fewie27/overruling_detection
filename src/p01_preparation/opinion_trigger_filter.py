"""Pre-LLM opinion text filter: keep cases whose majority opinion hints at overruling treatment."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from src.models import CaseDocument
from src.models.case_trace import PipelineInputRow

# Substrings checked after :func:`normalize_opinion_text` (casefold, hyphen-tolerant).
# Stems cover simple suffixes via compact match; keep forms that drop the stem -e.
DEFAULT_OVERRULING_TRIGGER_PHRASES: tuple[str, ...] = (
    "overrule",
    "overruling",
    "abrogate",
    "abrogating",
    "abrogation",
    "cannot stand",
    "can not stand",
    "no longer be followed",
    "should not be followed",
    "should not longer be followed",
    "no longer good law",
    "no longer controlling",
    "no longer the law",
    "ceases to be good law",
    "cease to be good law",
    "bad law",
    "disapprove",
    "holding is abandoned",
    "abandoned the rule",
    "abandon the rule",
    "no longer authoritative",
    "no longer valid",
    "superseded by",
)

_HYPHEN_CHARS = "-\u2010\u2011\u2012\u2013\u2014\u2212"
_APOSTROPHE_CHARS = "'\u2019\u2018\u201b\u201a"


def normalize_opinion_text(text: str) -> str:
    """Lowercase text and fold hyphens/apostrophes so ``over-ruled`` matches ``overruled``."""
    t = (text or "").casefold()
    for ch in _APOSTROPHE_CHARS:
        t = t.replace(ch, "")
    for ch in _HYPHEN_CHARS:
        t = t.replace(ch, "")
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def opinion_has_overruling_trigger(
    opinion_text: str,
    *,
    phrases: Iterable[str] | None = None,
) -> bool:
    if not (opinion_text or "").strip():
        return False
    norm = normalize_opinion_text(opinion_text)
    if not norm:
        return False
    compact = norm.replace(" ", "")
    for raw in phrases or DEFAULT_OVERRULING_TRIGGER_PHRASES:
        needle = normalize_opinion_text(raw)
        if not needle:
            continue
        if " " in needle:
            parts = needle.split()
            pattern = r"\b" + r"\s+".join(re.escape(p) for p in parts) + r"\b"
            if re.search(pattern, norm):
                return True
        elif needle in compact or re.search(rf"\b{re.escape(needle)}\b", norm):
            return True
    return False


@dataclass(frozen=True)
class OpinionTriggerFilterResult:
    kept: list[tuple[PipelineInputRow, CaseDocument]]
    skipped_row_nos: list[int]

    @property
    def skipped_count(self) -> int:
        return len(self.skipped_row_nos)


def filter_pairs_by_opinion_triggers(
    pairs: Iterable[tuple[PipelineInputRow, CaseDocument]],
    *,
    enabled: bool,
    phrases: Iterable[str] | None = None,
) -> OpinionTriggerFilterResult:
    """Drop cases whose majority opinion lacks any configured trigger phrase."""
    if not enabled:
        kept = list(pairs)
        return OpinionTriggerFilterResult(kept=kept, skipped_row_nos=[])

    kept: list[tuple[PipelineInputRow, CaseDocument]] = []
    skipped_row_nos: list[int] = []
    for row, case_doc in pairs:
        if opinion_has_overruling_trigger(case_doc.majority_opinion, phrases=phrases):
            kept.append((row, case_doc))
        else:
            skipped_row_nos.append(row.row_no)
    return OpinionTriggerFilterResult(kept=kept, skipped_row_nos=skipped_row_nos)
