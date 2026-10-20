from __future__ import annotations

import re
from dataclasses import dataclass

# Loose parse for a cite substring (volume + reporter + first page).
_CITE_CORE = re.compile(
    r"(\d+)\s+([A-Za-z][A-Za-z0-9.\s]*?)\s+(?:\([A-Za-z0-9.\s]+\)\s+)?(\d+)\b"
)


@dataclass(frozen=True)
class CitationSpan:
    """One regexp match: character span in the opinion and the cite text as it appeared."""

    start: int
    end: int
    surface: str


def normalize_reporter_token(reporter: str) -> str:
    """Group key for reporter: strip, remove spaces and dots, lower case."""
    t = reporter.strip().lower()
    t = re.sub(r"\s+", "", t)
    return t.replace(".", "")


def parse_volume_reporter_page(surface: str) -> tuple[str, str, str] | None:
    """Extract volume, reporter substring, first page from a matched cite string, or None."""
    s = (surface or "").strip()
    if not s:
        return None
    m = _CITE_CORE.search(s)
    if not m:
        return None
    return m.group(1), m.group(2).strip(), m.group(3)


def is_us_reporter_cite(surface: str) -> bool:
    """True iff the parsed reporter normalizes to ``us`` (U.S. Supreme Court reporter)."""
    p = parse_volume_reporter_page(surface)
    if p is None:
        return False
    _, rep, _ = p
    return normalize_reporter_token(rep) == "us"


def grouping_key_for_cite(surface: str) -> str | None:
    """
    Stable key: same volume + normalized reporter + page groups surface variants
    (e.g. ``530 U. S. 793`` vs ``530 U.S. 793``).
    """
    p = parse_volume_reporter_page(surface)
    if p is None:
        return None
    vol, rep, page = p
    return f"{vol}|{normalize_reporter_token(rep)}|{page}"


def group_us_reporter_citation_spans(
    text: str,
    pattern: re.Pattern[str],
) -> dict[str, list[CitationSpan]]:
    """
    Group regexp matches by (volume, normalized reporter, page). Only includes cites whose
    reporter normalizes to ``us``. Prompts use each match's *surface* string as found in text.
    """
    groups: dict[str, list[CitationSpan]] = {}
    for m in pattern.finditer(text):
        surface = m.group(0).strip()
        if not surface:
            continue
        if not is_us_reporter_cite(surface):
            continue
        key = grouping_key_for_cite(surface)
        if key is None:
            continue
        span = CitationSpan(start=m.start(), end=m.end(), surface=surface)
        groups.setdefault(key, []).append(span)
    return groups
