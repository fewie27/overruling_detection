import re
from typing import Optional

from src.models import CitationParts

# Optional ``(N How.)`` / ``(N Pet.)`` etc. between reporter and page (dual U.S. citations).
_CITATION_PARTS_REGEX = re.compile(
    r"(\d+)\s+([A-Z][A-Za-z0-9\.\s]{1,8}?)(?:\s+\([A-Za-z0-9., ]+\)\s+|\s+|,\sat\s)(\d+)"
)


def sanitize_reporter(reporter: str) -> str:
    return reporter.strip().lower().replace(".", "").replace(" ", "")


def parse_citation_parts(text: str) -> Optional[CitationParts]:
    if not text:
        return None
    m = _CITATION_PARTS_REGEX.search(normalize_citation_for_lookup(text))
    if not m:
        return None
    volume, reporter, first_page = m.group(1), m.group(2), m.group(3)
    return CitationParts(volume=volume, reporter=sanitize_reporter(reporter), first_page=first_page)


def normalize_citation_for_lookup(text: str) -> str:
    """Strip parenthetical fragments that break volume/reporter/page parsing or CL lookup."""
    if not text:
        return ""
    t = text.replace("\xa0", " ")
    t = re.sub(r"\s*\(\s*Mem(?:orandum)?\s*\)\s*", " ", t, flags=re.IGNORECASE)
    # Federal Appendix is often printed “App'x” / “App’x”; the citation regex does not allow `'`
    # inside the reporter token, and CAP / CourtListener expect “Appx” spelling.
    t = re.sub(r"(?i)App['\u2019]x\b", "Appx", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def parse_citation_lookup_parts(text: str) -> Optional[tuple[str, str, str]]:
    """Volume, reporter string as printed in the citation, and first page (for CourtListener-style APIs)."""
    if not text:
        return None
    m = _CITATION_PARTS_REGEX.search(normalize_citation_for_lookup(text))
    if not m:
        return None
    volume, reporter_raw, first_page = m.group(1), m.group(2).strip(), m.group(3)
    return (volume, reporter_raw, first_page)
