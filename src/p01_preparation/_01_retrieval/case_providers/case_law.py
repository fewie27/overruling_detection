from __future__ import annotations

from collections.abc import Callable
from typing import Final, Literal

import requests

from src.p01_preparation._01_retrieval.case_providers.base import CaseProvider
from src.p01_preparation._01_retrieval.citation_parser import parse_citation_parts, sanitize_reporter
from src.models import CaseDocument

# static.case.law uses per-page JSON slots ``…/cases/{page}-{nn}.json``; stop after the first 404.
_MAX_CASE_SLOTS_PER_PAGE = 99

# Reporters whose sanitized key maps to a different path segment on static.case.law
# (see https://static.case.law/ — e.g. Fed.Appx. → f-appx, not fedappx).
_REPORTER_SLUGS: Final[dict[str, tuple[str, ...]]] = {
    # “F. Appx.” / “F. App'x” (no “Fed.”) sanitizes to fappx — same CAP volume tree as Fed. Appx.
    "fappx": ("f-appx",),
    "fedappx": ("f-appx",),
    "fsupp": ("f-supp",),
    "fsupp2d": ("f-supp-2d",),
    "fsupp3d": ("f-supp-3d",),
}


def _static_reporter_slugs(sanitized_reporter: str) -> tuple[str, ...]:
    """Path segment(s) for ``https://static.case.law/{slug}/vol/cases/…`` (try in order)."""
    r = sanitize_reporter(sanitized_reporter)
    return _REPORTER_SLUGS.get(r, (r,))


class CaseLawStaticProvider(CaseProvider):
    def __init__(self, timeout_seconds: int = 30) -> None:
        self.timeout_seconds = timeout_seconds

    @property
    def label(self) -> str:
        return "case_law_project"

    def _fetch_slot(self, url: str) -> tuple[Literal["gone", "hit", "miss"], dict | None]:
        r = requests.get(url, timeout=self.timeout_seconds)
        if r.status_code == 404:
            return "gone", None
        if r.status_code == 200:
            return "hit", r.json()
        return "miss", None

    def get_case_by_citation(
        self,
        citation: str,
        *,
        case_law_caption_matcher: Callable[[str], bool] | None = None,
        require_full_casebody: bool = False,
    ) -> CaseDocument:
        parts = parse_citation_parts(citation)
        if parts is None:
            raise ValueError(f"Could not parse citation parts from: {citation}")

        page_num = int(parts.first_page)
        page_formats = [f"{page_num:04d}", str(page_num)]

        for slug in _static_reporter_slugs(parts.reporter):
            for page_fmt in page_formats:
                n = 1
                while n <= _MAX_CASE_SLOTS_PER_PAGE:
                    suffix = f"{n:02d}"
                    url = f"https://static.case.law/{slug}/{parts.volume}/cases/{page_fmt}-{suffix}.json"
                    status, payload = self._fetch_slot(url)
                    if status == "gone":
                        break
                    if status == "hit" and payload is not None:
                        doc = CaseDocument(citation=citation, source_url=url, raw_data=payload)
                        if case_law_caption_matcher is None or case_law_caption_matcher(doc.case_name):
                            return doc
                    n += 1

        raise RuntimeError(f"Could not retrieve case from case.law for citation: {citation}")
