from __future__ import annotations

import os
import re
from collections.abc import Callable
from typing import Any

import requests
from markdownify import markdownify

from src.p01_preparation._01_retrieval.case_providers.base import CaseProvider
from src.p01_preparation._01_retrieval.benchmark import case_matches_retrieved_caption
from src.p01_preparation._01_retrieval.citation_parser import (
    normalize_citation_for_lookup,
    parse_citation_lookup_parts,
)
from src.models import CaseDocument

_CITATION_LOOKUP_URL = "https://www.courtlistener.com/api/rest/v4/citation-lookup/"
_OPINION_BASE = "https://www.courtlistener.com/api/rest/v4/opinions"

_OPINION_TYPE_START_MARKERS: dict[str, tuple[str, ...]] = {
    "majority": (
        '<opinion type="majority">',
        "<opinion type='majority'>",
    ),
    "lead": (
        '<opinion type="lead">',
        "<opinion type='lead'>",
    ),
}
_PRIMARY_OPINION_TYPES: tuple[str, ...] = ("majority", "lead")

_ORDER_LIKE_MARKERS = (
    "motion of petitioner for leave to proceed in forma pauperis granted",
    "petition for writ of certiorari",
    "granted limited to the following question",
)


def _reporter_key(reporter: str) -> str:
    return (reporter or "").strip().lower().replace(".", "").replace(" ", "")


def _html_body_start(html: str) -> str:
    t = (html or "").lstrip("\ufeff \t\n\r")
    if t.startswith("<?xml"):
        close = t.find("?>")
        if close == -1:
            return ""
        t = t[close + 2 :].lstrip("\ufeff \n\r\t")
    return t


def html_leading_document_has_opinion_type(html: str, opinion_type: str) -> bool:
    """True when HTML starts with ``<opinion type=\"{opinion_type}\">`` (after optional XML declaration)."""
    markers = _OPINION_TYPE_START_MARKERS.get(opinion_type)
    if not markers:
        return False
    return _html_body_start(html).startswith(markers)


def html_leading_document_is_majority_opinion(html: str) -> bool:
    return html_leading_document_has_opinion_type(html, "majority")


def _opinion_type_from_html(html: str) -> str | None:
    """Parse ``type`` from a leading ``<opinion type=\"...\">`` tag, if present."""
    body = _html_body_start(html)
    m = re.match(r"<opinion\s+type=['\"]([^'\"]+)['\"]\s*>", body, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip().lower() or None
    return None


def _looks_like_order_not_substantive_opinion(markdown_text: str) -> bool:
    """
    Guard against short cert-order snippets that can still be wrapped in a majority tag.
    These are not useful for benchmark analysis and should trigger fallback retrieval.
    """
    text = (markdown_text or "").strip()
    if not text:
        return True
    norm = " ".join(text.lower().split())
    if len(norm) < 1600 and any(marker in norm for marker in _ORDER_LIKE_MARKERS):
        return True
    return False


class CourtListenerProvider(CaseProvider):
    """Fetch opinion HTML via citation lookup + opinions API (CourtListener REST v4).

    Default: majority/lead sub-opinion only. With ``require_full_casebody=True``, every
    cluster ``sub_opinion`` is fetched (majority first, then concurrence/dissent/etc.).
    """

    def __init__(self, *, api_token: str | None = None, timeout_seconds: int = 60) -> None:
        token = (api_token if api_token is not None else os.getenv("COURTLISTENER_API_TOKEN", "")).strip()
        if not token:
            raise RuntimeError(
                "CourtListener API token missing: set COURTLISTENER_API_TOKEN in the environment or pass api_token=."
            )
        self._token = token
        self.timeout_seconds = timeout_seconds
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Token {self._token}",
                "Accept": "application/json",
            }
        )

    @property
    def label(self) -> str:
        return "court_listener"

    def _citation_lookup_response(self, citation: str) -> list[Any]:
        parts = parse_citation_lookup_parts(citation)
        if parts is None:
            raise ValueError(f"Could not parse citation for CourtListener lookup: {citation}")
        volume, reporter, page = parts
        r = self._session.post(
            _CITATION_LOOKUP_URL,
            data={"volume": volume, "reporter": reporter, "page": page},
            timeout=self.timeout_seconds,
            headers={"Content-Type": None},
        )
        if r.status_code != 200:
            raise RuntimeError(
                f"CourtListener citation-lookup failed ({r.status_code}): {r.text[:500]}"
            )
        data = r.json()
        if not isinstance(data, list) or not data:
            raise RuntimeError(f"CourtListener citation-lookup returned no results for: {citation}")
        return data

    @staticmethod
    def _flatten_cluster_candidates(
        citation: str, data: list[Any]
    ) -> list[tuple[dict[str, Any], str]]:
        out: list[tuple[dict[str, Any], str]] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            resolved = (item.get("citation") or citation).strip()
            for cluster in item.get("clusters") or []:
                if isinstance(cluster, dict):
                    out.append((cluster, resolved))
        if not out:
            raise RuntimeError(f"CourtListener citation-lookup has no clusters for: {citation}")
        return out

    def _select_cluster_for_citation(
        self, citation: str, data: list[Any]
    ) -> tuple[dict[str, Any], str]:
        """
        When citation-lookup returns multiple clusters (same reporter volume/page),
        prefer the cluster whose caption matches the parties named in ``citation``.
        """
        candidates = self._flatten_cluster_candidates(citation, data)
        hint = normalize_citation_for_lookup(citation)
        for cluster, resolved in candidates:
            name = (cluster.get("case_name") or cluster.get("case_name_full") or "").strip()
            if name and hint and case_matches_retrieved_caption(hint, name):
                return cluster, resolved
        return candidates[0][0], candidates[0][1]

    def _lookup_first_cluster(self, citation: str) -> tuple[dict[str, Any], str]:
        data = self._citation_lookup_response(citation)
        return self._select_cluster_for_citation(citation, data)

    @staticmethod
    def _opinion_id_from_uri(uri: str) -> str:
        return uri.rstrip("/").split("/")[-1]

    def _fetch_opinion_fields(self, opinion_uri: str, fields: str) -> dict[str, Any]:
        oid = self._opinion_id_from_uri(opinion_uri)
        url = f"{_OPINION_BASE}/{oid}/"
        r = self._session.get(
            url,
            params={"fields": fields},
            timeout=self.timeout_seconds,
            headers={"Content-Type": None},
        )
        if r.status_code != 200:
            raise RuntimeError(f"CourtListener opinion GET failed ({r.status_code}) for {url}: {r.text[:300]}")
        return r.json()

    def _load_sub_opinion_htmls(self, sub_opinion_uris: list[str]) -> list[tuple[str, str]]:
        """Fetch non-empty ``html_with_citations`` for each sub-opinion URI (cluster order)."""
        fields = "html_with_citations"
        loaded: list[tuple[str, str]] = []
        for uri in sub_opinion_uris:
            try:
                payload = self._fetch_opinion_fields(uri, fields)
            except RuntimeError:
                continue
            html = (payload.get("html_with_citations") or "").strip()
            if html:
                loaded.append((html, uri))
        return loaded

    def _pick_majority_html(self, sub_opinion_uris: list[str]) -> tuple[str, str]:
        """Return HTML whose leading content is a majority or lead <opinion> document.

        When the cluster has only a single sub-opinion, treat it as the majority
        opinion regardless of its <opinion type=...> tag.
        """
        loaded = self._load_sub_opinion_htmls(sub_opinion_uris)

        if len(sub_opinion_uris) == 1 and loaded:
            return loaded[0]

        for opinion_type in _PRIMARY_OPINION_TYPES:
            for html, uri in loaded:
                if html_leading_document_has_opinion_type(html, opinion_type):
                    return html, uri

        raise RuntimeError(
            "CourtListener: no sub-opinion returned majority or lead opinion HTML "
            '(expected body starting with <opinion type="majority"> or <opinion type="lead">).'
        )

    def _all_opinions_markdown(
        self, sub_opinion_uris: list[str]
    ) -> tuple[list[dict[str, Any]], str]:
        """
        Convert every sub-opinion to markdown.

        Majority/lead is placed first (same selection as :meth:`_pick_majority_html`) so
        ``CaseDocument.majority_opinion`` stays consistent with the majority-only path.
        Returns ``(opinions, primary_source_uri)``.
        """
        loaded = self._load_sub_opinion_htmls(sub_opinion_uris)
        if not loaded:
            raise RuntimeError("CourtListener: no sub-opinion returned HTML with citations.")

        if len(sub_opinion_uris) == 1:
            primary_html, primary_uri = loaded[0]
        else:
            primary_html = None
            primary_uri = None
            for opinion_type in _PRIMARY_OPINION_TYPES:
                for html, uri in loaded:
                    if html_leading_document_has_opinion_type(html, opinion_type):
                        primary_html, primary_uri = html, uri
                        break
                if primary_uri is not None:
                    break
            if primary_uri is None or primary_html is None:
                raise RuntimeError(
                    "CourtListener: no sub-opinion returned majority or lead opinion HTML "
                    '(expected body starting with <opinion type="majority"> or <opinion type="lead">).'
                )

        ordered: list[tuple[str, str]] = [(primary_html, primary_uri)]
        ordered.extend((html, uri) for html, uri in loaded if uri != primary_uri)

        opinions: list[dict[str, Any]] = []
        for html, _uri in ordered:
            md = self._majority_html_to_markdown(html)
            if not md:
                continue
            entry: dict[str, Any] = {"text": md}
            op_type = _opinion_type_from_html(html)
            if op_type:
                entry["type"] = op_type
            opinions.append(entry)

        if not opinions:
            raise RuntimeError("CourtListener: markdown conversion produced empty opinions.")
        return opinions, primary_uri

    @staticmethod
    def _majority_html_to_markdown(html: str) -> str:
        """
        Convert opinion HTML to markdown to reduce prompt token usage.
        The conversion happens only after majority-opinion validation succeeded
        (majority-only path) or after opinions were loaded (full-casebody path).
        """
        md = markdownify(html or "", heading_style="ATX")
        return md.strip()

    def best_us_reporter_citation_for_case_law(self, citation: str) -> str | None:
        """Try to derive an official U.S.-reporter cite from CourtListener citation-lookup metadata."""
        try:
            data = self._citation_lookup_response(citation)
        except Exception:
            return None
        first = data[0] if data else None
        if not isinstance(first, dict):
            return None
        clusters = first.get("clusters") or []
        if not clusters or not isinstance(clusters[0], dict):
            return None
        cluster = clusters[0]
        resolved = (first.get("citation") or citation).strip()

        cites = list(cluster.get("citations") or [])
        for c in cites:
            rep = _reporter_key(str(c.get("reporter") or ""))
            vol = str(c.get("volume") or "").strip()
            page = str(c.get("page") or "").strip()
            if rep == "us" and vol and page:
                return f"{vol} U.S. {page}"

        if "U.S." in resolved:
            return resolved
        return None

    def get_case_by_citation(
        self,
        citation: str,
        *,
        case_law_caption_matcher: Callable[[str], bool] | None = None,
        require_full_casebody: bool = False,
    ) -> CaseDocument:
        cluster, cite_resolved = self._lookup_first_cluster(citation)
        case_name = (cluster.get("case_name") or cluster.get("case_name_full") or "").strip() or cite_resolved
        uris: list[str] = list(cluster.get("sub_opinions") or [])
        if not uris:
            raise RuntimeError("CourtListener cluster has empty sub_opinions.")

        if require_full_casebody:
            opinions, src_uri = self._all_opinions_markdown(uris)
            primary_markdown = str(opinions[0].get("text") or "")
            if _looks_like_order_not_substantive_opinion(primary_markdown):
                raise RuntimeError(
                    "CourtListener: primary-opinion body appears to be a short order/disposition, "
                    "not a substantive opinion."
                )
            payload = {
                "name": case_name,
                "name_abbreviation": case_name,
                "casebody": {"opinions": opinions},
                "courtlistener": {
                    "cluster_id": cluster.get("id"),
                    "opinion_uri": src_uri,
                    "sub_opinion_count": len(opinions),
                },
            }
            return CaseDocument(
                citation=cite_resolved,
                source_url=src_uri or _CITATION_LOOKUP_URL,
                raw_data=payload,
            )

        html, src_uri = self._pick_majority_html(uris)
        opinion_markdown = self._majority_html_to_markdown(html)
        if not opinion_markdown:
            raise RuntimeError("CourtListener: markdown conversion produced empty majority opinion.")
        if _looks_like_order_not_substantive_opinion(opinion_markdown):
            raise RuntimeError(
                "CourtListener: primary-opinion body appears to be a short order/disposition, not a substantive opinion."
            )
        payload = {
            "name": case_name,
            "name_abbreviation": case_name,
            "casebody": {"opinions": [{"text": opinion_markdown}]},
            "courtlistener": {"cluster_id": cluster.get("id"), "opinion_uri": src_uri},
        }
        return CaseDocument(
            citation=cite_resolved,
            source_url=src_uri or _CITATION_LOOKUP_URL,
            raw_data=payload,
        )
