from __future__ import annotations

import io
import json
import re
import zipfile
from dataclasses import dataclass, replace
from typing import Iterator

import requests

from src.models import CaseDocument
from src.models.case_trace import CaseTraceMetadata, PipelineInputRow
from src.p01_preparation._01_retrieval.case_providers.case_law import _static_reporter_slugs
from src.p01_preparation._01_retrieval.citation_parser import sanitize_reporter

_ZIP_MEMBER_RE = re.compile(r"^json/(\d+)-\d+\.json$")
_VOLUME_RANGE_RE = re.compile(r"^(\d+)\s*-\s*(\d+)$")


def parse_volume_spec(text: str) -> list[str]:
    """
    Parse a volume argument: one volume (``542``), comma list (``540,541,542``),
    inclusive range (``540-545``), or a mix (``540-542,550``).
    """
    t = (text or "").strip()
    if not t:
        raise ValueError("Volume spec is empty.")
    out: list[str] = []
    seen: set[str] = set()
    for part in (p.strip() for p in t.split(",")):
        if not part:
            raise ValueError("Empty entry in volume list (check for stray commas).")
        range_m = _VOLUME_RANGE_RE.fullmatch(part)
        if range_m:
            lo, hi = int(range_m.group(1)), int(range_m.group(2))
            if lo > hi:
                raise ValueError(f"Volume range start > end: {part!r}")
            for n in range(lo, hi + 1):
                vol = str(n)
                if vol not in seen:
                    seen.add(vol)
                    out.append(vol)
            continue
        if not part.isdigit():
            raise ValueError(f"Volume must be numeric or a range (e.g. 540-545): {part!r}")
        if part not in seen:
            seen.add(part)
            out.append(part)
    if not out:
        raise ValueError("No volumes parsed from volume spec.")
    return out


@dataclass(frozen=True)
class VolumeRetrievalConfig:
    reporter: str
    volume: str
    min_page: int | None = None
    max_page: int | None = None
    timeout_seconds: int = 120


def reporter_slug(reporter: str) -> str:
    """Map a reporter label to a static.case.law path segment (e.g. ``U.S.`` → ``us``, ``F. Appx.`` → ``f-appx``)."""
    return _static_reporter_slugs(sanitize_reporter(reporter))[0]


def volume_zip_url(reporter: str, volume: str) -> str:
    """CAP volume ZIP: ``https://static.case.law/{slug}/{volume}.zip``."""
    return f"https://static.case.law/{reporter_slug(reporter)}/{volume.strip()}.zip"


def _page_in_range(page: int, *, min_page: int | None, max_page: int | None) -> bool:
    if min_page is not None and page < min_page:
        return False
    if max_page is not None and page > max_page:
        return False
    return True


def _official_citation(raw: dict, *, volume: str, reporter: str, first_page: str) -> str:
    citations = raw.get("citations") or []
    if isinstance(citations, list):
        for item in citations:
            if isinstance(item, dict) and item.get("type") == "official":
                cite = str(item.get("cite") or "").strip()
                if cite:
                    return cite
    return f"{volume} {reporter.strip()} {first_page}"


def _trace_from_case(raw: dict, *, reporter: str, volume: str) -> CaseTraceMetadata:
    first_page = str(raw.get("first_page") or "").strip()
    case_id = raw.get("id")
    parsed_id = int(case_id) if case_id is not None else None
    decision_date = raw.get("decision_date")
    return CaseTraceMetadata(
        case_law_id=parsed_id,
        decision_date=str(decision_date).strip() if decision_date else None,
        volume=str(volume).strip(),
        reporter=reporter.strip(),
        page=first_page or None,
    )


def download_volume_zip(config: VolumeRetrievalConfig) -> bytes:
    url = volume_zip_url(config.reporter, config.volume)
    response = requests.get(url, timeout=config.timeout_seconds)
    if response.status_code != 200:
        raise RuntimeError(
            f"Could not download volume ZIP from {url!r} (HTTP {response.status_code}). "
            f"Check reporter and volume."
        )
    return response.content


def iter_volume_cases(config: VolumeRetrievalConfig) -> Iterator[tuple[PipelineInputRow, CaseDocument]]:
    """Download a reporter volume ZIP and yield pipeline rows with retrieved case documents."""
    payload = download_volume_zip(config)
    slug = reporter_slug(config.reporter)
    volume = str(config.volume).strip()
    row_no = 0

    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        members = sorted(
            name
            for name in zf.namelist()
            if _ZIP_MEMBER_RE.match(name) and not name.endswith("/")
        )
        for member in members:
            m = _ZIP_MEMBER_RE.match(member)
            if m is None:
                continue
            page = int(m.group(1))
            if not _page_in_range(page, min_page=config.min_page, max_page=config.max_page):
                continue
            raw = json.loads(zf.read(member))
            if not isinstance(raw, dict):
                continue
            first_page = str(raw.get("first_page") or page).strip()
            trace = _trace_from_case(raw, reporter=config.reporter, volume=volume)
            if not trace.page:
                trace.page = first_page
            citation = _official_citation(
                raw, volume=volume, reporter=config.reporter, first_page=first_page
            )
            source_url = f"https://static.case.law/{slug}/{volume}/cases/{member.split('/')[-1]}"
            case_doc = CaseDocument(
                citation=citation,
                source_url=source_url,
                raw_data=raw,
                retrieval_provider_label="case_law_volume",
            )
            display_name = (
                str(raw.get("name_abbreviation") or raw.get("name") or citation).strip()
            )
            row_no += 1
            row = PipelineInputRow(
                row_no=row_no,
                case_citation=citation,
                expected_classifications=[],
                display_case_name=display_name,
                lookup_citation=citation,
                trace=trace,
                raw_entry=raw,
            )
            yield row, case_doc


def load_volume_cases(config: VolumeRetrievalConfig) -> list[tuple[PipelineInputRow, CaseDocument]]:
    return list(iter_volume_cases(config))


def load_reporter_volume_cases(
    *,
    reporter: str,
    volumes: list[str],
    min_page: int | None = None,
    max_page: int | None = None,
    timeout_seconds: int = 120,
) -> list[tuple[PipelineInputRow, CaseDocument]]:
    """
    Load one or more reporter volumes. ``min_page`` / ``max_page`` apply only when
    exactly one volume is requested; they are ignored for multi-volume runs.
    """
    if not volumes:
        raise ValueError("At least one volume is required.")
    use_page_filter = len(volumes) == 1
    pairs: list[tuple[PipelineInputRow, CaseDocument]] = []
    row_no = 0
    for vol in volumes:
        config = VolumeRetrievalConfig(
            reporter=reporter,
            volume=vol,
            min_page=min_page if use_page_filter else None,
            max_page=max_page if use_page_filter else None,
            timeout_seconds=timeout_seconds,
        )
        for row, case_doc in iter_volume_cases(config):
            row_no += 1
            if row.row_no != row_no:
                row = replace(row, row_no=row_no)
            pairs.append((row, case_doc))
    return pairs
