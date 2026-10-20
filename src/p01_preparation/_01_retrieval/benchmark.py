"""Load the JSONL benchmark and provide DatasetInputRow records to the pipeline.

JSONL entry shape (one object per line):

    {
      "id": 1,
      "case_citation": "Childers v. Floyd, 642 F.3d 953 (2011)",
      "citations": [
        {"cite": "Espy v. Massac, 443 F.3d 1362 (2006)", "class": "overruled"}
      ],

      // optional retrieval/opinion overrides
      "lookup_citation": "642 F.3d 953",
      "case_name": "Childers v. Floyd",
      "cleaned_text_opinion": "..."
    }
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.models import ClassifiedCitation
from src.models.case_trace import PipelineInputRow

_SKIP_PARTY_WORDS = frozenset({"united", "states", "the", "of", "and", "in", "ex", "rel", "no"})


# Backward-compatible alias; benchmark rows are pipeline inputs without trace metadata.
DatasetInputRow = PipelineInputRow


def parse_indices_csv(text: str) -> list[int]:
    """Parse comma-separated 1-based ids and inclusive ranges (e.g. ``1-79,81,82-100``)."""
    t = (text or "").strip()
    if not t:
        return []
    nums: list[int] = []
    for p in (s.strip() for s in t.split(",")):
        if not p:
            raise ValueError("Empty entry in indices list (check for stray commas).")
        if "-" in p:
            lo_s, sep, hi_s = p.partition("-")
            if not sep or not lo_s.strip() or not hi_s.strip() or "-" in hi_s:
                raise ValueError(f"Invalid range in indices list: {p!r} (expected START-END).")
            try:
                lo = int(lo_s.strip())
                hi = int(hi_s.strip())
            except ValueError as exc:
                raise ValueError(f"Not an integer range in indices list: {p!r}") from exc
            if lo < 1 or hi < 1:
                raise ValueError(f"Row index must be >= 1, got range {lo}-{hi}")
            if hi < lo:
                raise ValueError(f"Range start must be <= end, got {p!r}")
            nums.extend(range(lo, hi + 1))
            continue
        try:
            n = int(p)
        except ValueError as exc:
            raise ValueError(f"Not an integer in indices list: {p!r}") from exc
        if n < 1:
            raise ValueError(f"Row index must be >= 1, got {n}")
        nums.append(n)
    seen: set[int] = set()
    out: list[int] = []
    for n in nums:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def load_dataset_inputs(
    dataset_path: str,
    *,
    from_idx: int | None = None,
    to_idx: int | None = None,
    indices: list[int] | None = None,
) -> list[DatasetInputRow]:
    entries = _load_benchmark_jsonl(dataset_path)
    selected = _select_entries(entries, from_idx=from_idx, to_idx=to_idx, indices=indices)
    return [_to_input_row(entry) for entry in selected]


def _load_benchmark_jsonl(path: str) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Benchmark JSONL not found: {path}")
    entries: list[dict[str, Any]] = []
    with open(p, encoding="utf-8") as f:
        for lineno, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {lineno} of {path}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Benchmark JSONL line {lineno} is not a JSON object")
            if "id" not in obj or "case_citation" not in obj or "citations" not in obj:
                raise ValueError(
                    f"Benchmark JSONL line {lineno} missing required fields "
                    f"'id', 'case_citation', 'citations'"
                )
            entries.append(obj)
    return entries


def _select_entries(
    entries: list[dict[str, Any]],
    *,
    from_idx: int | None,
    to_idx: int | None,
    indices: list[int] | None,
) -> list[dict[str, Any]]:
    if indices:
        by_id: dict[int, dict[str, Any]] = {}
        for e in entries:
            try:
                eid = int(e["id"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Benchmark entry has non-integer id: {e.get('id')!r}") from exc
            by_id[eid] = e
        out: list[dict[str, Any]] = []
        for n in indices:
            if n not in by_id:
                raise ValueError(f"Row index {n} is not present in benchmark (ids found: {sorted(by_id)}).")
            out.append(by_id[n])
        return out
    out = []
    for e in entries:
        try:
            eid = int(e["id"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Benchmark entry has non-integer id: {e.get('id')!r}") from exc
        if from_idx is not None and eid < from_idx:
            continue
        if to_idx is not None and eid > to_idx:
            continue
        out.append(e)
    return out


def _to_input_row(entry: dict[str, Any]) -> DatasetInputRow:
    row_no = int(entry["id"])
    case_citation = str(entry["case_citation"]).strip()
    raw_citations = entry.get("citations") or []
    if not isinstance(raw_citations, list):
        raise ValueError(f"Benchmark id={row_no}: 'citations' must be a list")
    classifications: list[ClassifiedCitation] = []
    for i, c in enumerate(raw_citations, start=1):
        if not isinstance(c, dict):
            raise ValueError(f"Benchmark id={row_no}, citations[{i}] must be an object")
        cc = ClassifiedCitation.from_dict(c)
        if not cc.cite or not cc.cls:
            raise ValueError(
                f"Benchmark id={row_no}, citations[{i}] must have non-empty 'cite' and 'class'"
            )
        classifications.append(cc)

    case_name_override = (entry.get("case_name") or "").strip() or None
    opinion_override_raw = entry.get("cleaned_text_opinion")
    opinion_text_override = (
        str(opinion_override_raw) if opinion_override_raw and str(opinion_override_raw).strip() else None
    )
    lookup_citation = (entry.get("lookup_citation") or "").strip() or case_citation

    return DatasetInputRow(
        row_no=row_no,
        case_citation=case_citation,
        expected_classifications=classifications,
        display_case_name=case_citation,
        lookup_citation=lookup_citation,
        case_name_override=case_name_override,
        opinion_text_override=opinion_text_override,
        raw_entry=entry,
    )


def case_matches_retrieved_caption(case_citation: str, retrieved_name: str) -> bool:
    """True iff the retrieved case caption plausibly refers to the benchmark case_citation."""
    a = _leading_case_title(case_citation)
    b = _leading_case_title(retrieved_name)
    if a and b and a == b:
        return True
    r = (retrieved_name or "").casefold()
    lead = a
    if " v " not in lead:
        return any(k in r for k in _party_keywords(lead))
    p1, p2 = lead.split(" v ", 1)
    for party in (p1.strip(), p2.strip()):
        for kw in _party_keywords(party):
            if kw in r:
                return True
    return False


def _leading_case_title(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    if "," in s:
        s = s.split(",")[0].strip()
    for ch in "\u2019\u2018\u201b\u201a":
        s = s.replace(ch, "'")
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    return s.casefold()


def _party_keywords(caption: str) -> list[str]:
    if not caption:
        return []
    out: list[str] = []
    for w in caption.split():
        w = w.strip(".,;")
        wl = w.casefold()
        if wl in _SKIP_PARTY_WORDS or len(w) < 3:
            continue
        out.append(wl)
        for part in re.split(r"[-\u2013\u2014]", w):
            pl = part.strip(".,;").casefold()
            if pl in _SKIP_PARTY_WORDS or len(pl) < 3 or pl == wl:
                continue
            out.append(pl)
    return list(dict.fromkeys(out))
