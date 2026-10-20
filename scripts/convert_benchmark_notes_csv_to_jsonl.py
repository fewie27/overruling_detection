#!/usr/bin/env python3
"""Convert `2026-05-06 Benchmark Notes.csv` (semicolon-separated) to benchmark JSONL.

Schema matches ``benchmarks/own_benchmark-001.jsonl``:
  {"id", "case_citation", "citations": [{"cite", "class"}, ...]}

``class`` uses prompt_i taxonomy (snake_case): affirmed, amended,
certiorari_granted, criticized, disapproved, disavowed,
overruled, overruling_recognized, remanded, repudiated, reversed,
subsequent_determination, superseded, vacated, withdrawn.

CSV layout (27 columns, ``;`` delimiter):
  0–2: primary affected case (name, reporter, year)
  3: Westlaw-style treatment blurb (maps to label)
  4–10: padding / markers (ignored except we skip ``(red-strip)`` rows in triplets)
  11–13: citing opinion (name, reporter, year)
  14: padding
  15–26: optional further affected cases as (name, reporter, year) triplets
  Column 15 may instead hold a ``;``-separated list of full citations, or a
  comma-separated list of ``United States v. …, cite`` pairs.

Rows whose treatment is *Abrogation Recognized* or *Abrogated by* (Westlaw-style) are
**omitted** from the JSONL (not remapped to other labels).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

# Longest / most specific phrases first.
_TREATMENT_TO_CLASS: list[tuple[str, str]] = [
    ("in subsequent determination", "subsequent_determination"),
    ("certiorari granted in part", "certiorari_granted"),
    ("certiorari granted, judgment vacated", "certiorari_granted"),
    ("certiorari granted, judgment reversed", "certiorari_granted"),
    ("certiorari granted", "certiorari_granted"),
    ("overruling recognized", "overruling_recognized"),
    ("opinion amended and superseded", "superseded"),
    ("vacated and superseded on rehearing", "superseded"),
    ("vacated and remanded", "vacated"),
    ("affirmed in part, reversed in part", "reversed"),
    ("reversed and remanded", "reversed"),
    ("withdrawn in part on rehearing", "withdrawn"),
    ("withdrawn in part by", "withdrawn"),
    ("withdrawn in part", "withdrawn"),
    ("order reversed", "reversed"),
    ("judgment reversed", "reversed"),
    ("overruled by", "overruled"),
    ("criticized by", "criticized"),
    ("repudiated by", "repudiated"),
    ("disapproved of by", "disapproved"),
    ("disapproved by", "disapproved"),
    ("disavowed by", "disavowed"),
    ("order vacated", "vacated"),
    ("judgment vacated", "vacated"),
    ("opinion vacated", "vacated"),
    ("superseded by statute", "superseded"),
    ("opinion amended", "amended"),
]


def _norm(s: str) -> str:
    return (s or "").replace("\xa0", " ").strip().lower()


def treatment_to_class(treatment: str) -> str:
    t = _norm(treatment)
    for needle, label in _TREATMENT_TO_CLASS:
        if needle in t:
            return label
    raise ValueError(f"Unmapped treatment text: {treatment!r}")


def treatment_is_abrogation_skipped(treatment: str) -> bool:
    """True when this CSV row should not appear in benchmark JSONL (abrogation taxonomy removed from prompt_i)."""
    t = _norm(treatment)
    return "abrogation recognized" in t or "abrogated by" in t


def _format_cite(name: str, reporter: str, year: str) -> str:
    name = (name or "").strip()
    reporter = (reporter or "").strip()
    year = (year or "").strip()
    if not name:
        return ""
    if reporter and year:
        return f"{name}, {reporter} ({year})"
    if reporter:
        return f"{name}, {reporter}"
    return name


_SKIP_NAME = frozenset({"(red-stripe)", "(red-strip)", "red-stripe", "red-strip"})


def _is_skip_triplet(name: str) -> bool:
    n = name.strip().lower()
    return not n or n in _SKIP_NAME or n.startswith("(red")


def _parse_paren_cite(chunk: str) -> tuple[str, str, str] | None:
    """Parse ``Name, reporter (year)``."""
    chunk = chunk.strip().rstrip(".")
    m = re.match(r"^(.+),\s*(.+)\s*\((\d{4})\)\s*$", chunk)
    if m:
        return m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
    return None


def _split_us_comma_list(cell: str) -> list[tuple[str, str, str]]:
    """``United States v. A, cite, United States v. B, cite`` (years often omitted)."""
    cell = cell.strip().rstrip(".")
    if ", united states v. " not in cell.lower():
        return []
    parts = re.split(r",\s*United States v\. ", cell, flags=re.IGNORECASE)
    out: list[tuple[str, str, str]] = []
    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
        if i > 0:
            part = "United States v. " + part
        # Last ", " before digits/F./U.S. as reporter
        m = re.match(r"^(.+?),\s*([\d\w.\s\-–]+)$", part)
        if not m:
            continue
        name, rep = m.group(1).strip(), m.group(2).strip()
        ym = re.search(r"\((\d{4})\)\s*$", rep)
        year = ym.group(1) if ym else ""
        if ym:
            rep = rep[: ym.start()].strip()
        out.append((name, rep, year))
    return out


def _extra_from_cell15(cell: str) -> list[tuple[str, str, str]]:
    """Column-15 overflow: ``;``-separated full cites or US-comma run."""
    cell = cell.strip()
    if not cell:
        return []
    if ";" in cell:
        triples: list[tuple[str, str, str]] = []
        for piece in cell.split(";"):
            piece = piece.strip()
            if not piece:
                continue
            p = _parse_paren_cite(piece)
            if p:
                triples.append(p)
            else:
                # ``Name, reporter`` without year
                if "," in piece:
                    name, rep = piece.rsplit(",", 1)
                    triples.append((name.strip(), rep.strip(), ""))
        return triples
    us = _split_us_comma_list(cell)
    if us:
        return us
    return []


def _triplets_from_row(row: list[str]) -> list[tuple[str, str, str]]:
    """Further affected: cols 15,18,21,... if (name,reporter,year) present."""
    out: list[tuple[str, str, str]] = []
    if len(row) > 15:
        blob = (row[15] or "").strip()
        if blob and (";" in blob or ", united states v. " in blob.lower()):
            out.extend(_extra_from_cell15(blob))
        else:
            for start in (15, 18, 21, 24):
                if start + 2 >= len(row):
                    break
                name, rep, yr = row[start], row[start + 1], row[start + 2]
                if not name.strip():
                    continue
                if _is_skip_triplet(name):
                    continue
                out.append((name, rep, yr))
    return out


def row_to_entry(row: list[str]) -> dict | None:
    if len(row) < 14:
        return None
    aff_name, aff_rep, aff_yr = row[0], row[1], row[2]
    cite_name, cite_rep, cite_yr = row[11], row[12], row[13]
    if not aff_name.strip() or not cite_name.strip():
        return None
    treatment = row[3] if len(row) > 3 else ""
    if treatment_is_abrogation_skipped(treatment):
        return None
    try:
        cls = treatment_to_class(treatment)
    except ValueError as e:
        raise ValueError(str(e)) from e

    cites: list[dict[str, str]] = []
    primary = _format_cite(aff_name, aff_rep, aff_yr)
    if primary:
        cites.append({"cite": primary, "class": cls})

    for name, rep, yr in _triplets_from_row(row):
        c = _format_cite(name, rep, yr)
        if c and c != primary:
            cites.append({"cite": c, "class": cls})

    def _cite_key(cite: str) -> str:
        s = cite.lower().replace("u.s.", "united states")
        s = re.sub(r"\s*\(\d{4}\)\s*$", "", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    # De-duplicate by normalized cite (U.S. vs United States, spacing)
    seen: set[str] = set()
    deduped: list[dict[str, str]] = []
    for item in cites:
        k = _cite_key(item["cite"])
        if k in seen:
            continue
        seen.add(k)
        deduped.append(item)

    case_citation = _format_cite(cite_name, cite_rep, cite_yr)
    return {"case_citation": case_citation, "citations": deduped}


def convert(csv_path: Path, jsonl_path: Path) -> int:
    written = 0
    last_json: str | None = None
    with csv_path.open(encoding="utf-8-sig", newline="") as f_in, jsonl_path.open(
        "w", encoding="utf-8"
    ) as f_out:
        rows = list(csv.reader(f_in, delimiter=";"))
        out_id = 0
        for row in rows:
            if not row or not (row[0] or "").strip():
                continue
            if (row[0].strip().lower().startswith("further negative")):
                continue
            try:
                entry = row_to_entry(row)
            except ValueError as err:
                print(err, file=sys.stderr)
                continue
            if not entry or not entry["citations"]:
                continue
            out_id += 1
            line = json.dumps(
                {
                    "id": out_id,
                    "case_citation": entry["case_citation"],
                    "citations": entry["citations"],
                },
                ensure_ascii=True,
            )
            if line == last_json:
                out_id -= 1
                continue
            last_json = line
            f_out.write(line + "\n")
            written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_csv", type=Path, nargs="?", default=None)
    parser.add_argument("output_jsonl", type=Path, nargs="?", default=None)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    csv_path = args.input_csv or Path.home() / "Downloads" / "2026-05-06 Benchmark Notes.csv"
    out_path = args.output_jsonl or root / "benchmarks" / "own_benchmark-002.jsonl"
    if not csv_path.is_file():
        print(f"Input file not found: {csv_path}", file=sys.stderr)
        sys.exit(2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = convert(csv_path, out_path)
    print(f"Wrote {n} entries to {out_path}")


if __name__ == "__main__":
    main()
