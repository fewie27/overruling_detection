#!/usr/bin/env python3
"""Convert a legacy CSV benchmark to the new JSONL format.

Legacy CSV columns:
    Order, Overruling Decision, Year of Overruling Decision,
    Overruled Decision(s), Year(s) of Overruled Decision(s)

Each row of the legacy file becomes one JSONL entry:

    {
      "id": <int>,
      "case_citation": "<Overruling Decision>",
      "citations": [
        {"cite": "<each citation in 'Overruled Decision(s)'>", "class": "overruled"},
        ...
      ]
    }

Usage:
    python scripts/convert_csv_benchmark_to_jsonl.py INPUT.csv [OUTPUT.jsonl]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def _split_citations(raw: str) -> list[str]:
    if not raw:
        return []
    return [p.strip() for p in raw.split(";") if p.strip()]


def convert(csv_path: Path, jsonl_path: Path, *, default_class: str = "overruled") -> int:
    written = 0
    with open(csv_path, encoding="utf-8", newline="") as f_in, open(jsonl_path, "w", encoding="utf-8") as f_out:
        reader = csv.DictReader(f_in)
        for row_no, row in enumerate(reader, start=1):
            order_raw = (row.get("Order") or "").strip()
            try:
                entry_id = int(order_raw) if order_raw else row_no
            except ValueError:
                entry_id = row_no
            case_citation = (row.get("Overruling Decision") or "").strip()
            if not case_citation:
                continue
            cites = _split_citations(row.get("Overruled Decision(s)") or "")
            entry = {
                "id": entry_id,
                "case_citation": case_citation,
                "citations": [{"cite": c, "class": default_class} for c in cites],
            }
            f_out.write(json.dumps(entry, ensure_ascii=True) + "\n")
            written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_csv", type=Path, help="Path to the legacy CSV benchmark")
    parser.add_argument(
        "output_jsonl",
        type=Path,
        nargs="?",
        help="Path to write the JSONL output (default: same stem with .jsonl extension)",
    )
    parser.add_argument(
        "--class",
        dest="default_class",
        default="overruled",
        help='Class label assigned to every CSV citation (default: "overruled").',
    )
    args = parser.parse_args()
    if not args.input_csv.is_file():
        print(f"Input file not found: {args.input_csv}", file=sys.stderr)
        sys.exit(2)
    output = args.output_jsonl or args.input_csv.with_suffix(".jsonl")
    written = convert(args.input_csv, output, default_class=args.default_class)
    print(f"Wrote {written} entries to {output}")


if __name__ == "__main__":
    main()
