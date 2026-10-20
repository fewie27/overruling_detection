#!/usr/bin/env python3
"""Print summary statistics for a benchmark JSONL file.

Each line is expected to match the project benchmark schema::

    {"id": int, "case_citation": str, "citations": [{"cite": str, "class": str}, ...]}

Reports:
  - number of entries (active / overruling cases)
  - total and average number of citations (passive / cited cases)
  - active cases per decade (year taken from ``case_citation``)
  - per-class citation counts (total occurrences)
  - per-class entry counts (entries with at least one citation of that class)

Usage:
    python scripts/analyze_benchmark_jsonl.py benchmarks/own_benchmark-002.jsonl
    python scripts/analyze_benchmark_jsonl.py benchmarks/own_benchmark-007.jsonl --json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

_CASE_YEAR_RE = re.compile(r"\((\d{4})\)")


def _year_from_case_citation(case_citation: str) -> int | None:
    """Return the decision year from the last ``(YYYY)`` in *case_citation*."""
    years = _CASE_YEAR_RE.findall((case_citation or "").strip())
    if not years:
        return None
    return int(years[-1])


def _decade_label(year: int) -> str:
    return f"{year // 10 * 10}s"


@dataclass(frozen=True)
class BenchmarkStats:
    path: Path
    num_entries: int
    total_citations: int
    avg_citations: float
    active_case_decade_counts: dict[str, int]
    active_cases_without_year: int
    class_citation_counts: dict[str, int]
    class_entry_counts: dict[str, int]

    @property
    def all_classes(self) -> list[str]:
        keys = set(self.class_citation_counts) | set(self.class_entry_counts)
        return sorted(keys, key=lambda c: (-self.class_citation_counts.get(c, 0), c))

    @property
    def decades_chronological(self) -> list[str]:
        decades = [d for d in self.active_case_decade_counts if d != "unknown"]
        return sorted(decades, key=lambda d: int(d.rstrip("s")))


def analyze_benchmark(path: Path) -> BenchmarkStats:
    num_entries = 0
    total_citations = 0
    active_case_decade_counts: Counter[str] = Counter()
    active_cases_without_year = 0
    class_citation_counts: Counter[str] = Counter()
    class_entry_counts: Counter[str] = Counter()

    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc

            case_citation = str(entry.get("case_citation", "")).strip()
            if not case_citation:
                raise ValueError(f"{path}:{line_no}: missing 'case_citation'")

            citations = entry.get("citations")
            if not isinstance(citations, list):
                raise ValueError(f"{path}:{line_no}: missing or invalid 'citations' list")

            num_entries += 1
            total_citations += len(citations)

            year = _year_from_case_citation(case_citation)
            if year is None:
                active_cases_without_year += 1
                active_case_decade_counts["unknown"] += 1
            else:
                active_case_decade_counts[_decade_label(year)] += 1

            classes_in_entry: set[str] = set()
            for cite in citations:
                if not isinstance(cite, dict):
                    raise ValueError(f"{path}:{line_no}: citation entry is not an object")
                cls = str(cite.get("class", "")).strip()
                if not cls:
                    raise ValueError(f"{path}:{line_no}: citation missing 'class'")
                class_citation_counts[cls] += 1
                classes_in_entry.add(cls)

            for cls in classes_in_entry:
                class_entry_counts[cls] += 1

    avg_citations = total_citations / num_entries if num_entries else 0.0
    return BenchmarkStats(
        path=path,
        num_entries=num_entries,
        total_citations=total_citations,
        avg_citations=avg_citations,
        active_case_decade_counts=dict(active_case_decade_counts),
        active_cases_without_year=active_cases_without_year,
        class_citation_counts=dict(class_citation_counts),
        class_entry_counts=dict(class_entry_counts),
    )


def _format_table(stats: BenchmarkStats) -> str:
    lines = [
        f"Benchmark: {stats.path}",
        f"Entries: {stats.num_entries}",
        f"Citations (total): {stats.total_citations}",
        f"Citations (avg per entry): {stats.avg_citations:.2f}",
        "",
        "Active cases by decade (from case_citation year):",
        f"{'decade':<12} {'entries':>10}",
        f"{'-' * 12} {'-' * 10}",
    ]
    for decade in stats.decades_chronological:
        lines.append(f"{decade:<12} {stats.active_case_decade_counts[decade]:>10}")
    if stats.active_cases_without_year:
        lines.append(f"{'unknown':<12} {stats.active_cases_without_year:>10}")
    lines.extend([
        "",
        f"{'class':<28} {'citations':>10} {'entries':>10}",
        f"{'-' * 28} {'-' * 10} {'-' * 10}",
    ])
    for cls in stats.all_classes:
        lines.append(
            f"{cls:<28} {stats.class_citation_counts.get(cls, 0):>10} "
            f"{stats.class_entry_counts.get(cls, 0):>10}"
        )
    return "\n".join(lines)


def _format_json(stats: BenchmarkStats) -> str:
    payload = {
        "path": str(stats.path),
        "num_entries": stats.num_entries,
        "total_citations": stats.total_citations,
        "avg_citations": round(stats.avg_citations, 4),
        "active_cases_by_decade": {
            decade: stats.active_case_decade_counts[decade]
            for decade in stats.decades_chronological
        },
        "active_cases_without_year": stats.active_cases_without_year,
        "classes": {
            cls: {
                "citation_count": stats.class_citation_counts.get(cls, 0),
                "entry_count": stats.class_entry_counts.get(cls, 0),
            }
            for cls in stats.all_classes
        },
    }
    return json.dumps(payload, indent=2, ensure_ascii=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input_jsonl", type=Path, help="Path to the benchmark JSONL file")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of a text table",
    )
    args = parser.parse_args()

    if not args.input_jsonl.is_file():
        print(f"Input file not found: {args.input_jsonl}", file=sys.stderr)
        sys.exit(2)

    try:
        stats = analyze_benchmark(args.input_jsonl)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    print(_format_json(stats) if args.json else _format_table(stats))


if __name__ == "__main__":
    main()
