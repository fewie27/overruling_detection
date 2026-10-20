"""Citation comparison + classification-aware evaluation.

The pipeline always carries citations together with their treatment-class label
(``ClassifiedCitation``). For benchmark evaluation a citation must be both:
  1) found by the model, and
  2) classified with the same class as the benchmark entry.

Predicted citations that do not appear in the benchmark are ignored unless their
class is in ``strict_classes`` (default: ``{"overruled", "partially overruled"}``);
unexpected predictions in those classes are counted as errors.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable, Tuple

from src.models import ClassifiedCitation, DEFAULT_STRICT_CLASSES, normalize_class_label

# When True, ``same_citation`` also requires caption/party compatibility after the reporter pin
# matches, and ``same_citation_loose`` may match on case names alone. When False, matching is
# pin-only (volume + reporter + page within tolerance) wherever a pin can be parsed.
_REQUIRE_PARTY_CAPTION_MATCH = False

_CITATION_PAGE_BOUNDARY_TAIL = r"(?:\(|,|;|\[|\]|$)"
_CITATION_PIN_REGEX = re.compile(
    rf"(\d+)\s+([A-Za-z0-9.' ]+?)\s+(?:\([A-Za-z0-9., ]+\)\s+)?(\d+)(?=\s*{_CITATION_PAGE_BOUNDARY_TAIL})"
)

_GENERIC_CAPTION_TOKENS = frozenset(
    {
        "united", "states", "state", "ex", "rel", "merits", "appellant", "respondent",
        "petitioner", "appellee", "plaintiff", "defendant", "the", "and", "for", "use",
        "board", "department", "commission", "county", "city", "inc", "llc", "ltd", "corp",
        "association", "assn",
    }
)


def _normalize_reporter_for_comparison(reporter: str) -> str:
    """Lowercase, strip spaces and periods so F.3d / F. 3d / f3d agree."""
    t = (reporter or "").strip().lower()
    return t.replace(".", "").replace(" ", "").replace("'", "")


def _first_valid_pin_match(s: str) -> re.Match[str] | None:
    """First volume/reporter/page pin in ``s`` (skips numeric-only "reporter" groups)."""
    for m in _CITATION_PIN_REGEX.finditer(s.strip()):
        rep = m.group(2).strip()
        if any(ch.isalpha() for ch in rep):
            return m
    return None


def _caption_before_reporter_pin(s: str) -> str:
    m = _first_valid_pin_match(s)
    if not m:
        return s.strip()
    return s[: m.start()].strip().rstrip(",")


def _citation_regex_groups(s: str) -> Tuple[str, str, str] | None:
    m = _first_valid_pin_match(s.strip())
    if not m:
        return None
    return (m.group(1), _normalize_reporter_for_comparison(m.group(2)), m.group(3))


def _levenshtein_distance(a: str, b: str) -> int:
    a, b = a.strip(), b.strip()
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i]
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[m]


def _case_title_key(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    if "," in s:
        s = s.split(",")[0].strip()
    for ch in "\u2019\u2018\u201b\u201a":
        s = s.replace(ch, "'")
    return s.casefold()


def _normalize_caption_for_overlap(s: str) -> str:
    s = (s or "").strip()
    for ch in "\u2019\u2018\u201b\u201a":
        s = s.replace(ch, "'")
    s = unicodedata.normalize("NFKC", s)
    s = s.casefold().replace(",", " ")
    s = re.sub(r"\bu\.?\s*s\.?\b", "united states", s, flags=re.UNICODE)
    s = re.sub(r"[^\w\s]+", " ", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()


def _substantive_caption_tokens(s: str) -> set[str]:
    norm = _normalize_caption_for_overlap(s)
    if not norm:
        return set()
    return {tok for tok in re.findall(r"\w{3,}", norm, flags=re.UNICODE) if tok not in _GENERIC_CAPTION_TOKENS}


def _captions_compatible_for_pin(a_caption: str, b_caption: str, *, max_levenshtein: int) -> bool:
    ac, bc = _normalize_caption_for_overlap(a_caption), _normalize_caption_for_overlap(b_caption)
    if not ac or not bc:
        return False
    shorter, longer = (ac, bc) if len(ac) <= len(bc) else (bc, ac)
    if len(shorter) >= 8 and shorter in longer:
        return True
    if _levenshtein_distance(ac[:500], bc[:500]) <= max_levenshtein:
        return True
    ta, tb = _substantive_caption_tokens(ac), _substantive_caption_tokens(bc)
    if not ta or not tb:
        return False
    inter = len(ta & tb)
    jacc = inter / len(ta | tb)
    if jacc >= 0.22 or (jacc >= 0.14 and inter):
        return True
    ka, kb = _case_title_key(a_caption), _case_title_key(b_caption)
    return bool(ka and kb and ka == kb)


def same_citation_loose(a: str, b: str) -> bool:
    """Allow matching by case caption alone when reporter pin matching fails."""
    if same_citation(a, b):
        return True
    if not _REQUIRE_PARTY_CAPTION_MATCH:
        return False
    ta, tb = _case_title_key(a), _case_title_key(b)
    if not ta or not tb:
        return False
    if ta == tb:
        return True
    if ta in tb or tb in ta:
        return min(len(ta), len(tb)) >= 8
    return False


def same_citation(a: str, b: str, max_levenshtein: int = 12, page_tolerance: int = 1) -> bool:
    a, b = a.strip(), b.strip()
    if not a or not b:
        return False
    ga = _citation_regex_groups(a)
    gb = _citation_regex_groups(b)
    if ga is None or gb is None:
        return False
    volume_a, reporter_a, page_a = ga
    volume_b, reporter_b, page_b = gb
    if volume_a != volume_b or reporter_a != reporter_b:
        return False
    if abs(int(page_a) - int(page_b)) > page_tolerance:
        return False
    if not _REQUIRE_PARTY_CAPTION_MATCH:
        return True
    return _captions_compatible_for_pin(
        _caption_before_reporter_pin(a),
        _caption_before_reporter_pin(b),
        max_levenshtein=max_levenshtein,
    )


@dataclass(frozen=True)
class EvaluationCounts:
    matched: int        # benchmark citations the model also identified with the right class
    missing: int        # benchmark citations the model failed to identify with the right class
    unexpected: int     # extra predictions in strict_classes that are not in the benchmark


def evaluate_classified_citations(
    expected: list[ClassifiedCitation],
    predicted: list[ClassifiedCitation],
    *,
    strict_classes: Iterable[str] = DEFAULT_STRICT_CLASSES,
    use_loose: bool = False,
) -> tuple[str, EvaluationCounts]:
    """Compare predictions against expected (benchmark) citations.

    Returns (verdict, counts) where verdict is one of MATCH / PARTIAL / WRONG.
    """
    matcher = same_citation_loose if use_loose else same_citation
    strict = {normalize_class_label(c) for c in strict_classes}

    unmatched_pred = list(range(len(predicted)))
    matched_count = 0
    missing_count = 0
    for e in expected:
        hit = None
        for j in unmatched_pred:
            p = predicted[j]
            if matcher(e.cite, p.cite) and e.cls == p.cls:
                hit = j
                break
        if hit is None:
            missing_count += 1
        else:
            matched_count += 1
            unmatched_pred.remove(hit)

    unexpected_count = 0
    for j in unmatched_pred:
        p = predicted[j]
        if p.cls not in strict:
            continue
        # Predictions whose cite already covers an expected citation are not "extras";
        # the wrong-class case is already counted as missing on the expected side.
        if any(matcher(p.cite, e.cite) for e in expected):
            continue
        unexpected_count += 1

    counts = EvaluationCounts(
        matched=matched_count,
        missing=missing_count,
        unexpected=unexpected_count,
    )
    if missing_count == 0 and unexpected_count == 0:
        verdict = "MATCH"
    elif matched_count > 0:
        verdict = "PARTIAL"
    else:
        verdict = "WRONG"
    return verdict, counts


def expected_classification_for_chunk(
    expected: list[ClassifiedCitation],
    citation_surfaces: list[str],
) -> ClassifiedCitation | None:
    """Return the benchmark entry whose cite matches any surface in ``citation_surfaces``."""
    if not citation_surfaces:
        return None
    for e in expected:
        for s in citation_surfaces:
            if same_citation(e.cite, s):
                return e
    return None


def chunk_citation_state(
    expected: list[ClassifiedCitation],
    citation_surfaces: list[str],
    predicted: list[ClassifiedCitation],
    *,
    strict_classes: Iterable[str] = DEFAULT_STRICT_CLASSES,
) -> str:
    """Per-chunk MATCH/WRONG: did the model classify this surface correctly relative to gold?"""
    if not citation_surfaces:
        raise ValueError("citation_surfaces must be non-empty")
    strict = {normalize_class_label(c) for c in strict_classes}
    gold = expected_classification_for_chunk(expected, citation_surfaces)
    pred_cls = predicted[0].cls if predicted else None

    if gold is None:
        # No benchmark expectation for this surface: prediction in strict_classes is wrong, else fine.
        if pred_cls is None or pred_cls not in strict:
            return "MATCH"
        return "WRONG"
    return "MATCH" if pred_cls == gold.cls else "WRONG"


def format_classified_citations(items: list[ClassifiedCitation]) -> str:
    """Human-friendly ``cite [class]`` joined with ``"; "``."""
    return "; ".join(f"{c.cite} [{c.cls}]" for c in items if c.cite)


NOT_EXPECTED_LABEL = "not expected"
NOT_PREDICTED_LABEL = "not predicted"


def pair_expected_predicted(
    expected: list[ClassifiedCitation],
    predicted: list[ClassifiedCitation],
    *,
    use_loose: bool = False,
) -> tuple[
    list[tuple[ClassifiedCitation, ClassifiedCitation]],
    list[ClassifiedCitation],
    list[ClassifiedCitation],
]:
    """1:1 pair expected↔predicted by citation surface.

    Phase 1 prefers same-class matches; phase 2 pairs remaining cites regardless of class.
    Returns ``(pairs, unmatched_expected, unmatched_predicted)``.
    """
    matcher = same_citation_loose if use_loose else same_citation
    unmatched_pred = list(range(len(predicted)))
    pairs: list[tuple[ClassifiedCitation, ClassifiedCitation]] = []
    unmatched_expected: list[ClassifiedCitation] = []

    def _take_hit(e: ClassifiedCitation, *, require_same_class: bool) -> bool:
        for j in unmatched_pred:
            p = predicted[j]
            if not matcher(e.cite, p.cite):
                continue
            if require_same_class and e.cls != p.cls:
                continue
            pairs.append((e, p))
            unmatched_pred.remove(j)
            return True
        return False

    still_open: list[ClassifiedCitation] = []
    for e in expected:
        if not _take_hit(e, require_same_class=True):
            still_open.append(e)
    for e in still_open:
        if not _take_hit(e, require_same_class=False):
            unmatched_expected.append(e)

    unmatched_predicted = [predicted[j] for j in unmatched_pred]
    return pairs, unmatched_expected, unmatched_predicted


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    return _ratio(2 * precision * recall, precision + recall)


def compute_class_metrics_and_confusion(
    cases: Iterable[tuple[list[ClassifiedCitation], list[ClassifiedCitation]]],
    *,
    strict_classes: Iterable[str] = DEFAULT_STRICT_CLASSES,
    use_loose: bool = False,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Aggregate per-class TP/FP/FN metrics and a confusion matrix over many cases.

    Returns ``(class_level_metrics, aggregate_class_metrics, confusion_matrix)``.

    Strict classes count FP; non-strict classes set ``fp``/``precision``/``f1`` to null.
    Confusion rows = expected labels + ``not expected``; cols = predicted labels + ``not predicted``.
    Labels are the union of classes observed in this run.
    """
    strict = {normalize_class_label(c) for c in strict_classes}
    tp: dict[str, int] = {}
    fp: dict[str, int] = {}
    fn: dict[str, int] = {}
    observed_labels: set[str] = set()
    # (expected_label, predicted_label) -> count; extras use NOT_* constants
    cell_counts: dict[tuple[str, str], int] = {}

    def _bump_cell(row: str, col: str, n: int = 1) -> None:
        cell_counts[(row, col)] = cell_counts.get((row, col), 0) + n

    def _bump(d: dict[str, int], key: str, n: int = 1) -> None:
        d[key] = d.get(key, 0) + n

    for expected, predicted in cases:
        for c in expected:
            observed_labels.add(c.cls)
        for c in predicted:
            observed_labels.add(c.cls)

        pairs, unmatched_exp, unmatched_pred = pair_expected_predicted(
            expected, predicted, use_loose=use_loose
        )
        for e, p in pairs:
            _bump_cell(e.cls, p.cls)
            if e.cls == p.cls:
                _bump(tp, e.cls)
            else:
                _bump(fn, e.cls)
                if p.cls in strict:
                    _bump(fp, p.cls)
        for e in unmatched_exp:
            _bump_cell(e.cls, NOT_PREDICTED_LABEL)
            _bump(fn, e.cls)
        for p in unmatched_pred:
            _bump_cell(NOT_EXPECTED_LABEL, p.cls)
            if p.cls in strict:
                _bump(fp, p.cls)

    labels = sorted(observed_labels)
    class_level_metrics: dict[str, dict[str, Any]] = {}
    total_tp = 0
    total_fn = 0
    for label in labels:
        is_strict = label in strict
        tp_n = tp.get(label, 0)
        fn_n = fn.get(label, 0)
        fp_n = fp.get(label, 0) if is_strict else None
        recall = _ratio(tp_n, tp_n + fn_n)
        precision = None if fp_n is None else _ratio(tp_n, tp_n + fp_n)
        class_level_metrics[label] = {
            "strict": is_strict,
            "tp": tp_n,
            "fp": fp_n,
            "fn": fn_n,
            "precision": precision,
            "recall": recall,
            "f1": _f1(precision, recall),
        }
        total_tp += tp_n
        total_fn += fn_n

    aggregate_class_metrics = {
        "tp": total_tp,
        "fn": total_fn,
        "recall": _ratio(total_tp, total_tp + total_fn),
    }

    row_labels = labels + [NOT_EXPECTED_LABEL]
    col_labels = labels + [NOT_PREDICTED_LABEL]
    counts = [
        [cell_counts.get((row, col), 0) for col in col_labels]
        for row in row_labels
    ]
    confusion_matrix = {
        "row_labels": row_labels,
        "col_labels": col_labels,
        "counts": counts,
    }
    return class_level_metrics, aggregate_class_metrics, confusion_matrix
