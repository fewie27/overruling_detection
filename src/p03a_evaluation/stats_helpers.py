"""Confidence-interval offsets for proportion metrics and bootstrapped F1."""

from __future__ import annotations

import math
import random
from typing import Any, NamedTuple, Sequence

DEFAULT_Z = 1.96
DEFAULT_BOOTSTRAP_B = 10_000
DEFAULT_BOOTSTRAP_SEED = 0


class OffsetInterval(NamedTuple):
    """Deviation of the CI bounds from the point estimate: ``estimate ± offsets``."""

    lower: float
    upper: float


def _offsets(estimate: float, lo: float, hi: float) -> OffsetInterval:
    return OffsetInterval(estimate - lo, hi - estimate)


def _require_binomial(successes: int, n: int) -> None:
    if n < 0 or successes < 0 or successes > n:
        raise ValueError(f"need 0 <= successes <= n; got successes={successes}, n={n}")
    if n == 0:
        raise ValueError("n must be > 0")


def _alpha_from_z(z: float) -> float:
    """Two-sided tail mass for a normal critical value ``z`` (e.g. 1.96 → ≈0.05)."""
    return 1.0 - math.erf(z / math.sqrt(2.0))


def _percentile(sorted_vals: Sequence[float], p: float) -> float:
    if not sorted_vals:
        raise ValueError("empty sample")
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"percentile p must be in [0, 1]; got {p}")
    n = len(sorted_vals)
    if n == 1:
        return float(sorted_vals[0])
    k = (n - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_vals[int(k)])
    return float(sorted_vals[f]) * (c - k) + float(sorted_vals[c]) * (k - f)


def wilson_offsets(successes: int, n: int, *, z: float = DEFAULT_Z) -> OffsetInterval:
    """Wilson score interval offsets for a binomial proportion ``successes / n``."""
    _require_binomial(successes, n)
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n)
    return _offsets(p, center - margin, center + margin)


def wald_margin(successes: int, n: int, *, z: float = DEFAULT_Z) -> float:
    """Symmetric Wald half-width ``z * sqrt(p(1-p)/n)`` (not clipped to [0, 1])."""
    _require_binomial(successes, n)
    p = successes / n
    return z * math.sqrt(p * (1.0 - p) / n)


def wald_offsets(
    successes: int,
    n: int,
    *,
    z: float = DEFAULT_Z,
    clip: bool = True,
) -> OffsetInterval:
    """Wald (normal approximation) interval offsets for ``successes / n``."""
    _require_binomial(successes, n)
    p = successes / n
    margin = wald_margin(successes, n, z=z)
    lo, hi = p - margin, p + margin
    if clip:
        lo, hi = max(0.0, lo), min(1.0, hi)
    return _offsets(p, lo, hi)


def f1_score(tp: int, fp: int, fn: int) -> float:
    """Binary F1 from confusion counts; 0.0 when precision or recall is undefined."""
    if min(tp, fp, fn) < 0:
        raise ValueError(f"counts must be >= 0; got tp={tp}, fp={fp}, fn={fn}")
    prec_den = tp + fp
    rec_den = tp + fn
    if prec_den == 0 or rec_den == 0:
        return 0.0
    precision = tp / prec_den
    recall = tp / rec_den
    denom = precision + recall
    return 0.0 if denom == 0.0 else 2.0 * precision * recall / denom


def bootstrap_offsets(
    replicates: Sequence[float],
    estimate: float,
    *,
    z: float = DEFAULT_Z,
) -> OffsetInterval:
    """Percentile-CI offsets from bootstrap replicates around ``estimate``."""
    if not replicates:
        raise ValueError("replicates must be non-empty")
    alpha = _alpha_from_z(z)
    ordered = sorted(replicates)
    lo = _percentile(ordered, alpha / 2.0)
    hi = _percentile(ordered, 1.0 - alpha / 2.0)
    return _offsets(estimate, lo, hi)


def bootstrap_f1_offsets(
    tp: int,
    fp: int,
    fn: int,
    *,
    B: int = DEFAULT_BOOTSTRAP_B,
    z: float = DEFAULT_Z,
    seed: int | None = None,
) -> OffsetInterval:
    """Sample-bootstrap percentile offsets for F1 (resample TP/FP/FN instances)."""
    if min(tp, fp, fn) < 0:
        raise ValueError(f"counts must be >= 0; got tp={tp}, fp={fp}, fn={fn}")
    if B < 1:
        raise ValueError(f"B must be >= 1; got {B}")

    estimate = f1_score(tp, fp, fn)
    n = tp + fp + fn
    if n == 0:
        return OffsetInterval(0.0, 0.0)

    rng = random.Random(seed)
    categories = (0, 1, 2)
    weights = (tp, fp, fn)
    replicates: list[float] = []
    for _ in range(B):
        counts = [0, 0, 0]
        for label in rng.choices(categories, weights=weights, k=n):
            counts[label] += 1
        replicates.append(f1_score(counts[0], counts[1], counts[2]))
    return bootstrap_offsets(replicates, estimate, z=z)


def precision_wilson_offsets(tp: int, fp: int, *, z: float = DEFAULT_Z) -> OffsetInterval:
    return wilson_offsets(tp, tp + fp, z=z)


def precision_wald_offsets(tp: int, fp: int, *, z: float = DEFAULT_Z) -> OffsetInterval:
    return wald_offsets(tp, tp + fp, z=z)


def recall_wilson_offsets(tp: int, fn: int, *, z: float = DEFAULT_Z) -> OffsetInterval:
    return wilson_offsets(tp, tp + fn, z=z)


def recall_wald_offsets(tp: int, fn: int, *, z: float = DEFAULT_Z) -> OffsetInterval:
    return wald_offsets(tp, tp + fn, z=z)


def accuracy_wilson_offsets(
    tp: int, tn: int, fp: int, fn: int, *, z: float = DEFAULT_Z
) -> OffsetInterval:
    return wilson_offsets(tp + tn, tp + tn + fp + fn, z=z)


def accuracy_wald_offsets(
    tp: int, tn: int, fp: int, fn: int, *, z: float = DEFAULT_Z
) -> OffsetInterval:
    return wald_offsets(tp + tn, tp + tn + fp + fn, z=z)


def ci_meta(
    *,
    z: float = DEFAULT_Z,
    bootstrap_b: int = DEFAULT_BOOTSTRAP_B,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, float | int]:
    return {"z": z, "bootstrap_b": bootstrap_b, "bootstrap_seed": bootstrap_seed}


def _proportion_ci_fields(
    successes: int,
    n: int,
    *,
    z: float,
) -> tuple[list[float] | None, float | None]:
    if n <= 0:
        return None, None
    w = wilson_offsets(successes, n, z=z)
    return [w.lower, w.upper], wald_margin(successes, n, z=z)


def enrich_confusion_metric_block(
    block: dict[str, Any],
    *,
    z: float = DEFAULT_Z,
    bootstrap_b: int = DEFAULT_BOOTSTRAP_B,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> None:
    """Add Wilson/Wald/bootstrap offset fields in-place for a TP/FP/FN metric block.

    Only emits CI keys for metrics already present on ``block``.
    """
    tp = block.get("tp")
    fp = block.get("fp")
    fn = block.get("fn")
    if not isinstance(tp, int) or not isinstance(fn, int):
        raise ValueError("metric block requires integer tp and fn")

    if "precision" in block:
        if block.get("precision") is None or not isinstance(fp, int):
            block["precision_wilson"] = None
            block["precision_wald"] = None
        else:
            wilson, wald = _proportion_ci_fields(tp, tp + fp, z=z)
            block["precision_wilson"] = wilson
            block["precision_wald"] = wald

    if "recall" in block:
        if block.get("recall") is None:
            block["recall_wilson"] = None
            block["recall_wald"] = None
        else:
            wilson, wald = _proportion_ci_fields(tp, tp + fn, z=z)
            block["recall_wilson"] = wilson
            block["recall_wald"] = wald

    if "f1" in block:
        if block.get("f1") is None or not isinstance(fp, int):
            block["f1_boot"] = None
        else:
            off = bootstrap_f1_offsets(
                tp, fp, fn, B=bootstrap_b, z=z, seed=bootstrap_seed
            )
            block["f1_boot"] = [off.lower, off.upper]


def enrich_evaluation_metadata(
    metadata: dict[str, Any],
    *,
    z: float = DEFAULT_Z,
    bootstrap_b: int = DEFAULT_BOOTSTRAP_B,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> bool:
    """Enrich ``class_level_metrics`` / ``aggregate_class_metrics`` with CI offsets.

    Returns False if the metadata lacks usable TP/FN counts (older runs).
    """
    class_metrics = metadata.get("class_level_metrics")
    if not isinstance(class_metrics, dict) or not class_metrics:
        return False

    sample = next(iter(class_metrics.values()))
    if not isinstance(sample, dict) or "tp" not in sample or "fn" not in sample:
        return False

    for block in class_metrics.values():
        if not isinstance(block, dict):
            continue
        enrich_confusion_metric_block(
            block, z=z, bootstrap_b=bootstrap_b, bootstrap_seed=bootstrap_seed
        )

    aggregate = metadata.get("aggregate_class_metrics")
    if isinstance(aggregate, dict) and "tp" in aggregate and "fn" in aggregate:
        enrich_confusion_metric_block(
            aggregate, z=z, bootstrap_b=bootstrap_b, bootstrap_seed=bootstrap_seed
        )

    metadata["ci"] = ci_meta(z=z, bootstrap_b=bootstrap_b, bootstrap_seed=bootstrap_seed)
    return True
