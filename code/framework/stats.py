"""Statistical analysis utilities for credit-method comparisons.

Phase-1/2 reported only mean ± stdev over seeds, which cannot answer
"is method A *significantly* better than B on this pipeline?". Phase-3
adds:

* Bootstrap confidence intervals (percentile method).
* Paired permutation test for mean differences on matched seeds.
* Holm–Bonferroni correction when comparing more than two methods.

No numpy/scipy dependency — everything runs on Python's stdlib.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from random import Random
from typing import Iterable, Sequence


# -----------------------------------------------------------------------------
# Bootstrap CIs
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class CI:
    """Percentile-bootstrap confidence interval."""

    mean: float
    low: float
    high: float
    level: float  # e.g. 0.95

    def __str__(self) -> str:
        return f"{self.mean:.3f} [{self.low:.3f}, {self.high:.3f}] (level={self.level:.2f})"


def bootstrap_ci(
    data: Sequence[float],
    *,
    n_resamples: int = 2000,
    level: float = 0.95,
    rng: Random | None = None,
) -> CI:
    """Percentile-bootstrap CI on the mean.

    Returns NaN bounds if ``data`` is empty. For a single observation the
    interval collapses to the point.
    """
    if not data:
        return CI(mean=math.nan, low=math.nan, high=math.nan, level=level)
    if len(data) == 1:
        v = data[0]
        return CI(mean=v, low=v, high=v, level=level)
    rng = rng or Random(0)
    means = []
    n = len(data)
    for _ in range(n_resamples):
        sample = [data[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    alpha = 1.0 - level
    lo_idx = int(alpha / 2.0 * n_resamples)
    hi_idx = int((1.0 - alpha / 2.0) * n_resamples) - 1
    return CI(
        mean=statistics.fmean(data),
        low=means[lo_idx],
        high=means[hi_idx],
        level=level,
    )


# -----------------------------------------------------------------------------
# Paired permutation test
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class PairedComparison:
    """Outcome of a paired two-sided permutation test."""

    mean_a: float
    mean_b: float
    diff: float  # mean_a - mean_b
    p_value: float
    n_pairs: int

    def describe(self) -> str:
        return (
            f"Δ={self.diff:+.3f} (a={self.mean_a:.3f}, b={self.mean_b:.3f}, "
            f"n={self.n_pairs}, p={self.p_value:.3f})"
        )


def paired_permutation_test(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_permutations: int = 5000,
    rng: Random | None = None,
) -> PairedComparison:
    """Two-sided paired permutation test on mean difference.

    Entries must be aligned: ``a[i]`` and ``b[i]`` are the two methods'
    values on the same seed. For each permutation each pair is independently
    swapped with probability 0.5; the p-value is the fraction of
    permutations where ``|mean_diff|`` is at least as large as the observed.
    """
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    if not a:
        return PairedComparison(
            mean_a=math.nan, mean_b=math.nan, diff=math.nan,
            p_value=math.nan, n_pairs=0,
        )
    rng = rng or Random(0)
    diffs = [ai - bi for ai, bi in zip(a, b)]
    observed = abs(sum(diffs) / len(diffs))
    n = len(diffs)
    hits = 0
    for _ in range(n_permutations):
        signs = [1 if rng.random() < 0.5 else -1 for _ in range(n)]
        m = abs(sum(s * d for s, d in zip(signs, diffs)) / n)
        if m >= observed:
            hits += 1
    # Add-one smoothing so we never report p=0 exactly.
    p = (hits + 1) / (n_permutations + 1)
    return PairedComparison(
        mean_a=statistics.fmean(a), mean_b=statistics.fmean(b),
        diff=statistics.fmean(a) - statistics.fmean(b),
        p_value=p, n_pairs=n,
    )


# -----------------------------------------------------------------------------
# Multiple-comparisons correction
# -----------------------------------------------------------------------------


def holm_bonferroni(
    p_values: Iterable[tuple[str, float]], *, alpha: float = 0.05
) -> list[tuple[str, float, float, bool]]:
    """Apply Holm–Bonferroni step-down correction.

    Args:
        p_values: Iterable of (label, p). Labels must be unique.
        alpha: Family-wise error rate.

    Returns:
        List of (label, raw_p, adjusted_p, reject) in input order. ``reject``
        is True when the adjusted p-value is ≤ alpha.
    """
    pairs = list(p_values)
    if not pairs:
        return []
    labels = [p[0] for p in pairs]
    if len(set(labels)) != len(labels):
        raise ValueError("Duplicate labels in p_values")

    indexed = sorted(enumerate(pairs), key=lambda t: t[1][1])
    m = len(pairs)
    adjusted: dict[int, float] = {}
    running_max = 0.0
    for rank, (orig_idx, (_, p)) in enumerate(indexed):
        adj = p * (m - rank)
        running_max = max(running_max, adj)
        adjusted[orig_idx] = min(1.0, running_max)

    return [
        (label, raw, adjusted[i], adjusted[i] <= alpha)
        for i, (label, raw) in enumerate(pairs)
    ]
