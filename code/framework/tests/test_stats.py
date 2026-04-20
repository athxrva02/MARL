"""Tests for bootstrap CIs, paired permutation, Holm–Bonferroni."""

from __future__ import annotations

import math
from random import Random

import pytest

from framework.stats import (
    bootstrap_ci,
    holm_bonferroni,
    paired_permutation_test,
)


def test_bootstrap_ci_contains_true_mean_on_symmetric_data():
    data = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    ci = bootstrap_ci(data, n_resamples=1000, rng=Random(0))
    assert ci.mean == pytest.approx(0.25)
    assert ci.low <= 0.25 <= ci.high


def test_bootstrap_ci_empty_returns_nan():
    ci = bootstrap_ci([], rng=Random(0))
    assert math.isnan(ci.mean)
    assert math.isnan(ci.low)


def test_bootstrap_ci_single_point():
    ci = bootstrap_ci([0.42], rng=Random(0))
    assert ci.low == ci.high == ci.mean == 0.42


def test_paired_permutation_detects_large_effect():
    a = [1.0, 1.0, 1.0, 1.0, 1.0]
    b = [0.0, 0.0, 0.0, 0.0, 0.0]
    result = paired_permutation_test(a, b, n_permutations=2000, rng=Random(0))
    assert result.diff == pytest.approx(1.0)
    assert result.p_value < 0.1  # 1/32 ≈ 0.03 with 5 pairs


def test_paired_permutation_no_effect_gives_large_p():
    a = [0.5, 0.6, 0.4, 0.55]
    b = [0.5, 0.6, 0.4, 0.55]
    result = paired_permutation_test(a, b, n_permutations=500, rng=Random(0))
    assert result.diff == 0.0
    # With zero difference, p-value should be at the maximum (≈ 1.0 minus
    # add-one smoothing).
    assert result.p_value >= 0.9


def test_paired_permutation_length_mismatch_raises():
    with pytest.raises(ValueError):
        paired_permutation_test([1.0], [1.0, 2.0])


def test_holm_bonferroni_rejects_in_order():
    # Raw ps: 0.01 (strongest), 0.02, 0.04; alpha=0.05, m=3.
    # Holm: 0.01*3=0.03, 0.02*2=0.04, 0.04*1=0.04 -> all reject.
    results = holm_bonferroni(
        [("a", 0.01), ("b", 0.02), ("c", 0.04)], alpha=0.05
    )
    assert {r[0]: r[3] for r in results} == {"a": True, "b": True, "c": True}


def test_holm_bonferroni_respects_running_max():
    # If adj_p for rank k is less than adj_p for rank k-1, should clamp up.
    results = holm_bonferroni(
        [("a", 0.03), ("b", 0.001)], alpha=0.05
    )
    # Sorted: b(p=0.001) rank 0 -> adj=0.002; a(p=0.03) rank 1 -> adj=0.03
    adjusted = {r[0]: r[2] for r in results}
    assert adjusted["b"] < adjusted["a"]


def test_holm_bonferroni_empty():
    assert holm_bonferroni([]) == []


def test_holm_bonferroni_rejects_duplicate_labels():
    with pytest.raises(ValueError):
        holm_bonferroni([("a", 0.01), ("a", 0.02)])
