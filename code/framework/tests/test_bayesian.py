"""Tests for the Beta-Bernoulli Thompson sampling learner."""

from __future__ import annotations

from random import Random

import pytest

from framework.learner.bayesian import BetaBernoulliThompson


def test_uniform_prior_is_beta_1_1():
    l = BetaBernoulliThompson()
    # Before any updates, posterior mean of an unseen arm is 0.5 (Beta(1,1)).
    assert l.mean("agent", "action") == pytest.approx(0.5)


def test_success_update_increments_alpha():
    l = BetaBernoulliThompson()
    l.update("a", "x", credit=1.0)
    # α=2, β=1 -> mean = 2/3
    assert l.mean("a", "x") == pytest.approx(2.0 / 3.0)


def test_failure_update_increments_beta():
    l = BetaBernoulliThompson()
    l.update("a", "x", credit=0.0)
    # α=1, β=2 -> mean = 1/3
    assert l.mean("a", "x") == pytest.approx(1.0 / 3.0)


def test_soft_credit_is_well_defined():
    l = BetaBernoulliThompson()
    l.update("a", "x", credit=0.3)
    # α = 1 + 0.3 = 1.3, β = 1 + 0.7 = 1.7 -> mean = 1.3/3.0
    assert l.mean("a", "x") == pytest.approx(1.3 / 3.0)


def test_posterior_converges_with_many_updates():
    # With many successes, posterior mean should be close to 1.
    l = BetaBernoulliThompson()
    for _ in range(100):
        l.update("a", "x", credit=1.0)
    assert l.mean("a", "x") > 0.98


def test_choose_picks_high_mean_arm_after_enough_data():
    rng = Random(42)
    l = BetaBernoulliThompson()
    # Saturate arm 'good' with successes, 'bad' with failures.
    for _ in range(200):
        l.update("a", "good", credit=1.0)
        l.update("a", "bad", credit=0.0)
    picks = [l.choose("a", ["good", "bad"], rng) for _ in range(200)]
    assert picks.count("good") / len(picks) > 0.9


def test_credit_out_of_range_rejected():
    l = BetaBernoulliThompson()
    with pytest.raises(ValueError):
        l.update("a", "x", credit=1.5)
    with pytest.raises(ValueError):
        l.update("a", "x", credit=-0.1)


def test_invalid_prior_rejected():
    with pytest.raises(ValueError):
        BetaBernoulliThompson(alpha0=0.0, beta0=1.0)
    with pytest.raises(ValueError):
        BetaBernoulliThompson(alpha0=1.0, beta0=-1.0)


def test_choose_is_deterministic_under_seed():
    l = BetaBernoulliThompson()
    l.update("a", "x", 1.0)
    l.update("a", "y", 0.0)
    rng1 = Random(123)
    rng2 = Random(123)
    seq1 = [l.choose("a", ["x", "y"], rng1) for _ in range(20)]
    seq2 = [l.choose("a", ["x", "y"], rng2) for _ in range(20)]
    assert seq1 == seq2


def test_snapshot_shape():
    l = BetaBernoulliThompson()
    l.update("a", "x", 1.0)
    l.update("b", "y", 0.5)
    snap = l.snapshot()
    assert set(snap) == {"a", "b"}
    assert snap["a"]["x"] == [2.0, 1.0]
    assert snap["b"]["y"] == [1.5, 1.5]
