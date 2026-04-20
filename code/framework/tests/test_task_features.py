"""Tests for feature-conditioned tasks + agents."""

from __future__ import annotations

from random import Random

import pytest

from framework.agent_features import FeatureAwareAgent, FeatureEffect
from framework.agent import Observation
from framework.environment import Task
from framework.task_generator import (
    TaskFeatureDistribution,
    TaskFeatures,
    TaskGenerator,
)


def test_task_generator_is_deterministic_given_seed():
    gen_a = TaskGenerator(seed=42)
    gen_b = TaskGenerator(seed=42)
    tasks_a = list(gen_a.stream(50))
    tasks_b = list(gen_b.stream(50))
    assert [t.task_id for t in tasks_a] == [t.task_id for t in tasks_b]
    assert [t.context["features"] for t in tasks_a] == \
           [t.context["features"] for t in tasks_b]


def test_task_generator_respects_distribution():
    # A skewed distribution should yield biased samples.
    dist = TaskFeatureDistribution(
        doc_size_probs={"small": 0.0, "medium": 0.0, "large": 1.0},
        log_volume_probs={"low": 0.0, "medium": 1.0, "high": 0.0},
        pii_prob=1.0,
    )
    gen = TaskGenerator(seed=0, distribution=dist)
    tasks = list(gen.stream(20))
    for t in tasks:
        feats = t.context["features"]
        assert feats["doc_size"] == "large"
        assert feats["log_volume"] == "medium"
        assert feats["has_pii"] is True


def test_task_generator_id_prefix():
    gen = TaskGenerator(seed=0, id_prefix="myexp")
    first = next(iter(gen.stream(1)))
    assert first.task_id.startswith("myexp")


def test_feature_aware_agent_varies_by_feature():
    # Base p_success = 0.9. Effect: -0.8 when doc_size=large -> effective 0.1.
    agent = FeatureAwareAgent(
        name="LogParser",
        actions=["shallow", "deep"],
        base_p_success={"shallow": 0.9, "deep": 0.9},
        effects={"shallow": [FeatureEffect(when={"doc_size": "large"}, delta=-0.8)]},
        confidence_noise=0.0,
    )
    small_successes = 0
    large_successes = 0
    rng = Random(0)
    N = 400
    for _ in range(N):
        small_obs = Observation(
            payload={}, context={"features": {"doc_size": "small"},
                                 "action_for:LogParser": "shallow"}
        )
        large_obs = Observation(
            payload={}, context={"features": {"doc_size": "large"},
                                 "action_for:LogParser": "shallow"}
        )
        if agent.act(small_obs, rng).success:
            small_successes += 1
        if agent.act(large_obs, rng).success:
            large_successes += 1
    # Expect ~90% for small, ~10% for large.
    assert small_successes / N > 0.8
    assert large_successes / N < 0.2


def test_feature_aware_agent_honours_injected_action():
    agent = FeatureAwareAgent(
        name="A",
        actions=["a1", "a2"],
        base_p_success={"a1": 1.0, "a2": 0.0},
        confidence_noise=0.0,
    )
    obs = Observation(
        payload={}, context={"features": {}, "action_for:A": "a2"}
    )
    out = agent.act(obs, Random(0))
    assert out.action_id == "a2"
    assert out.success is False


def test_feature_aware_agent_rejects_unknown_injected_action():
    agent = FeatureAwareAgent(
        name="A",
        actions=["a1"],
        base_p_success={"a1": 1.0},
        confidence_noise=0.0,
    )
    obs = Observation(
        payload={}, context={"features": {}, "action_for:A": "does-not-exist"}
    )
    with pytest.raises(ValueError):
        agent.act(obs, Random(0))


def test_feature_aware_agent_clamps_p_to_valid_range():
    # Effect drives p below 0 or above 1 — should clip.
    agent = FeatureAwareAgent(
        name="A",
        actions=["a"],
        base_p_success={"a": 0.1},
        effects={"a": [FeatureEffect(when={"k": "v"}, delta=-1.0)]},
        confidence_noise=0.0,
    )
    obs = Observation(
        payload={}, context={"features": {"k": "v"}, "action_for:A": "a"}
    )
    # Should never succeed (effective p = 0), but shouldn't raise either.
    for _ in range(50):
        assert agent.act(obs, Random()).success is False
