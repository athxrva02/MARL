"""Tests for pipeline execution and trace semantics."""

from __future__ import annotations

from random import Random

import pytest

from framework.agent import SyntheticAgent
from framework.environment import Task
from framework.pipeline import Pipeline


def _always_agent(name: str, *, succeed: bool) -> SyntheticAgent:
    return SyntheticAgent(
        name=name,
        actions=["only"],
        p_success={"only": 1.0 if succeed else 0.0},
        confidence_noise=0.0,
    )


def test_duplicate_agent_names_rejected():
    a = _always_agent("dup", succeed=True)
    b = _always_agent("dup", succeed=True)
    with pytest.raises(ValueError):
        Pipeline([a, b])


def test_all_success_gives_reward_one():
    pipeline = Pipeline([_always_agent("a", succeed=True),
                         _always_agent("b", succeed=True)])
    rng = Random(0)
    _, outcome = pipeline.run(Task(task_id="t", payload={}), rng)
    assert outcome.success is True
    assert outcome.reward == 1.0


def test_any_failure_gives_reward_zero():
    pipeline = Pipeline([_always_agent("a", succeed=True),
                         _always_agent("b", succeed=False)])
    rng = Random(0)
    _, outcome = pipeline.run(Task(task_id="t", payload={}), rng)
    assert outcome.success is False
    assert outcome.reward == 0.0


def test_run_is_deterministic_under_seed():
    pipeline = Pipeline([
        SyntheticAgent(
            name="a", actions=["x", "y"], p_success={"x": 0.5, "y": 0.5},
            confidence_noise=0.1,
        ),
        SyntheticAgent(
            name="b", actions=["p", "q"], p_success={"p": 0.5, "q": 0.5},
            confidence_noise=0.1,
        ),
    ])
    rng1 = Random(2026)
    rng2 = Random(2026)
    t = Task(task_id="t", payload={})
    trace1, out1 = pipeline.run(t, rng1)
    trace2, out2 = pipeline.run(t, rng2)
    assert out1 == out2
    assert [s.output for s in trace1.steps] == [s.output for s in trace2.steps]


def test_rerun_from_preserves_prefix():
    pipeline = Pipeline([
        _always_agent("a", succeed=True),
        _always_agent("b", succeed=True),
        _always_agent("c", succeed=False),
    ])
    rng = Random(1)
    trace, _ = pipeline.run(Task(task_id="t", payload={}), rng)
    # Re-run from "b" should preserve step "a" verbatim.
    trace2, _ = pipeline.rerun_from(trace, "b", Random(99))
    assert trace2.steps[0] == trace.steps[0]
    assert trace2.steps[1].agent_name == "b"
    assert trace2.steps[2].agent_name == "c"
