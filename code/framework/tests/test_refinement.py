"""Tests for the refinement loop and iteration-aware credit."""

from __future__ import annotations

from random import Random

import pytest

from framework.agent import SyntheticAgent
from framework.credit.iteration_discounted import IterationDiscountedCredit
from framework.environment import Task
from framework.pipeline import Pipeline
from framework.refinement import (
    CompletenessDecisionPolicy,
    EpisodeRun,
    RefinementLoop,
)


def _perfect_pipeline() -> Pipeline:
    return Pipeline([
        SyntheticAgent("a", ["x"], {"x": 1.0}, confidence_noise=0.0),
        SyntheticAgent(
            "b", ["x"], {"x": 1.0},
            confidence_noise=0.0, upstream_failure_penalty=0.0,
        ),
    ])


def _broken_pipeline() -> Pipeline:
    # Deterministic failure on agent b.
    return Pipeline([
        SyntheticAgent("a", ["x"], {"x": 1.0}, confidence_noise=0.0),
        SyntheticAgent(
            "b", ["x"], {"x": 0.0},
            confidence_noise=0.0, upstream_failure_penalty=0.0,
        ),
    ])


def test_completeness_policy_stops_on_success():
    policy = CompletenessDecisionPolicy(complete_threshold=1.0, max_iterations=3)
    loop = RefinementLoop(_perfect_pipeline(), policy)
    run = loop.run_episode(Task(task_id="t", payload={}), Random(0))
    assert run.num_iterations == 1
    assert run.stopped_reason == "complete"
    assert run.final_reward == 1.0


def test_completeness_policy_caps_at_max_iterations():
    policy = CompletenessDecisionPolicy(complete_threshold=1.0, max_iterations=3,
                                        stagnation_window=99)  # disable stagnation
    loop = RefinementLoop(_broken_pipeline(), policy)
    run = loop.run_episode(Task(task_id="t", payload={}), Random(0))
    assert run.num_iterations == 3
    assert run.stopped_reason == "stagnant"
    assert run.final_reward == 0.0


def test_completeness_policy_detects_stagnation_window():
    policy = CompletenessDecisionPolicy(
        complete_threshold=1.0, max_iterations=10, stagnation_window=2
    )
    loop = RefinementLoop(_broken_pipeline(), policy)
    run = loop.run_episode(Task(task_id="t", payload={}), Random(0))
    # Both the first two iterations return reward 0 (identical) -> stagnant.
    assert run.num_iterations == 2
    assert run.stopped_reason == "stagnant"


def test_refinement_context_carries_iteration_index():
    policy = CompletenessDecisionPolicy(complete_threshold=1.0, max_iterations=3,
                                        stagnation_window=99)
    loop = RefinementLoop(_broken_pipeline(), policy)
    run = loop.run_episode(
        Task(task_id="t", payload={}, context={"foo": "bar"}),
        Random(0),
    )
    # Inspect the observation seen by the first agent on each iteration.
    for k, it in enumerate(run.iterations):
        first_obs = it.trace.steps[0].observation
        assert first_obs.context["iteration_index"] == k
        assert first_obs.context["is_refinement"] is (k > 0)
        assert first_obs.context["foo"] == "bar"


def test_iteration_discounted_weights_early_success_more():
    # Construct a run where agent "a" succeeds iteration 0 and agent "b"
    # succeeds only iteration 2. With gamma=0.5, a should get more credit
    # than b even though each succeeded once.
    policy = CompletenessDecisionPolicy(complete_threshold=1.0, max_iterations=3,
                                        stagnation_window=99)
    # Use alternating-success hand-built iterations via a controlled pipeline.
    # Simpler: just build EpisodeRun manually.
    from framework.environment import default_outcome
    from framework.pipeline import Trace, TraceStep
    from framework.agent import AgentOutput, Observation
    from framework.refinement import Iteration

    def step(name: str, success: bool) -> TraceStep:
        return TraceStep(
            agent_name=name,
            observation=Observation(payload={}, context={}),
            output=AgentOutput(
                value={}, confidence=0.5, action_id="x", success=success
            ),
        )

    def iter_of(index: int, a_ok: bool, b_ok: bool) -> Iteration:
        trace = Trace(task_id="t", steps=(step("a", a_ok), step("b", b_ok)))
        outcome = default_outcome({"a": a_ok, "b": b_ok})
        return Iteration(index=index, trace=trace, outcome=outcome)

    run = EpisodeRun(task_id="t")
    run.iterations = [
        iter_of(0, a_ok=True, b_ok=False),
        iter_of(1, a_ok=False, b_ok=False),
        iter_of(2, a_ok=False, b_ok=True),
    ]
    run.final_outcome = run.iterations[-1].outcome

    credits = IterationDiscountedCredit(gamma=0.5).assign(run, Random(0))
    # Weights: a = 1·0.5^0 = 1.0 over total = 1 + 0.5 + 0.25 = 1.75 -> 0.571
    #          b = 1·0.5^2 = 0.25 / 1.75                          = 0.143
    # Then blended 50/50 with final_reward = 0 (b_ok alone is partial).
    assert credits["a"] > credits["b"]


def test_iteration_discounted_invalid_gamma():
    with pytest.raises(ValueError):
        IterationDiscountedCredit(gamma=0.0)
    with pytest.raises(ValueError):
        IterationDiscountedCredit(gamma=1.5)


def test_iteration_discounted_empty_run_returns_empty_dict():
    credits = IterationDiscountedCredit().assign(EpisodeRun(task_id="t"), Random(0))
    assert credits == {}
