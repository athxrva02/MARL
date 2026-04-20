"""Tests for the learnable Supervisor (policy-hyperparameter arm)."""

from __future__ import annotations

from random import Random

import pytest

from framework.agent import SyntheticAgent
from framework.environment import Task, default_outcome
from framework.pipeline import Pipeline, Trace, TraceStep
from framework.agent import AgentOutput, Observation
from framework.refinement import EpisodeRun, Iteration, RefinementLoop
from framework.supervisor_agent import (
    DEFAULT_PRESETS,
    SUPERVISOR_AGENT_NAME,
    SupervisorAgent,
    SupervisorPreset,
    preset_to_policy,
    supervisor_credit,
)


def test_default_presets_cover_three_operating_points():
    names = set(DEFAULT_PRESETS.keys())
    assert names == {"aggressive", "balanced", "conservative"}
    # Thresholds strictly decrease and iteration caps strictly increase.
    thresholds = [DEFAULT_PRESETS[k].complete_threshold
                  for k in ("aggressive", "balanced", "conservative")]
    max_iters = [DEFAULT_PRESETS[k].max_iterations
                 for k in ("aggressive", "balanced", "conservative")]
    assert thresholds == sorted(thresholds, reverse=True)
    assert max_iters == sorted(max_iters)


def test_supervisor_agent_exposes_named_actions():
    sup = SupervisorAgent()
    assert sup.name == SUPERVISOR_AGENT_NAME
    assert sup.actions == ["aggressive", "balanced", "conservative"]


def test_supervisor_agent_raises_on_empty_presets():
    with pytest.raises(ValueError):
        SupervisorAgent(presets={})


def test_preset_to_policy_builds_matching_completeness_policy():
    preset = SupervisorPreset(complete_threshold=0.7, max_iterations=4)
    policy = preset_to_policy(preset)
    # Drive a trivially-complete run through it to verify threshold honoured.
    run = EpisodeRun(task_id="t")
    step = TraceStep(
        agent_name="a",
        observation=Observation(payload={}, context={}),
        output=AgentOutput(value={}, confidence=1.0, action_id="x", success=True),
    )
    trace = Trace(task_id="t", steps=(step,))
    run.iterations.append(
        Iteration(index=0, trace=trace, outcome=default_outcome({"a": True}))
    )
    assert policy.decide(run, Random(0)) == "complete"  # reward 1.0 > 0.7


def test_supervisor_credit_rewards_fast_success():
    run = EpisodeRun(task_id="t")
    run.iterations.append(_iter(0, success=True))
    run.final_outcome = run.iterations[-1].outcome
    fast = supervisor_credit(run, max_iterations_hint=5)

    slow_run = EpisodeRun(task_id="t")
    for k in range(5):
        slow_run.iterations.append(_iter(k, success=(k == 4)))
    slow_run.final_outcome = slow_run.iterations[-1].outcome
    slow = supervisor_credit(slow_run, max_iterations_hint=5)

    # Both succeeded, but the 1-iteration run should score higher.
    assert fast > slow
    assert fast == pytest.approx(1.0)
    assert 0.0 <= slow <= 1.0


def test_supervisor_credit_punishes_stagnant_failure():
    run = EpisodeRun(task_id="t")
    for k in range(3):
        run.iterations.append(_iter(k, success=False))
    run.final_outcome = run.iterations[-1].outcome
    c = supervisor_credit(run, max_iterations_hint=3)
    assert c == 0.0


def test_supervisor_credit_is_clamped():
    run = EpisodeRun(task_id="t")
    run.iterations.append(_iter(0, success=True))
    run.final_outcome = run.iterations[-1].outcome
    c = supervisor_credit(run, iteration_cost_weight=10.0, max_iterations_hint=5)
    assert 0.0 <= c <= 1.0


def test_supervisor_preset_drives_refinement_loop_termination():
    # Pipeline that always fails. With stagnation disabled, aggressive stops
    # at max=2 iters and conservative at max=5. This isolates the
    # max_iterations knob from the stagnation detector (tested elsewhere).
    p = Pipeline([SyntheticAgent("a", ["x"], {"x": 0.0}, confidence_noise=0.0)])
    aggressive = preset_to_policy(SupervisorPreset(
        complete_threshold=1.0, max_iterations=2, stagnation_window=99
    ))
    conservative = preset_to_policy(SupervisorPreset(
        complete_threshold=0.5, max_iterations=5, stagnation_window=99
    ))

    run_a = RefinementLoop(p, aggressive).run_episode(
        Task(task_id="t", payload={}), Random(0)
    )
    run_c = RefinementLoop(p, conservative).run_episode(
        Task(task_id="t", payload={}), Random(0)
    )
    assert run_a.num_iterations == 2
    assert run_c.num_iterations == 5


# --- helpers -----------------------------------------------------------------

def _iter(index: int, *, success: bool) -> Iteration:
    step = TraceStep(
        agent_name="a",
        observation=Observation(payload={}, context={}),
        output=AgentOutput(
            value={}, confidence=0.5, action_id="x", success=success
        ),
    )
    trace = Trace(task_id="t", steps=(step,))
    return Iteration(index=index, trace=trace,
                     outcome=default_outcome({"a": success}))
