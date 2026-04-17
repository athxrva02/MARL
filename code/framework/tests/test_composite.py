"""Tests for CompositeAgent (nested pipelines as a single Agent)."""

from __future__ import annotations

from random import Random

from framework.agent import Observation, SyntheticAgent
from framework.composite_agent import CompositeAgent, CompositeOutput, sub_trace_of
from framework.environment import Task
from framework.pipeline import Pipeline


def _sub_pipeline() -> Pipeline:
    a = SyntheticAgent("inner_a", ["x"], {"x": 1.0}, confidence_noise=0.0)
    b = SyntheticAgent(
        "inner_b", ["x"], {"x": 1.0},
        confidence_noise=0.0, upstream_failure_penalty=0.0,
    )
    return Pipeline([a, b])


def test_composite_agent_exposes_sub_trace():
    comp = CompositeAgent(name="Composite", sub_pipeline=_sub_pipeline())
    rng = Random(0)
    obs = Observation(payload={}, context={})
    out = comp.act(obs, rng)
    assert isinstance(out.value, CompositeOutput)
    sub = sub_trace_of(out)
    assert sub is not None
    assert [s.agent_name for s in sub.steps] == ["inner_a", "inner_b"]
    # Composite should succeed when both sub-agents succeed.
    assert out.success is True


def test_composite_agent_propagates_sub_failure():
    a = SyntheticAgent("inner_a", ["x"], {"x": 0.0}, confidence_noise=0.0)
    b = SyntheticAgent(
        "inner_b", ["x"], {"x": 1.0},
        confidence_noise=0.0, upstream_failure_penalty=1.0,
    )
    comp = CompositeAgent(name="Composite", sub_pipeline=Pipeline([a, b]))
    rng = Random(0)
    out = comp.act(Observation(payload={}, context={}), rng)
    assert out.success is False
    sub = sub_trace_of(out)
    assert sub is not None
    assert sub.steps[0].output.success is False


def test_composite_in_outer_pipeline_runs_end_to_end():
    outer_pre = SyntheticAgent("pre", ["x"], {"x": 1.0}, confidence_noise=0.0)
    comp = CompositeAgent(name="MidComposite", sub_pipeline=_sub_pipeline())
    outer_post = SyntheticAgent(
        "post", ["x"], {"x": 1.0},
        confidence_noise=0.0, upstream_failure_penalty=0.0,
    )
    outer = Pipeline([outer_pre, comp, outer_post])
    rng = Random(0)
    trace, outcome = outer.run(Task(task_id="t", payload={}), rng)
    names = [s.agent_name for s in trace.steps]
    assert names == ["pre", "MidComposite", "post"]
    # From the outer view, the composite is a single step with a rich value.
    mid_output = trace.output_of("MidComposite")
    assert sub_trace_of(mid_output) is not None
    assert outcome.reward == 1.0


def test_composite_action_strategies():
    sub = _sub_pipeline()
    ensemble = CompositeAgent("C1", sub_pipeline=sub, action_id_strategy="ensemble")
    first = CompositeAgent("C2", sub_pipeline=sub, action_id_strategy="first_action")
    rng = Random(0)
    obs = Observation(payload={}, context={})
    assert ensemble.act(obs, rng).action_id == "ensemble"
    out2 = first.act(obs, Random(0))
    # first_action should pull from the first sub-agent's action_id.
    assert out2.action_id == "x"
