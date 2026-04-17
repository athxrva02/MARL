"""Tests for ValidatorAnchoredCredit (hierarchical credit on composite agents)."""

from __future__ import annotations

from random import Random

from framework.agent import SyntheticAgent
from framework.composite_agent import CompositeAgent
from framework.credit.validator_anchored import ValidatorAnchoredCredit
from framework.environment import Task
from framework.pipeline import Pipeline


def _composite_with_validator(*, validator_succeeds: bool) -> CompositeAgent:
    schema = SyntheticAgent("SchemaAgent", ["x"], {"x": 1.0}, confidence_noise=0.0)
    builder = SyntheticAgent(
        "BuilderAgent", ["x"], {"x": 1.0},
        confidence_noise=0.0, upstream_failure_penalty=0.0,
    )
    validator = SyntheticAgent(
        "ValidatorAgent", ["x"], {"x": 1.0 if validator_succeeds else 0.0},
        confidence_noise=0.0, upstream_failure_penalty=0.0,
    )
    qa = SyntheticAgent(
        "QAAgent", ["x"], {"x": 1.0},
        confidence_noise=0.0, upstream_failure_penalty=0.0,
    )
    return CompositeAgent(
        name="CodeAgent",
        sub_pipeline=Pipeline([schema, builder, validator, qa]),
    )


def test_validator_anchored_flat_agents_use_per_step_success():
    p = Pipeline([
        SyntheticAgent("a", ["x"], {"x": 1.0}, confidence_noise=0.0),
        SyntheticAgent(
            "b", ["x"], {"x": 0.0},
            confidence_noise=0.0, upstream_failure_penalty=0.0,
        ),
    ])
    rng = Random(0)
    trace, outcome = p.run(Task(task_id="t", payload={}), rng)
    credits = ValidatorAnchoredCredit().assign(trace, outcome, p, rng)
    assert credits == {"a": 1.0, "b": 0.0}


def test_validator_anchored_blesses_composite_when_validator_succeeds():
    comp = _composite_with_validator(validator_succeeds=True)
    p = Pipeline([comp])
    rng = Random(0)
    trace, outcome = p.run(Task(task_id="t", payload={}), rng)
    credits = ValidatorAnchoredCredit().assign(trace, outcome, p, rng)
    assert credits["CodeAgent"] == 1.0


def test_validator_anchored_blames_composite_when_validator_fails():
    comp = _composite_with_validator(validator_succeeds=False)
    p = Pipeline([comp])
    rng = Random(0)
    trace, outcome = p.run(Task(task_id="t", payload={}), rng)
    credits = ValidatorAnchoredCredit().assign(trace, outcome, p, rng)
    assert credits["CodeAgent"] == 0.0


def test_validator_anchored_falls_back_when_no_validator():
    schema = SyntheticAgent("S", ["x"], {"x": 1.0}, confidence_noise=0.0)
    builder = SyntheticAgent(
        "B", ["x"], {"x": 0.0},
        confidence_noise=0.0, upstream_failure_penalty=0.0,
    )
    comp = CompositeAgent("NoVal", sub_pipeline=Pipeline([schema, builder]))
    p = Pipeline([comp])
    rng = Random(0)
    trace, outcome = p.run(Task(task_id="t", payload={}), rng)
    credits = ValidatorAnchoredCredit().assign(trace, outcome, p, rng)
    # Fallback: mean of per-step success = (1 + 0) / 2 = 0.5
    assert credits["NoVal"] == 0.5
