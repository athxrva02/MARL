"""Tests for HierarchicalCredit (per-leaf credit across composites)."""

from __future__ import annotations

from random import Random

import pytest

from framework.agent import SyntheticAgent
from framework.composite_agent import CompositeAgent
from framework.credit import (
    CounterfactualCredit,
    EndToEndCredit,
    HierarchicalCredit,
    ValidatorAnchoredCredit,
)
from framework.environment import Task
from framework.pipeline import Pipeline


def _composite(*, schema_ok=True, builder_ok=True, validator_ok=True, qa_ok=True) -> CompositeAgent:
    schema = SyntheticAgent(
        "SchemaAgent", ["x"], {"x": 1.0 if schema_ok else 0.0},
        confidence_noise=0.0,
    )
    builder = SyntheticAgent(
        "BuilderAgent", ["x"], {"x": 1.0 if builder_ok else 0.0},
        confidence_noise=0.0, upstream_failure_penalty=0.0,
    )
    validator = SyntheticAgent(
        "ValidatorAgent", ["x"], {"x": 1.0 if validator_ok else 0.0},
        confidence_noise=0.0, upstream_failure_penalty=0.0,
    )
    qa = SyntheticAgent(
        "QAAgent", ["x"], {"x": 1.0 if qa_ok else 0.0},
        confidence_noise=0.0, upstream_failure_penalty=0.0,
    )
    return CompositeAgent(
        name="CodeAgent",
        sub_pipeline=Pipeline([schema, builder, validator, qa]),
    )


def test_hierarchical_returns_per_leaf_credit_not_composite():
    doc = SyntheticAgent("Doc", ["x"], {"x": 1.0}, confidence_noise=0.0)
    comp = _composite()
    p = Pipeline([doc, comp])
    rng = Random(0)
    trace, outcome = p.run(Task(task_id="t", payload={}), rng)
    credits = HierarchicalCredit(EndToEndCredit()).assign(trace, outcome, p, rng)
    # Top-level flat agent is present, composite NAME is replaced by its leaves.
    assert "Doc" in credits
    assert "CodeAgent" not in credits
    assert {"SchemaAgent", "BuilderAgent", "ValidatorAgent", "QAAgent"} <= credits.keys()


def test_hierarchical_end_to_end_credits_equal_reward_on_success():
    doc = SyntheticAgent("Doc", ["x"], {"x": 1.0}, confidence_noise=0.0)
    comp = _composite()
    p = Pipeline([doc, comp])
    rng = Random(0)
    trace, outcome = p.run(Task(task_id="t", payload={}), rng)
    credits = HierarchicalCredit(EndToEndCredit(), outer_weight=0.0).assign(
        trace, outcome, p, rng
    )
    assert outcome.reward == 1.0
    for v in credits.values():
        assert v == 1.0


def test_hierarchical_inner_assigner_differentiates_sub_agents_on_failure():
    # Only BuilderAgent fails. Outer end-to-end gives 0 to everyone; inner
    # end-to-end also gives 0; so end-to-end hierarchical cannot discriminate.
    # But ValidatorAnchored as the INNER strategy should blame Builder more
    # (because when Builder fails -> validator fails -> inner credit 0).
    # Meanwhile outer stays 0 for the composite. We expect Builder to be
    # strictly no better than the other sub-agents.
    doc = SyntheticAgent("Doc", ["x"], {"x": 1.0}, confidence_noise=0.0)
    comp = _composite(builder_ok=False)
    p = Pipeline([doc, comp])
    rng = Random(0)
    trace, outcome = p.run(Task(task_id="t", payload={}), rng)
    credits = HierarchicalCredit(
        outer_assigner=EndToEndCredit(),
        inner_assigner=ValidatorAnchoredCredit(),
        outer_weight=0.0,
    ).assign(trace, outcome, p, rng)
    # Doc succeeded at outer level: outer_weight=0 means Doc gets its outer
    # credit (reward=0 since composite failed). Builder should tie or be
    # lowest among the leaves per validator-anchored semantics.
    inner_leaves = ["SchemaAgent", "BuilderAgent", "ValidatorAgent", "QAAgent"]
    inner_vals = [credits[a] for a in inner_leaves]
    assert min(inner_vals) == credits["BuilderAgent"] or len(set(inner_vals)) == 1


def test_hierarchical_outer_weight_zero_ignores_outer_credit():
    doc = SyntheticAgent("Doc", ["x"], {"x": 1.0}, confidence_noise=0.0)
    comp = _composite(builder_ok=False)
    p = Pipeline([doc, comp])
    rng = Random(0)
    trace, outcome = p.run(Task(task_id="t", payload={}), rng)
    # With validator-anchored as the inner assigner, Schema (succeeded
    # per-step) gets 1.0 and Builder (failed) gets 0.0 regardless of the
    # composite's outer credit when outer_weight=0.
    credits = HierarchicalCredit(
        outer_assigner=EndToEndCredit(),
        inner_assigner=ValidatorAnchoredCredit(),
        outer_weight=0.0,
    ).assign(trace, outcome, p, rng)
    # Validator-anchored falls back to per-step success on flat sub-pipelines.
    # Schema succeeded; Builder failed; Validator/QA failed downstream.
    assert credits["SchemaAgent"] == 1.0
    assert credits["BuilderAgent"] == 0.0

    # With outer_weight=1 every leaf inherits the composite's outer credit
    # (0.0 on failure), regardless of inner signal.
    credits_w1 = HierarchicalCredit(
        outer_assigner=EndToEndCredit(),
        inner_assigner=ValidatorAnchoredCredit(),
        outer_weight=1.0,
    ).assign(trace, outcome, p, rng)
    for a in ("SchemaAgent", "BuilderAgent", "ValidatorAgent", "QAAgent"):
        assert credits_w1[a] == 0.0


def test_hierarchical_rejects_invalid_outer_weight():
    with pytest.raises(ValueError):
        HierarchicalCredit(EndToEndCredit(), outer_weight=-0.1)
    with pytest.raises(ValueError):
        HierarchicalCredit(EndToEndCredit(), outer_weight=1.5)


def test_hierarchical_handles_pipeline_with_no_composites():
    p = Pipeline([
        SyntheticAgent("a", ["x"], {"x": 1.0}, confidence_noise=0.0),
        SyntheticAgent(
            "b", ["x"], {"x": 0.0},
            confidence_noise=0.0, upstream_failure_penalty=0.0,
        ),
    ])
    rng = Random(0)
    trace, outcome = p.run(Task(task_id="t", payload={}), rng)
    credits = HierarchicalCredit(EndToEndCredit()).assign(trace, outcome, p, rng)
    # Falls back to outer assignment for flat pipelines.
    assert credits == {"a": 0.0, "b": 0.0}


def test_hierarchical_uses_counterfactual_inside_composite():
    # Just a smoke test: ensure the wiring holds when the inner method is
    # rollout-based (counterfactual), not just structural (validator).
    doc = SyntheticAgent("Doc", ["x"], {"x": 1.0}, confidence_noise=0.0)
    comp = _composite()
    p = Pipeline([doc, comp])
    rng = Random(0)
    trace, outcome = p.run(Task(task_id="t", payload={}), rng)
    credits = HierarchicalCredit(
        outer_assigner=EndToEndCredit(),
        inner_assigner=CounterfactualCredit(n_samples=2),
        outer_weight=0.5,
    ).assign(trace, outcome, p, rng)
    # All 5 leaves present and in [0, 1].
    for a in ("Doc", "SchemaAgent", "BuilderAgent", "ValidatorAgent", "QAAgent"):
        assert 0.0 <= credits[a] <= 1.0
