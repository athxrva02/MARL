"""Tests for SelfDistillationCredit (SDPO-inspired calibration credit)."""

from __future__ import annotations

import math
from random import Random

import pytest

from framework.agent import SyntheticAgent
from framework.credit import SelfDistillationCredit
from framework.environment import Task
from framework.pipeline import Pipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(agents, *, task_id="t", seed=0):
    p = Pipeline(agents)
    rng = Random(seed)
    trace, outcome = p.run(Task(task_id=task_id, payload={}), rng)
    return p, trace, outcome


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


def test_invalid_temperature_zero():
    with pytest.raises(ValueError, match="temperature"):
        SelfDistillationCredit(temperature=0.0)


def test_invalid_temperature_negative():
    with pytest.raises(ValueError, match="temperature"):
        SelfDistillationCredit(temperature=-1.0)


def test_invalid_eps_zero():
    with pytest.raises(ValueError, match="eps"):
        SelfDistillationCredit(eps=0.0)


def test_invalid_eps_too_large():
    with pytest.raises(ValueError, match="eps"):
        SelfDistillationCredit(eps=0.6)


# ---------------------------------------------------------------------------
# Output invariants
# ---------------------------------------------------------------------------


def test_credits_in_unit_interval():
    p, trace, outcome = _run([
        SyntheticAgent("a", ["x"], {"x": 0.8}),
        SyntheticAgent("b", ["x"], {"x": 0.4}),
        SyntheticAgent("c", ["x"], {"x": 0.9}),
    ], seed=42)
    credits = SelfDistillationCredit().assign(trace, outcome, p, Random(0))
    for name, c in credits.items():
        assert 0.0 <= c <= 1.0, f"credit[{name}] = {c} outside [0, 1]"


def test_all_agents_receive_credit():
    p, trace, outcome = _run([
        SyntheticAgent("a", ["x"], {"x": 0.7}),
        SyntheticAgent("b", ["x"], {"x": 0.5}),
    ])
    credits = SelfDistillationCredit().assign(trace, outcome, p, Random(0))
    assert set(credits.keys()) == {"a", "b"}


def test_single_agent_pipeline():
    p, trace, outcome = _run([SyntheticAgent("only", ["x"], {"x": 0.6})])
    credits = SelfDistillationCredit().assign(trace, outcome, p, Random(0))
    assert "only" in credits
    assert 0.0 <= credits["only"] <= 1.0


# ---------------------------------------------------------------------------
# Core SDPO signal: calibration direction
# ---------------------------------------------------------------------------


def test_overconfident_on_failure_gets_low_credit():
    # Agent always succeeds (p=1), no upstream, so outcome.reward will be 1.
    # For SDPO signal we need: high confidence + low outcome.
    # Construct directly by running a failing pipeline and checking the
    # overconfident agent gets credit < 0.5.
    #
    # Pipeline: a (always fails, no noise) → b (always succeeds, no noise).
    # Outcome = 0 (because a failed).  Agent b reports high confidence (~1.0)
    # because it succeeded locally, but the outcome is 0.
    # SelfDistillationCredit: p_teacher = clip(0, eps, 1-eps) ≈ eps
    #                          p_student = high confidence ≈ 1.0
    #                          log_ratio = log(eps / 1.0) << 0 → credit << 0.5
    p = Pipeline([
        SyntheticAgent("a", ["x"], {"x": 0.0}, confidence_noise=0.0,
                       upstream_failure_penalty=0.0),
        SyntheticAgent("b", ["x"], {"x": 1.0}, confidence_noise=0.0,
                       upstream_failure_penalty=0.0),
    ])
    rng = Random(0)
    trace, outcome = p.run(Task(task_id="t", payload={}), rng)
    assert outcome.reward == 0.0, "expected pipeline failure"

    credits = SelfDistillationCredit(eps=1e-3).assign(trace, outcome, p, rng)
    # b had high confidence; outcome was bad → b is "overconfident" → low credit
    assert credits["b"] < 0.3, f"expected low credit for overconfident-on-failure b, got {credits['b']:.3f}"


def test_underconfident_on_success_gets_high_credit():
    # Inverse case: agent with LOW confidence, good outcome.
    # SyntheticAgent with low p_success (0.15) produces low confidence on success
    # (conf ≈ p + noise ≈ 0.15 for success case) while outcome.reward = 1.0.
    # log_ratio = log(1.0 / 0.15) >> 0 → credit >> 0.5.
    #
    # Force a success by running many trials and picking a successful one.
    p = Pipeline([
        SyntheticAgent("a", ["x"], {"x": 0.15}, confidence_noise=0.0),
    ])
    rng = Random(99)
    for _ in range(200):
        trace, outcome = p.run(Task(task_id="t", payload={}), rng)
        if outcome.reward == 1.0:
            credits = SelfDistillationCredit(eps=1e-3).assign(trace, outcome, p, rng)
            # Agent was underconfident (conf ≈ 0.15) but succeeded → teacher
            # says "should have been more confident" → high credit
            assert credits["a"] > 0.7, (
                f"expected high credit for underconfident-on-success a, "
                f"got {credits['a']:.3f} (conf={trace.steps[0].output.confidence:.3f})"
            )
            return
    pytest.fail("No successful episode found in 200 trials")


def test_calibrated_agent_near_neutral():
    # When confidence ≈ outcome.reward the log-ratio is near 0 → credit ≈ 0.5.
    # Use a medium-probability agent (p=0.6) which reports conf ≈ 0.6 on success.
    # Run until we get a success (outcome=1), then credit should be close to 0.5.
    p = Pipeline([
        SyntheticAgent("a", ["x"], {"x": 0.6}, confidence_noise=0.0),
    ])
    rng = Random(5)
    for _ in range(100):
        trace, outcome = p.run(Task(task_id="t", payload={}), rng)
        if outcome.reward == 1.0:
            conf = trace.steps[0].output.confidence
            # Manual prediction: log(1.0 / conf) → credit = sigmoid(log(1/conf))
            expected = _sigmoid(math.log(1.0 - 1e-3) - math.log(conf))
            credits = SelfDistillationCredit(eps=1e-3).assign(trace, outcome, p, rng)
            assert abs(credits["a"] - expected) < 1e-9
            return
    pytest.fail("No successful episode found in 100 trials")


# ---------------------------------------------------------------------------
# Silent-failure detection (key Phase-3 use case)
# ---------------------------------------------------------------------------


def test_silent_failure_gets_very_low_credit():
    # A silent failure: agent reports high confidence (p + noise) but output
    # is actually bad (success=False).  The pipeline outcome is therefore 0.
    # SelfDistillationCredit should assign very low credit to such an agent.
    # Use silent_failure_rate=1.0 so the effect is deterministic.
    p = Pipeline([
        SyntheticAgent(
            "tricky",
            ["x"],
            {"x": 0.0},           # always fails
            confidence_noise=0.0,
            silent_failure_rate=1.0,  # but always reports high confidence
        ),
    ])
    rng = Random(0)
    trace, outcome = p.run(Task(task_id="t", payload={}), rng)
    assert outcome.reward == 0.0

    # Confidence should be high (silent_failure_rate=1.0 → conf ≈ p_success + noise = 0.0 + 0.05*gauss)
    # With confidence_noise=0.0 and p=0.0, silent conf = p + gauss(0,0) = 0.0, then clipped...
    # Let me check: SyntheticAgent silent branch: conf = min(1.0, max(0.0, p + gauss(0, noise)))
    # With p=0.0 and noise=0.0: conf = 0.0.  That's not high.
    # Use p=0.8 with silent_failure_rate=1.0 instead:
    p2 = Pipeline([
        SyntheticAgent(
            "tricky",
            ["x"],
            {"x": 0.8},
            confidence_noise=0.0,
            silent_failure_rate=1.0,
            upstream_failure_penalty=0.0,
        ),
    ])
    # Override success: SyntheticAgent draws success = rng.random() < p.
    # With p=0.8 we might get a success; we need a failure.  Force by seeding.
    rng2 = Random(1)  # seed 1 gives rng.random() > 0.8 on first draw
    for _ in range(50):
        trace2, outcome2 = p2.run(Task(task_id="t", payload={}), Random(rng2.randint(0, 10000)))
        if outcome2.reward == 0.0:
            conf = trace2.steps[0].output.confidence
            credits = SelfDistillationCredit(eps=1e-3).assign(trace2, outcome2, p2, Random(0))
            # High confidence (≈ 0.8) + bad outcome (0) → very low credit
            assert conf > 0.5, f"expected high silent-failure confidence, got {conf:.3f}"
            assert credits["tricky"] < 0.15, (
                f"silent failure should get very low credit, got {credits['tricky']:.3f}"
            )
            return
    pytest.fail("Could not obtain a silent failure episode in 50 trials")


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------


def test_higher_temperature_increases_discrimination():
    # Same trace; higher temperature should push credits further from 0.5.
    p = Pipeline([
        SyntheticAgent("a", ["x"], {"x": 0.0}, confidence_noise=0.0,
                       upstream_failure_penalty=0.0),
        SyntheticAgent("b", ["x"], {"x": 1.0}, confidence_noise=0.0,
                       upstream_failure_penalty=0.0),
    ])
    rng = Random(0)
    trace, outcome = p.run(Task(task_id="t", payload={}), rng)

    low_temp = SelfDistillationCredit(temperature=0.5, eps=1e-3)
    high_temp = SelfDistillationCredit(temperature=3.0, eps=1e-3)
    c_low = low_temp.assign(trace, outcome, p, rng)
    c_high = high_temp.assign(trace, outcome, p, rng)

    # b has high conf + bad outcome: high_temp should push credit further below 0.5
    assert c_high["b"] < c_low["b"], (
        f"higher temperature should sharpen signal: "
        f"high={c_high['b']:.3f} should be < low={c_low['b']:.3f}"
    )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_deterministic_given_trace_and_outcome():
    p, trace, outcome = _run([
        SyntheticAgent("a", ["x"], {"x": 0.7}),
        SyntheticAgent("b", ["x"], {"x": 0.5}),
    ])
    assigner = SelfDistillationCredit()
    c1 = assigner.assign(trace, outcome, p, Random(0))
    c2 = assigner.assign(trace, outcome, p, Random(99))  # different rng, same result
    assert c1 == c2, "SelfDistillationCredit should be deterministic (rng unused)"


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_satisfies_credit_assigner_protocol():
    from framework.credit.base import CreditAssigner
    assert isinstance(SelfDistillationCredit(), CreditAssigner)


def test_name_attribute():
    assert SelfDistillationCredit.name == "self_distillation"
