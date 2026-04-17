"""Smoke test for the end-to-end experiment loop on the AIDI pipeline."""

from __future__ import annotations

import sys
from pathlib import Path
from random import Random

# Make ``code/`` importable when running pytest directly from repo root.
_REPO_CODE = Path(__file__).resolve().parents[2]
if str(_REPO_CODE) not in sys.path:
    sys.path.insert(0, str(_REPO_CODE))

from aidi_instance import build_aidi_pipeline
from framework.credit import CounterfactualCredit, EndToEndCredit
from framework.environment import Task
from framework.experiment import run_episode
from framework.learner import BetaBernoulliThompson


def test_aidi_pipeline_has_four_agents():
    p = build_aidi_pipeline()
    assert p.agent_names == [
        "DocumentAnalysisAgent",
        "LogParserAgent",
        "DocumentOnboardingSupervisor",
        "CodeAgent",
    ]


def test_episode_updates_learner_and_returns_credit():
    rng = Random(0)
    pipeline = build_aidi_pipeline()
    learner = BetaBernoulliThompson()
    credit_assigner = EndToEndCredit()

    task = Task(task_id="t0", payload={}, context={})
    trace, outcome, credit, actions = run_episode(
        pipeline, task, learner, credit_assigner, rng
    )
    assert set(credit) == set(pipeline.agent_names)
    assert set(actions) == set(pipeline.agent_names)
    # End-to-end: all credits equal.
    assert len(set(credit.values())) == 1
    snap = learner.snapshot()
    assert set(snap) == set(pipeline.agent_names)


def test_counterfactual_assigns_differentiated_credit():
    # On the AIDI pipeline a counterfactual assigner should (on average)
    # differentiate between agents rather than give them all the same value
    # the way end-to-end does.
    rng = Random(0)
    pipeline = build_aidi_pipeline()
    learner = BetaBernoulliThompson()
    credit_assigner = CounterfactualCredit(n_samples=4)

    seen_differentiated = False
    for i in range(30):
        task = Task(task_id=f"t{i}", payload={}, context={})
        _, _, credit, _ = run_episode(
            pipeline, task, learner, credit_assigner, rng
        )
        if len(set(round(v, 6) for v in credit.values())) > 1:
            seen_differentiated = True
            break
    assert seen_differentiated, "counterfactual should differentiate at least once"


def test_run_episode_deterministic_under_fixed_seed():
    def one_run(seed: int):
        rng = Random(seed)
        pipeline = build_aidi_pipeline()
        learner = BetaBernoulliThompson()
        credit_assigner = EndToEndCredit()
        results = []
        for i in range(10):
            task = Task(task_id=f"t{i}", payload={}, context={})
            _, outcome, credit, actions = run_episode(
                pipeline, task, learner, credit_assigner, rng
            )
            results.append((outcome.reward, tuple(sorted(actions.items())),
                            tuple(sorted(credit.items()))))
        return results

    assert one_run(42) == one_run(42)
