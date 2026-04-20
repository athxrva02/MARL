"""Tests for SQLite-backed ExperimentStore."""

from __future__ import annotations

from pathlib import Path

import pytest

from framework.learner import BetaBernoulliThompson
from framework.persistence import (
    EpisodeRow,
    ExperimentStore,
    PosteriorRow,
    posterior_rows_from_learner,
)


def test_store_round_trips_episode(tmp_path: Path):
    store = ExperimentStore(tmp_path / "test.sqlite")
    with store:
        row = EpisodeRow(
            experiment="exp1", method="end_to_end", seed=1, episode=0,
            task_id="t0",
            features={"doc_size": "small", "log_volume": "low", "has_pii": False},
            reward=1.0, num_iterations=2, stopped_reason="complete",
            planted_blame=None,
            credits={"a": 1.0, "b": 0.8},
            actions={"a": "x", "b": "y"},
            action_credits={"a": 1.0, "b": 0.8},
        )
        ep_id = store.record_episode(row)
        assert ep_id >= 1
        assert store.episode_count(experiment="exp1") == 1

        rows = store.fetch_episodes(experiment="exp1")
        assert len(rows) == 1
        assert rows[0]["reward"] == 1.0
        assert rows[0]["method"] == "end_to_end"
        assert rows[0]["planted_blame"] is None


def test_store_filters_by_method(tmp_path: Path):
    store = ExperimentStore(tmp_path / "test.sqlite")
    with store:
        for m in ("a", "b"):
            store.record_episode(EpisodeRow(
                experiment="exp", method=m, seed=1, episode=0, task_id=f"t-{m}",
                features=None, reward=0.5, num_iterations=1, stopped_reason="complete",
                planted_blame=None, credits={}, actions={}, action_credits={},
            ))
        assert len(store.fetch_episodes(experiment="exp")) == 2
        assert len(store.fetch_episodes(experiment="exp", method="a")) == 1


def test_posterior_snapshot_from_learner(tmp_path: Path):
    learner = BetaBernoulliThompson()
    learner.update("A", "x", 1.0)
    learner.update("A", "y", 0.0)
    rows = posterior_rows_from_learner(
        learner, experiment="e", method="m", seed=0, episode=5
    )
    names = {(r.agent, r.action) for r in rows}
    assert ("A", "x") in names
    assert ("A", "y") in names
    for r in rows:
        assert r.episode == 5

    store = ExperimentStore(tmp_path / "t.sqlite")
    with store:
        store.snapshot_posteriors(rows)
        # Basic integrity: round-trip via raw SQL.
        conn = store._conn
        assert conn is not None
        cur = conn.execute(
            "SELECT COUNT(*) FROM posterior_snapshots WHERE experiment = ?", ("e",)
        )
        assert cur.fetchone()[0] == 2


def test_store_persists_across_reopens(tmp_path: Path):
    path = tmp_path / "t.sqlite"
    with ExperimentStore(path) as store:
        store.record_episode(EpisodeRow(
            experiment="e", method="m", seed=0, episode=0, task_id="x",
            features=None, reward=1.0, num_iterations=1, stopped_reason="complete",
            planted_blame=None, credits={}, actions={}, action_credits={},
        ))
    # Re-open and verify data is still there.
    with ExperimentStore(path) as store:
        assert store.episode_count() == 1


def test_store_requires_context_for_writes(tmp_path: Path):
    store = ExperimentStore(tmp_path / "t.sqlite")
    with pytest.raises(AssertionError):
        store.record_episode(EpisodeRow(
            experiment="e", method="m", seed=0, episode=0, task_id="x",
            features=None, reward=0.0, num_iterations=1, stopped_reason="",
            planted_blame=None, credits={}, actions={}, action_credits={},
        ))
