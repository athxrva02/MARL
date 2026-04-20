"""End-to-end smoke test for the Phase-3 wiring.

Runs a tiny :mod:`experiments.rq3_phase3` simulation and checks:

  * The SQLite store ends up with the expected number of episode rows
    (``n_methods × n_seeds × n_episodes``).
  * The Supervisor arm receives updates (posterior_snapshots contain its
    per-preset posteriors).
  * The summary JSON has a per-method CI and a pairwise-vs-baseline block
    with one entry per non-baseline method.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from experiments.rq3_phase3 import main as phase3_main


@pytest.fixture
def tiny_config(tmp_path: Path) -> Path:
    cfg = {
        "experiment": {
            "name": "phase3_it_test",
            "seed": 1,
            "n_episodes": 6,
            "n_seeds": 2,
            "baseline": "end_to_end",
            "snapshot_every": 3,
            "bootstrap_resamples": 50,
            "permutation_resamples": 100,
            "alpha": 0.05,
        },
        "pipeline": {
            "upstream_failure_penalty": 0.3,
            "code_silent_failure_rate": 0.2,
            "max_iterations_hint": 5,
        },
        "learner": {"kind": "beta_bernoulli_thompson", "alpha0": 1.0, "beta0": 1.0},
        "tasks": {
            "doc_size_probs": {"small": 0.5, "medium": 0.3, "large": 0.2},
            "log_volume_probs": {"low": 0.3, "medium": 0.5, "high": 0.2},
            "pii_prob": 0.3,
        },
        "credit_methods": [
            {"kind": "end_to_end"},
            {"kind": "validator_anchored", "validator_name": "ValidatorAgent"},
        ],
    }
    p = tmp_path / "tiny.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_phase3_end_to_end_runs_and_persists(tmp_path: Path, tiny_config: Path):
    out_dir = tmp_path / "out"
    rc = phase3_main([
        "--config", str(tiny_config),
        "--output-dir", str(out_dir),
    ])
    assert rc == 0

    db_path = out_dir / "episodes.sqlite"
    assert db_path.exists()

    summary = json.loads((out_dir / "summary.json").read_text())
    assert set(summary["per_method"]) == {"end_to_end", "validator_anchored"}
    # Non-baseline methods should each have a pairwise row.
    assert {row["method"] for row in summary["pairwise_vs_baseline"]} == {
        "validator_anchored"
    }

    # Inspect the DB: 2 methods × 2 seeds × 6 episodes = 24 rows.
    import sqlite3
    conn = sqlite3.connect(db_path)
    try:
        (n_ep,) = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()
        assert n_ep == 24
        # Actions table carries one row per (agent, episode); we have 6
        # leaf agents (Doc, LogParser, Schema, Builder, Validator, QA) +
        # Supervisor = 7 entries per episode.
        (n_act,) = conn.execute("SELECT COUNT(*) FROM actions").fetchone()
        assert n_act == 24 * 7
        # Supervisor posteriors must be snapshot at least once.
        (n_sup,) = conn.execute(
            "SELECT COUNT(*) FROM posterior_snapshots WHERE agent = 'Supervisor'"
        ).fetchone()
        assert n_sup > 0
    finally:
        conn.close()
