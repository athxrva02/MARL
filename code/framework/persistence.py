"""SQLite persistence for experiment episode logs and learner posteriors.

Per the assignment's note "to learn from previous mistakes and find
optimal policies, we should store the errors and correct decisions in an
appropriate database (Take SQLite for now)", Phase-3 persists every
episode's outcome, the action each agent chose, and periodic learner
posterior snapshots.

Schema (three tables):

    episodes
        id INTEGER PRIMARY KEY
        experiment TEXT   -- e.g. "rq3_supervised_feature_smoke"
        method TEXT       -- credit method name
        seed INTEGER
        episode INTEGER   -- episode index within this (experiment, method, seed)
        task_id TEXT
        features TEXT     -- JSON-encoded TaskFeatures
        reward REAL
        num_iterations INTEGER
        stopped_reason TEXT
        planted_blame TEXT   -- NULL if not planted
        credits TEXT      -- JSON-encoded dict[agent, credit]
        created_at TEXT   -- ISO 8601

    actions
        id INTEGER PRIMARY KEY
        episode_id INTEGER REFERENCES episodes(id)
        agent TEXT
        action TEXT
        credit REAL      -- credit this agent received this episode

    posterior_snapshots
        id INTEGER PRIMARY KEY
        experiment TEXT
        method TEXT
        seed INTEGER
        episode INTEGER   -- snapshot taken AFTER this episode
        agent TEXT
        action TEXT
        alpha REAL
        beta REAL

The DB is intended to be inspected via plain SQL or pandas; no heavy ORM.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping


_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment TEXT NOT NULL,
    method TEXT NOT NULL,
    seed INTEGER NOT NULL,
    episode INTEGER NOT NULL,
    task_id TEXT NOT NULL,
    features TEXT,
    reward REAL NOT NULL,
    num_iterations INTEGER NOT NULL,
    stopped_reason TEXT,
    planted_blame TEXT,
    credits TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_episodes_lookup
    ON episodes(experiment, method, seed, episode);

CREATE TABLE IF NOT EXISTS actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id INTEGER NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    agent TEXT NOT NULL,
    action TEXT NOT NULL,
    credit REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_actions_episode ON actions(episode_id);

CREATE TABLE IF NOT EXISTS posterior_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment TEXT NOT NULL,
    method TEXT NOT NULL,
    seed INTEGER NOT NULL,
    episode INTEGER NOT NULL,
    agent TEXT NOT NULL,
    action TEXT NOT NULL,
    alpha REAL NOT NULL,
    beta REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_posterior_lookup
    ON posterior_snapshots(experiment, method, seed, episode);
"""


@dataclass
class EpisodeRow:
    experiment: str
    method: str
    seed: int
    episode: int
    task_id: str
    features: Mapping | None
    reward: float
    num_iterations: int
    stopped_reason: str
    planted_blame: str | None
    credits: Mapping[str, float]
    actions: Mapping[str, str]  # agent -> action
    action_credits: Mapping[str, float]  # agent -> credit given this episode


@dataclass
class PosteriorRow:
    experiment: str
    method: str
    seed: int
    episode: int
    agent: str
    action: str
    alpha: float
    beta: float


class ExperimentStore:
    """SQLite-backed episode + posterior store.

    Intended usage::

        store = ExperimentStore(Path("experiments.sqlite"))
        with store:
            store.record_episode(row)
            ...
            store.snapshot_posteriors(rows)

    The same file is appended to across runs; use the (experiment, method,
    seed) triple to scope queries.
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._conn: sqlite3.Connection | None = None

    # -- context management ------------------------------------------------

    def __enter__(self) -> "ExperimentStore":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def open(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.commit()
            self._conn.close()
            self._conn = None

    # -- writes ------------------------------------------------------------

    def record_episode(self, row: EpisodeRow) -> int:
        """Persist one episode's outcome + actions. Returns the ep id."""
        assert self._conn is not None, "call open() or use as context manager"
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        cur = self._conn.execute(
            """
            INSERT INTO episodes (
                experiment, method, seed, episode, task_id, features,
                reward, num_iterations, stopped_reason, planted_blame,
                credits, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row.experiment, row.method, row.seed, row.episode, row.task_id,
                json.dumps(row.features) if row.features is not None else None,
                row.reward, row.num_iterations, row.stopped_reason,
                row.planted_blame, json.dumps(dict(row.credits)), now,
            ),
        )
        episode_id = cur.lastrowid
        self._conn.executemany(
            "INSERT INTO actions (episode_id, agent, action, credit) VALUES (?, ?, ?, ?)",
            [
                (episode_id, agent, action, float(row.action_credits.get(agent, 0.5)))
                for agent, action in row.actions.items()
            ],
        )
        self._conn.commit()
        return episode_id

    def snapshot_posteriors(self, rows: Iterable[PosteriorRow]) -> None:
        assert self._conn is not None, "call open() or use as context manager"
        self._conn.executemany(
            """
            INSERT INTO posterior_snapshots (
                experiment, method, seed, episode, agent, action, alpha, beta
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (r.experiment, r.method, r.seed, r.episode, r.agent, r.action,
                 r.alpha, r.beta)
                for r in rows
            ],
        )
        self._conn.commit()

    # -- reads -------------------------------------------------------------

    def episode_count(self, *, experiment: str | None = None) -> int:
        assert self._conn is not None
        if experiment is None:
            cur = self._conn.execute("SELECT COUNT(*) FROM episodes")
        else:
            cur = self._conn.execute(
                "SELECT COUNT(*) FROM episodes WHERE experiment = ?", (experiment,)
            )
        return int(cur.fetchone()[0])

    def fetch_episodes(
        self, *, experiment: str, method: str | None = None
    ) -> list[dict]:
        assert self._conn is not None
        query = "SELECT * FROM episodes WHERE experiment = ?"
        params: list = [experiment]
        if method is not None:
            query += " AND method = ?"
            params.append(method)
        query += " ORDER BY seed, episode"
        cur = self._conn.execute(query, params)
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


def posterior_rows_from_learner(
    learner,
    *,
    experiment: str,
    method: str,
    seed: int,
    episode: int,
) -> list[PosteriorRow]:
    """Flatten a :class:`BetaBernoulliThompson` snapshot into rows."""
    rows: list[PosteriorRow] = []
    snap = learner.snapshot()
    for agent, action_map in snap.items():
        for action, (alpha, beta) in action_map.items():
            rows.append(PosteriorRow(
                experiment=experiment, method=method, seed=seed, episode=episode,
                agent=agent, action=action, alpha=alpha, beta=beta,
            ))
    return rows
