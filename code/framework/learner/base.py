"""Protocol for per-agent action-selection learners."""

from __future__ import annotations

from random import Random
from typing import Protocol, runtime_checkable


@runtime_checkable
class Learner(Protocol):
    """Selects actions for each agent and updates from credit signals."""

    def choose(self, agent_name: str, actions: list[str], rng: Random) -> str:
        ...  # pragma: no cover

    def update(self, agent_name: str, action: str, credit: float) -> None:
        ...  # pragma: no cover

    def snapshot(self) -> dict:
        """Serializable view of learner state (for analysis/plotting)."""
        ...  # pragma: no cover
