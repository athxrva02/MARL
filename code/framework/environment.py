"""Tasks and outcomes for pipeline experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Task:
    """A single end-to-end job for the pipeline.

    Attributes:
        task_id: Unique identifier (for logging / reproducibility).
        payload: Initial input to the first agent.
        context: Metadata accessible to any agent (document type, vendor,
            etc.). Useful later for RQ2 contextual bandits; here it is
            threaded through unchanged.
        planted_blame: Ground-truth name of the agent (if any) intended to
            fail on this task. ``None`` means no blame is planted and any
            failure is incidental.
    """

    task_id: str
    payload: Any
    context: dict[str, Any] = field(default_factory=dict)
    planted_blame: str | None = None


@dataclass(frozen=True)
class Outcome:
    """Result of running a pipeline on a task.

    ``reward`` ∈ [0, 1] is the scalar signal used by end-to-end credit. For
    the default outcome model it equals 1.0 iff *every* agent succeeded; more
    forgiving reward shapes can be provided via a custom ``outcome_fn``.
    """

    reward: float
    success: bool
    details: dict[str, Any] = field(default_factory=dict)


def default_outcome(per_agent_success: dict[str, bool]) -> Outcome:
    """All-or-nothing outcome: reward = 1 iff all agents succeeded.

    This is the strictest credit-assignment setting and the best stress-test
    for comparing end-to-end vs counterfactual: a single upstream failure
    tanks the reward, so end-to-end credit cannot distinguish "I failed" from
    "I was given bad input."
    """
    all_ok = all(per_agent_success.values())
    return Outcome(
        reward=1.0 if all_ok else 0.0,
        success=all_ok,
        details={"per_agent_success": dict(per_agent_success)},
    )
