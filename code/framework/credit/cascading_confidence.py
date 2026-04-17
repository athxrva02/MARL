"""Cascading-confidence credit (stub).

assignment.md lists this as a candidate method: credit_i = product of all
agents' confidences. Not implemented in Phase 1 per user scope, but present
here to prove the :class:`CreditAssigner` interface accommodates it without
refactor. Raises :class:`NotImplementedError` when invoked.
"""

from __future__ import annotations

from random import Random

from framework.environment import Outcome
from framework.pipeline import Pipeline, Trace


class CascadingConfidence:
    name = "cascading_confidence"

    def assign(
        self,
        trace: Trace,
        outcome: Outcome,
        pipeline: Pipeline,
        rng: Random,
    ) -> dict[str, float]:
        raise NotImplementedError(
            "Cascading-confidence credit is out of Phase-1 scope. "
            "Interface is stable; implement by returning the product of "
            "agent confidences along the trace."
        )
