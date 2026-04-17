"""Base protocol for credit-assignment strategies."""

from __future__ import annotations

from random import Random
from typing import Protocol, runtime_checkable

from framework.environment import Outcome
from framework.pipeline import Pipeline, Trace


@runtime_checkable
class CreditAssigner(Protocol):
    """Map a pipeline execution to per-agent credit in [0, 1].

    All assigners receive ``rng`` — some (e.g., counterfactual) need it to
    resample; deterministic assigners may ignore it.
    """

    name: str

    def assign(
        self,
        trace: Trace,
        outcome: Outcome,
        pipeline: Pipeline,
        rng: Random,
    ) -> dict[str, float]:  # pragma: no cover
        ...
