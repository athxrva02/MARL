"""End-to-end credit baseline: every agent receives the pipeline outcome.

This is the naive baseline from assignment.md. It is noisy because an agent
is rewarded/punished for upstream and downstream behaviour it did not cause.
The research question is whether a smarter assigner recovers faster.
"""

from __future__ import annotations

from random import Random

from framework.environment import Outcome
from framework.pipeline import Pipeline, Trace


class EndToEndCredit:
    name = "end_to_end"

    def assign(
        self,
        trace: Trace,
        outcome: Outcome,
        pipeline: Pipeline,
        rng: Random,
    ) -> dict[str, float]:
        r = float(outcome.reward)
        return {step.agent_name: r for step in trace.steps}
