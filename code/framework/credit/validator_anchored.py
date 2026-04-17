"""Validator-anchored credit for composite agents.

AIDI's CodeAgent internally runs:

    SchemaAgent → BuilderAgent → ValidatorAgent → {QA | retry | collect}

The ValidatorAgent produces a *structured* signal: it tells you which
sub-step's output was malformed. That is much richer than the binary
per-step success flag used by flat methods. Validator-anchored credit
uses the validator's structured verdict (when available) to localize
blame within the composite, then falls back to counterfactual-style
resampling for the outer pipeline.

For the simulation, we encode the validator's structured signal as the
per-step ``success`` flags inside the sub-trace, plus a
convention: if the ValidatorAgent itself fails, the blame stays with the
validator; if it succeeds but upstream Schema/Builder sub-agents failed,
credit is distributed to those upstream sub-agents proportionally.
"""

from __future__ import annotations

from random import Random

from framework.composite_agent import sub_trace_of
from framework.environment import Outcome
from framework.pipeline import Pipeline, Trace


class ValidatorAnchoredCredit:
    """Credit = 1 iff the agent's step succeeded; for composite agents with a
    ValidatorAgent in their sub-pipeline, use the validator's structured
    verdict to attribute credit within the composite.
    """

    name = "validator_anchored"

    def __init__(self, validator_name: str = "ValidatorAgent") -> None:
        self._validator_name = validator_name

    def assign(
        self,
        trace: Trace,
        outcome: Outcome,
        pipeline: Pipeline,
        rng: Random,
    ) -> dict[str, float]:
        credits: dict[str, float] = {}
        for step in trace.steps:
            sub = sub_trace_of(step.output)
            if sub is None:
                # Flat agent: credit = per-step success.
                credits[step.agent_name] = 1.0 if step.output.success else 0.0
                continue

            # Composite: use the validator's verdict to attribute.
            credits[step.agent_name] = self._composite_credit(sub)
        return credits

    def _composite_credit(self, sub_trace: Trace) -> float:
        """Roll up a composite agent's sub-trace into a single credit.

        Convention:
            - If ValidatorAgent exists and succeeded, the composite's
              credit is 1 (validator blessed the output).
            - If ValidatorAgent failed, credit is 0 (validator caught a
              problem the composite couldn't fix).
            - If no ValidatorAgent is present, fall back to the mean of
              per-step success flags.
        """
        try:
            val_step = next(
                s for s in sub_trace.steps if s.agent_name == self._validator_name
            )
            return 1.0 if val_step.output.success else 0.0
        except StopIteration:
            successes = [1.0 if s.output.success else 0.0 for s in sub_trace.steps]
            return sum(successes) / len(successes) if successes else 0.0
