"""Hierarchical credit wrapper for pipelines containing composite agents.

Wraps any flat :class:`CreditAssigner` and runs it on (a) the outer trace
and (b) each composite's embedded sub-trace, merging the two assignments
into a single leaf-level credit dict.

Why this exists: flat credit methods treat a :class:`CompositeAgent` as one
unit, which is correct for the outer pipeline but leaves all sub-agents
inside the composite receiving identical credit (the composite's credit
uniformly propagated). With the wrapper, each sub-agent receives a signal
derived from its own sub-trace — e.g. leave-one-out counterfactual
*within* the CodeAgent's Schema→Builder→Validator→QA pipeline — which
the flat-only pipeline setup cannot produce.

Two design choices worth calling out:

1. The wrapper runs the SAME flat assigner inside and outside by default;
   experiments can pass a different ``inner_assigner`` (e.g. use
   ``CounterfactualCredit`` on outer, ``ValidatorAnchoredCredit`` on inner)
   to mix strategies.

2. Outer-level composite credit is NOT discarded: the wrapper combines it
   with the per-leaf inner credit via a convex blend (``outer_weight``
   in [0, 1]). Setting ``outer_weight=0`` uses the inner signal exclusively;
   ``outer_weight=1`` recovers the uniform-propagation behaviour of the
   legacy experiment runner.
"""

from __future__ import annotations

from random import Random
from typing import Optional

from framework.composite_agent import CompositeAgent, sub_trace_of
from framework.environment import Outcome, default_outcome
from framework.pipeline import Pipeline, Trace

from framework.credit.base import CreditAssigner


class HierarchicalCredit:
    """Flatten per-leaf credit across composite-agent boundaries."""

    def __init__(
        self,
        outer_assigner: CreditAssigner,
        *,
        inner_assigner: Optional[CreditAssigner] = None,
        outer_weight: float = 0.3,
    ) -> None:
        """
        Args:
            outer_assigner: Run on the top-level trace. Gives credit to the
                three top-level entries (Doc, Log, CodeAgent-composite).
            inner_assigner: Run on each composite's sub-trace. Defaults to
                ``outer_assigner`` itself for convenience.
            outer_weight: Blend weight for the composite's outer credit
                when combining with its inner per-leaf credit. Leaf credit
                = outer_weight · outer_composite_credit + (1 − outer_weight)
                · inner_leaf_credit.
        """
        if not 0.0 <= outer_weight <= 1.0:
            raise ValueError("outer_weight must be in [0, 1]")
        self._outer = outer_assigner
        self._inner = inner_assigner or outer_assigner
        self._outer_weight = outer_weight
        self.name = f"hierarchical({outer_assigner.name})"

    def assign(
        self,
        trace: Trace,
        outcome: Outcome,
        pipeline: Pipeline,
        rng: Random,
    ) -> dict[str, float]:
        outer_credit = self._outer.assign(trace, outcome, pipeline, rng)
        result: dict[str, float] = {}

        for step in trace.steps:
            outer_agent = pipeline.agent_by_name(step.agent_name)

            if isinstance(outer_agent, CompositeAgent):
                sub_trace = sub_trace_of(step.output)
                if sub_trace is None:
                    # Defensive: composite with no embedded sub-trace;
                    # fall back to uniform propagation.
                    result[step.agent_name] = outer_credit.get(step.agent_name, 0.5)
                    continue

                sub_outcome = default_outcome(sub_trace.per_agent_success())
                inner_credit = self._inner.assign(
                    sub_trace, sub_outcome, outer_agent.sub_pipeline, rng
                )
                outer_c = outer_credit.get(step.agent_name, 0.5)
                w = self._outer_weight
                for leaf_name, leaf_c in inner_credit.items():
                    blended = w * outer_c + (1.0 - w) * leaf_c
                    result[leaf_name] = max(0.0, min(1.0, blended))
                # The composite itself is not a leaf; intentionally skipped.
            else:
                result[step.agent_name] = outer_credit.get(step.agent_name, 0.5)

        return result
