"""Composite agents: agents that internally run a nested pipeline.

This models AIDI's :class:`CodeAgent`, which is itself a sub-pipeline
(SchemaAgent → BuilderAgent → ValidatorAgent → QAAgent). From the outer
pipeline's perspective, :class:`CompositeAgent` is a single
:class:`Agent` — it emits one :class:`AgentOutput`. Internally, it
produces a full sub-trace that credit assigners can inspect for
hierarchical attribution.

This is the minimum extension needed to study AIDI's real architecture,
where the credit-assignment problem is recursive: attribute credit
*across* top-level agents AND *within* the CodeAgent sub-pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random

from framework.agent import AgentOutput, Observation
from framework.environment import Outcome
from framework.pipeline import Pipeline, Trace


@dataclass(frozen=True)
class CompositeOutput:
    """Output payload of a :class:`CompositeAgent`.

    Mirrors :class:`AgentOutput`'s shape externally but carries the full
    internal trace for use by hierarchical credit assigners.
    """

    value: object
    confidence: float
    action_id: str
    success: bool
    sub_trace: Trace
    sub_outcome: Outcome


class CompositeAgent:
    """An :class:`Agent` whose :meth:`act` runs an internal sub-pipeline.

    The composite's overall success is defined by its ``success_fn`` applied
    to the sub-outcome; confidence is aggregated via ``confidence_fn``
    (default: mean of sub-step confidences).
    """

    def __init__(
        self,
        name: str,
        sub_pipeline: Pipeline,
        *,
        action_id_strategy: str = "ensemble",
    ) -> None:
        """
        Args:
            name: Outer-pipeline-level agent name.
            sub_pipeline: The internal :class:`Pipeline`.
            action_id_strategy: How to summarise the sub-run into an
                ``action_id`` for the outer learner. ``ensemble`` emits
                a fixed label so the outer learner sees the composite as
                a single arm; ``first_action`` emits the first sub-agent's
                chosen action (useful if the outer learner should pick the
                sub-pipeline's entry strategy).
        """
        self.name = name
        self._sub = sub_pipeline
        self._action_strategy = action_id_strategy
        # Exposed for API compatibility with :class:`SyntheticAgent`.
        self.actions = [action_id_strategy]

    @property
    def sub_pipeline(self) -> Pipeline:
        return self._sub

    def act(self, obs: Observation, rng: Random) -> AgentOutput:
        # Construct a sub-task from the outer observation.
        from framework.environment import Task  # local import to avoid cycle

        sub_task = Task(
            task_id=f"{self.name}:sub",
            payload=obs.payload,
            context=dict(obs.context),
        )
        sub_trace, sub_outcome = self._sub.run(sub_task, rng)

        # Aggregate confidence as the mean over sub-steps.
        confs = [s.output.confidence for s in sub_trace.steps]
        conf = sum(confs) / len(confs) if confs else 0.0

        if self._action_strategy == "first_action":
            action_id = sub_trace.steps[0].output.action_id if sub_trace.steps else self.name
        else:
            action_id = "ensemble"

        composite_value = CompositeOutput(
            value={"ok": sub_outcome.success, "sub": sub_trace.task_id},
            confidence=conf,
            action_id=action_id,
            success=sub_outcome.success,
            sub_trace=sub_trace,
            sub_outcome=sub_outcome,
        )
        # The outer-pipeline API requires AgentOutput; stuff the sub-trace
        # into the value payload so credit assigners can reach it.
        return AgentOutput(
            value=composite_value,
            confidence=conf,
            action_id=action_id,
            success=sub_outcome.success,
        )


def sub_trace_of(outer_output: AgentOutput) -> Trace | None:
    """Extract the embedded sub-trace from a composite agent's output, if any."""
    v = outer_output.value
    if isinstance(v, CompositeOutput):
        return v.sub_trace
    return None
