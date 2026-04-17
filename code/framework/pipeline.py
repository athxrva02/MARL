"""Sequential pipeline composition and execution.

A :class:`Pipeline` runs its agents in order and emits a :class:`Trace` — the
full record of per-agent inputs, outputs, and confidences. Traces are
deliberately immutable and fully describe a single pipeline execution so
that the counterfactual credit assigner can replay from any prefix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Callable

from framework.agent import Agent, AgentOutput, Observation
from framework.environment import Outcome, Task, default_outcome


@dataclass(frozen=True)
class TraceStep:
    """Record of a single agent's execution within a pipeline run."""

    agent_name: str
    observation: Observation
    output: AgentOutput


@dataclass(frozen=True)
class Trace:
    """Ordered sequence of :class:`TraceStep`s for one pipeline run."""

    task_id: str
    steps: tuple[TraceStep, ...]

    def per_agent_success(self) -> dict[str, bool]:
        return {s.agent_name: s.output.success for s in self.steps}

    def output_of(self, agent_name: str) -> AgentOutput:
        for s in self.steps:
            if s.agent_name == agent_name:
                return s.output
        raise KeyError(agent_name)

    def index_of(self, agent_name: str) -> int:
        for i, s in enumerate(self.steps):
            if s.agent_name == agent_name:
                return i
        raise KeyError(agent_name)


OutcomeFn = Callable[[dict[str, bool]], Outcome]


class Pipeline:
    """A sequential pipeline of agents.

    The pipeline is transport-only: it threads each agent's output into the
    next agent's observation (as ``payload``), and marks ``upstream_failed``
    in the context whenever any prior agent reported failure. Actual
    failure-propagation behaviour (i.e., *how* downstream agents react to
    bad inputs) belongs inside the agents themselves — see
    :class:`framework.agent.SyntheticAgent`.
    """

    def __init__(
        self,
        agents: list[Agent],
        *,
        outcome_fn: OutcomeFn = default_outcome,
    ) -> None:
        names = [a.name for a in agents]
        if len(names) != len(set(names)):
            raise ValueError(f"Agent names must be unique, got {names}")
        self._agents = list(agents)
        self._outcome_fn = outcome_fn

    @property
    def agents(self) -> list[Agent]:
        return list(self._agents)

    @property
    def agent_names(self) -> list[str]:
        return [a.name for a in self._agents]

    def agent_by_name(self, name: str) -> Agent:
        for a in self._agents:
            if a.name == name:
                return a
        raise KeyError(name)

    def run(self, task: Task, rng: Random) -> tuple[Trace, Outcome]:
        """Execute the full pipeline on ``task``."""
        steps = list(self._run_from(0, initial_payload=task.payload,
                                    base_context=task.context, rng=rng))
        trace = Trace(task_id=task.task_id, steps=tuple(steps))
        outcome = self._outcome_fn(trace.per_agent_success())
        return trace, outcome

    def rerun_from(
        self,
        trace: Trace,
        agent_name: str,
        rng: Random,
        override_output: AgentOutput | None = None,
    ) -> tuple[Trace, Outcome]:
        """Replay the pipeline starting at ``agent_name``.

        Steps before ``agent_name`` are reused verbatim from ``trace``. The
        step for ``agent_name`` is either re-sampled (default) or replaced
        with ``override_output``. Steps after are re-executed fresh.

        This is the workhorse for the counterfactual credit assigner.
        """
        idx = trace.index_of(agent_name)
        prefix = list(trace.steps[:idx])

        if override_output is not None:
            # Use the override as the agent_name step's output, reconstructing
            # the observation that the original run presented to that agent.
            original_step = trace.steps[idx]
            target_step = TraceStep(
                agent_name=agent_name,
                observation=original_step.observation,
                output=override_output,
            )
            prefix.append(target_step)
            start_at = idx + 1
            initial_payload = override_output.value
            upstream_failed = any(not s.output.success for s in prefix)
        else:
            # Re-execute agent_name under new randomness.
            start_at = idx
            if prefix:
                initial_payload = prefix[-1].output.value
                upstream_failed = any(not s.output.success for s in prefix)
            else:
                # Re-running from the first agent: go back to task inputs.
                initial_payload = trace.steps[0].observation.payload
                upstream_failed = trace.steps[0].observation.context.get(
                    "upstream_failed", False
                )

        # Base context: inherit from the original step's observation so that
        # any task-level metadata is preserved. Strip upstream_failed since
        # we recompute it below.
        base_context = dict(trace.steps[idx].observation.context)
        base_context.pop("upstream_failed", None)

        new_steps = self._run_from(
            start_at,
            initial_payload=initial_payload,
            base_context=base_context,
            rng=rng,
            initial_upstream_failed=upstream_failed,
        )
        all_steps = tuple(prefix + list(new_steps))
        new_trace = Trace(task_id=trace.task_id, steps=all_steps)
        outcome = self._outcome_fn(new_trace.per_agent_success())
        return new_trace, outcome

    # ------------------------------------------------------------------

    def _run_from(
        self,
        start_idx: int,
        *,
        initial_payload,
        base_context: dict,
        rng: Random,
        initial_upstream_failed: bool = False,
    ):
        payload = initial_payload
        upstream_failed = initial_upstream_failed
        for agent in self._agents[start_idx:]:
            ctx = dict(base_context)
            ctx["upstream_failed"] = upstream_failed
            obs = Observation(payload=payload, context=ctx)
            out = agent.act(obs, rng)
            yield TraceStep(agent_name=agent.name, observation=obs, output=out)
            payload = out.value
            upstream_failed = upstream_failed or (not out.success)
