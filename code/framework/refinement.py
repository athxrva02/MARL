"""Refinement-loop orchestrator: AIDI's supervisor pattern.

The real AIDI supervisor runs a LangGraph of the form:

    Doc → Log → Code → DecisionEngine ──complete──→ End
                             │
                             └─refine──→ Doc (with clarification context)

``RefinementLoop`` simulates this: it repeatedly runs an underlying pipeline
and asks a :class:`DecisionPolicy` whether to stop or refine. Each iteration
produces a full :class:`Trace`; the episode-level record is
:class:`EpisodeRun` (list of iteration traces + final outcome).

This is the minimum extension needed to make credit-assignment studies
iteration-aware. It does NOT try to be a faithful LangGraph; it captures
the two features that matter for credit:

  1. The same agents may run multiple times on the same task.
  2. The decision to refine vs stop is itself a choice with credit
     implications (refining when the pipeline was already good wastes LLM
     spend; stopping early when refinement would have succeeded is a
     miss).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Callable, Protocol, runtime_checkable

from framework.environment import Outcome, Task
from framework.pipeline import Pipeline, Trace


@dataclass(frozen=True)
class Iteration:
    """Record of one pass through the underlying pipeline."""

    index: int
    trace: Trace
    outcome: Outcome


@dataclass
class EpisodeRun:
    """The full record of one refinement-loop episode."""

    task_id: str
    iterations: list[Iteration] = field(default_factory=list)
    final_outcome: Outcome | None = None
    stopped_reason: str = ""

    @property
    def num_iterations(self) -> int:
        return len(self.iterations)

    @property
    def final_reward(self) -> float:
        return float(self.final_outcome.reward) if self.final_outcome else 0.0


@runtime_checkable
class DecisionPolicy(Protocol):
    """Controls whether the refinement loop continues or stops.

    Mirrors AIDI's :class:`DecisionEngine` in spirit: consumes the history
    of iterations and returns one of ``"complete"`` / ``"refine"`` /
    ``"stagnant"``.
    """

    def decide(self, run: EpisodeRun, rng: Random) -> str:  # pragma: no cover
        ...


class CompletenessDecisionPolicy:
    """Stops when reward clears ``complete_threshold``; refines otherwise,
    up to ``max_iterations``. Declares stagnation if the last
    ``stagnation_window`` iterations all tied on reward.

    This mirrors AIDI's current policy (completeness + stagnation), without
    wiring any learning into the decision itself.
    """

    name = "completeness"

    def __init__(
        self,
        complete_threshold: float = 1.0,
        max_iterations: int = 3,
        stagnation_window: int = 2,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        self._thresh = complete_threshold
        self._max = max_iterations
        self._stag_window = stagnation_window

    def decide(self, run: EpisodeRun, rng: Random) -> str:
        if not run.iterations:
            return "refine"
        last = run.iterations[-1].outcome.reward
        if last >= self._thresh:
            return "complete"
        if run.num_iterations >= self._max:
            return "stagnant"
        if run.num_iterations >= self._stag_window:
            recent = [i.outcome.reward for i in run.iterations[-self._stag_window:]]
            if len(set(recent)) == 1:
                return "stagnant"
        return "refine"


ContextBuilder = Callable[[EpisodeRun, Task], Task]


def default_refinement_context(prev: EpisodeRun, original_task: Task) -> Task:
    """Augment the task with a counter so agents can see which iteration
    they're on. Real AIDI does much more (clarifications, missing-attr
    hints), but the counter is enough for the simulation."""
    ctx = dict(original_task.context)
    ctx["iteration_index"] = prev.num_iterations
    ctx["is_refinement"] = prev.num_iterations > 0
    return Task(
        task_id=original_task.task_id,
        payload=original_task.payload,
        context=ctx,
        planted_blame=original_task.planted_blame,
    )


class RefinementLoop:
    """Simulates AIDI's supervisor-orchestrated refinement loop."""

    def __init__(
        self,
        pipeline: Pipeline,
        decision_policy: DecisionPolicy,
        *,
        context_builder: ContextBuilder = default_refinement_context,
    ) -> None:
        self._pipeline = pipeline
        self._policy = decision_policy
        self._context_builder = context_builder

    @property
    def pipeline(self) -> Pipeline:
        return self._pipeline

    def run_episode(self, task: Task, rng: Random) -> EpisodeRun:
        run = EpisodeRun(task_id=task.task_id)
        while True:
            current_task = self._context_builder(run, task)
            trace, outcome = self._pipeline.run(current_task, rng)
            run.iterations.append(
                Iteration(index=run.num_iterations, trace=trace, outcome=outcome)
            )
            decision = self._policy.decide(run, rng)
            if decision in ("complete", "stagnant"):
                run.final_outcome = outcome
                run.stopped_reason = decision
                return run
            # else: "refine" — loop
