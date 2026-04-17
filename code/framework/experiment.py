"""Experiment runner: couples pipeline, learner, and credit assigner.

One ``run_episode`` call executes the full loop for a single task:
    1. Ask the learner to pick an action per agent (via Thompson sampling).
    2. Inject those choices into the task context.
    3. Run the pipeline.
    4. Score the outcome.
    5. Assign credit.
    6. Update the learner.

All randomness flows through a single ``Random`` instance so experiments are
bit-for-bit reproducible given a seed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Any

from framework.agent import SyntheticAgent
from framework.credit.base import CreditAssigner
from framework.environment import Outcome, Task
from framework.learner.base import Learner
from framework.pipeline import Pipeline, Trace


@dataclass
class EpisodeRecord:
    """Per-episode log for analysis."""

    task_id: str
    planted_blame: str | None
    actions: dict[str, str]
    per_agent_success: dict[str, bool]
    outcome_reward: float
    credit: dict[str, float]
    # Derived attribution metric: credit assigned to the planted-blame agent.
    # Lower is better (blame-worthy agent should get LOW credit).
    blame_credit: float | None = None


@dataclass
class RunResult:
    """Aggregated output of an experiment run."""

    config: dict[str, Any]
    episodes: list[EpisodeRecord] = field(default_factory=list)

    def mean_reward(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(e.outcome_reward for e in self.episodes) / len(self.episodes)

    def blame_attribution_accuracy(self) -> float | None:
        """Fraction of planted-blame episodes where the lowest-credit agent
        was the actually-to-blame agent."""
        scored = [e for e in self.episodes if e.planted_blame is not None]
        if not scored:
            return None
        hits = 0
        for e in scored:
            min_agent = min(e.credit, key=e.credit.get)
            if min_agent == e.planted_blame:
                hits += 1
        return hits / len(scored)


def run_episode(
    pipeline: Pipeline,
    task: Task,
    learner: Learner,
    credit_assigner: CreditAssigner,
    rng: Random,
) -> tuple[Trace, Outcome, dict[str, float], dict[str, str]]:
    """Run one learning episode end-to-end.

    Returns:
        trace, outcome, per-agent credit, and the action each agent chose.
    """
    # 1. Choose an action per agent via the learner. We inject these choices
    #    into the task context so the pipeline's agents honour them. Only
    #    SyntheticAgent currently reads these keys; arbitrary wrapped agents
    #    can do the same by convention.
    action_by_agent: dict[str, str] = {}
    injected_context = dict(task.context)
    for agent in pipeline.agents:
        if isinstance(agent, SyntheticAgent):
            action = learner.choose(agent.name, agent.actions, rng)
        else:
            action = learner.choose(agent.name, getattr(agent, "actions", []), rng)
        action_by_agent[agent.name] = action
        injected_context[f"action_for:{agent.name}"] = action

    # 2. Run pipeline with injected actions.
    injected_task = Task(
        task_id=task.task_id,
        payload=task.payload,
        context=injected_context,
        planted_blame=task.planted_blame,
    )
    trace, outcome = pipeline.run(injected_task, rng)

    # 3. Credit assignment.
    credit = credit_assigner.assign(trace, outcome, pipeline, rng)

    # 4. Update learner.
    for agent_name, action in action_by_agent.items():
        learner.update(agent_name, action, credit[agent_name])

    return trace, outcome, credit, action_by_agent
