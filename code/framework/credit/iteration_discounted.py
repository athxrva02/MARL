"""Iteration-discounted credit for refinement-loop episodes.

AIDI's supervisor runs agents multiple times per task via the refinement
loop. A flat credit method (end-to-end, counterfactual) does not
distinguish "this agent's iteration-1 output was already good" from "this
agent only worked after iteration-3 clarification." Both matter for
training — but not equally. An agent that succeeds early deserves more
credit than one that needed multiple retries.

This assigner operates on an :class:`EpisodeRun` (NOT a single
:class:`Trace`). It assigns credit to each (agent, iteration) pair and
aggregates per-agent by discounted sum.
"""

from __future__ import annotations

from random import Random

from framework.refinement import EpisodeRun


class IterationDiscountedCredit:
    """Credit = Σ_k γ^k · (per-agent success at iteration k).

    Agents are credited more for succeeding early. A stagnant / failed
    episode leaves the posterior flat (credit = 0.5 neutral).
    """

    name = "iteration_discounted"

    def __init__(self, gamma: float = 0.7) -> None:
        if not 0.0 < gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        self._gamma = gamma

    def assign(self, run: EpisodeRun, rng: Random) -> dict[str, float]:
        if not run.iterations:
            return {}
        # Collect per-agent per-iteration success.
        # Agents are the same across iterations by construction.
        first = run.iterations[0].trace
        agent_names = [s.agent_name for s in first.steps]

        weights: dict[str, float] = {a: 0.0 for a in agent_names}
        total_weight = 0.0
        for k, it in enumerate(run.iterations):
            w = self._gamma ** k
            total_weight += w
            for step in it.trace.steps:
                if step.output.success:
                    weights[step.agent_name] += w

        # Normalise into [0, 1] per agent.
        if total_weight == 0:
            return {a: 0.5 for a in agent_names}

        # Also blend in the episode's final reward so a "stagnant/failed"
        # ending is penalised even if some agents succeeded along the way.
        final = run.final_reward
        credits: dict[str, float] = {}
        for a in agent_names:
            raw = weights[a] / total_weight  # ∈ [0, 1]
            # Blend per-agent performance with final outcome (50/50).
            credits[a] = 0.5 * raw + 0.5 * final
        return credits
