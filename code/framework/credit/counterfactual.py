"""Counterfactual (leave-one-out) credit assignment.

For each agent ``i``, sample ``n_samples`` alternative outputs for that agent
and replay the pipeline from ``i`` onward. The agent's credit is

    credit_i = reward_actual - E[reward | agent_i resampled]

clamped to [0, 1]. Intuitively: if resampling this agent makes outcomes
better on average, the actual draw was bad, so credit is low. If the actual
draw was already near-optimal, resampling rarely improves matters and credit
stays high.

This is the COMA-style counterfactual baseline (Foerster et al. 2017) adapted
to a non-differentiable sequential pipeline of black-box agents. Unlike COMA
it does not marginalize analytically; it uses Monte-Carlo resampling, which
is necessary because our agents are not differentiable policies.

Cost is O(N * n_samples * avg_downstream_length) per task, i.e. linear in
the number of agents — not exponential like exact Shapley.
"""

from __future__ import annotations

from random import Random

from framework.environment import Outcome
from framework.pipeline import Pipeline, Trace


class CounterfactualCredit:
    name = "counterfactual"

    def __init__(self, n_samples: int = 5) -> None:
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")
        self._n = n_samples

    def assign(
        self,
        trace: Trace,
        outcome: Outcome,
        pipeline: Pipeline,
        rng: Random,
    ) -> dict[str, float]:
        actual_r = float(outcome.reward)
        credits: dict[str, float] = {}
        for step in trace.steps:
            rewards = []
            for _ in range(self._n):
                _, alt_outcome = pipeline.rerun_from(trace, step.agent_name, rng)
                rewards.append(float(alt_outcome.reward))
            avg_alt = sum(rewards) / len(rewards)
            # Credit interpretation:
            #   If actual_r ≈ avg_alt  → agent made no difference (credit neutral).
            #   If actual_r > avg_alt  → this agent did better than its distribution.
            #   If actual_r < avg_alt  → this agent did worse than its distribution
            #                             (bears blame).
            # Map to [0, 1] by affine shift: 0.5 + 0.5 * (actual - avg).
            delta = actual_r - avg_alt
            credit = 0.5 + 0.5 * delta
            credits[step.agent_name] = max(0.0, min(1.0, credit))
        return credits
