"""The Supervisor as a learnable agent.

Phase-2 modelled the Supervisor as a hard-coded :class:`RefinementLoop`
orchestrator. Phase-3 makes its decision hyperparameters themselves a
learnable arm: the learner chooses one of a small preset table of
``(complete_threshold, max_iterations)`` combinations *before* the
refinement loop runs, and receives credit based on the
reward-vs-iteration-cost trade-off the chosen preset produced.

This mirrors AIDI's real deployment setting: the DecisionEngine's
thresholds are tunable, the intern will want Bayesian evidence for which
preset wins on which task distribution, and that is precisely what a
Beta-Bernoulli Thompson learner delivers.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Mapping

from framework.refinement import (
    CompletenessDecisionPolicy,
    DecisionPolicy,
    EpisodeRun,
)


# Name used by the learner, experiment runner, and credit-propagation logic.
SUPERVISOR_AGENT_NAME = "Supervisor"


@dataclass(frozen=True)
class SupervisorPreset:
    """A named bundle of completeness-decision hyperparameters.

    ``complete_threshold`` — reward threshold above which the loop stops
    (and declares success). ``max_iterations`` — hard cap on refinement
    passes. ``stagnation_window`` — number of recent equal-reward
    iterations that triggers early exit as "stagnant".
    """

    complete_threshold: float
    max_iterations: int
    stagnation_window: int = 2


# Default action space for the Supervisor learning arm. Tuned so the three
# presets sit at meaningfully different points on the
# latency-vs-completeness trade-off curve.
DEFAULT_PRESETS: Mapping[str, SupervisorPreset] = {
    "aggressive":   SupervisorPreset(complete_threshold=1.0, max_iterations=2),
    "balanced":     SupervisorPreset(complete_threshold=0.8, max_iterations=3),
    "conservative": SupervisorPreset(complete_threshold=0.5, max_iterations=5),
}


def preset_to_policy(preset: SupervisorPreset) -> DecisionPolicy:
    """Instantiate the concrete :class:`DecisionPolicy` implied by a preset."""
    return CompletenessDecisionPolicy(
        complete_threshold=preset.complete_threshold,
        max_iterations=preset.max_iterations,
        stagnation_window=preset.stagnation_window,
    )


class SupervisorAgent:
    """Action-arm wrapper for the Supervisor's decision-hyperparameter choice.

    Unlike :class:`SyntheticAgent` and :class:`CompositeAgent`, the
    :class:`SupervisorAgent` never runs inside a :class:`Pipeline`. It is
    consulted once per episode by the experiment runner to pick a preset;
    the :class:`RefinementLoop` is then built from that preset. The class
    still exposes ``name``, ``actions``, and a ``choose`` method so that
    the existing learner plumbing works unchanged.
    """

    def __init__(
        self,
        *,
        name: str = SUPERVISOR_AGENT_NAME,
        presets: Mapping[str, SupervisorPreset] = DEFAULT_PRESETS,
    ) -> None:
        if not presets:
            raise ValueError("presets must be non-empty")
        self.name = name
        self._presets = dict(presets)
        self.actions = list(self._presets.keys())

    def preset_of(self, action_id: str) -> SupervisorPreset:
        try:
            return self._presets[action_id]
        except KeyError as e:  # pragma: no cover - guarded by learner arm
            raise ValueError(
                f"Unknown supervisor action {action_id!r}; expected one of "
                f"{self.actions}"
            ) from e

    def choose(self, rng: Random) -> str:
        """Uniform-random fallback for when no learner is attached. The
        experiment runner normally bypasses this by calling the learner
        directly; this is kept for symmetry with :class:`SyntheticAgent`.
        """
        return rng.choice(self.actions)


def supervisor_credit(
    run: EpisodeRun,
    *,
    iteration_cost_weight: float = 0.3,
    max_iterations_hint: int | None = None,
) -> float:
    """Score a Supervisor-preset episode on reward-minus-cost.

    credit = final_reward − iteration_cost_weight · (num_iters − 1) / (max − 1)

    (The ``−1`` terms normalise so that a one-shot success scores pure reward
    and a max-iteration failure pays the full cost.) Clamped to [0, 1].

    Defaults make a conservative preset that burns 5 iterations for reward 1.0
    score 0.7 — still good, but worse than aggressive preset nailing it in 1
    iteration (score 1.0). An early stagnant exit (reward 0, 2 iters) scores 0.
    """
    max_iters = max_iterations_hint or max(run.num_iterations, 1)
    if max_iters <= 1:
        cost = 0.0
    else:
        cost = iteration_cost_weight * (run.num_iterations - 1) / (max_iters - 1)
    raw = run.final_reward - cost
    return max(0.0, min(1.0, raw))
