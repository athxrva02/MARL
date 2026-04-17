"""Per-agent Beta-Bernoulli Thompson sampling learner.

Mirror (by name) of AIDI's dormant ``BayesianAttributeLearner`` (see the
internship proposal, §1.2). Each (agent, action) pair has a Beta(α, β) prior
over its success probability; credit ∈ [0, 1] is treated as a soft Bernoulli
success: α += credit, β += 1 - credit. At decision time we Thompson-sample
by drawing a probability from each arm's posterior and picking the argmax.

Soft (non-integer) updates are well-defined for conjugate Beta: they amount
to an expected-sufficient-statistic update and remain a valid Beta posterior.
"""

from __future__ import annotations

from random import Random


class BetaBernoulliThompson:
    """Thompson sampling over per-(agent, action) Beta posteriors."""

    def __init__(self, alpha0: float = 1.0, beta0: float = 1.0) -> None:
        if alpha0 <= 0 or beta0 <= 0:
            raise ValueError("Beta prior parameters must be positive")
        self._alpha0 = alpha0
        self._beta0 = beta0
        # Nested dict: agent_name -> action -> [alpha, beta]
        self._posteriors: dict[str, dict[str, list[float]]] = {}

    # -- action selection --------------------------------------------------

    def choose(self, agent_name: str, actions: list[str], rng: Random) -> str:
        best_action = None
        best_sample = -1.0
        for a in actions:
            alpha, beta = self._get(agent_name, a)
            sample = _beta_sample(alpha, beta, rng)
            if sample > best_sample:
                best_sample = sample
                best_action = a
        assert best_action is not None
        return best_action

    # -- posterior update --------------------------------------------------

    def update(self, agent_name: str, action: str, credit: float) -> None:
        if not 0.0 <= credit <= 1.0:
            raise ValueError(f"credit must be in [0, 1], got {credit}")
        alpha, beta = self._get(agent_name, action)
        self._posteriors[agent_name][action] = [
            alpha + credit,
            beta + (1.0 - credit),
        ]

    # -- introspection -----------------------------------------------------

    def mean(self, agent_name: str, action: str) -> float:
        alpha, beta = self._get(agent_name, action)
        return alpha / (alpha + beta)

    def snapshot(self) -> dict:
        return {
            agent: {action: list(params) for action, params in actions.items()}
            for agent, actions in self._posteriors.items()
        }

    # -- helpers -----------------------------------------------------------

    def _get(self, agent_name: str, action: str) -> list[float]:
        by_action = self._posteriors.setdefault(agent_name, {})
        if action not in by_action:
            by_action[action] = [self._alpha0, self._beta0]
        return by_action[action]


def _beta_sample(alpha: float, beta: float, rng: Random) -> float:
    """Sample from Beta(alpha, beta) using Python's stdlib Random.

    We avoid depending on numpy so the framework stays lightweight.
    """
    return rng.betavariate(alpha, beta)
