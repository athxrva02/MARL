"""Feature-conditioned synthetic agent.

:class:`FeatureAwareAgent` is a drop-in replacement for
:class:`framework.agent.SyntheticAgent` whose per-action success probability
depends on the task's features. Concretely, ``p_success[action]`` is a
base probability that can be modified additively by a feature-effects
table — e.g. the LogParser might succeed 20% less often when log_volume
is "high", or the SchemaAgent might struggle with PII.

The Bayesian learner in this repo is *feature-blind* by design: it keeps
one Beta posterior per (agent, action). The learned optimum is therefore
the average over the task-feature distribution. Making the agent
feature-aware while leaving the learner feature-blind lets us study how
much mileage the Thompson-sampling baseline gets just from the marginal —
a useful anchor before any contextual-bandit extension (RQ2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Mapping

from framework.agent import AgentOutput, Observation


@dataclass
class FeatureEffect:
    """Additive effect on ``p_success`` for a specific feature value.

    All effects are summed and clipped to [0, 1]. Positive values make the
    action more likely to succeed in this feature bucket; negative values
    punish it.
    """

    when: dict[str, object]  # e.g. {"doc_size": "large"}
    delta: float  # e.g. -0.2


class FeatureAwareAgent:
    """Like :class:`SyntheticAgent`, but ``p_success`` is feature-conditioned."""

    def __init__(
        self,
        name: str,
        actions: list[str],
        base_p_success: dict[str, float],
        *,
        effects: Mapping[str, list[FeatureEffect]] | None = None,
        upstream_failure_penalty: float = 0.0,
        confidence_noise: float = 0.05,
        silent_failure_rate: float = 0.0,
    ) -> None:
        """
        Args:
            effects: Per-action list of :class:`FeatureEffect` entries. A
                missing action means no feature effect (constant base p).
        """
        if not actions:
            raise ValueError("actions must be non-empty")
        missing = set(actions) - base_p_success.keys()
        if missing:
            raise ValueError(f"base_p_success missing: {sorted(missing)}")
        self.name = name
        self.actions = list(actions)
        self._base = dict(base_p_success)
        self._effects: dict[str, list[FeatureEffect]] = {
            a: list(effects.get(a, [])) if effects else []
            for a in actions
        }
        self._upstream_penalty = upstream_failure_penalty
        self._conf_noise = confidence_noise
        self._silent_rate = silent_failure_rate

    def _effective_p(self, action: str, features: dict) -> float:
        p = self._base[action]
        for fx in self._effects.get(action, []):
            if all(features.get(k) == v for k, v in fx.when.items()):
                p += fx.delta
        return max(0.0, min(1.0, p))

    def choose_action(self, obs: Observation, rng: Random) -> str:
        chosen = obs.context.get(f"action_for:{self.name}")
        if chosen is not None:
            if chosen not in self.actions:
                raise ValueError(
                    f"Injected action {chosen!r} not in action set for "
                    f"agent {self.name!r}"
                )
            return chosen
        return rng.choice(self.actions)

    def act(self, obs: Observation, rng: Random) -> AgentOutput:
        action = self.choose_action(obs, rng)
        features = obs.context.get("features", {}) or {}
        p = self._effective_p(action, features)
        if obs.context.get("upstream_failed", False) and self._upstream_penalty > 0:
            p *= 1.0 - self._upstream_penalty
        success = rng.random() < p

        if success:
            conf = min(1.0, max(0.0, p + rng.gauss(0.0, self._conf_noise)))
            value = {"ok": True, "action": action}
        else:
            if rng.random() < self._silent_rate:
                conf = min(1.0, max(0.0, p + rng.gauss(0.0, self._conf_noise)))
            else:
                conf = max(0.0, min(1.0, (1.0 - p) + rng.gauss(0.0, self._conf_noise)))
                conf = 1.0 - conf
            value = {"ok": False, "action": action}

        return AgentOutput(
            value=value, confidence=conf, action_id=action, success=success
        )
