"""Agent interfaces for the credit-assignment framework.

An Agent is a stateless, seedable function mapping an Observation to an
AgentOutput. Statelessness is a deliberate design choice: the counterfactual
credit assigner needs to re-execute an agent (or its downstream successors)
many times under varying random draws, which is awkward if agents carry
hidden state between calls.

Real-world LLM agents can be wrapped by implementing ``Agent`` and caching
deterministically on seed; for the Phase-1 simulation study we only need
``SyntheticAgent``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class Observation:
    """Input to an agent.

    ``payload`` is intentionally untyped so concrete pipelines can pass through
    whatever their upstream agents emit (dict, string, dataclass, ...). The
    framework only inspects ``context`` for bookkeeping.
    """

    payload: Any
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentOutput:
    """Output of a single agent step.

    Attributes:
        value: The agent's actual output, passed to the next agent as
            ``Observation.payload``.
        confidence: Self-reported confidence in ``value``. Used by the
            (future) cascading-confidence credit assigner; end-to-end and
            counterfactual ignore it.
        action_id: A discrete identifier of the action this agent took. The
            Bayesian learner keeps separate priors per (agent, action_id), so
            the granularity here controls how fine-grained learning is.
        success: Ground-truth per-step success flag, known only in synthetic
            mode. Downstream outcome computation uses this; credit-assignment
            methods must NOT read it (it is the hidden oracle we score
            attribution against).
    """

    value: Any
    confidence: float
    action_id: str
    success: bool


@runtime_checkable
class Agent(Protocol):
    """Protocol for any agent that can participate in a :class:`Pipeline`.

    Implementations must be stateless w.r.t. inputs; any required state is
    passed in via ``obs.context`` or initialised at construction time.
    """

    name: str

    def act(self, obs: Observation, rng: Random) -> AgentOutput:  # pragma: no cover
        ...


class SyntheticAgent:
    """A stochastic synthetic agent used for the simulation study.

    The agent has a discrete action set. For each action ``a`` the agent has a
    base success probability ``p_success[a]``. A failure from any upstream
    agent (signalled by ``obs.context["upstream_failed"] = True``) degrades
    this probability by ``upstream_failure_penalty``, simulating the realistic
    case where bad inputs make the best-in-class downstream agent still fail.

    This is the stressor that makes end-to-end credit bad: the "blame" should
    go to the upstream agent, not the downstream one that merely suffered
    cascading failure.
    """

    def __init__(
        self,
        name: str,
        actions: list[str],
        p_success: dict[str, float],
        *,
        upstream_failure_penalty: float = 0.0,
        confidence_noise: float = 0.05,
        silent_failure_rate: float = 0.0,
    ) -> None:
        """
        Args:
            name: Agent identifier. Must be unique within a :class:`Pipeline`.
            actions: Discrete action space.
            p_success: Per-action base success probability (values in [0, 1]).
            upstream_failure_penalty: Multiplicative penalty applied to
                p_success when the upstream trace contains a failed step.
                Set > 0 to induce cascading failure; 0 means agents are
                independent.
            confidence_noise: Std-dev of Gaussian noise added to the reported
                confidence, clipped to [0, 1]. Models miscalibration.
            silent_failure_rate: Probability of reporting high confidence on a
                failed step. This is the "silent failure" mode called out in
                the plan for the CodeAgent (regex that parses but is wrong).
        """
        if not actions:
            raise ValueError("actions must be non-empty")
        missing = set(actions) - p_success.keys()
        if missing:
            raise ValueError(f"p_success missing entries for: {sorted(missing)}")
        for a, p in p_success.items():
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"p_success[{a!r}] = {p} outside [0, 1]")

        self.name = name
        self.actions = list(actions)
        self._p_success = dict(p_success)
        self._upstream_penalty = upstream_failure_penalty
        self._conf_noise = confidence_noise
        self._silent_rate = silent_failure_rate

    def choose_action(self, obs: Observation, rng: Random) -> str:
        """Pick an action. Default: pick deterministically from context if the
        learner has chosen one, else uniform random. Exposed so the experiment
        runner can inject Thompson-sampled action choices.
        """
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
        p = self._p_success[action]
        if obs.context.get("upstream_failed", False) and self._upstream_penalty > 0:
            p *= 1.0 - self._upstream_penalty
        success = rng.random() < p

        if success:
            conf = min(1.0, max(0.0, p + rng.gauss(0.0, self._conf_noise)))
            value = {"ok": True, "action": action}
        else:
            if rng.random() < self._silent_rate:
                # Silent failure: high confidence, bad output.
                conf = min(1.0, max(0.0, p + rng.gauss(0.0, self._conf_noise)))
            else:
                conf = max(0.0, min(1.0, (1.0 - p) + rng.gauss(0.0, self._conf_noise)))
                conf = 1.0 - conf  # report LOW confidence on honest failures
            value = {"ok": False, "action": action}

        return AgentOutput(
            value=value, confidence=conf, action_id=action, success=success
        )
