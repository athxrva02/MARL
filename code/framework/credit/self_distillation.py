"""Self-distillation credit (SDPO-inspired).

Approximates the per-agent advantage from Self-Distillation Policy
Optimization (SDPO; Hubotter et al. 2026, arXiv:2601.20802) for the
simulation setting.

In SDPO for LLM agents the same model acts as both student (without feedback)
and teacher (with environment feedback prepended). The KL between the two
distributions yields a per-token advantage:

    A_SDPO(t) = log π_teacher(token_t) - log π_student(token_t)

Tokens the feedback-informed teacher would have elevated get positive
advantage; tokens it would have suppressed get negative advantage, producing
a dense training signal from a single scalar reward.

Simulation approximation
------------------------
Agents in this framework emit a scalar ``confidence`` (self-assessed quality)
rather than a full token distribution.  We approximate the SDPO log-ratio at
the agent level by treating the pipeline outcome as the "teacher's hindsight
probability" for each agent's action:

    p_student_i = clip(confidence_i, ε, 1-ε)
    p_teacher_i = clip(outcome.reward, ε, 1-ε)
    A_i         = log(p_teacher_i / p_student_i)
    credit_i    = σ(temperature · A_i)

where σ is the logistic sigmoid.  The teacher probability models the LLM
self-teacher in SDPO: a copy of the agent conditioned on outcome feedback
that would assign higher probability to correct actions and lower probability
to incorrect ones.

Properties
----------
* **Overconfident on failure** (high conf, low reward) → large negative A
  → credit ≈ 0.  Specifically catches silent failures, where an agent reports
  high confidence on a bad output.
* **Correctly calibrated** (conf ≈ reward) → A ≈ 0 → credit ≈ 0.5.
* **Underconfident on success** (low conf, high reward) → large positive A
  → credit ≈ 1.  The teacher (seeing success) says "you should have been
  more confident" — this is the SDPO gradient direction.

Relationship to other methods
------------------------------
CounterfactualCredit answers "which agent *caused* the outcome?" by causal
intervention (leave-one-out resampling).  SelfDistillationCredit answers a
complementary question: "was each agent's uncertainty well-calibrated?"
The two are designed to be used together via HierarchicalCredit.

This method is registered as a TRACE_METHOD in the experiment runner and
is therefore automatically wrapped with HierarchicalCredit for composite
pipelines (where the inner call uses the sub-trace reward, giving finer-
grained attribution within composite agents).

Limitations
-----------
Unlike true SDPO this method does not perform causal intervention; every
agent's credit depends on the same ``outcome.reward`` scalar.  Agents in
a run cannot be distinguished by outcome alone — only by how well their
confidence predicted it.  For causal attribution prefer CounterfactualCredit
or ValidatorAnchoredCredit; this method adds the calibration signal on top.
"""

from __future__ import annotations

import math
from random import Random

from framework.environment import Outcome
from framework.pipeline import Pipeline, Trace


class SelfDistillationCredit:
    """SDPO-inspired calibration credit for LLM agent pipelines.

    See module docstring for the full derivation and design rationale.
    """

    name = "self_distillation"

    def __init__(
        self,
        *,
        temperature: float = 1.0,
        eps: float = 1e-3,
    ) -> None:
        """
        Args:
            temperature: Scales the log-ratio before the sigmoid.  Higher
                values push credits toward 0/1 (sharper discrimination
                between calibrated and miscalibrated agents).  Default 1.0.
            eps: Probability floor applied to both teacher and student via
                clip(p, eps, 1-eps) to avoid log(0).  Default 1e-3.
        """
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature!r}")
        if not (0.0 < eps < 0.5):
            raise ValueError(f"eps must be in (0, 0.5), got {eps!r}")
        self._temp = temperature
        self._eps = eps

    def assign(
        self,
        trace: Trace,
        outcome: Outcome,
        pipeline: Pipeline,
        rng: Random,
    ) -> dict[str, float]:
        """Return per-agent credit in [0, 1] based on confidence calibration.

        Args:
            trace: Executed pipeline trace (confidence values are read from
                each step's ``output.confidence``).
            outcome: Pipeline outcome; ``outcome.reward`` serves as the
                teacher's hindsight probability.
            pipeline: Unused by this assigner (present for interface
                compatibility with HierarchicalCredit and the dispatcher).
            rng: Unused (assigner is deterministic given trace + outcome).
        """
        p_teacher = max(self._eps, min(1.0 - self._eps, float(outcome.reward)))
        credits: dict[str, float] = {}
        for step in trace.steps:
            p_student = max(
                self._eps, min(1.0 - self._eps, float(step.output.confidence))
            )
            # SDPO log-ratio advantage at agent level:
            #   positive  → teacher more confident than agent (calibration deficit)
            #   negative  → teacher less confident (agent was overconfident)
            log_ratio = math.log(p_teacher) - math.log(p_student)
            credits[step.agent_name] = 1.0 / (1.0 + math.exp(-self._temp * log_ratio))
        return credits
