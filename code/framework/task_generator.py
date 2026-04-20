"""Feature-conditioned synthetic task generator.

Phase-1/2 used empty-payload tasks: every task looked identical from an
agent's perspective, so the Bayesian learner's per-action posterior
converged to a single global optimum. Phase-3 introduces *task features*
so that an agent's best action can depend on the task — the prerequisite
for RQ2-style contextual-bandit experiments and a better simulation of
AIDI's real heterogeneous workload (small vs large documents, noisy vs
clean logs, PII vs non-PII).

Each generated :class:`Task` carries a feature dict in ``context['features']``.
A :class:`FeatureAwareAgent` (see :mod:`framework.agent_features`) looks up
its per-action success probability via a feature-conditioned table. The
Bayesian learner in this repo is feature-blind — its posteriors are per
(agent, action) and ignore task features — so the learned optimum is an
average over the feature distribution. This is *intended*: Phase-3 is about
making the task distribution non-degenerate so that subsequent phases can
measure context-aware learners against this baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Iterator, Mapping

from framework.environment import Task


# Feature value domains. Kept small and categorical so the feature space
# is enumerable for diagnostic plots.
DOC_SIZES = ("small", "medium", "large")
LOG_VOLUMES = ("low", "medium", "high")
PII_FLAGS = (False, True)


@dataclass(frozen=True)
class TaskFeatures:
    """Discrete task features influencing per-agent success."""

    doc_size: str
    log_volume: str
    has_pii: bool

    def as_dict(self) -> dict:
        return {
            "doc_size": self.doc_size,
            "log_volume": self.log_volume,
            "has_pii": self.has_pii,
        }


@dataclass
class TaskFeatureDistribution:
    """Marginal distribution over each feature. Independence assumed."""

    doc_size_probs: Mapping[str, float] = field(
        default_factory=lambda: {"small": 0.4, "medium": 0.4, "large": 0.2}
    )
    log_volume_probs: Mapping[str, float] = field(
        default_factory=lambda: {"low": 0.3, "medium": 0.5, "high": 0.2}
    )
    pii_prob: float = 0.2

    def sample(self, rng: Random) -> TaskFeatures:
        return TaskFeatures(
            doc_size=_sample_discrete(self.doc_size_probs, rng),
            log_volume=_sample_discrete(self.log_volume_probs, rng),
            has_pii=rng.random() < self.pii_prob,
        )


def _sample_discrete(probs: Mapping[str, float], rng: Random) -> str:
    r = rng.random()
    acc = 0.0
    last = None
    for k, p in probs.items():
        acc += p
        last = k
        if r < acc:
            return k
    return last  # floating-point fallback


@dataclass
class TaskGenerator:
    """Yield reproducible synthetic tasks.

    The generator is deterministic given its ``seed``: consecutive calls
    produce the same sequence of tasks. This lets different credit methods
    be compared on the *same* task stream.
    """

    seed: int = 0
    distribution: TaskFeatureDistribution = field(default_factory=TaskFeatureDistribution)
    id_prefix: str = "t"

    def stream(self, n: int) -> Iterator[Task]:
        rng = Random(self.seed)
        for i in range(n):
            feats = self.distribution.sample(rng)
            yield Task(
                task_id=f"{self.id_prefix}{i:05d}",
                payload={"features": feats.as_dict()},
                context={"features": feats.as_dict()},
            )
