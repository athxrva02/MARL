"""Phase-3 RQ3 experiment: wires all Phase-3 components together.

What's new versus :mod:`rq3_supervised`:

  * Feature-conditioned synthetic tasks via :class:`TaskGenerator` — the
    optimal action is task-dependent, so the feature-blind Thompson
    baseline converges to a distribution-averaged optimum that future
    contextual learners can be compared against.
  * Supervisor as a *learnable arm*: per-episode the learner picks one of
    N decision-policy presets (``aggressive``/``balanced``/``conservative``),
    the refinement loop runs under that preset, and the preset is credited
    with a reward-minus-iteration-cost scalar.
  * Hierarchical credit: every trace-level credit method is wrapped with
    :class:`HierarchicalCredit` so sub-agents inside the CodeAgent composite
    receive signals derived from the sub-trace, not the composite's average.
  * SQLite persistence: every episode + periodic learner posterior
    snapshots go into a single ``.sqlite`` file for offline analysis.
  * Statistical summary: bootstrap CIs on mean reward per method, paired
    permutation test vs the ``end_to_end`` baseline, Holm-Bonferroni
    correction across the family of comparisons.

Usage::

    python code/experiments/rq3_phase3.py \\
        --config code/experiments/configs/phase3_smoke.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import yaml  # noqa: E402

from aidi_instance import build_feature_aware_pipeline  # noqa: E402
from framework.agent_features import FeatureAwareAgent  # noqa: E402
from framework.composite_agent import CompositeAgent  # noqa: E402
from framework.credit import (  # noqa: E402
    CounterfactualCredit,
    EndToEndCredit,
    HierarchicalCredit,
    IterationDiscountedCredit,
    SelfDistillationCredit,
    ValidatorAnchoredCredit,
)
from framework.environment import Task  # noqa: E402
from framework.learner import BetaBernoulliThompson  # noqa: E402
from framework.persistence import (  # noqa: E402
    EpisodeRow,
    ExperimentStore,
    posterior_rows_from_learner,
)
from framework.pipeline import Pipeline  # noqa: E402
from framework.refinement import EpisodeRun, RefinementLoop  # noqa: E402
from framework.stats import (  # noqa: E402
    bootstrap_ci,
    holm_bonferroni,
    paired_permutation_test,
)
from framework.supervisor_agent import (  # noqa: E402
    DEFAULT_PRESETS,
    SUPERVISOR_AGENT_NAME,
    SupervisorAgent,
    preset_to_policy,
    supervisor_credit,
)
from framework.task_generator import (  # noqa: E402
    TaskFeatureDistribution,
    TaskGenerator,
)


TRACE_METHODS = {"end_to_end", "counterfactual", "validator_anchored", "self_distillation"}
EPISODE_METHODS = {"iteration_discounted"}


def _build_assigner(kind: str, spec: dict):
    if kind == "end_to_end":
        return EndToEndCredit()
    if kind == "counterfactual":
        return CounterfactualCredit(n_samples=int(spec.get("n_samples", 4)))
    if kind == "validator_anchored":
        return ValidatorAnchoredCredit(
            validator_name=spec.get("validator_name", "ValidatorAgent"),
        )
    if kind == "iteration_discounted":
        return IterationDiscountedCredit(gamma=float(spec.get("gamma", 0.7)))
    if kind == "self_distillation":
        return SelfDistillationCredit(
            temperature=float(spec.get("temperature", 1.0)),
            eps=float(spec.get("eps", 1e-3)),
        )
    raise ValueError(f"Unknown credit kind: {kind!r}")


def _build_method(spec: dict):
    """Build the concrete (possibly hierarchical) CreditAssigner."""
    outer_kind = spec["kind"]
    outer = _build_assigner(outer_kind, spec)
    if spec.get("hierarchical", True) and outer_kind in TRACE_METHODS:
        inner_spec = spec.get("inner") or {"kind": outer_kind}
        inner = _build_assigner(inner_spec["kind"], inner_spec)
        return HierarchicalCredit(
            outer, inner_assigner=inner,
            outer_weight=float(spec.get("outer_weight", 0.3)),
        )
    return outer


# ---------------------------------------------------------------------------
# Agent/leaf utilities.
# ---------------------------------------------------------------------------


def _leaf_agents(pipeline: Pipeline) -> list[FeatureAwareAgent]:
    out: list[FeatureAwareAgent] = []
    for a in pipeline.agents:
        if isinstance(a, CompositeAgent):
            out.extend(_leaf_agents(a.sub_pipeline))
        else:
            out.append(a)
    return out


def _choose_leaf_actions(
    pipeline: Pipeline, learner: BetaBernoulliThompson, rng: Random
) -> dict[str, str]:
    actions: dict[str, str] = {}
    for agent in _leaf_agents(pipeline):
        actions[agent.name] = learner.choose(agent.name, agent.actions, rng)
    return actions


def _inject_context(task: Task, actions: dict[str, str]) -> Task:
    ctx = dict(task.context)
    for name, action in actions.items():
        ctx[f"action_for:{name}"] = action
    return Task(
        task_id=task.task_id,
        payload=task.payload,
        context=ctx,
        planted_blame=task.planted_blame,
    )


def _compute_credit(
    kind: str, assigner, run: EpisodeRun, pipeline: Pipeline, rng: Random
) -> dict[str, float]:
    if kind in EPISODE_METHODS:
        # Episode-level assigners have their own signature; they're not
        # wrapped in HierarchicalCredit (the wrapping path is only for
        # trace-level methods). Drive them directly on the EpisodeRun.
        return assigner.assign(run, rng)
    last = run.iterations[-1]
    return assigner.assign(last.trace, last.outcome, pipeline, rng)


# ---------------------------------------------------------------------------
# Episode bookkeeping (kept lightweight; SQLite is the source of truth).
# ---------------------------------------------------------------------------


@dataclass
class SeedResult:
    method: str
    seed: int
    rewards: list[float] = field(default_factory=list)
    iterations: list[int] = field(default_factory=list)

    @property
    def mean_reward(self) -> float:
        return sum(self.rewards) / len(self.rewards) if self.rewards else 0.0

    @property
    def mean_iterations(self) -> float:
        return sum(self.iterations) / len(self.iterations) if self.iterations else 0.0


# ---------------------------------------------------------------------------
# Core driver.
# ---------------------------------------------------------------------------


def run_one_seed(
    *,
    experiment_name: str,
    method_name: str,
    credit_spec: dict,
    pipeline_spec: dict,
    learner_spec: dict,
    task_spec: dict,
    n_episodes: int,
    seed: int,
    store: ExperimentStore,
    snapshot_every: int,
) -> SeedResult:
    rng = Random(seed)
    pipeline = build_feature_aware_pipeline(
        upstream_failure_penalty=float(pipeline_spec["upstream_failure_penalty"]),
        code_silent_failure_rate=float(pipeline_spec["code_silent_failure_rate"]),
    )
    max_iters_cfg = int(pipeline_spec.get("max_iterations_hint", 5))

    learner = BetaBernoulliThompson(
        alpha0=float(learner_spec.get("alpha0", 1.0)),
        beta0=float(learner_spec.get("beta0", 1.0)),
    )
    assigner = _build_method(credit_spec)
    supervisor = SupervisorAgent(presets=DEFAULT_PRESETS)

    # Task stream: deterministic given seed.
    generator = TaskGenerator(
        seed=seed,
        distribution=TaskFeatureDistribution(
            doc_size_probs=dict(task_spec.get("doc_size_probs",
                {"small": 0.4, "medium": 0.4, "large": 0.2})),
            log_volume_probs=dict(task_spec.get("log_volume_probs",
                {"low": 0.3, "medium": 0.5, "high": 0.2})),
            pii_prob=float(task_spec.get("pii_prob", 0.2)),
        ),
        id_prefix=f"{method_name}-{seed}-",
    )

    result = SeedResult(method=method_name, seed=seed)

    for ep, task in enumerate(generator.stream(n_episodes)):
        # 1. Learner picks the Supervisor preset and the per-leaf actions.
        sup_action = learner.choose(
            SUPERVISOR_AGENT_NAME, supervisor.actions, rng
        )
        preset = supervisor.preset_of(sup_action)
        loop = RefinementLoop(
            pipeline=pipeline, decision_policy=preset_to_policy(preset)
        )
        leaf_actions = _choose_leaf_actions(pipeline, learner, rng)

        # 2. Run the episode with injected actions (features already set).
        task_with_actions = _inject_context(task, leaf_actions)
        run = loop.run_episode(task_with_actions, rng)

        # 3. Credit assignment & learner updates.
        credit = _compute_credit(credit_spec["kind"], assigner, run, pipeline, rng)
        for leaf_name, leaf_action in leaf_actions.items():
            learner.update(leaf_name, leaf_action, credit.get(leaf_name, 0.5))
        sup_c = supervisor_credit(run, max_iterations_hint=max_iters_cfg)
        learner.update(SUPERVISOR_AGENT_NAME, sup_action, sup_c)

        # 4. Persist this episode.
        actions_for_row = dict(leaf_actions)
        actions_for_row[SUPERVISOR_AGENT_NAME] = sup_action
        credits_for_row = dict(credit)
        credits_for_row[SUPERVISOR_AGENT_NAME] = sup_c
        store.record_episode(EpisodeRow(
            experiment=experiment_name,
            method=method_name,
            seed=seed,
            episode=ep,
            task_id=task.task_id,
            features=task.context.get("features"),
            reward=run.final_reward,
            num_iterations=run.num_iterations,
            stopped_reason=run.stopped_reason,
            planted_blame=None,  # feature-aware tasks don't plant blame
            credits=credits_for_row,
            actions=actions_for_row,
            action_credits=credits_for_row,
        ))

        # 5. Periodic posterior snapshot for learning-curve plots.
        if snapshot_every > 0 and (ep + 1) % snapshot_every == 0:
            store.snapshot_posteriors(posterior_rows_from_learner(
                learner, experiment=experiment_name,
                method=method_name, seed=seed, episode=ep,
            ))

        result.rewards.append(run.final_reward)
        result.iterations.append(run.num_iterations)

    # Final snapshot regardless of interval.
    store.snapshot_posteriors(posterior_rows_from_learner(
        learner, experiment=experiment_name,
        method=method_name, seed=seed, episode=n_episodes - 1,
    ))

    return result


# ---------------------------------------------------------------------------
# Aggregation.
# ---------------------------------------------------------------------------


def _summarise(
    per_method: dict[str, list[SeedResult]],
    *,
    baseline: str,
    bootstrap_resamples: int,
    permutation_resamples: int,
    alpha: float,
) -> dict[str, Any]:
    out: dict[str, Any] = {"per_method": {}, "pairwise_vs_baseline": []}
    rng = Random(0)

    # Per-method bootstrap CI on mean reward (input: one mean per seed).
    for method, runs in per_method.items():
        seed_means = [r.mean_reward for r in runs]
        ci = bootstrap_ci(seed_means, n_resamples=bootstrap_resamples, rng=rng)
        out["per_method"][method] = {
            "n_seeds": len(runs),
            "mean_reward": ci.mean,
            "ci_low": ci.low,
            "ci_high": ci.high,
            "mean_iterations": (
                sum(r.mean_iterations for r in runs) / len(runs) if runs else 0.0
            ),
        }

    # Paired permutation test: each non-baseline method vs baseline, on the
    # per-seed mean reward. Seeds must be aligned — they are, by construction
    # (same seed range for every method).
    if baseline in per_method:
        baseline_means_by_seed = {r.seed: r.mean_reward for r in per_method[baseline]}
        pairwise_p: list[tuple[str, float]] = []
        pairwise_meta: list[tuple[str, float, float, int]] = []
        for method, runs in per_method.items():
            if method == baseline:
                continue
            a = [r.mean_reward for r in runs]
            b = [baseline_means_by_seed[r.seed] for r in runs]
            pc = paired_permutation_test(
                a, b, n_permutations=permutation_resamples, rng=rng
            )
            pairwise_p.append((method, pc.p_value))
            pairwise_meta.append((method, pc.diff, pc.p_value, pc.n_pairs))
        adjusted = holm_bonferroni(pairwise_p, alpha=alpha)
        adjusted_by_label = {r[0]: r for r in adjusted}
        for method, diff, p, n in pairwise_meta:
            _, raw_p, adj_p, reject = adjusted_by_label[method]
            out["pairwise_vs_baseline"].append({
                "method": method,
                "baseline": baseline,
                "diff": diff,
                "raw_p": raw_p,
                "adjusted_p": adj_p,
                "reject_at_alpha": reject,
                "n_pairs": n,
            })

    return out


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args(argv)

    with args.config.open() as f:
        cfg = yaml.safe_load(f)

    exp_name = cfg["experiment"]["name"]
    base_seed = int(cfg["experiment"]["seed"])
    n_episodes = int(cfg["experiment"]["n_episodes"])
    n_seeds = int(cfg["experiment"]["n_seeds"])
    baseline = cfg["experiment"].get("baseline", "end_to_end")
    snapshot_every = int(cfg["experiment"].get("snapshot_every", 100))
    bootstrap_resamples = int(cfg["experiment"].get("bootstrap_resamples", 2000))
    permutation_resamples = int(cfg["experiment"].get("permutation_resamples", 5000))
    alpha = float(cfg["experiment"].get("alpha", 0.05))

    output_dir = args.output_dir or (args.config.parent.parent / "results" / exp_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = output_dir / "episodes.sqlite"

    per_method: dict[str, list[SeedResult]] = defaultdict(list)
    with ExperimentStore(db_path) as store:
        for credit_spec in cfg["credit_methods"]:
            method_name = credit_spec.get("name", credit_spec["kind"])
            for s in range(n_seeds):
                seed = base_seed + s
                print(f"  running {method_name} seed={seed}...", flush=True)
                r = run_one_seed(
                    experiment_name=exp_name,
                    method_name=method_name,
                    credit_spec=credit_spec,
                    pipeline_spec=cfg["pipeline"],
                    learner_spec=cfg["learner"],
                    task_spec=cfg.get("tasks", {}),
                    n_episodes=n_episodes,
                    seed=seed,
                    store=store,
                    snapshot_every=snapshot_every,
                )
                per_method[method_name].append(r)

    summary = _summarise(
        per_method,
        baseline=baseline,
        bootstrap_resamples=bootstrap_resamples,
        permutation_resamples=permutation_resamples,
        alpha=alpha,
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Human-readable console view.
    print("\n=== Per-method (mean reward with 95% bootstrap CI) ===")
    for method, s in summary["per_method"].items():
        print(
            f"  {method:>22s}: {s['mean_reward']:.3f} "
            f"[{s['ci_low']:.3f}, {s['ci_high']:.3f}]  "
            f"iters={s['mean_iterations']:.2f}  n_seeds={s['n_seeds']}"
        )
    print(f"\n=== Pairwise paired-permutation vs {baseline} (Holm-Bonferroni @ α={alpha}) ===")
    if not summary["pairwise_vs_baseline"]:
        print("  (no comparisons — baseline absent or sole method)")
    for row in summary["pairwise_vs_baseline"]:
        mark = "*" if row["reject_at_alpha"] else " "
        print(
            f"  {mark} {row['method']:>22s}  Δ={row['diff']:+.3f}  "
            f"p={row['raw_p']:.3f}  adj={row['adjusted_p']:.3f}"
        )
    print(f"\nWrote: {db_path}\n       {output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
