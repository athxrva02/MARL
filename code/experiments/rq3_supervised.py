"""RQ3 experiment (revised): credit assignment in the faithful AIDI
supervisor-orchestrated refinement loop.

Unlike :mod:`rq3_e2e_vs_counterfactual` (which used a flat 4-agent pipeline),
this runner respects the real AIDI architecture:

  * Supervisor as :class:`RefinementLoop` (not a peer agent).
  * 3 top-level agents: DocumentAnalysisAgent, LogParserAgent, CodeAgent.
  * CodeAgent is a composite of Schema → Builder → Validator → QA.
  * Iterative refinement up to ``max_iterations`` before giving up.

It compares four credit methods:
    * end_to_end        (baseline: all agents get the final reward)
    * counterfactual    (leave-one-out Monte Carlo on the final trace)
    * validator_anchored(uses ValidatorAgent verdict inside CodeAgent)
    * iteration_discounted (weights early-iteration success more)

Usage:
    python code/experiments/rq3_supervised.py \\
        --config code/experiments/configs/supervised_smoke.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import yaml  # noqa: E402

from aidi_instance import build_aidi_supervisor  # noqa: E402
from framework.agent import SyntheticAgent  # noqa: E402
from framework.composite_agent import CompositeAgent  # noqa: E402
from framework.credit import (  # noqa: E402
    CounterfactualCredit,
    EndToEndCredit,
    IterationDiscountedCredit,
    ValidatorAnchoredCredit,
)
from framework.environment import Task  # noqa: E402
from framework.learner import BetaBernoulliThompson  # noqa: E402
from framework.refinement import EpisodeRun, RefinementLoop  # noqa: E402


# ---------------------------------------------------------------------------
# Credit method dispatch — trace-level vs episode-level assigners.
# ---------------------------------------------------------------------------

TRACE_METHODS = {"end_to_end", "counterfactual", "validator_anchored"}
EPISODE_METHODS = {"iteration_discounted"}


def _build_credit_method(spec: dict):
    kind = spec["kind"]
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
    raise ValueError(f"Unknown credit method kind: {kind!r}")


# ---------------------------------------------------------------------------
# Episode bookkeeping.
# ---------------------------------------------------------------------------


@dataclass
class EpisodeRecord:
    task_id: str
    planted_blame: str | None
    actions: dict[str, str]
    per_agent_success: dict[str, bool]  # from final iteration
    final_reward: float
    num_iterations: int
    stopped_reason: str
    credit: dict[str, float]
    blame_credit: float | None = None


@dataclass
class RunResult:
    config: dict[str, Any]
    episodes: list[EpisodeRecord] = field(default_factory=list)

    def mean_reward(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(e.final_reward for e in self.episodes) / len(self.episodes)

    def mean_iterations(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(e.num_iterations for e in self.episodes) / len(self.episodes)

    def blame_attribution_accuracy(self) -> float | None:
        scored = [e for e in self.episodes if e.planted_blame is not None]
        if not scored:
            return None
        hits = 0
        for e in scored:
            if not e.credit:
                continue
            min_agent = min(e.credit, key=e.credit.get)
            if min_agent == e.planted_blame:
                hits += 1
        return hits / len(scored)


# ---------------------------------------------------------------------------
# Action injection and learner wiring.
# ---------------------------------------------------------------------------


def _all_leaf_agents(pipeline) -> list[SyntheticAgent]:
    """Flatten out SyntheticAgents, descending into composite sub-pipelines."""
    out: list[SyntheticAgent] = []
    for a in pipeline.agents:
        if isinstance(a, SyntheticAgent):
            out.append(a)
        elif isinstance(a, CompositeAgent):
            out.extend(_all_leaf_agents(a.sub_pipeline))
    return out


def _choose_actions(
    loop: RefinementLoop, learner: BetaBernoulliThompson, rng: Random
) -> dict[str, str]:
    """Sample an action per leaf SyntheticAgent (top-level + inside composite)."""
    actions: dict[str, str] = {}
    for agent in _all_leaf_agents(loop.pipeline):
        actions[agent.name] = learner.choose(agent.name, agent.actions, rng)
    return actions


def _plant_blame_on_leaf(
    loop: RefinementLoop, agent_name: str, degraded_p: float
) -> tuple[SyntheticAgent, dict[str, float]]:
    """Temporarily drop an agent's success probabilities; return restore info."""
    for agent in _all_leaf_agents(loop.pipeline):
        if agent.name == agent_name:
            original = dict(agent._p_success)  # type: ignore[attr-defined]
            agent._p_success = {a: degraded_p for a in agent.actions}  # type: ignore[attr-defined]
            return agent, original
    raise KeyError(agent_name)


def _clear_blame_on_leaf(
    agent: SyntheticAgent, restore: dict[str, float]
) -> None:
    agent._p_success = restore  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Credit: map top-level credit onto all leaf agents for learner update.
#
# The outer trace has 3 steps (Doc, LogParser, CodeAgent). Sub-agents inside
# CodeAgent share its credit (no inner attribution in phase 1 trace-level
# methods). Iteration-discounted works on the outer trace too.
# ---------------------------------------------------------------------------


def _propagate_credit_to_leaves(
    outer_credit: dict[str, float], loop: RefinementLoop
) -> dict[str, float]:
    flat: dict[str, float] = {}
    for agent in loop.pipeline.agents:
        if isinstance(agent, CompositeAgent):
            c = outer_credit.get(agent.name, 0.5)
            for sub in _all_leaf_agents(agent.sub_pipeline):
                flat[sub.name] = c
        else:
            flat[agent.name] = outer_credit.get(agent.name, 0.5)
    return flat


# ---------------------------------------------------------------------------
# Episode driver.
# ---------------------------------------------------------------------------


def _inject_actions(task: Task, actions: dict[str, str]) -> Task:
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
    kind: str, assigner, run: EpisodeRun, loop: RefinementLoop, rng: Random
) -> dict[str, float]:
    if kind in EPISODE_METHODS:
        return assigner.assign(run, rng)
    # Trace-level: use the final iteration's trace + outcome.
    last = run.iterations[-1]
    return assigner.assign(last.trace, last.outcome, loop.pipeline, rng)


def run_one(
    *,
    credit_spec: dict,
    pipeline_spec: dict,
    learner_spec: dict,
    blame_spec: dict,
    n_episodes: int,
    seed: int,
) -> RunResult:
    rng = Random(seed)
    loop = build_aidi_supervisor(
        upstream_failure_penalty=float(pipeline_spec["upstream_failure_penalty"]),
        code_silent_failure_rate=float(pipeline_spec["code_silent_failure_rate"]),
        max_iterations=int(pipeline_spec.get("max_iterations", 3)),
    )
    learner = BetaBernoulliThompson(
        alpha0=float(learner_spec.get("alpha0", 1.0)),
        beta0=float(learner_spec.get("beta0", 1.0)),
    )
    assigner = _build_credit_method(credit_spec)

    result = RunResult(
        config={
            "credit": credit_spec,
            "pipeline": pipeline_spec,
            "learner": learner_spec,
            "blame": blame_spec,
            "seed": seed,
            "n_episodes": n_episodes,
        }
    )

    leaf_names = [a.name for a in _all_leaf_agents(loop.pipeline)]
    blame_rate = float(blame_spec["rate"])
    degraded_p = float(blame_spec["degraded_p"])

    for ep in range(n_episodes):
        planted: str | None = None
        restore: tuple[SyntheticAgent, dict[str, float]] | None = None
        if rng.random() < blame_rate:
            planted = rng.choice(leaf_names)
            restore = _plant_blame_on_leaf(loop, planted, degraded_p)

        actions = _choose_actions(loop, learner, rng)
        task = _inject_actions(
            Task(task_id=f"ep{ep:05d}", payload={}, context={},
                 planted_blame=planted),
            actions,
        )
        try:
            run: EpisodeRun = loop.run_episode(task, rng)
        finally:
            if restore is not None:
                _clear_blame_on_leaf(*restore)

        credit = _compute_credit(credit_spec["kind"], assigner, run, loop, rng)
        flat_credit = _propagate_credit_to_leaves(credit, loop)

        # Update learner with the credit each leaf saw this episode. If a leaf
        # didn't receive credit (e.g., methods operating only on the outer
        # trace don't populate sub-agents directly — handled by propagation
        # above), fall back to neutral 0.5 so the posterior is untouched.
        for name, action in actions.items():
            c = flat_credit.get(name, 0.5)
            learner.update(name, action, c)

        last_trace = run.iterations[-1].trace
        per_agent_success = last_trace.per_agent_success()

        record = EpisodeRecord(
            task_id=task.task_id,
            planted_blame=planted,
            actions=actions,
            per_agent_success=per_agent_success,
            final_reward=run.final_reward,
            num_iterations=run.num_iterations,
            stopped_reason=run.stopped_reason,
            # For blame-attribution scoring we use the LEAF credit map, so
            # planted blame on e.g. BuilderAgent (inside CodeAgent) is
            # discoverable.
            credit=flat_credit,
            blame_credit=flat_credit.get(planted) if planted else None,
        )
        result.episodes.append(record)
    return result


# ---------------------------------------------------------------------------
# Aggregation & main.
# ---------------------------------------------------------------------------


def aggregate(results: dict[str, list[RunResult]]) -> dict:
    summary: dict = {}
    for method, runs in results.items():
        rewards = [r.mean_reward() for r in runs]
        iters = [r.mean_iterations() for r in runs]
        acc = [a for a in (r.blame_attribution_accuracy() for r in runs) if a is not None]
        summary[method] = {
            "n_seeds": len(runs),
            "mean_reward": statistics.fmean(rewards),
            "reward_stdev": statistics.stdev(rewards) if len(rewards) > 1 else 0.0,
            "mean_iterations": statistics.fmean(iters),
            "blame_attribution_accuracy": statistics.fmean(acc) if acc else None,
            "blame_attribution_stdev": statistics.stdev(acc) if len(acc) > 1 else 0.0,
        }
    return summary


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

    output_dir = args.output_dir or (args.config.parent.parent / "results" / exp_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, list[RunResult]] = defaultdict(list)
    for credit_spec in cfg["credit_methods"]:
        method_name = credit_spec["kind"]
        for s in range(n_seeds):
            seed = base_seed + s
            print(f"  running {method_name} seed={seed}...", flush=True)
            run = run_one(
                credit_spec=credit_spec,
                pipeline_spec=cfg["pipeline"],
                learner_spec=cfg["learner"],
                blame_spec=cfg["blame_injection"],
                n_episodes=n_episodes,
                seed=seed,
            )
            results[method_name].append(run)

    csv_path = output_dir / "summary.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "method", "seed", "episode", "reward",
            "num_iterations", "stopped_reason",
            "planted_blame", "lowest_credit_agent", "blame_credit",
        ])
        for method, runs in results.items():
            for run in runs:
                seed = run.config["seed"]
                for i, e in enumerate(run.episodes):
                    lowest = min(e.credit, key=e.credit.get) if e.credit else ""
                    w.writerow([
                        method, seed, i, f"{e.final_reward:.4f}",
                        e.num_iterations, e.stopped_reason,
                        e.planted_blame or "", lowest,
                        f"{e.blame_credit:.4f}" if e.blame_credit is not None else "",
                    ])

    summary = aggregate(results)
    json_path = output_dir / "summary.json"
    with json_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Results ===")
    for method, stats in summary.items():
        acc = stats["blame_attribution_accuracy"]
        acc_str = f"{acc:.3f}" if acc is not None else "n/a"
        print(
            f"  {method:>22s}: reward={stats['mean_reward']:.3f} "
            f"±{stats['reward_stdev']:.3f}  iters={stats['mean_iterations']:.2f}  "
            f"blame_acc={acc_str} ±{stats['blame_attribution_stdev']:.3f}"
        )
    print(f"\nWrote: {csv_path}\n       {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
