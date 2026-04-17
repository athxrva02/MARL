"""RQ3 experiment: end-to-end vs counterfactual credit in an AIDI-shaped
4-agent pipeline.

Runs each credit method over ``n_seeds`` repetitions, each with
``n_episodes`` learning episodes. Records per-episode reward and blame-
attribution accuracy (fraction of planted-blame episodes where the lowest-
credit agent was the true culprit).

Usage:
    python code/experiments/rq3_e2e_vs_counterfactual.py \
        --config code/experiments/configs/smoke.yaml

Outputs (written next to the config under ``results/<experiment.name>/``):
    summary.csv   -- per (method × seed × episode) record
    summary.json  -- aggregated means and CIs
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from random import Random

# Repo is not installed as a package; make ``code/`` importable.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import yaml  # noqa: E402

from aidi_instance import build_aidi_pipeline  # noqa: E402
from framework.agent import SyntheticAgent  # noqa: E402
from framework.credit import CounterfactualCredit, EndToEndCredit  # noqa: E402
from framework.environment import Task  # noqa: E402
from framework.experiment import EpisodeRecord, RunResult, run_episode  # noqa: E402
from framework.learner import BetaBernoulliThompson  # noqa: E402


def _build_credit_method(spec: dict):
    kind = spec["kind"]
    if kind == "end_to_end":
        return EndToEndCredit()
    if kind == "counterfactual":
        return CounterfactualCredit(n_samples=int(spec.get("n_samples", 4)))
    raise ValueError(f"Unknown credit method kind: {kind!r}")


def _plant_blame(pipeline, agent_name: str, degraded_p: float) -> None:
    """Temporarily override an agent's success probabilities in-place."""
    agent: SyntheticAgent = pipeline.agent_by_name(agent_name)
    agent._original_p = dict(agent._p_success)  # type: ignore[attr-defined]
    agent._p_success = {a: degraded_p for a in agent.actions}  # type: ignore[attr-defined]


def _clear_blame(pipeline, agent_name: str) -> None:
    agent: SyntheticAgent = pipeline.agent_by_name(agent_name)
    agent._p_success = agent._original_p  # type: ignore[attr-defined]
    del agent._original_p  # type: ignore[attr-defined]


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
    pipeline = build_aidi_pipeline(
        upstream_failure_penalty=float(pipeline_spec["upstream_failure_penalty"]),
        code_silent_failure_rate=float(pipeline_spec["code_silent_failure_rate"]),
    )
    learner = BetaBernoulliThompson(
        alpha0=float(learner_spec.get("alpha0", 1.0)),
        beta0=float(learner_spec.get("beta0", 1.0)),
    )
    credit_assigner = _build_credit_method(credit_spec)

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

    blame_rate = float(blame_spec["rate"])
    degraded_p = float(blame_spec["degraded_p"])
    agent_names = pipeline.agent_names

    for ep in range(n_episodes):
        planted: str | None = None
        if rng.random() < blame_rate:
            planted = rng.choice(agent_names)
            _plant_blame(pipeline, planted, degraded_p)

        task = Task(
            task_id=f"ep{ep:05d}",
            payload={},
            context={},
            planted_blame=planted,
        )
        try:
            trace, outcome, credit, actions = run_episode(
                pipeline, task, learner, credit_assigner, rng
            )
        finally:
            if planted is not None:
                _clear_blame(pipeline, planted)

        record = EpisodeRecord(
            task_id=task.task_id,
            planted_blame=planted,
            actions=actions,
            per_agent_success=trace.per_agent_success(),
            outcome_reward=outcome.reward,
            credit=credit,
            blame_credit=credit.get(planted) if planted else None,
        )
        result.episodes.append(record)
    return result


def aggregate(results: dict[str, list[RunResult]]) -> dict:
    summary: dict = {}
    for method, runs in results.items():
        rewards_per_seed = [r.mean_reward() for r in runs]
        acc_per_seed = [
            a for a in (r.blame_attribution_accuracy() for r in runs) if a is not None
        ]
        summary[method] = {
            "n_seeds": len(runs),
            "mean_reward": statistics.fmean(rewards_per_seed),
            "reward_stdev": (
                statistics.stdev(rewards_per_seed) if len(rewards_per_seed) > 1 else 0.0
            ),
            "blame_attribution_accuracy": (
                statistics.fmean(acc_per_seed) if acc_per_seed else None
            ),
            "blame_attribution_stdev": (
                statistics.stdev(acc_per_seed) if len(acc_per_seed) > 1 else 0.0
            ),
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

    # Write per-episode CSV.
    csv_path = output_dir / "summary.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "method", "seed", "episode", "reward",
            "planted_blame", "lowest_credit_agent",
            "blame_credit",
        ])
        for method, runs in results.items():
            for run in runs:
                seed = run.config["seed"]
                for i, e in enumerate(run.episodes):
                    lowest = min(e.credit, key=e.credit.get) if e.credit else ""
                    w.writerow([
                        method, seed, i, f"{e.outcome_reward:.4f}",
                        e.planted_blame or "", lowest,
                        f"{e.blame_credit:.4f}" if e.blame_credit is not None else "",
                    ])

    # Aggregated JSON.
    summary = aggregate(results)
    json_path = output_dir / "summary.json"
    with json_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Results ===")
    for method, stats in summary.items():
        acc = stats["blame_attribution_accuracy"]
        acc_str = f"{acc:.3f}" if acc is not None else "n/a"
        print(
            f"  {method:>16s}: mean_reward={stats['mean_reward']:.3f} "
            f"±{stats['reward_stdev']:.3f}  blame_acc={acc_str} "
            f"±{stats['blame_attribution_stdev']:.3f}"
        )
    print(f"\nWrote: {csv_path}\n       {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
