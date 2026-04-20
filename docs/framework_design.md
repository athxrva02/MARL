# Framework Design — Credit-Assignment Research Testbed for AIDI

Status: Phase-3. Building on Phase-2's faithful AIDI architecture
(Supervisor as `RefinementLoop`, 3 top-level agents, `CompositeAgent`
CodeAgent, four credit methods), Phase-3 adds: (1) the Supervisor as a
*learnable arm* over decision-policy presets; (2) `HierarchicalCredit`,
a wrapper that lets any flat credit method descend into a composite's
sub-trace; (3) feature-conditioned synthetic tasks (`TaskGenerator` +
`FeatureAwareAgent`) so the optimal action becomes task-dependent;
(4) SQLite persistence (`ExperimentStore`) for every episode + periodic
learner posterior snapshots; (5) a stdlib-only statistics module
(bootstrap CIs, paired permutation tests, Holm-Bonferroni). 78/78 unit
tests pass; three smoke experiments (flat 4-agent, supervised
refinement, Phase-3 feature-aware) run in seconds on commodity hardware.
The research question "what is the best way to inject information into
the RL framework?" is answered in the companion design note
[`docs/information_injection.md`](information_injection.md).

This document is an architecture decision record (ADR) for the simulation
framework in [`code/framework/`](../code/framework/). It exists because
research readers and the intern's successor will need to understand **why**
the interfaces are shaped the way they are — not just what they do.

---

## 1. Design goals

1. **Cheap & reproducible.** No LLM calls in Phase 1. A full run of the
   smoke config completes in under a second and is bit-for-bit reproducible
   given a seed.
2. **Pluggable credit assignment.** The whole point of the study is to
   compare credit strategies, so the `CreditAssigner` protocol must be the
   narrowest reasonable interface.
3. **Generic over pipelines.** AIDI's 4-agent pipeline is *one* instance.
   The same framework must accept 3-, 6-, or 8-agent pipelines without
   refactoring so we can study how credit methods scale with depth.
4. **Mirror AIDI interfaces by name only.** The framework must not depend
   on the AIDI repository, but class names and method shapes should be
   close enough that the intern can port learnings back cheaply (see
   `BetaBernoulliThompson` ↔ `BayesianAttributeLearner`).

## 2. Core abstractions

```
 Task ─► Pipeline ─► Trace ─► Outcome
                       │         │
                       ▼         ▼
                CreditAssigner ─► per-agent credit ─► Learner
```

### 2.1 `Agent`

A `Protocol` with a single `act(obs, rng) -> AgentOutput` method. Stateless
on purpose: the counterfactual assigner re-executes agents with fresh
randomness, and hidden state makes that unsound.

**Why `AgentOutput.success`?** The framework keeps ground-truth per-step
success because the simulation has oracle knowledge. Credit assigners must
*not* read `success` — it's the hidden target they are being evaluated
against. The experiment runner uses it only to compute `Outcome`.

### 2.2 `Pipeline`

Owns agent order, threads outputs forward, and records traces. Two
execution primitives:

- `run(task, rng)` → full execution.
- `rerun_from(trace, agent_name, rng, override_output=None)` → replay
  preserving prefix; either resample the target agent (no override) or
  substitute a given output. This is the workhorse for counterfactual
  credit.

### 2.3 `Trace` and `TraceStep`

Immutable. Steps carry the observation that was actually fed in (including
`upstream_failed`) so assigners can audit the exact conditions each agent
ran under.

### 2.4 `CreditAssigner`

Narrow protocol returning `dict[agent_name, float]` with values in `[0, 1]`.
Choosing `[0, 1]` (rather than `[-1, 1]`) lets the Bayesian learner consume
credit directly as a soft-Bernoulli outcome, which keeps the learner
agnostic to which assigner is in use.

### 2.5 `Learner`

`Learner` selects an action per agent and ingests credit. The Phase-1
concrete class is `BetaBernoulliThompson` — matches AIDI's dormant
`BayesianAttributeLearner` mentioned in the internship proposal.

Soft-Bernoulli update `α += credit; β += 1 − credit` is mathematically
well-defined (it matches the expected sufficient statistic for a
fractional observation) and avoids the discretization artefacts you get
from a hard threshold at 0.5.

### 2.6 `CompositeAgent` — a pipeline that *is* an Agent

AIDI's real `CodeAgent` is itself a pipeline (Schema → Builder → Validator
→ QA). Modelling it as a flat peer of DocAnalysis / LogParser was
structurally wrong: the Validator's per-step verdict is a *structured*
credit signal that flat credit methods cannot use. We therefore introduce
`CompositeAgent`, which:

- Implements the `Agent` protocol externally (single `act(obs, rng)`).
- Runs a nested `Pipeline` internally and embeds the full sub-trace in its
  `AgentOutput.value` (via `CompositeOutput`).
- Exposes `sub_trace_of(output)` so hierarchical credit assigners (e.g.
  `ValidatorAnchoredCredit`) can descend into the composite without
  breaking the flat-assigner contract.

Flat credit methods (`EndToEndCredit`, `CounterfactualCredit`) treat a
composite as a single unit — their credit goes to the composite, and the
experiment runner propagates it uniformly to all leaves for learner
updates. Hierarchical methods override `_composite_credit(sub_trace)` to
use the structured signal.

### 2.7 `RefinementLoop` — Supervisor as orchestrator, not peer

Real AIDI runs a LangGraph of the form:

```
Doc → Log → Code → DecisionEngine ─complete─► End
                         │
                         └─refine─► Doc (with clarification)
```

`RefinementLoop` captures the two features that matter for credit:

1. The same agents may run multiple times on the same task.
2. The decision to refine vs stop is itself a choice with credit
   implications.

Its episode-level record is `EpisodeRun = list[Iteration] + final_outcome`.
Trace-level credit assigners (`end_to_end`, `counterfactual`,
`validator_anchored`) consume the final iteration's trace; episode-level
assigners (`IterationDiscountedCredit`) consume the whole `EpisodeRun`.
The experiment runner dispatches on the assigner's shape — see
`rq3_supervised.TRACE_METHODS` vs `EPISODE_METHODS`.

### 2.8 `SupervisorAgent` — Supervisor as a learnable arm (Phase 3)

Phase-2 hard-coded the Supervisor as a single `CompletenessDecisionPolicy`
with fixed hyperparameters. The assignment's note "RL should learn across
the entire multi-agent system, including the Supervisor" forced a choice:
either (a) make the Supervisor itself a RL agent with a continuous
hyperparameter action space, or (b) expose a *discrete* set of named
presets and let the existing Thompson-sampling learner pick among them.

We chose (b): `SupervisorAgent` exposes three presets — `aggressive`
(threshold 1.0, ≤2 iters), `balanced` (0.8, ≤3), `conservative`
(0.5, ≤5) — and the learner treats them as it treats any other agent's
action set. Per-episode the runner samples a preset, builds the
`RefinementLoop` from it, runs the episode, then credits the Supervisor
arm with `supervisor_credit(run) = reward − cost·(n_iters−1)/(max−1)`.

**Why presets, not a continuous policy:** (1) Beta-Bernoulli conjugacy
only covers discrete arms; a continuous Supervisor would require a
different learner and break the "one learner for the whole system"
invariant. (2) The presets are named so the research reader can reason
about them; a continuous threshold is harder to interpret. (3) The
reward-minus-cost shape of `supervisor_credit` already rewards cheap
success and penalises wasted iterations — the trade-off the continuous
learner would be discovering anyway.

### 2.9 `HierarchicalCredit` — flat methods descend into composites

Flat credit assigners treat a `CompositeAgent` as one unit: their credit
lands on the composite and the runner propagates it uniformly to every
leaf. That is fine for the outer pipeline but wastes the sub-trace that
the composite embeds in its `AgentOutput`.

`HierarchicalCredit(outer, inner_assigner=..., outer_weight=w)` wraps any
flat assigner and:

1. Runs the outer assigner on the top-level trace.
2. For each `CompositeAgent` step, runs the inner assigner on the
   embedded `sub_trace`.
3. Blends: `leaf_credit = w · outer_composite_credit + (1-w) · inner_leaf_credit`.

This preserves the flat-assigner API (Phase-1/2 experiments keep working
unchanged) while making per-leaf signals available to the learner. Inner
and outer can be *different* methods — e.g. `CounterfactualCredit`
outside, `ValidatorAnchoredCredit` inside — which is the easiest way to
express a "hierarchical-hybrid" credit method without writing a new
class.

### 2.10 `FeatureAwareAgent` + `TaskGenerator` (Phase 3)

Phase-1/2 tasks had empty payloads: every task looked identical, so the
learned optimum was a single global best action per agent.
`FeatureAwareAgent` takes a base `p_success` table plus a list of
`FeatureEffect` entries that additively shift `p_success` based on task
features (e.g. `drain3_shallow: −0.15 when log_volume="high"`).
`TaskGenerator` emits a stream of `Task` objects carrying
`context["features"] = {doc_size, log_volume, has_pii}`, sampled from a
configurable marginal distribution.

**Deliberate feature-blindness of the learner.** The Beta-Bernoulli
learner continues to ignore features: its posteriors are per
(agent, action), not per (agent, action, feature-bucket). The learned
optimum is therefore the *average over the feature distribution*. This
is intentional — Phase-3 is about making the task distribution
non-degenerate so that a future contextual learner (RQ2) can be measured
against this feature-blind anchor. See
[`docs/information_injection.md`](information_injection.md) §3 for the
full argument.

### 2.11 `ExperimentStore` — SQLite persistence (Phase 3)

The assignment's note "to learn from previous mistakes and find optimal
policies, we should store errors and correct decisions in an appropriate
database (Take SQLite for now)" is operationalised as a three-table
schema:

- `episodes`     — one row per episode: task id, features, reward,
  iteration count, credits JSON, planted-blame ground truth.
- `actions`      — one row per (episode, agent): the chosen action + the
  credit that agent received on this episode.
- `posterior_snapshots` — periodic `(agent, action, α, β)` rows so
  learning curves can be reconstructed offline.

The store is append-only and the same file is shared across runs;
queries scope by `(experiment, method, seed)`. No ORM — downstream
analysis uses plain SQL or pandas.

**Logging is not learning.** These rows are *persisted* from the loop,
not *injected* back into it. Re-credit / re-score passes can reconstruct
everything from stored traces + seeds without re-executing the pipeline.

### 2.12 `framework.stats` — statistical summaries (Phase 3)

Phase-1/2 reported mean ± stdev over seeds, which cannot answer "is A
significantly better than B?". Phase-3 adds:

- `bootstrap_ci(data)`         — percentile-bootstrap CI on the mean.
- `paired_permutation_test(a,b)` — two-sided paired permutation test;
  aligns seeds so the comparison is matched.
- `holm_bonferroni(pairs, α)`  — step-down correction when multiple
  methods are compared against one baseline.

Implemented on Python stdlib (`random`, `statistics`) so the framework
stays dependency-light.

## 3. Key design decisions

### 3.1 Why a single `Random` instance threaded through the whole episode?

Every stochastic element (agent action, pipeline draw, credit resampling)
pulls from the same RNG. This guarantees that `run_episode(seed=k)` is
fully deterministic. It is slightly awkward for counterfactual replay
(resampling agent *i* consumes draws that the next episode would otherwise
see) but the determinism win is worth it; users who need IID counterfactual
rollouts can fork the RNG at call time. Empirically-measured
reproducibility is asserted in `test_run_episode_deterministic_under_fixed_seed`.

### 3.2 Why not use `numpy`?

Framework stays on stdlib `random` so the dependency surface is tiny.
`Random.betavariate` is well-tested and sufficient for Thompson sampling at
this scale. If a future RQ needs vectorised Monte-Carlo, swap in numpy
then — the `Learner` protocol isolates the choice.

### 3.3 Action injection through `Task.context`

The learner decides an action per agent **before** the pipeline runs.
Rather than threading an "action selection" callback through `Pipeline`,
we inject each choice into `Task.context[f"action_for:{agent_name}"]` and
have the agent read it. This keeps `Pipeline` pure transport and lets
non-synthetic agents opt in by convention.

Trade-off: agents that don't honour the convention silently pick their own
actions. This is acceptable for Phase 1 (only `SyntheticAgent` exists);
when real LLM agents are wrapped, their wrapper must either (a) respect
the injected action or (b) expose a degenerate single-action space.

### 3.4 Counterfactual method: leave-one-out with fresh downstream sampling

`CounterfactualCredit` replays the pipeline from agent *i* with no override,
which means both agent *i*'s output **and** every downstream agent are
re-sampled. This is stronger than the strict COMA "marginalise agent *i*'s
action, everything else equal" notion because our downstream agents are
stochastic and the framework does not snapshot their randomness.

**Known limitation (validated by tests):** this method cannot cleanly
separate a deterministic upstream agent from its stochastic downstream
neighbours — resampling either produces the same distribution over
outcomes. The test
`test_counterfactual_blames_downstream_of_failure_not_above_it` codifies
this: on a `[det, stoch, det]` pipeline the method correctly assigns more
blame to the middle (stochastic) agent than to the trailing deterministic
one, but cannot distinguish the head deterministic agent from the middle
stochastic one. For AIDI, where every real agent has non-trivial
stochasticity, this corner is unlikely to bite; it is documented here so
future readers don't re-derive it.

A strict-COMA variant (override output, keep downstream randomness fixed)
can be added by using the `override_output` path of `rerun_from` plus
seeded per-agent RNGs — the interface already supports this. It is
deliberately not Phase-1 scope.

### 3.5 Counterfactual credit has *low observed variance* on this task

Smoke experiment result (`code/experiments/configs/smoke.yaml`, 5 seeds,
400 episodes, 4 agents):

| method         | late mean reward | blame-attr. acc |
|----------------|------------------|-----------------|
| end_to_end     | 0.186 ± 0.03     | 0.259 ± 0.024   |
| counterfactual | 0.154 ± 0.03     | 0.280 ± 0.026   |

A longer 1500-episode run (no blame injection) shows end-to-end *beating*
counterfactual on late-training reward (0.416 vs 0.239). Root cause:
counterfactual credit hovers near 0.5 on this pipeline, so each Bayesian
update is tiny (α += 0.5, β += 0.5) and the posterior barely sharpens.
End-to-end gives a saturated 0/1 signal per episode, updating the
posterior by a full unit.

This is a **real research finding** for the survey — not a bug. The
framework behaves as specified; the credit method itself is the issue.
Remedies to investigate in the next iteration:

- **Baseline subtraction:** credit = reward − E[reward | resample], not
  clamped to [0,1]. Use a sigmoid or affine rescale tuned to observed
  variance.
- **Use the learner's own posterior to draw counterfactual actions**, not
  agent defaults. This aligns the counterfactual with what the learner
  would actually do, tightening the signal.
- **Difference rewards (Wolpert/Tumer):** replace agent *i* with a fixed
  "default" action rather than resampling its distribution.
- **Shapley value with exact enumeration** over 2⁴=16 coalitions — still
  cheap enough; provides a principled alternative. The `CreditAssigner`
  protocol accepts it unchanged.

### 3.6 Supervised-refinement smoke results (revised architecture)

Smoke experiment (`code/experiments/configs/supervised_smoke.yaml`,
5 seeds, 400 episodes, 3 top-level agents, CodeAgent as a 4-stage
composite, 40% blame-injection rate over 6 leaves):

| method               | mean reward  | mean iters | blame-attr. acc |
|----------------------|--------------|------------|-----------------|
| end_to_end           | 0.206 ± 0.02 | 1.89       | 0.161 ± 0.031   |
| counterfactual       | 0.183 ± 0.02 | 1.90       | 0.227 ± 0.024   |
| validator_anchored   | 0.225 ± 0.03 | 1.87       | 0.301 ± 0.050   |
| iteration_discounted | 0.223 ± 0.04 | 1.87       | **0.465 ± 0.015** |

Chance baseline is 1/6 ≈ 0.167. Observations:

- `end_to_end` is at chance — as expected for a method that cannot
  differentiate agents.
- `counterfactual` recovers a real but modest signal (~36% above chance).
- `validator_anchored` roughly doubles accuracy over end-to-end by
  exploiting the ValidatorAgent's structured verdict inside the
  composite. This is the cheapest hierarchical method and should be
  the first thing plugged into real AIDI.
- `iteration_discounted` wins decisively (~2.9× chance). Planted-blame
  agents fail repeatedly across all refinement iterations; honest
  stochastic agents do not. Temporal repetition is a strong, cheap
  signal that no pure single-trace method can see.

Implication for AIDI: **iteration-aware credit should be activated
together with any trace-level method** when the refinement loop is in
play. The two are complementary — one uses cross-iteration repetition,
the other uses within-iteration structure.

### 3.7 Phase-3 smoke results (feature-aware tasks, learnable Supervisor)

Smoke experiment (`code/experiments/configs/phase3_smoke.yaml`, 5 seeds,
300 episodes, feature-conditioned tasks, Supervisor preset chosen by
the learner, trace methods wrapped in `HierarchicalCredit(outer_weight=0.3)`):

| method                 | mean reward [95% bootstrap CI] | iters | Δ vs e2e | adj-p |
|------------------------|--------------------------------|-------|----------|-------|
| end_to_end             | 0.337 [0.308, 0.365]           | 1.80  | —        | —     |
| counterfactual         | 0.271 [0.252, 0.294]           | 1.86  | −0.066   | 0.360 |
| validator_anchored     | 0.362 [0.331, 0.396]           | 1.80  | +0.025   | 0.570 |
| iteration_discounted   | 0.308 [0.293, 0.321]           | 1.84  | −0.029   | 0.490 |

(Paired permutation, Holm-Bonferroni across three comparisons vs
`end_to_end`. No adjusted p clears α=0.05 at N=5 seeds × 300 episodes;
this is expected at smoke-scale and motivates the full-scale sweep.)

Observations specific to Phase-3:

- The *blame-attribution* signal is now subsumed by the reward-CI
  comparison: without planted blame on feature-aware tasks, the
  performance difference across methods is itself the quantity of
  interest, and the CIs are the principled way to report it.
- `validator_anchored` under the hierarchical wrapper remains the best
  point estimate — consistent with Phase-2's finding that structured
  per-step verdicts dominate on the composite architecture.
- The Supervisor's posterior over `{aggressive, balanced, conservative}`
  (read from `posterior_snapshots`) converges to a method-dependent
  preference: methods that push reward upward make `aggressive`
  attractive (cheap success), while less-informative methods keep
  `conservative` competitive (extra iterations as a hedge). This is the
  first measurable result of the Supervisor being learnable rather than
  fixed.

## 4. Directory layout

```
MARL/
├── code/
│   ├── framework/
│   │   ├── agent.py                  # Agent protocol + SyntheticAgent
│   │   ├── agent_features.py         # FeatureAwareAgent (Phase 3)
│   │   ├── composite_agent.py        # CompositeAgent (nested pipeline)
│   │   ├── refinement.py             # RefinementLoop, EpisodeRun, DecisionPolicy
│   │   ├── supervisor_agent.py       # SupervisorAgent learnable arm (Phase 3)
│   │   ├── task_generator.py         # Feature-conditioned tasks (Phase 3)
│   │   ├── environment.py            # Task, Outcome, default_outcome
│   │   ├── pipeline.py               # Pipeline, Trace, TraceStep
│   │   ├── experiment.py             # run_episode, EpisodeRecord, RunResult
│   │   ├── persistence.py            # ExperimentStore (SQLite, Phase 3)
│   │   ├── stats.py                  # Bootstrap CI, permutation, Holm (Phase 3)
│   │   ├── credit/
│   │   │   ├── base.py               # CreditAssigner protocol
│   │   │   ├── end_to_end.py         # EndToEndCredit (baseline)
│   │   │   ├── counterfactual.py     # CounterfactualCredit (leave-one-out)
│   │   │   ├── validator_anchored.py # Hierarchical (uses Validator verdict)
│   │   │   ├── iteration_discounted.py # Episode-level, γ^k weighting
│   │   │   ├── hierarchical.py       # HierarchicalCredit wrapper (Phase 3)
│   │   │   └── cascading_confidence.py # stub; interface compat
│   │   ├── learner/
│   │   │   ├── base.py               # Learner protocol
│   │   │   └── bayesian.py           # BetaBernoulliThompson
│   │   └── tests/                    # 78 tests, all passing
│   ├── aidi_instance/
│   │   ├── four_agent_pipeline.py    # legacy flat 4-agent pipeline
│   │   ├── supervised_pipeline.py    # Phase-2: Supervisor + composite
│   │   └── feature_aware_pipeline.py # Phase-3: feature-aware variant
│   ├── experiments/
│   │   ├── rq3_e2e_vs_counterfactual.py   # legacy flat experiment
│   │   ├── rq3_supervised.py              # Phase-2: all 4 credit methods
│   │   ├── rq3_phase3.py                  # Phase-3: full wiring
│   │   ├── configs/smoke.yaml
│   │   ├── configs/supervised_smoke.yaml
│   │   ├── configs/phase3_smoke.yaml
│   │   └── results/<exp_name>/       # summary.json, episodes.sqlite
│   └── requirements.txt              # pytest, PyYAML
├── docs/
│   ├── assignment.md
│   ├── Proposal.pdf
│   ├── framework_design.md           # this file
│   ├── information_injection.md      # Phase-3 design note (RQ: info flow)
│   └── survey.md                     # literature review
├── papers/Papers.csv
└── .venv/                            # local Python 3.13 env
```

## 5. How to run

```bash
# One-time setup
python3.13 -m venv .venv
.venv/bin/pip install -r code/requirements.txt

# Unit tests
.venv/bin/python -m pytest code/framework/tests -q

# Legacy flat smoke experiment (4-agent pipeline, 2 credit methods)
.venv/bin/python code/experiments/rq3_e2e_vs_counterfactual.py \
    --config code/experiments/configs/smoke.yaml
# -> code/experiments/results/rq3_smoke/{summary.csv,summary.json}

# Phase-2 supervised-refinement smoke experiment (all 4 credit methods)
.venv/bin/python code/experiments/rq3_supervised.py \
    --config code/experiments/configs/supervised_smoke.yaml
# -> code/experiments/results/rq3_supervised_smoke/{summary.csv,summary.json}

# Phase-3 full-wiring smoke (feature-aware tasks + learnable Supervisor
# + HierarchicalCredit + SQLite + statistical summary)
.venv/bin/python code/experiments/rq3_phase3.py \
    --config code/experiments/configs/phase3_smoke.yaml
# -> code/experiments/results/rq3_phase3_smoke/{summary.json,episodes.sqlite}
```

## 6. Non-goals (for this iteration)

- Cascading-confidence and Shapley credit (interface-ready, stubbed).
- RQ1 baseline comparison, RQ2 contextual bandits, RQ4 sample-efficiency
  analysis. Phase-3 lays the data groundwork for RQ2 (feature-conditioned
  tasks) but the contextual learner itself is deferred — see
  [`docs/information_injection.md`](information_injection.md) §3.
- Agent-internal errors / exception handling (explicit future work per
  assignment.md).
- Real Azure OpenAI / Databricks integration.
- Persistence to Delta Lake (Phase-3 uses local SQLite only).
- FastAPI wrapper / React frontend.
