# Framework Design — Credit-Assignment Research Testbed for AIDI

Status: Phase-2 scaffold. The framework now reflects the faithful AIDI
architecture: Supervisor as orchestrator (`RefinementLoop`), 3 top-level agents with the
CodeAgent realized as a nested `CompositeAgent` sub-pipeline
(Schema → Builder → Validator → QA), and four credit methods: end-to-end
(baseline), counterfactual (leave-one-out), validator-anchored
(structured-signal hierarchical), and iteration-discounted (temporal
weighting across refinement iterations). 40/40 unit tests pass; two smoke
experiments (flat 4-agent and supervised refinement) run in seconds on
commodity hardware.

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

## 4. Directory layout

```
MARL/
├── code/
│   ├── framework/
│   │   ├── agent.py                  # Agent protocol + SyntheticAgent
│   │   ├── composite_agent.py        # CompositeAgent (nested pipeline)
│   │   ├── refinement.py             # RefinementLoop, EpisodeRun, DecisionPolicy
│   │   ├── environment.py            # Task, Outcome, default_outcome
│   │   ├── pipeline.py               # Pipeline, Trace, TraceStep
│   │   ├── experiment.py             # run_episode, EpisodeRecord, RunResult
│   │   ├── credit/
│   │   │   ├── base.py               # CreditAssigner protocol
│   │   │   ├── end_to_end.py         # EndToEndCredit (baseline)
│   │   │   ├── counterfactual.py     # CounterfactualCredit (leave-one-out)
│   │   │   ├── validator_anchored.py # Hierarchical (uses Validator verdict)
│   │   │   ├── iteration_discounted.py # Episode-level, γ^k weighting
│   │   │   └── cascading_confidence.py # stub; interface compat
│   │   ├── learner/
│   │   │   ├── base.py               # Learner protocol
│   │   │   └── bayesian.py           # BetaBernoulliThompson
│   │   └── tests/                    # 40 tests, all passing
│   ├── aidi_instance/
│   │   ├── four_agent_pipeline.py    # legacy flat 4-agent pipeline
│   │   └── supervised_pipeline.py    # faithful AIDI: Supervisor + composite
│   ├── experiments/
│   │   ├── rq3_e2e_vs_counterfactual.py   # legacy flat experiment
│   │   ├── rq3_supervised.py              # revised: all 4 credit methods
│   │   ├── configs/smoke.yaml
│   │   ├── configs/supervised_smoke.yaml
│   │   └── results/<exp_name>/       # summary.csv, summary.json
│   └── requirements.txt              # pytest, PyYAML
├── docs/
│   ├── assignment.md
│   ├── Proposal.pdf
│   ├── framework_design.md           # this file
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

# Revised supervised-refinement smoke experiment (all 4 credit methods)
.venv/bin/python code/experiments/rq3_supervised.py \
    --config code/experiments/configs/supervised_smoke.yaml
# -> code/experiments/results/rq3_supervised_smoke/{summary.csv,summary.json}
```

## 6. Non-goals (for this iteration)

- Cascading-confidence and Shapley credit (interface-ready, stubbed).
- RQ1 baseline comparison, RQ2 contextual bandits, RQ4 sample-efficiency
  analysis. The framework is ready for all of them.
- Real Azure OpenAI / Databricks integration.
- Persistence to Delta Lake.
- FastAPI wrapper / React frontend.
