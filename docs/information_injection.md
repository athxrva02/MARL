# Design note: how information flows into and out of the RL loop

*Answering assignment.md's research question: "What is the best way to
inject information into the RL framework we will make?"*

## 0. Why this note exists

The testbed has three categories of signal that *could* be learned from:

1. **Per-episode reward** — did the pipeline, end-to-end, succeed?
2. **Per-step structure** — which agent failed, which succeeded, what did
   the validator say, did the refinement loop settle on the first pass?
3. **Task features** — is the document small or large? noisy logs? PII?

The question is not "is each one useful" (they all are) but "which of them
should be a *training signal* versus a *feature* versus merely *logged for
debugging*." Phase-3 locks in this split so later phases don't accidentally
double-count the same signal from two directions.

## 1. Four injection points

```
              ┌──────────────────────────────────────────────────────────┐
              │                     RL loop                              │
              │                                                          │
   features ──┼──► (A) learner CHOICE input ──► per-agent action         │
              │                                                          │
   env/task ──┼──► (B) pipeline payload ──► per-agent act(obs)           │
              │                                                          │
   outcome ───┼──► (C) credit assignment ──► per-agent credit            │
              │                                                          │
   posterior ─┼──► (D) learner update ──► new (α, β) per (agent, action) │
              └──────────────────────────────────────────────────────────┘
```

Each arrow is a place where information can be injected. The testbed
distinguishes them explicitly:

| Point | Mechanism | What flows in |
|-------|-----------|---------------|
| **A** Choice input | `Learner.choose(agent, actions, rng)` | Features would go here *if* we built a contextual learner; currently NOT wired (see §3). |
| **B** Payload | `Task.payload`, `Task.context`, and `context["action_for:..."]` | Task features + learner's chosen action per agent. |
| **C** Credit | `CreditAssigner.assign(trace, outcome, pipeline, rng)` | Trace structure, composite sub-traces, iteration history. |
| **D** Update | `Learner.update(agent, action, credit)` | One scalar in [0, 1] per (agent, action) per episode. |

## 2. The current recipe (Phase 3)

What we inject, at which point, and why each choice was made:

### 2.1 Point A (choice): nothing. Deliberately.

`BetaBernoulliThompson.choose` ignores task features. Its arms are per
(agent, action). This keeps the baseline *context-blind*: the learned
optimum is the average-over-features optimum, not the per-feature optimum.

**Why:** (a) it matches AIDI's dormant `BayesianAttributeLearner` exactly;
(b) it gives RQ2 a well-defined comparison point — a contextual learner
would be strictly better *iff* the per-feature optima differ, which is a
measurable claim.

### 2.2 Point B (payload): features + injected action.

Each `Task` carries `context["features"] = {doc_size, log_volume, has_pii}`.
`FeatureAwareAgent` reads those features and modulates its own
`p_success`. `context["action_for:<agent>"]` carries the learner's chosen
arm for that agent, which the agent must honour.

**Why features in context not payload:** `payload` is agent-to-agent data
flow (Doc → Log → Code). Features are global task metadata. Conflating
them breaks the pipeline-transport abstraction.

**Why actions in context not as an argument:** the `Pipeline` interface
is pure transport — threading an action-selection callback through it would
couple action selection to pipeline mechanics and make plain agents (no
learner attached) awkward to run. Convention-by-context keeps agents
simple and lets non-synthetic agents opt in.

### 2.3 Point C (credit): maximally rich, method-specific.

Credit assigners receive the full `Trace` (per-step observations, outputs,
confidences, success flags) *plus* any embedded `CompositeOutput.sub_trace`
*plus* — for episode-level assigners — the entire `EpisodeRun` with all
refinement iterations. Everything structural that could inform credit is
available here.

**Why so rich:** this is the research surface. If a new credit method
wants to use, say, a sub-pipeline's Validator verdict to override the
composite's outer reward (→ `ValidatorAnchoredCredit`), the information
must be reachable. Narrowing the interface would foreclose methods.

### 2.4 Point D (update): exactly one scalar in [0, 1].

`Learner.update(agent, action, credit)` consumes a single number. All
signal is compressed by the credit assigner into this one slot.

**Why so narrow:** Beta-Bernoulli conjugacy works because the update is a
single-parameter Bernoulli expectation. A multi-dimensional update would
break conjugacy and require approximate inference — premature
complication for the question we are asking. If a richer update proves
necessary (e.g. per-feature posteriors), it is a deliberate future
extension, not an accidental one.

## 3. What we are *not* injecting and why

* **Features into the learner (contextual bandit).** Explicitly deferred
  to RQ2. Phase-3's feature-conditioned tasks generate the *data* a
  contextual learner would need, but the learner itself is still
  context-blind. This preserves the Phase-3 baseline as a monotonic
  anchor.
* **Per-step confidences into the credit assigner.** They are *available*
  via `TraceStep.output.confidence` but no current assigner reads them —
  the stub `CascadingConfidence` does, and it raises `NotImplementedError`.
  This is intentional: confidences are miscalibrated in the simulation
  (silent-failure mode injects high confidence on failures) and will be in
  real AIDI too. Activating them before calibration is measured would
  confound future comparisons.
* **Agent-internal errors / exceptions.** Assignment explicitly says this
  is future work.

## 4. Logging vs. learning

We persist everything — task features, actions, per-agent credit, reward,
iteration count, posterior snapshots — to SQLite. This is *not* the same
as "injecting" it into the loop:

* Logged data is available for offline analysis, ad-hoc SQL, and learning-
  curve plots.
* It is *not* seen by the learner during training.

The distinction matters because it lets us retroactively re-assign credit
(using a new method) or re-score an old run without re-executing the
pipeline. Stored `Trace` rows + per-episode seeds are enough to
reconstruct everything.

## 5. The one-line answer

> Features and actions go through the pipeline's `Task.context`; trace
> structure flows into the credit assigner; exactly one scalar in [0, 1]
> flows from the credit assigner into the learner; everything else is
> logged for retrospection, not fed back online. Contextual features into
> the learner are deliberately deferred to RQ2 so that the Phase-3
> Thompson-sampling baseline remains a clean comparison point.

## 6. Verifiable consequences

If the note is correctly operationalised in code:

1. Adding a new `CreditAssigner` requires zero changes to `Pipeline`,
   `Learner`, or the experiment runner beyond a dispatch entry.
2. Adding a new task-feature dimension requires zero changes to any
   existing agent or credit method — only to the feature-effect tables of
   the agents that should react to it.
3. A contextual learner would be a drop-in for `BetaBernoulliThompson`
   provided it honours the same `choose/update` protocol, and would
   require no changes elsewhere.
4. The SQLite store can reconstruct any episode's credit and learner
   state from (experiment, method, seed, episode) without re-running
   the pipeline.

Tests in `code/framework/tests/` exercise (1)–(4).
