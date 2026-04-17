# Survey: Credit Assignment for Sequential LLM Pipelines (AIDI / RQ3)

> Draft literature review and critical assessment for the AIDI internship
> ([proposal](AIDI_Internship_Proposal%201.pdf), [assignment](assignment.md)).
> Target length 8–12 pages when expanded; this file is the research-ready
> outline plus a critical taxonomy of the papers in
> [`papers/Papers.csv`](../papers/Papers.csv).

---

## 1. Problem statement

AIDI is a sequential, role-specialised LLM pipeline:

```
  DocumentAnalysisAgent → LogParserAgent → DocumentOnboardingSupervisor → CodeAgent
           (RAG)             (Drain3+LLM)        (orchestration)          (regex for Cribl)
```

When an extraction fails, **which agent should the learner blame, and by
how much?** The internship proposal lists three candidates (end-to-end,
cascading confidence, counterfactual) and asks us to determine which
enables effective learning.

This is a **credit-assignment** problem — distinct from (though related to)
the *failure-attribution* problem of identifying the responsible agent
post-hoc for debugging. Credit assignment is a *training signal* problem:
we need a per-agent scalar that, when fed to a per-agent learner,
converges the pipeline toward better policies faster than a naive global
signal would.

### 1.1 What makes the AIDI setting unusual

Compared to canonical cooperative multi-agent RL (MARL):

| Property                       | Canonical MARL        | AIDI                                |
|--------------------------------|-----------------------|-------------------------------------|
| Agent structure                | Differentiable policies | Black-box LLM calls              |
| Training signal                | Gradient-based        | Bayesian posterior updates          |
| Agent interaction              | Simultaneous           | Strictly sequential (fixed DAG)     |
| Per-step observability         | Centralized critic    | Per-agent confidence only           |
| Rollout cost                   | Cheap (env simulation) | Expensive (LLM API calls)          |
| Number of agents               | 2–50+                 | 4 (currently)                       |
| Failure mode of interest       | Policy suboptimality  | Cascading data corruption + silent semantic errors |

Four implications follow: (i) gradient-based credit methods (COMA, Shapley
Q-learning) do not transfer algorithmically; (ii) exact Shapley over
coalitions is computationally **trivial** at N=4 (2⁴=16) — worth
benchmarking; (iii) rollout-based counterfactuals must be budgeted
carefully for any real-LLM deployment; (iv) silent failures (CodeAgent
emits parse-valid but semantically wrong regex) break confidence-product
methods, since the offending agent self-reports high confidence.

## 2. Taxonomy of the reviewed literature

The papers in [`papers/Papers.csv`](../papers/Papers.csv) partition cleanly
into three schools plus philosophical framing. The table below summarises
each and its applicability to AIDI RQ3.

### 2.1 School A: Cooperative MARL credit assignment

| Work | Core idea | AIDI applicability |
|---|---|---|
| **Foerster et al., COMA (2017/arXiv 1705.08926)** | Centralised critic + counterfactual baseline: marginalise out one agent's action while keeping others fixed; policy gradient against that baseline. | Conceptual inspiration for leave-one-out credit. Algorithm requires differentiable softmax policies and a shared Q-function — AIDI's agents are LLM calls with no gradient and no shared critic. **Transfer: concept ✓, algorithm ✗.** |
| **Wang et al., Shapley Q-value / SQDDPG (AAAI 2020, arXiv 1907.05707)** | Cast global-reward MARL as an *extended convex game*; distribute per-agent credit via Shapley values; derive actor-critic (SQDDPG). | Principled axiomatic guarantees (efficiency, symmetry, dummy-agent detection). Again assumes differentiable actors/critics. At N=4, *exact* Shapley over 16 coalitions is tractable as a baseline on the simulation framework. |
| **Wang, Shapley-value MARL thesis (2022)** | Extends Shapley Q to partial observability (SHAQ, SMFPPO) with an energy-network application. | Same algorithmic mismatch as SQDDPG. Use as theoretical grounding for the axioms in the survey. |
| **Chhablani (2023)** | Shows COMA updates ≈ regret minimisation; proposes Neural Replicator Dynamics variant on StarCraft II. | Narrow applicability — interesting for a PG learner on differentiable policies; not relevant for the AIDI Bayesian-bandit setting. |

### 2.2 School B: Sequential blame attribution

| Work | Core idea | AIDI applicability |
|---|---|---|
| **Triantafyllou, Singla, Radanovic (NeurIPS 2021)** | Formal definitions of blame in accountable *sequential* multi-agent decision making. | Setting matches AIDI's sequential pipeline. Formalisations (actual cause, but-for cause) are adoptable directly. **Primary theoretical anchor** for the survey's §3. |
| **Natan, Stern, Kalech (2023, MAPF blame)** | Shapley approximation via coalition subsampling for multi-agent path-finding execution failures. | Demonstrates that Shapley approximation becomes necessary only at large N. For AIDI (N=4) we can afford exact enumeration; this paper guides scaling to future larger pipelines. |

### 2.3 School C: LLM-pipeline traceability and root-cause analysis

| Work | Core idea | AIDI applicability |
|---|---|---|
| **Barrak, Traceability & Accountability in Planner→Executor→Critic pipelines (2025, arXiv 2510.07614)** | Evaluates 3-stage LLM pipelines; introduces structured handoffs and per-role repair/harm metrics. | Closest structural analogue in the corpus. AIDI adds a 4th stage (CodeAgent) downstream; the methodology (per-role accuracy, harm rate) transfers directly and should be a reporting template for our experiments. |
| **Wang, AgentTrace (2026, arXiv 2603.14688)** | Post-hoc causal-graph tracing of deployed multi-agent logs for RCA without LLM inference at debug time. | Relevant for *debugging* AIDI in production, complementary to the *training-time* credit problem. ⚠ **Authenticity check required** before citing heavily — arXiv ID prefix is unusually recent; I have not been able to independently verify the paper. |
| **Zhang et al., Who & When (2025, arXiv 2505.00212)** | Benchmark + dataset (127 multi-agent failure logs) for automated failure attribution. Best method: 53.5% agent-level / **14.2% step-level** accuracy; o1/R1 fail to deliver practical usability. | **Anchors the motivation:** a general-purpose LLM-driven failure attributor is not yet solved. AIDI should exploit pipeline-specific structure (known topology, per-agent confidences, ground-truth feedback) rather than chase a general solver. Our simulation framework's oracle `success` flag provides the ground truth that Who&When had to annotate manually. |
| **Ikram et al., Microservices RCA via causal discovery** | Hierarchical, localised causal graph for root-cause detection in microservice failures. | Structurally analogous: microservice pipelines are DAGs of stochastic components with cascading failure. The hierarchical/localised search is a template we could use if AIDI grows in complexity. |

### 2.4 Framing / ethics

| Work | Relevance |
|---|---|
| **Hallamaa & Kalliokoski (2022), Placing Blame in MAS** | Distinguishes blame, responsibility, and accountability. Useful vocabulary for §1 of the survey. Non-technical. |

### 2.5 Housekeeping notes on the CSV

- **Triantafyllou (NeurIPS 2021) is listed twice** (`TE7HIHU5`, `3783VF27`); merge in the Zotero library.
- **AgentTrace (arXiv 2603.14688)** — verify before final submission. If the paper cannot be retrieved, drop the citation and rely on Ikram et al. for the RCA framing.

## 3. Critical assessment: what fits AIDI, what doesn't

1. **Gradient-based MARL methods (COMA, SQDDPG, SHAQ) are structurally incompatible with AIDI's Bayesian-bandit learner and black-box LLM agents.** Their theoretical contributions (counterfactual marginalisation; Shapley axioms) transfer at the *concept* level but not as drop-in algorithms. Over-citing them without this caveat would be misleading.
2. **Triantafyllou (2021) and Barrak (2025) are the right primary references.** They match the *sequential, discrete-action, per-step accountability* setting AIDI operates in. The survey should lead with these.
3. **Exact Shapley is cheap here.** Papers treating Shapley as exotic (Natan 2023's subset-sampling) over-index on large-N settings. At N=4, 16 coalitions is trivial — we should include it as a baseline in Phase 2 even though it was dropped from Phase 1.
4. **The big gap: no paper in this corpus empirically compares end-to-end vs counterfactual credit in an LLM-pipeline simulation with Bayesian updates.** That is precisely the measurement the simulation framework makes.
5. **Who&When's 14.2% step-level accuracy is sobering.** It is the strongest evidence that *general-purpose* automated attribution is hard. AIDI can succeed where Who&When fails because AIDI knows (a) the pipeline topology, (b) per-agent confidences, and (c) ground-truth feedback from SOC analysts. The survey should frame AIDI's advantage as "exploit structure the benchmark denied the attributor."

## 4. Hypotheses entering the experiments

H1. **End-to-end credit suffers as pipeline depth grows** because a single failure taints the signal for every agent; counterfactual credit should stay informative.

H2. **Counterfactual credit as currently implemented (leave-one-out with fresh downstream sampling) will have low variance and therefore under-update a conjugate Bayesian learner.** Some form of baseline subtraction or variance stretching will be required.

H3. **Silent failures (CodeAgent emits parse-valid but semantically wrong regex) will destroy cascading-confidence credit**, since the offending agent self-reports high confidence. Counterfactual and Shapley should be robust.

H4. **Exact Shapley at N=4 will dominate leave-one-out for cleanliness of the per-agent attribution**, at a cost of 4× more rollouts. If rollouts are cheap (simulation) or batchable (LLM), Shapley is preferable.

## 5. Empirical findings to date

From `code/experiments/results/rq3_smoke/`:

| method         | late mean reward | blame-attr. acc (planted blame) |
|----------------|------------------|----------------------------------|
| end_to_end     | 0.186 ± 0.03     | 0.259 ± 0.024                    |
| counterfactual | 0.154 ± 0.03     | 0.280 ± 0.026                    |

And a 1500-episode no-blame run (5 seeds): **end-to-end 0.416 vs counterfactual 0.239 late mean reward.**

**Preliminary conclusion:** on this pipeline and with a Beta-Bernoulli learner, **leave-one-out counterfactual credit as naively formulated is *worse* than end-to-end baseline**. The counterfactual signal clusters around 0.5 and the Bayesian posterior barely moves per episode. This confirms H2 and motivates the remedies in [framework_design.md §3.5](framework_design.md).

This is an honest negative result worth reporting in the final deliverable — it challenges the assumption (implicit in the internship proposal) that fancier credit methods will out-learn the baseline by default.

## 6. Proposed next experiments

| Exp | Method variant | Expected signal |
|-----|----------------|-----------------|
| E1  | Leave-one-out with **baseline subtraction and variance rescaling** | Tighter credit → faster posterior convergence |
| E2  | Leave-one-out driven by the **learner's own action distribution** (not the agent's default) | Counterfactual aligned with what the learner would actually do |
| E3  | **Difference rewards** (Wolpert/Tumer): replace agent with a fixed default rather than resampling | Lower variance at the cost of distributional mismatch |
| E4  | **Exact Shapley** over the 16 coalitions | Axiomatic baseline; more expensive but principled |
| E5  | Cascading-confidence with the CodeAgent's silent-failure mode on vs. off | Quantify H3 |
| E6  | Sweep pipeline depth N ∈ {3, 4, 6, 8} × methods | Test H1 |

Each is one additional `CreditAssigner` class + one YAML; no framework
refactor needed.

## 7. Recommendations for the internship's implementation phase

1. **Do not default to activating counterfactual credit in production.** The naive version is worse than end-to-end in the simulation; further tuning is required before betting LLM spend on it.
2. **Ship end-to-end + Beta-Bernoulli Thompson first.** It already out-learns in the current experiments and matches AIDI's dormant `BayesianAttributeLearner` cleanly.
3. **Reserve counterfactual for offline analysis / debugging first.** Use it to diagnose which agent is most limiting on a held-out set, before wiring it into the online training loop.
4. **Log per-agent confidences persistently** even while cascading-confidence is not the training signal. They become a cheap feature for later learners (RQ2 contextual bandits) and for post-hoc attribution.
5. **Treat Who&When-style step-level attribution as a stretch goal.** Aim first for agent-level credit that learns policies; step-level blame is a harder problem that may not move the business metric.

## 8. Open questions for supervisors (Akash / Kwidama / Tom Viering)

- How much human-feedback volume can we realistically expect per month? (Determines whether RQ4 sample-efficiency bounds are a blocker.)
- Are per-agent confidences already logged in Databricks? If not, that is a cheap prerequisite for most of §6.
- Is there appetite for an offline Shapley-based RCA tool as an interim debugging aid, independent of the online learning loop?

---

*Appendix materials to add in the expanded-page version:* figure of the
pipeline, full experiment CSV plots (learning curves per method, per-agent
posterior evolution), complete bibliography formatted as BibTeX, and a
short methods section on the reproducibility protocol (seeds, Python
version, hardware).
