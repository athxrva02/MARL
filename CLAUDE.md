# CLAUDE.md — Claude Code Working Guidelines for MARL

This file documents conventions and token-efficiency tips for working with Claude on the AIDI credit-assignment research framework.

## 1. Key Project Constraints

- **No Claude co-authorship.** Never add `Co-Authored-By: Claude ...` trailers or "Generated with Claude Code" lines to commits or PRs. This is enforced in all git work.
- **Local-only files.** `docs/aidi-architecture.md` must never be committed; it is a local-only reference. It is already in `.git/info/exclude`.
- **Framework philosophy.** The framework is *research infrastructure*, not production code. Prioritise clarity and reproducibility over generalisation.

## 2. Token-Efficient Workflows

### 2.1 Use Memory System
- **Always check `/Users/atharva/.claude/projects/-Users-atharva-MARL/memory/MEMORY.md`** at session start to recall prior decisions and avoid re-asking clarifying questions.
- Save non-obvious findings (architectural choices, user preferences, gotchas) as memory files so future sessions stay aligned.
- Update memory proactively when you discover something worth preserving.

### 2.2 Focused File Reading
- Use `Read` with `limit` and `offset` to read only the relevant portion of large files (e.g. test files, experiment runners).
- Never read entire files when you only need a method signature or config schema — use `Grep` to find the section first.
- When exploring a new module, read the module docstring + one concrete example, not the entire API.

### 2.3 Minimal Diffs
- **Edit, don't rewrite.** Use the `Edit` tool for surgical changes; reserve `Write` for new files only.
- `replace_all: false` (default) keeps diffs smaller and easier to review.
- Batch related edits in a single tool call when they don't depend on each other's output.

### 2.4 Experiment Running
- **Smoke tests before full runs.** Always validate a new experiment script with `phase3_smoke.yaml` (tiny config: 6 episodes × 2 seeds) before claiming success.
- The smoke run takes ~5 seconds and catches 90% of bugs (import errors, schema mismatches, off-by-one iteration counts).
- Full runs (300+ episodes × 5+ seeds) should only be started after smoke passes and you're confident in the implementation.

### 2.5 Testing Strategy
- **Run the full suite once per session.** `pytest code/framework/tests/ -q` takes ~0.2s and catches regressions early.
- For targeted work (e.g. adding a new `CreditAssigner`), run only the relevant test file: `pytest code/framework/tests/test_credit.py -q`.
- Use `git diff` to check what you've changed *before* writing tests — this prevents writing tests for code that doesn't exist yet.

### 2.6 Use Agents Selectively
- **Don't spawn a general-purpose agent for simple tasks.** Use direct tool calls (Glob, Grep, Read) for file searches and small edits.
- **Spawn an Explore agent** only when you're hunting for a pattern across multiple files and don't know where to start (e.g. "find all places where we instantiate a learner").
- **Never spawn an agent to write code.** Write code directly; agents are for research and exploration.

## 3. Documentation & Research

### 3.1 When to Document
- Document *why*, not *what*. The code shows what; a comment should explain an unusual choice.
- Example: "We use stdlib Random, not numpy, so the dependency surface stays tiny" is worth a comment; "increment counter by 1" is not.
- Design decisions that affect multiple modules go in the relevant `*.py` docstring; architectural choices go in `docs/framework_design.md`.

### 3.2 Referencing Prior Work
- The `docs/survey.md` and `docs/information_injection.md` files are your canonical references.
- Don't re-read papers or re-derive logic — check the survey first. If the answer isn't there, read the paper *once* and update the survey so future sessions don't repeat the work.

## 4. Phase Boundaries & Scope

### Phase-1 (Done)
Flat 4-agent pipeline, end-to-end + counterfactual credit, no learner, no stats.

### Phase-2 (Done)
Faithful AIDI (Supervisor + composite CodeAgent), 4 credit methods, Beta-Bernoulli learner, plain CSV output.

### Phase-3 (Done)
Feature-aware agents, learnable Supervisor arm, HierarchicalCredit, SQLite persistence, statistical summaries.

### Phase-4+ (Deferred)
- **RQ2 contextual bandits:** Plug a feature-aware learner into the Phase-3 framework; compare against the feature-blind baseline.
- **Agent-internal error handling:** Explicit future work per assignment.md.
- **Shapley credit & cascading confidence:** Interface-ready stubs; implement only if measured to help.

## 5. Common Patterns

### New Experiment
1. Create a new `code/experiments/rq<N>_<name>.py` that imports phase components.
2. Add a YAML config in `code/experiments/configs/<name>_smoke.yaml` with tiny N.
3. Run smoke test: `python code/experiments/rq<N>_<name>.py --config code/experiments/configs/<name>_smoke.yaml`.
4. Add an integration test in `code/framework/tests/test_<name>_integration.py` that runs smoke via the module's `main()`.
5. Run `pytest code/framework/tests/test_<name>_integration.py -q` and full suite `pytest code/framework/tests/ -q`.
6. Expand YAML `n_episodes` and `n_seeds` for the full run.

### New Credit Method
1. Create `code/framework/credit/<method>.py` with a class `SomethingCredit(CreditAssigner)`.
2. Implement `assign(trace, outcome, pipeline, rng) -> dict[str, float]`.
3. Test in `code/framework/tests/test_credit.py` (or a new test file if complex).
4. Add to `code/framework/credit/__init__.py` and update the dispatcher in the experiment runner.
5. Add a line to the YAML config and re-run the smoke experiment.

### New Agent Type
1. Create `code/framework/<agent_type>.py` with a class implementing the `Agent` protocol (`name`, `act(obs, rng) -> AgentOutput`).
2. Test in `code/framework/tests/test_<agent_type>.py`.
3. Use it in a pipeline builder (e.g. `code/aidi_instance/<pipeline>.py`).
4. No changes needed elsewhere — the framework is transport-agnostic.

## 6. Token Budget Tips

- **Per-session:** Budget ~50k tokens for a full Phase task (one of the 8). Stay under by using memory and avoiding re-reads.
- **Per-query:** Keep questions specific. "How do I add a credit method?" is better than "Explain the whole framework to me."
- **Avoid:** Asking for explanations of code you've already read. Use memory or re-read the file yourself.
- **Reuse:** If a question is non-obvious, save the answer to memory so you don't ask again in a future session.

## 7. When to Ask for Help

- **Stuck on a test failure?** Run the test locally, capture the full error message, and paste it. Don't ask for blind debugging.
- **Architectural decision?** Read `docs/framework_design.md` and `docs/information_injection.md` first; they answer most design Qs.
- **Unsure if a change is right?** Use `git diff` to show me the delta, not a narrative description.

---

**Last updated:** Phase-3 completion (2026-04-20).
