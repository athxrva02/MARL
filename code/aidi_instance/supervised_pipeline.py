"""Faithful AIDI simulation: 3 top-level agents + CodeAgent sub-pipeline +
supervisor-driven refinement loop.

Why this file exists separately from :mod:`four_agent_pipeline`:

After reviewing AIDI's real architecture, the flat 4-agent model was
revealed to be structurally wrong:

  * The Supervisor is a :class:`GraphOrchestrator`, not a peer agent in the
    flow. It owns the graph and the :class:`DecisionEngine`.
  * The pipeline is **iterative** — Doc → Log → Code → Decision can loop
    back to Doc with a clarification.
  * The CodeAgent is itself a sub-pipeline (Schema → Builder → Validator →
    QA) with its own retry loop and bandit selector.

This module models all three of those properties.

Pipeline structure::

    Supervisor (refinement loop, max 3 iterations)
      └── DocumentAnalysisAgent   (flat, 3 actions: RAG strategy)
          LogParserAgent          (flat, 3 actions: Drain3 depth)
          CodeAgent               (COMPOSITE of:)
              ├── SchemaAgent     (flat, 2 actions: docs lookup / best-practice)
              ├── BuilderAgent    (flat, 3 actions: template / LLM / hybrid)
              ├── ValidatorAgent  (flat, 2 actions: static / static+API)
              └── QAAgent         (flat, 2 actions: strict / lenient)
"""

from __future__ import annotations

from framework.agent import SyntheticAgent
from framework.composite_agent import CompositeAgent
from framework.pipeline import Pipeline
from framework.refinement import CompletenessDecisionPolicy, RefinementLoop


def build_code_subpipeline(
    *,
    upstream_failure_penalty: float = 0.3,
    silent_failure_rate: float = 0.25,
) -> Pipeline:
    """The CodeAgent's internal Schema→Builder→Validator→QA pipeline."""
    schema = SyntheticAgent(
        name="SchemaAgent",
        actions=["docs_lookup", "best_practice"],
        p_success={"docs_lookup": 0.78, "best_practice": 0.70},
        confidence_noise=0.05,
    )
    builder = SyntheticAgent(
        name="BuilderAgent",
        actions=["template", "llm", "hybrid"],
        p_success={"template": 0.65, "llm": 0.72, "hybrid": 0.80},
        upstream_failure_penalty=upstream_failure_penalty,
        confidence_noise=0.07,
        # Silent failure is the signature CodeAgent issue: a regex that
        # parses but is semantically wrong. Lives in the Builder, caught
        # (sometimes) by the Validator.
        silent_failure_rate=silent_failure_rate,
    )
    validator = SyntheticAgent(
        name="ValidatorAgent",
        actions=["static_only", "static_plus_api"],
        p_success={"static_only": 0.70, "static_plus_api": 0.85},
        upstream_failure_penalty=upstream_failure_penalty,
        confidence_noise=0.05,
    )
    qa = SyntheticAgent(
        name="QAAgent",
        actions=["strict", "lenient"],
        p_success={"strict": 0.78, "lenient": 0.85},
        upstream_failure_penalty=upstream_failure_penalty,
        confidence_noise=0.06,
    )
    return Pipeline([schema, builder, validator, qa])


def build_supervisor_pipeline(
    *,
    upstream_failure_penalty: float = 0.4,
    code_silent_failure_rate: float = 0.25,
) -> Pipeline:
    """The top-level 3-agent pipeline the Supervisor orchestrates."""
    doc = SyntheticAgent(
        name="DocumentAnalysisAgent",
        actions=["rag_chunk_small", "rag_chunk_large", "rag_hybrid"],
        p_success={
            "rag_chunk_small": 0.72,
            "rag_chunk_large": 0.60,
            "rag_hybrid": 0.80,
        },
        confidence_noise=0.05,
    )
    log_parser = SyntheticAgent(
        name="LogParserAgent",
        actions=["drain3_shallow", "drain3_deep", "llm_only"],
        p_success={
            "drain3_shallow": 0.65,
            "drain3_deep": 0.78,
            "llm_only": 0.55,
        },
        upstream_failure_penalty=upstream_failure_penalty,
        confidence_noise=0.07,
    )
    code_sub = build_code_subpipeline(
        upstream_failure_penalty=upstream_failure_penalty,
        silent_failure_rate=code_silent_failure_rate,
    )
    code_agent = CompositeAgent(
        name="CodeAgent",
        sub_pipeline=code_sub,
        action_id_strategy="ensemble",
    )
    return Pipeline([doc, log_parser, code_agent])


def build_aidi_supervisor(
    *,
    upstream_failure_penalty: float = 0.4,
    code_silent_failure_rate: float = 0.25,
    max_iterations: int = 3,
) -> RefinementLoop:
    """Build the full Supervisor-orchestrated refinement loop."""
    pipeline = build_supervisor_pipeline(
        upstream_failure_penalty=upstream_failure_penalty,
        code_silent_failure_rate=code_silent_failure_rate,
    )
    policy = CompletenessDecisionPolicy(
        complete_threshold=1.0,
        max_iterations=max_iterations,
        stagnation_window=2,
    )
    return RefinementLoop(pipeline=pipeline, decision_policy=policy)


# Best-action reference (used for oracle regret computation in experiments).
BEST_ACTIONS: dict[str, str] = {
    "DocumentAnalysisAgent": "rag_hybrid",
    "LogParserAgent": "drain3_deep",
    "CodeAgent": "ensemble",  # composite: best via sub-agent selection
    # Inside CodeAgent:
    "SchemaAgent": "docs_lookup",
    "BuilderAgent": "hybrid",
    "ValidatorAgent": "static_plus_api",
    "QAAgent": "lenient",
}
