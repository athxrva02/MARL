"""Phase-3 AIDI instance: feature-aware agents over the faithful architecture.

Structurally identical to :mod:`supervised_pipeline` but with every leaf
agent built as :class:`FeatureAwareAgent`. The feature-effect tables below
encode domain intuitions that make the optimal action genuinely
task-dependent — the precondition for any future contextual-bandit study
(RQ2).

Examples encoded below:

  * Large documents penalise ``rag_chunk_large`` (too much context, worse
    retrieval) and favour ``rag_hybrid``.
  * High log volumes favour ``drain3_deep`` (more templates to cluster).
  * PII-bearing tasks punish ``llm_only`` (leaks risk + lower accuracy),
    and make ``static_plus_api`` validation more valuable than cheaper
    ``static_only``.

The feature-blind Thompson-sampling baseline learns an average-over-
distribution optimum for each agent. A future contextual learner would
be strictly better iff these per-feature optima differ — they do, by
construction.

This module intentionally exposes only :func:`build_feature_aware_pipeline`
(the top-level :class:`Pipeline`). The Phase-3 runner builds the
:class:`RefinementLoop` itself, per-episode, from the Supervisor preset
that the learner picks.
"""

from __future__ import annotations

from framework.agent_features import FeatureAwareAgent, FeatureEffect
from framework.composite_agent import CompositeAgent
from framework.pipeline import Pipeline


def _build_code_subpipeline(
    *,
    upstream_failure_penalty: float,
    silent_failure_rate: float,
) -> Pipeline:
    schema = FeatureAwareAgent(
        name="SchemaAgent",
        actions=["docs_lookup", "best_practice"],
        base_p_success={"docs_lookup": 0.78, "best_practice": 0.70},
        effects={
            # PII tasks benefit from consulting the real docs (which carry
            # the PII-handling rules) rather than best-practice heuristics.
            "docs_lookup": [FeatureEffect(when={"has_pii": True}, delta=+0.05)],
            "best_practice": [FeatureEffect(when={"has_pii": True}, delta=-0.10)],
        },
        confidence_noise=0.05,
    )
    builder = FeatureAwareAgent(
        name="BuilderAgent",
        actions=["template", "llm", "hybrid"],
        base_p_success={"template": 0.65, "llm": 0.72, "hybrid": 0.80},
        effects={
            # Pure-LLM builds leak PII into prompts more easily; penalised.
            "llm": [FeatureEffect(when={"has_pii": True}, delta=-0.12)],
            "hybrid": [FeatureEffect(when={"has_pii": True}, delta=-0.04)],
            # Templates stay constant; no effect table.
        },
        upstream_failure_penalty=upstream_failure_penalty,
        confidence_noise=0.07,
        silent_failure_rate=silent_failure_rate,
    )
    validator = FeatureAwareAgent(
        name="ValidatorAgent",
        actions=["static_only", "static_plus_api"],
        base_p_success={"static_only": 0.70, "static_plus_api": 0.85},
        effects={
            # PII sharpens the value of the API-backed validator (which
            # actually checks redaction) over the cheap static linter.
            "static_only": [FeatureEffect(when={"has_pii": True}, delta=-0.10)],
            "static_plus_api": [FeatureEffect(when={"has_pii": True}, delta=+0.05)],
        },
        upstream_failure_penalty=upstream_failure_penalty,
        confidence_noise=0.05,
    )
    qa = FeatureAwareAgent(
        name="QAAgent",
        actions=["strict", "lenient"],
        base_p_success={"strict": 0.78, "lenient": 0.85},
        effects={
            # Strict QA catches more on large payloads where lenient lets
            # silent failures slip.
            "strict": [FeatureEffect(when={"doc_size": "large"}, delta=+0.05)],
            "lenient": [FeatureEffect(when={"doc_size": "large"}, delta=-0.08)],
        },
        upstream_failure_penalty=upstream_failure_penalty,
        confidence_noise=0.06,
    )
    return Pipeline([schema, builder, validator, qa])


def build_feature_aware_pipeline(
    *,
    upstream_failure_penalty: float = 0.4,
    code_silent_failure_rate: float = 0.25,
) -> Pipeline:
    """Top-level feature-aware pipeline (Doc → LogParser → CodeAgent-composite)."""
    doc = FeatureAwareAgent(
        name="DocumentAnalysisAgent",
        actions=["rag_chunk_small", "rag_chunk_large", "rag_hybrid"],
        base_p_success={
            "rag_chunk_small": 0.72,
            "rag_chunk_large": 0.60,
            "rag_hybrid": 0.80,
        },
        effects={
            # Large docs break rag_chunk_large (too much stuffed context);
            # hybrid scales best.
            "rag_chunk_large": [FeatureEffect(when={"doc_size": "large"}, delta=-0.20)],
            "rag_chunk_small": [FeatureEffect(when={"doc_size": "large"}, delta=-0.08)],
            "rag_hybrid": [FeatureEffect(when={"doc_size": "large"}, delta=+0.05)],
        },
        confidence_noise=0.05,
    )
    log_parser = FeatureAwareAgent(
        name="LogParserAgent",
        actions=["drain3_shallow", "drain3_deep", "llm_only"],
        base_p_success={
            "drain3_shallow": 0.65,
            "drain3_deep": 0.78,
            "llm_only": 0.55,
        },
        effects={
            # High-volume logs reward deeper clustering; llm_only chokes.
            "drain3_deep": [FeatureEffect(when={"log_volume": "high"}, delta=+0.08)],
            "drain3_shallow": [
                FeatureEffect(when={"log_volume": "high"}, delta=-0.15),
                # At low volumes the shallow cluster is fine and cheaper.
                FeatureEffect(when={"log_volume": "low"}, delta=+0.05),
            ],
            "llm_only": [FeatureEffect(when={"log_volume": "high"}, delta=-0.20)],
        },
        upstream_failure_penalty=upstream_failure_penalty,
        confidence_noise=0.07,
    )
    code_sub = _build_code_subpipeline(
        upstream_failure_penalty=upstream_failure_penalty,
        silent_failure_rate=code_silent_failure_rate,
    )
    code_agent = CompositeAgent(
        name="CodeAgent",
        sub_pipeline=code_sub,
        action_id_strategy="ensemble",
    )
    return Pipeline([doc, log_parser, code_agent])
