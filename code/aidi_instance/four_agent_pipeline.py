"""AIDI-shaped 4-agent pipeline used by the RQ3 experiments.

Pipeline order (per [docs/AIDI_Internship_Proposal 1.pdf] and user
clarification):

    DocumentAnalysisAgent  -> extracts security events
    LogParserAgent         -> parses/classifies log samples
    DocumentOnboardingSupervisor -> reconciles / orchestrates
    CodeAgent              -> emits regex code for Cribl pipelines

Each agent is modelled as a :class:`SyntheticAgent` with a small discrete
action space standing in for "which strategy the agent chose on this step"
(e.g., which RAG chunking for DocumentAnalysis, which Drain3 depth for
LogParser). The CodeAgent has an elevated ``silent_failure_rate`` — its
regex can parse but be semantically wrong, which is the scenario that
motivates counterfactual credit over confidence-based credit.
"""

from __future__ import annotations

from framework.agent import SyntheticAgent
from framework.pipeline import Pipeline


def build_aidi_pipeline(
    *,
    upstream_failure_penalty: float = 0.4,
    code_silent_failure_rate: float = 0.3,
) -> Pipeline:
    """Construct the 4-agent AIDI pipeline with reasonable defaults.

    Args:
        upstream_failure_penalty: Multiplicative penalty applied to each
            agent's success probability when any upstream agent failed. Set
            to 0 to make agents independent; AIDI's real cascading failure
            mode motivates values in (0, 1).
        code_silent_failure_rate: Probability the CodeAgent reports high
            confidence on a failed step — the "parses but wrong regex"
            scenario.
    """
    doc_analysis = SyntheticAgent(
        name="DocumentAnalysisAgent",
        actions=["rag_chunk_small", "rag_chunk_large", "rag_hybrid"],
        p_success={
            "rag_chunk_small": 0.72,
            "rag_chunk_large": 0.60,
            "rag_hybrid": 0.80,
        },
        upstream_failure_penalty=0.0,  # first agent: no upstream to fail
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
    supervisor = SyntheticAgent(
        name="DocumentOnboardingSupervisor",
        actions=["reclarify_aggressive", "reclarify_conservative", "trust_upstream"],
        p_success={
            "reclarify_aggressive": 0.70,
            "reclarify_conservative": 0.82,
            "trust_upstream": 0.60,
        },
        upstream_failure_penalty=upstream_failure_penalty,
        confidence_noise=0.06,
    )
    code_agent = SyntheticAgent(
        name="CodeAgent",
        actions=["template_regex", "llm_regex", "hybrid_regex"],
        p_success={
            "template_regex": 0.68,
            "llm_regex": 0.74,
            "hybrid_regex": 0.82,
        },
        upstream_failure_penalty=upstream_failure_penalty,
        confidence_noise=0.08,
        silent_failure_rate=code_silent_failure_rate,
    )
    return Pipeline(agents=[doc_analysis, log_parser, supervisor, code_agent])


# Convenience: the best-case action for each agent (used by smoke tests as
# the "oracle" reference for regret computation).
BEST_ACTIONS: dict[str, str] = {
    "DocumentAnalysisAgent": "rag_hybrid",
    "LogParserAgent": "drain3_deep",
    "DocumentOnboardingSupervisor": "reclarify_conservative",
    "CodeAgent": "hybrid_regex",
}
