"""Concrete AIDI-shaped pipeline instances for experiments.

Two instances are provided:

* :func:`build_aidi_pipeline` — the flat 4-agent pipeline (historical Phase-1
  model; kept for backwards-compatible experiments).
* :func:`build_aidi_supervisor` — the faithful AIDI architecture:
  Supervisor-orchestrated refinement loop over Doc → Log → CodeAgent
  (composite).
"""

from aidi_instance.feature_aware_pipeline import build_feature_aware_pipeline
from aidi_instance.four_agent_pipeline import build_aidi_pipeline
from aidi_instance.supervised_pipeline import (
    BEST_ACTIONS,
    build_aidi_supervisor,
    build_code_subpipeline,
    build_supervisor_pipeline,
)

__all__ = [
    "BEST_ACTIONS",
    "build_aidi_pipeline",
    "build_aidi_supervisor",
    "build_code_subpipeline",
    "build_feature_aware_pipeline",
    "build_supervisor_pipeline",
]
