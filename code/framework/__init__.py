"""Generic N-agent sequential pipeline framework for credit-assignment research.

This package provides the building blocks for the AIDI credit-assignment study
described in docs/assignment.md. The framework is deliberately minimal: it
models a sequential pipeline of black-box agents, supports pluggable credit
assigners, and couples them to per-agent Bayesian learners.

See docs/framework_design.md for the architectural rationale.
"""

from framework.agent import Agent, AgentOutput, Observation, SyntheticAgent
from framework.environment import Outcome, Task
from framework.pipeline import Pipeline, Trace, TraceStep

__all__ = [
    "Agent",
    "AgentOutput",
    "Observation",
    "Outcome",
    "Pipeline",
    "SyntheticAgent",
    "Task",
    "Trace",
    "TraceStep",
]
