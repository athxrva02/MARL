"""Credit-assignment strategies.

Each strategy implements :class:`CreditAssigner` and maps a (trace, outcome,
pipeline) triple to a scalar credit per agent. Credit is always in [0, 1] so
that it can be consumed directly by the Beta-Bernoulli learner as a
pseudo-Bernoulli success signal.
"""

from framework.credit.base import CreditAssigner
from framework.credit.cascading_confidence import CascadingConfidence
from framework.credit.counterfactual import CounterfactualCredit
from framework.credit.end_to_end import EndToEndCredit
from framework.credit.hierarchical import HierarchicalCredit
from framework.credit.iteration_discounted import IterationDiscountedCredit
from framework.credit.validator_anchored import ValidatorAnchoredCredit

__all__ = [
    "CascadingConfidence",
    "CounterfactualCredit",
    "CreditAssigner",
    "EndToEndCredit",
    "HierarchicalCredit",
    "IterationDiscountedCredit",
    "ValidatorAnchoredCredit",
]
