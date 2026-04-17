"""Downstream learners that consume credit signals."""

from framework.learner.base import Learner
from framework.learner.bayesian import BetaBernoulliThompson

__all__ = ["BetaBernoulliThompson", "Learner"]
