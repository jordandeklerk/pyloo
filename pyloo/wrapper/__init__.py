"""Wrappers for different model backends to support LOO-CV computations."""

from .laplace import Laplace
from .pymc import PyMCWrapper

__all__ = ["PyMCWrapper", "Laplace"]
