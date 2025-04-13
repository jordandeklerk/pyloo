"""Wrappers for different model backends to support LOO-CV computations."""

from .pymc.laplace import Laplace
from .pymc.pymc import PyMCWrapper

__all__ = ["PyMCWrapper", "Laplace"]
