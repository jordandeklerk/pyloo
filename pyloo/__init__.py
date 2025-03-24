"""Python implementation of the R package 'loo' for LOO-CV and WAIC."""

import logging

if not logging.root.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


from .base import ISMethod, compute_importance_weights
from .elpd import ELPDData
from .loo import loo
from .loo_approximate_posterior import loo_approximate_posterior
from .loo_kfold import _kfold_split_random, _kfold_split_stratified, kfold
from .loo_moment_match import loo_moment_match, loo_moment_match_split
from .loo_predictive_metric import loo_predictive_metric
from .loo_score import loo_score
from .plots import plot_loo
from .psis import psislw
from .reloo import reloo
from .sis import sislw
from .tis import tislw
from .waic import waic
from .wrapper.pymc import PyMCWrapper

__all__ = [
    "compute_importance_weights",
    "ELPDData",
    "ISMethod",
    "kfold",
    "_kfold_split_random",
    "_kfold_split_stratified",
    "loo",
    "loo_approximate_posterior",
    "loo_moment_match",
    "loo_moment_match_split",
    "loo_predictive_metric",
    "loo_score",
    "PyMCWrapper",
    "psislw",
    "reloo",
    "sislw",
    "tislw",
    "waic",
    "plot_loo",
]
