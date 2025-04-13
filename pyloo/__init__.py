"""Modules for LOO-CV."""

import logging

if not logging.root.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

from .base import ISMethod, compute_importance_weights
from .compare import loo_compare
from .e_loo import e_loo
from .elpd import ELPDData
from .helpers import (
    ParameterConverter,
    ShiftAndCovResult,
    ShiftAndScaleResult,
    ShiftResult,
    UpdateQuantitiesResult,
    compute_updated_r_eff,
    extract_log_likelihood_for_observation,
    log_lik_i_upars,
    log_prob_upars,
)
from .loo import loo
from .loo_approximate_posterior import loo_approximate_posterior
from .loo_group import loo_group
from .loo_kfold import (
    _kfold_split_grouped,
    _kfold_split_random,
    _kfold_split_stratified,
    loo_kfold,
)
from .loo_moment_match import loo_moment_match, loo_moment_match_split
from .loo_nonfactor import loo_nonfactor
from .loo_predictive_metric import loo_predictive_metric
from .loo_score import loo_score
from .loo_subsample import loo_subsample
from .plots import influence_plot, loo_difference_plot, loo_plot
from .psis import psislw
from .reloo import reloo
from .sis import sislw
from .tis import tislw
from .waic import waic
from .wrapper.pymc.laplace import Laplace
from .wrapper.pymc.pymc import PyMCWrapper

__all__ = [
    "compute_importance_weights",
    "loo_compare",
    "e_loo",
    "ELPDData",
    "ISMethod",
    "loo_kfold",
    "_kfold_split_random",
    "_kfold_split_stratified",
    "_kfold_split_grouped",
    "loo",
    "loo_subsample",
    "loo_approximate_posterior",
    "loo_group",
    "loo_moment_match",
    "loo_moment_match_split",
    "loo_nonfactor",
    "loo_predictive_metric",
    "loo_score",
    "PyMCWrapper",
    "Laplace",
    "psislw",
    "reloo",
    "sislw",
    "tislw",
    "waic",
    "loo_plot",
    "influence_plot",
    "loo_difference_plot",
    "ParameterConverter",
    "ShiftAndCovResult",
    "ShiftAndScaleResult",
    "ShiftResult",
    "UpdateQuantitiesResult",
    "log_lik_i_upars",
    "log_prob_upars",
    "compute_updated_r_eff",
    "extract_log_likelihood_for_observation",
]
