"""Utility functions for moment matching."""

from typing import TypedDict

import numpy as np
import xarray as xr

from .wrapper.pymc_wrapper import PyMCWrapper

__all__ = [
    "_compute_log_likelihood",
    "_compute_log_prob",
    "_initialize_array",
    "ShiftResult",
    "ShiftAndScaleResult",
    "ShiftAndCovResult",
    "UpdateQuantitiesResult",
]


class SplitMomentMatchResult(TypedDict):
    """Result of split moment matching."""

    lwi: np.ndarray
    lwfi: np.ndarray
    log_liki: np.ndarray
    r_eff_i: float


class UpdateQuantitiesResult(TypedDict):
    """Result of updating quantities for observation i."""

    lwi: np.ndarray
    lwfi: np.ndarray
    ki: float
    kfi: int
    log_liki: np.ndarray


class ShiftResult(TypedDict):
    """Result of shift transformation."""

    upars: np.ndarray
    shift: np.ndarray


class ShiftAndScaleResult(TypedDict):
    """Result of shift and scale transformation."""

    upars: np.ndarray
    shift: np.ndarray
    scaling: np.ndarray


class ShiftAndCovResult(TypedDict):
    """Result of shift and covariance transformation."""

    upars: np.ndarray
    shift: np.ndarray
    mapping: np.ndarray


def _compute_log_likelihood(wrapper: PyMCWrapper, i: int) -> np.ndarray:
    """Compute log likelihood for observation i.

    Parameters
    ----------
    wrapper : PyMCWrapper
        PyMC model wrapper instance
    i : int
        Index of the observation

    Returns
    -------
    np.ndarray
        Log likelihood values for observation i across all samples
    """
    log_liki = wrapper.log_likelihood_i(i, wrapper.idata)
    log_liki = log_liki.stack(__sample__=("chain", "draw"))

    return log_liki.values.flatten()


def _compute_log_prob(wrapper: PyMCWrapper, upars: np.ndarray) -> np.ndarray:
    """Compute log probability for parameters.

    This function converts unconstrained parameters to constrained space and
    computes the log probability for each parameter set.

    Parameters
    ----------
    wrapper : PyMCWrapper
        PyMC model wrapper instance that provides access to model variables and transformations
    upars : np.ndarray
        Unconstrained parameters matrix with shape (n_samples, n_parameters)

    Returns
    -------
    np.ndarray
        Log probability for each parameter set, shape (n_samples,)
    """
    constrained_params = {}
    param_names = list(wrapper.get_unconstrained_parameters().keys())
    for j, name in enumerate(param_names):
        constrained_params[name] = xr.DataArray(
            upars[:, j],
            dims=["sample"],
            coords={"sample": np.arange(len(upars))},
        )
    constrained = wrapper.constrain_parameters(constrained_params)

    log_prob = np.zeros(len(upars))
    for name, param in constrained.items():
        var = wrapper.get_variable(name)
        if var is not None and hasattr(var, "logp"):
            if isinstance(param, xr.DataArray):
                param = param.values
            log_prob_part = var.logp(param).eval()
            log_prob += log_prob_part

    return log_prob


def _initialize_array(arr, default_func, dim):
    """Initialize array with default values if shape doesn't match."""
    return arr if arr.shape[0] == dim else default_func(dim)
