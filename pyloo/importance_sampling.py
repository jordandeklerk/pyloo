"""Utilities for different importance sampling methods."""

from typing import Optional, Tuple, Union

import numpy as np

from .ess import psis_eff_size
from .psis import psislw
from .utils import _logsumexp

# TODO: Implement truncated importance sampling (TIS)
try:
    from .tis import tislw
except ImportError:
    tislw = None

from .sis import sislw


def ImportanceSampling(
    log_ratios: np.ndarray, r_eff: Union[float, np.ndarray], method: str = "psis"
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Perform importance sampling using the specified method.

    Parameters
    ----------
    log_ratios : np.ndarray
        Array of shape (n_samples, n_observations) containing log importance
        ratios (for example, log-likelihood values).
    r_eff : Union[float, np.ndarray]
        Relative MCMC efficiency (effective sample size / total samples).
        Can be a scalar or array of length n_observations.
    method : str, optional
        The importance sampling method to use. Options are:
        - "psis": Pareto Smoothed Importance Sampling (default)
        - "tis": Truncated Importance Sampling
        - "sis": Standard Importance Sampling
    cores : int, optional
        Number of cores to use for parallelization. Default is 1.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]
        - Array of smoothed log weights
        - Array of diagnostic values
        - Optional array of effective sample sizes

    Raises
    ------
    ValueError
        If the specified method is not implemented or if inputs are invalid.
    """
    implemented_methods = ["psis", "tis", "sis"]
    if method not in implemented_methods:
        raise ValueError(
            f"Importance sampling method '{method}' is not implemented. "
            f"Implemented methods: {', '.join(implemented_methods)}"
        )

    if not isinstance(log_ratios, np.ndarray):
        log_ratios = np.asarray(log_ratios)

    if log_ratios.ndim == 1:
        log_ratios = log_ratios.reshape(-1, 1)

    if log_ratios.ndim != 2:
        raise ValueError("log_ratios must be 1D or 2D array")

    _, n_obs = log_ratios.shape

    if isinstance(r_eff, (int, float)):
        r_eff = float(r_eff)
    elif len(r_eff) != n_obs:
        raise ValueError("r_eff must be a scalar or have length equal to n_observations")

    if method == "psis":
        weights, diagnostics, ess = psislw(log_ratios, r_eff)
    elif method == "tis":
        if tislw is None:
            raise NotImplementedError("Truncated Importance Sampling (TIS) is not yet implemented")
        weights, diagnostics, ess = tislw(log_ratios, r_eff)
    elif method == "sis":
        weights, diagnostics, ess = sislw(log_ratios, r_eff)
    else:
        raise ValueError(f"Method {method} not properly implemented")

    return weights, diagnostics, ess


def importance_sampling_object(
    log_weights: np.ndarray,
    pareto_k: np.ndarray,
    tail_len: Optional[Union[int, np.ndarray]],
    r_eff: Union[float, np.ndarray],
    method: str,
) -> dict:
    """Create an importance sampling results object.

    Parameters
    ----------
    log_weights : np.ndarray
        Array of unnormalized log weights
    pareto_k : np.ndarray
        Array of diagnostic values (e.g., Pareto k values for PSIS)
    tail_len : Optional[Union[int, np.ndarray]]
        Length of tail used for fitting (if applicable)
    r_eff : Union[float, np.ndarray]
        Relative MCMC efficiency values
    method : str
        Name of importance sampling method used

    Returns
    -------
    dict
        Dictionary containing importance sampling results and diagnostics
    """
    if not isinstance(log_weights, np.ndarray):
        raise TypeError("log_weights must be a numpy array")

    norm_const_log = _logsumexp(log_weights, axis=0)

    weights = np.exp(log_weights - norm_const_log)
    if isinstance(r_eff, (int, float)):
        ess = psis_eff_size(weights, r_eff)
    else:
        ess = np.array([psis_eff_size(weights[:, i], r_eff[i]) for i in range(weights.shape[1])])

    return {
        "log_weights": log_weights,
        "diagnostics": {
            "pareto_k": pareto_k,
            "n_eff": ess,
            "r_eff": r_eff,
        },
        "norm_const_log": norm_const_log,
        "tail_len": tail_len,
        "dims": log_weights.shape,
        "method": method,
    }


def _get_weights(
    obj: dict,
    log: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """Extract importance sampling weights from results object."""
    weights = obj["log_weights"].copy()
    if normalize:
        weights = weights - obj["norm_const_log"]
    if not log:
        weights = np.exp(weights)
    return weights
