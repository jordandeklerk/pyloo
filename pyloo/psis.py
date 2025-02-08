"""Functions for Pareto smoothed importance sampling (PSIS)."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np

from .utils import _logsumexp


@dataclass
class PSISData:
    """Object containing smoothed log weights and diagnostics from PSIS calculations.

    Attributes
    ----------
    log_weights : np.ndarray
        Smoothed log weights
    pareto_k : np.ndarray
        Estimated shape parameters for the generalized Pareto distribution
    ess : Optional[np.ndarray]
        Effective sample size estimate (if computed)
    r_eff : Optional[np.ndarray]
        Relative efficiency estimate (if provided)
    tail_len : Optional[Union[int, np.ndarray]]
        Length of the tail for Pareto smoothing. Can be a scalar or array.
    mcse_elpd_loo : Optional[np.ndarray]
        Monte Carlo standard error estimates for PSIS-LOO
    influence_pareto_k : Optional[np.ndarray]
        Pareto k influence values
    """

    log_weights: np.ndarray
    pareto_k: np.ndarray
    ess: Optional[np.ndarray] = None
    r_eff: Optional[np.ndarray] = None
    tail_len: Optional[Union[int, np.ndarray]] = None
    mcse_elpd_loo: Optional[np.ndarray] = None
    influence_pareto_k: Optional[np.ndarray] = None


def psislw(
    log_weights: np.ndarray,
    reff: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pareto smoothed importance sampling (PSIS).

    Parameters
    ----------
    log_weights : np.ndarray
        Array of size (n_observations, n_samples)
    reff : float, default 1
        relative MCMC efficiency, ``ess / n``

    Returns
    -------
    lw_out : np.ndarray
        Smoothed, truncated and normalized log weights.
    kss : np.ndarray
        Estimates of the shape parameter *k* of the generalized Pareto
        distribution.

    References
    ----------
    * Vehtari et al. (2024). Pareto smoothed importance sampling. Journal of Machine
      Learning Research, 25(72):1-58.

    See Also
    --------
    loo : Compute Pareto-smoothed importance sampling leave-one-out cross-validation (PSIS-LOO-CV).
    """
    log_weights = deepcopy(log_weights)

    if log_weights.ndim == 1:
        log_weights = log_weights.reshape(-1, 1)
    elif log_weights.ndim != 2:
        raise ValueError("log_weights must be 1D or 2D array")

    n_samples = log_weights.shape[-1]
    shape = log_weights.shape[:-1]

    cutoff_ind = -int(np.ceil(min(n_samples / 5.0, 3 * (n_samples / reff) ** 0.5))) - 1
    cutoffmin = np.log(np.finfo(float).tiny)

    smoothed_log_weights = np.empty_like(log_weights)
    pareto_k = np.empty(shape)

    for idx in np.ndindex(shape):
        x = log_weights[idx]
        smoothed_log_weights[idx], pareto_k[idx] = _psislw(x, cutoff_ind, cutoffmin)

    if log_weights.shape[0] == 1:
        smoothed_log_weights = smoothed_log_weights.ravel()
        pareto_k = pareto_k.ravel()

    return smoothed_log_weights, pareto_k


def _psislw(log_weights: np.ndarray, cutoff_ind: int, cutoffmin: float) -> Tuple[np.ndarray, float]:
    """Pareto smoothed importance sampling (PSIS) for a 1D vector.

    Parameters
    ----------
    log_weights: array
        Array of length n_observations
    cutoff_ind: int
        Index for tail cutoff
    cutoffmin: float
        Minimum cutoff value

    Returns
    -------
    lw_out: array
        Smoothed log weights
    kss: float
        Pareto tail index
    """
    x = np.asarray(log_weights)
    x -= np.max(x)
    x_sort_ind = np.argsort(x)
    xcutoff = max(x[x_sort_ind[cutoff_ind]], cutoffmin)

    expxcutoff = np.exp(xcutoff)
    (tailinds,) = np.where(x > xcutoff)
    x_tail = x[tailinds]
    tail_len = len(x_tail)

    if tail_len <= 4:
        # not enough tail samples for gpdfit
        k = np.inf
    else:
        x_tail_si = np.argsort(x_tail)
        x_tail = np.exp(x_tail) - expxcutoff
        k, sigma = _gpdfit(x_tail[x_tail_si])

        if np.isfinite(k):
            # no smoothing if GPD fit failed
            # compute ordered statistic for the fit
            sti = np.arange(0.5, tail_len) / tail_len
            smoothed_tail = _gpinv(sti, k, sigma)
            smoothed_tail = np.log(smoothed_tail + expxcutoff)
            # place the smoothed tail into the output array
            x[tailinds[x_tail_si]] = smoothed_tail
            x[x > 0] = 0

    x -= _logsumexp(x)

    return x, k


def _gpdfit(x: np.ndarray) -> Tuple[float, float]:
    """Estimate generalized Pareto distribution parameters using method of moments."""
    prior_bs = 3
    prior_k = 10
    n = len(x)
    m_est = 30 + int(n**0.5)

    b_ary = 1 - np.sqrt(m_est / (np.arange(1, m_est + 1) - 0.5))
    b_ary /= prior_bs * x[int(n / 4 + 0.5) - 1]
    b_ary += 1 / x[-1]

    k_ary = np.mean(np.log1p(-b_ary[:, None] * x), axis=1)
    len_scale = n * (np.log(-b_ary / k_ary) - k_ary - 1)
    weights = 1 / np.sum(np.exp(len_scale - len_scale[:, None]), axis=1)

    real_idxs = weights >= 10 * np.finfo(float).eps
    if not np.all(real_idxs):
        weights = weights[real_idxs]
        b_ary = b_ary[real_idxs]

    weights = weights / np.sum(weights)
    b_post = np.sum(b_ary * weights)
    k_post = np.mean(np.log1p(-b_post * x))

    sigma = -k_post / b_post
    k_post = (n * k_post + prior_k * 0.5) / (n + prior_k)

    return k_post, sigma


def _gpinv(probs: np.ndarray, kappa: float, sigma: float) -> np.ndarray:
    """Compute inverse generalized Pareto distribution function."""
    x = np.full_like(probs, np.nan)
    if sigma <= 0:
        return x

    ok = (probs > 0) & (probs < 1)
    if np.all(ok):
        if np.abs(kappa) < np.finfo(float).eps:
            x = -np.log1p(-probs)
        else:
            x = np.expm1(-kappa * np.log1p(-probs)) / kappa
        x *= sigma
    else:
        if np.abs(kappa) < np.finfo(float).eps:
            x[ok] = -np.log1p(-probs[ok])
        else:
            x[ok] = np.expm1(-kappa * np.log1p(-probs[ok])) / kappa
        x *= sigma
        x[probs == 0] = 0
        x[probs == 1] = np.inf if kappa >= 0 else -sigma / kappa

    return x
