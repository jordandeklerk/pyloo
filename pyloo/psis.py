"""Functions for Pareto smoothed importance sampling (PSIS) with adaptations from ArviZ."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import xarray as xr

from .utils import _logsumexp, wrap_xarray_ufunc


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


def psislw(log_weights, reff=1.0):
    """
    Pareto smoothed importance sampling (PSIS).

    Notes
    -----
    If the ``log_weights`` input is an :class:`~xarray.DataArray` with a dimension
    named ``__sample__`` (recommended) ``psislw`` will interpret this dimension as samples,
    and all other dimensions as dimensions of the observed data, looping over them to
    calculate the psislw of each observation. If no ``__sample__`` dimension is present or
    the input is a numpy array, the last dimension will be interpreted as ``__sample__``.

    Parameters
    ----------
    log_weights : DataArray or (..., N) array-like
        Array of size (n_observations, n_samples)
    reff : float, default 1
        relative MCMC efficiency, ``ess / n``

    Returns
    -------
    lw_out : DataArray or (..., N) ndarray
        Smoothed, truncated and normalized log weights.
    kss : DataArray or (...) ndarray
        Estimates of the shape parameter *k* of the generalized Pareto
        distribution.

    References
    ----------
    * Vehtari et al. (2024). Pareto smoothed importance sampling. Journal of Machine
      Learning Research, 25(72):1-58.

    See Also
    --------
    loo : Compute Pareto-smoothed importance sampling leave-one-out cross-validation (PSIS-LOO-CV).
    sis : Standard Importance Sampling
    tis : Truncated Importance Sampling

    Examples
    --------
    Get Pareto smoothed importance sampling (PSIS) log weights:

    .. ipython::

        In [1]: import pyloo as loo
           ...: data = az.load_arviz_data("non_centered_eight")
           ...: log_likelihood = data.log_likelihood["obs"].stack(
           ...:     __sample__=["chain", "draw"]
           ...: )
           ...: loo.psislw(-log_likelihood, reff=0.8)

    """
    log_weights = deepcopy(log_weights)
    if hasattr(log_weights, "__sample__"):
        n_samples = len(log_weights.__sample__)
        shape = [size for size, dim in zip(log_weights.shape, log_weights.dims) if dim != "__sample__"]
    else:
        n_samples = log_weights.shape[-1]
        shape = log_weights.shape[:-1]
    cutoff_ind = -int(np.ceil(min(n_samples / 5.0, 3 * (n_samples / reff) ** 0.5))) - 1
    cutoffmin = np.log(np.finfo(float).tiny)

    out = np.empty_like(log_weights), np.empty(shape)

    func_kwargs = {"cutoff_ind": cutoff_ind, "cutoffmin": cutoffmin, "out": out}
    ufunc_kwargs = {"n_dims": 1, "n_output": 2, "ravel": False, "check_shape": False}
    kwargs = {"input_core_dims": [["__sample__"]], "output_core_dims": [["__sample__"], []]}
    log_weights, pareto_shape = wrap_xarray_ufunc(
        _psislw,
        log_weights,
        ufunc_kwargs=ufunc_kwargs,
        func_kwargs=func_kwargs,
        **kwargs,
    )
    if isinstance(log_weights, xr.DataArray):
        log_weights = log_weights.rename("log_weights")
    if isinstance(pareto_shape, xr.DataArray):
        pareto_shape = pareto_shape.rename("pareto_shape")
    return log_weights, pareto_shape


def _psislw(log_weights, cutoff_ind, cutoffmin):
    """
    Pareto smoothed importance sampling (PSIS) for a 1D vector.

    Parameters
    ----------
    log_weights: array
        Array of length n_observations
    cutoff_ind: int
    cutoffmin: float
    k_min: float

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
            x[tailinds[x_tail_si]] = smoothed_tail
            x[x > 0] = 0
    x -= _logsumexp(x)

    return x, k


def _gpdfit(ary):
    """Estimate the parameters for the Generalized Pareto Distribution (GPD).

    Empirical Bayes estimate for the parameters of the generalized Pareto
    distribution given the data.

    Parameters
    ----------
    ary: array
        sorted 1D data array

    Returns
    -------
    k: float
        estimated shape parameter
    sigma: float
        estimated scale parameter
    """
    prior_bs = 3
    prior_k = 10
    n = len(ary)
    m_est = 30 + int(n**0.5)

    b_ary = 1 - np.sqrt(m_est / (np.arange(1, m_est + 1, dtype=float) - 0.5))
    b_ary /= prior_bs * ary[int(n / 4 + 0.5) - 1]
    b_ary += 1 / ary[-1]

    k_ary = np.log1p(-b_ary[:, None] * ary).mean(axis=1)
    len_scale = n * (np.log(-(b_ary / k_ary)) - k_ary - 1)
    weights = 1 / np.exp(len_scale - len_scale[:, None]).sum(axis=1)

    real_idxs = weights >= 10 * np.finfo(float).eps
    if not np.all(real_idxs):
        weights = weights[real_idxs]
        b_ary = b_ary[real_idxs]
    weights /= weights.sum()

    # posterior mean for b
    b_post = np.sum(b_ary * weights)
    # estimate for k
    k_post = np.log1p(-b_post * ary).mean()
    # add prior for k_post
    sigma = -k_post / b_post
    k_post = (n * k_post + prior_k * 0.5) / (n + prior_k)

    return k_post, sigma


def _gpinv(probs, kappa, sigma):
    """Inverse Generalized Pareto distribution function."""
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
