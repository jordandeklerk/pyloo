"""Truncated Importance Sampling (TIS) implementation."""

from copy import deepcopy

import numpy as np
import xarray as xr

from .utils import _logsumexp, wrap_xarray_ufunc


def tislw(log_weights):
    """
    Truncated importance sampling (TIS).

    Notes
    -----
    If the ``log_weights`` input is an :class:`~xarray.DataArray` with a dimension
    named ``__sample__`` (recommended) ``tislw`` will interpret this dimension as samples,
    and all other dimensions as dimensions of the observed data, looping over them to
    calculate the tislw of each observation. If no ``__sample__`` dimension is present or
    the input is a numpy array, the last dimension will be interpreted as ``__sample__``.

    Parameters
    ----------
    log_weights : DataArray or (..., N) array-like
        Array of size (n_observations, n_samples)

    Returns
    -------
    lw_out : DataArray or (..., N) ndarray
        Truncated and normalized log weights
    ess : DataArray or (...) ndarray
        Effective sample sizes

    References
    ----------
    .. [1] Ionides, Edward L. (2008). Truncated importance sampling.
           Journal of Computational and Graphical Statistics 17(2): 295--311.

    See Also
    --------
    psis : Pareto Smoothed Importance Sampling
    sis : Standard Importance Sampling

    Examples
    --------
    Get Truncated importance sampling (TIS) log weights:

    .. ipython::

        In [1]: import pyloo as pl
           ...: data = az.load_arviz_data("non_centered_eight")
           ...: log_likelihood = data.log_likelihood["obs"].stack(
           ...:     __sample__=["chain", "draw"]
           ...: )
           ...: pl.tislw(-log_likelihood)
    """
    log_weights = deepcopy(log_weights)
    if hasattr(log_weights, "__sample__"):
        n_samples = len(log_weights.__sample__)
        shape = [size for size, dim in zip(log_weights.shape, log_weights.dims) if dim != "__sample__"]
    else:
        n_samples = log_weights.shape[-1]
        shape = log_weights.shape[:-1]

    out = np.empty_like(log_weights), np.empty(shape)

    func_kwargs = {"n_samples": n_samples, "out": out}
    ufunc_kwargs = {"n_dims": 1, "n_output": 2, "ravel": False, "check_shape": False}
    kwargs = {"input_core_dims": [["__sample__"]], "output_core_dims": [["__sample__"], []]}
    log_weights, ess = wrap_xarray_ufunc(
        _tislw,
        log_weights,
        ufunc_kwargs=ufunc_kwargs,
        func_kwargs=func_kwargs,
        **kwargs,
    )
    if isinstance(log_weights, xr.DataArray):
        log_weights = log_weights.rename("log_weights")
    if isinstance(ess, xr.DataArray):
        ess = ess.rename("ess")
    return log_weights, ess


def _tislw(log_weights, n_samples):
    """
    Truncated importance sampling (TIS) for a 1D vector.

    Parameters
    ----------
    log_weights: array
        Array of length n_observations
    n_samples: int
        Number of samples

    Returns
    -------
    lw_out: array
        Truncated and normalized log weights
    ess: float
        Effective sample size
    """
    x = np.asarray(log_weights)
    x -= np.max(x)

    # Compute normalization term (c-hat in Ionides 2008 appendix)
    log_Z = _logsumexp(x) - np.log(n_samples)

    log_cutpoint = log_Z + 0.5 * np.log(n_samples)
    x = np.minimum(x, log_cutpoint)
    x -= _logsumexp(x)

    weights = np.exp(x)
    ess = 1 / np.sum(weights**2)
    return x, ess
