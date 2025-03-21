"""Standard Importance Sampling (SIS) implementation."""

from copy import deepcopy

import numpy as np
import xarray as xr

from .utils import _logsumexp, wrap_xarray_ufunc


def sislw(log_weights):
    """Standard importance sampling (SIS).

    Parameters
    ----------
    log_weights : DataArray or (..., N) array-like
        Array of size (n_observations, n_samples)

    Returns
    -------
    lw_out : DataArray or (..., N) ndarray
        Normalized log weights
    ess : DataArray or (...) ndarray
        Effective sample sizes

    Notes
    -----
    If the ``log_weights`` input is an :class:`~xarray.DataArray` with a dimension
    named ``__sample__`` (recommended) ``sislw`` will interpret this dimension as samples,
    and all other dimensions as dimensions of the observed data, looping over them to
    calculate the sislw of each observation. If no ``__sample__`` dimension is present or
    the input is a numpy array, the last dimension will be interpreted as ``__sample__``.

    See Also
    --------
    psis : Pareto Smoothed Importance Sampling
    tis : Truncated Importance Sampling

    Examples
    --------
    Get Standard importance sampling (SIS) log weights:

    .. code-block:: python

        import pyloo as pl

        data = az.load_arviz_data("non_centered_eight")

        log_likelihood = data.log_likelihood["obs"].stack(
            __sample__=["chain", "draw"]
        )
        pl.sislw(-log_likelihood)
    """
    log_weights = deepcopy(log_weights)
    if hasattr(log_weights, "__sample__"):
        shape = [
            size
            for size, dim in zip(log_weights.shape, log_weights.dims)
            if dim != "__sample__"
        ]
    else:
        shape = log_weights.shape[:-1]

    out = np.empty_like(log_weights), np.empty(shape)

    func_kwargs = {"out": out}
    ufunc_kwargs = {"n_dims": 1, "n_output": 2, "ravel": False, "check_shape": False}
    kwargs = {
        "input_core_dims": [["__sample__"]],
        "output_core_dims": [["__sample__"], []],
    }
    log_weights, ess = wrap_xarray_ufunc(
        _sislw,
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


def _sislw(log_weights):
    """Standard importance sampling (SIS) for a 1D vector.

    Parameters
    ----------
    log_weights: array
        Array of length n_observations

    Returns
    -------
    lw_out: array
        Normalized log weights
    ess: float
        Effective sample size
    """
    x = np.asarray(log_weights)
    x -= np.max(x)
    x -= _logsumexp(x)
    weights = np.exp(x)
    ess = 1 / np.sum(weights**2)
    return x, ess
