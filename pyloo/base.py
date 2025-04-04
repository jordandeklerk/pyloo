"""Unified importance sampling module supporting multiple methods."""

from copy import deepcopy
from enum import Enum
from typing import Callable, cast

import numpy as np
import xarray as xr

from .psis import _psislw
from .sis import _sislw
from .tis import _tislw
from .utils import wrap_xarray_ufunc

__all__ = ["ISMethod", "compute_importance_weights"]


class ISMethod(str, Enum):
    """Enumeration of supported importance sampling methods."""

    PSIS = "psis"
    SIS = "sis"
    TIS = "tis"


ImplFunc = Callable[..., tuple[np.ndarray, float | np.ndarray]]


def compute_importance_weights(
    log_weights: xr.DataArray | np.ndarray | None = None,
    method: ISMethod | str = ISMethod.PSIS,
    reff: float = 1.0,
) -> tuple[xr.DataArray | np.ndarray, xr.DataArray | np.ndarray]:
    """Unified importance sampling computation that supports multiple methods.

    Parameters
    ----------
    log_weights : DataArray or (..., N) array-like
        Array of size (n_observations, n_samples)
    method : {'psis', 'sis', 'tis'}, default 'psis'
        The importance sampling method to use:
        - 'psis': Pareto Smoothed Importance Sampling
        - 'sis': Standard Importance Sampling
        - 'tis': Truncated Importance Sampling
    reff : float, default 1.0
        Relative MCMC efficiency (only used for PSIS method)

    Returns
    -------
    lw_out : DataArray or (..., N) ndarray
        Processed log weights (smoothed/truncated/normalized depending on method)
    diagnostic : DataArray or (...) ndarray
        Method-specific diagnostic value:
        - PSIS: Pareto shape parameter (k)
        - SIS/TIS: Effective sample size (ESS)

    Notes
    -----
    If the ``log_weights`` input is an :class:`~xarray.DataArray` with a dimension
    named ``__sample__`` (recommended) this function will interpret this dimension as samples,
    and all other dimensions as dimensions of the observed data, looping over them to
    calculate the importance weights for each observation. If no ``__sample__`` dimension is
    present or the input is a numpy array, the last dimension will be interpreted as ``__sample__``.

    See Also
    --------
    psislw : Pareto Smoothed Importance Sampling (original implementation)
    sislw : Standard Importance Sampling (original implementation)
    tislw : Truncated Importance Sampling (original implementation)

    Examples
    --------
    Get importance sampling weights using different methods

    .. code-block:: python

        import pyloo as pl

        data = az.load_arviz_data("non_centered_eight")
        log_likelihood = data.log_likelihood["obs"].stack(
            __sample__=["chain", "draw"]
        )

        # Using PSIS (default)
        lw_psis, k = pl.compute_importance_weights(-log_likelihood)

        # Using SIS
        lw_sis, ess = pl.compute_importance_weights(-log_likelihood, method="sis")

        # Using TIS
        lw_tis, ess = pl.compute_importance_weights(-log_likelihood, method="tis")
    """
    if isinstance(log_weights, xr.DataArray):
        if "__sample__" not in log_weights.dims:
            if "chain" in log_weights.dims and "draw" in log_weights.dims:
                log_weights = log_weights.stack(__sample__=("chain", "draw"))
            else:
                raise ValueError("log_weights must have a __sample__ dimension")

    if isinstance(method, str):
        try:
            method = ISMethod(method.lower())
        except ValueError:
            valid_methods = ", ".join(m.value for m in ISMethod)
            raise ValueError(
                f"Invalid method '{method}'. Must be one of: {valid_methods}"
            )

    if log_weights is None:
        raise ValueError("log_weights must be provided when variational=False")

    log_weights = deepcopy(log_weights)

    if hasattr(log_weights, "__sample__"):
        n_samples = len(log_weights.__sample__)
        shape = [
            size
            for size, dim in zip(log_weights.shape, log_weights.dims)
            if dim != "__sample__"
        ]
    else:
        n_samples = log_weights.shape[-1]
        shape = log_weights.shape[:-1]

    out = np.empty_like(log_weights), np.empty(shape)

    ufunc_kwargs: dict[str, bool | int] = {
        "n_dims": 1,
        "n_output": 2,
        "ravel": False,
        "check_shape": False,
    }
    kwargs: dict[str, list[list[str]]] = {
        "input_core_dims": [["__sample__"]],
        "output_core_dims": [["__sample__"], []],
    }

    if method == ISMethod.PSIS:
        cutoff_ind = (
            -int(np.ceil(min(n_samples / 5.0, 3 * (n_samples / reff) ** 0.5))) - 1
        )
        cutoffmin = np.log(np.finfo(float).tiny)
        func_kwargs = {"cutoff_ind": cutoff_ind, "cutoffmin": cutoffmin, "out": out}
        impl_func = cast(ImplFunc, _psislw)

    elif method == ISMethod.SIS:
        func_kwargs = {"out": out}
        impl_func = cast(ImplFunc, _sislw)

    elif method == ISMethod.TIS:
        func_kwargs = {"n_samples": n_samples, "out": out}
        impl_func = cast(ImplFunc, _tislw)

    else:
        raise ValueError(
            f"Method {method} is not supported for standard importance sampling. "
            "Use 'psis', 'sis', or 'tis' instead."
        )

    log_weights, diagnostic = wrap_xarray_ufunc(
        impl_func,
        log_weights,
        ufunc_kwargs=ufunc_kwargs,
        func_kwargs=func_kwargs,
        **kwargs,
    )

    if isinstance(log_weights, xr.DataArray):
        log_weights = log_weights.rename("log_weights")
    if isinstance(diagnostic, xr.DataArray):
        diagnostic = diagnostic.rename(
            "pareto_shape" if method == ISMethod.PSIS else "ess"
        )

    return log_weights, diagnostic
