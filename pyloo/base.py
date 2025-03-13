"""Unified importance sampling module supporting multiple methods."""

from copy import deepcopy
from enum import Enum
from typing import Callable, Literal, cast

import numpy as np
import xarray as xr

from .psis import _psislw, vi_psis_sampling
from .sis import _sislw
from .tis import _tislw
from .utils import wrap_xarray_ufunc

__all__ = ["ISMethod", "compute_importance_weights"]


class ISMethod(str, Enum):
    """Enumeration of supported importance sampling methods."""

    PSIS = "psis"
    SIS = "sis"
    TIS = "tis"
    PSIR = "psir"
    IDENTITY = "identity"


ImplFunc = Callable[..., tuple[np.ndarray, float | np.ndarray]]


def compute_importance_weights(
    log_weights: xr.DataArray | np.ndarray | None = None,
    method: ISMethod | str = ISMethod.PSIS,
    reff: float = 1.0,
    variational: bool = False,
    samples: np.ndarray | None = None,
    logP: np.ndarray | None = None,
    logQ: np.ndarray | None = None,
    num_draws: int | None = None,
    random_seed: int | None = None,
) -> tuple[xr.DataArray | np.ndarray, xr.DataArray | np.ndarray]:
    """
    Unified importance sampling computation that supports multiple methods.

    Parameters
    ----------
    log_weights : DataArray or (..., N) array-like
        Array of size (n_observations, n_samples)
    method : {'psis', 'sis', 'tis', 'psir', 'identity'}, default 'psis'
        The importance sampling method to use:
        - 'psis': Pareto Smoothed Importance Sampling
        - 'sis': Standard Importance Sampling
        - 'tis': Truncated Importance Sampling
        - 'psir': Pareto Smoothed Importance Resampling (for variational inference)
        - 'identity': Apply log importance weights directly (for variational inference)
    reff : float, default 1.0
        Relative MCMC efficiency (only used for PSIS method)
    variational : bool, default False
        Whether to use variational inference specific importance sampling.
        If True, samples, logP, logQ, and num_draws must be provided.
    samples : array-like, optional
        Samples from proposal distribution, shape (L, M, N) where:
        - L is the number of chains/paths
        - M is the number of draws per chain
        - N is the number of parameters
        Required when variational=True.
    logP : array-like, optional
        Log probability values of target distribution, shape (L, M)
        Required when variational=True.
    logQ : array-like, optional
        Log probability values of proposal distribution, shape (L, M)
        Required when variational=True.
    num_draws : int, optional
        Number of draws to return where num_draws <= samples.shape[0] * samples.shape[1]
        Required when variational=True.
    random_seed : int, optional
        Random seed for reproducibility in variational inference sampling.

    Returns
    -------
    lw_out : DataArray or (..., N) ndarray
        Processed log weights (smoothed/truncated/normalized depending on method)
    diagnostic : DataArray or (...) ndarray
        Method-specific diagnostic value:
        - PSIS/PSIR: Pareto shape parameter (k)
        - SIS/TIS: Effective sample size (ESS)
        - IDENTITY: None

    Notes
    -----
    If the ``log_weights`` input is an :class:`~xarray.DataArray` with a dimension
    named ``__sample__`` (recommended) this function will interpret this dimension as samples,
    and all other dimensions as dimensions of the observed data, looping over them to
    calculate the importance weights for each observation. If no ``__sample__`` dimension is
    present or the input is a numpy array, the last dimension will be interpreted as ``__sample__``.

    For variational inference models, use the variational=True parameter and provide the
    required samples, logP, logQ, and num_draws parameters. This will use the vi_psis_sampling
    function which is specifically designed for variational inference.

    See Also
    --------
    psislw : Pareto Smoothed Importance Sampling (original implementation)
    sislw : Standard Importance Sampling (original implementation)
    tislw : Truncated Importance Sampling (original implementation)
    vi_psis_sampling : Importance sampling for variational inference

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

    For variational inference

    .. code-block:: python

        import pyloo as pl
        import numpy as np

        # Simulate samples from a variational approximation
        samples = np.random.normal(size=(4, 1000, 10))  # 4 paths, 1000 draws, 10 parameters
        logP = np.random.normal(size=(4, 1000))  # Log prob from target distribution
        logQ = np.random.normal(size=(4, 1000))  # Log prob from proposal distribution

        result = pl.compute_importance_weights(
            method="psis",
            variational=True,
            samples=samples,
            logP=logP,
            logQ=logQ,
            num_draws=1000
        )
    """
    if isinstance(method, str):
        try:
            method = ISMethod(method.lower())
        except ValueError:
            valid_methods = ", ".join(m.value for m in ISMethod)
            raise ValueError(
                f"Invalid method '{method}'. Must be one of: {valid_methods}"
            )

    if variational:
        if samples is None or logP is None or logQ is None or num_draws is None:
            raise ValueError(
                "When variational=True, samples, logP, logQ, and num_draws must be"
                " provided"
            )

        if method == ISMethod.PSIS:
            vi_method = "psis"
        elif method == ISMethod.PSIR:
            vi_method = "psir"
        elif method == ISMethod.IDENTITY:
            vi_method = "identity"
        elif method == ISMethod.SIS or method == ISMethod.TIS:
            raise ValueError(
                f"Method {method} is not supported for variational inference. "
                "Use 'psis', 'psir', or 'identity' instead."
            )
        else:
            vi_method = None

        vi_method_literal = cast(Literal["psis", "psir", "identity", None], vi_method)

        result = vi_psis_sampling(
            samples=samples,
            logP=logP,
            logQ=logQ,
            num_draws=num_draws,
            method=vi_method_literal,
            random_seed=random_seed,
        )

        return result.log_weights, result.pareto_k

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
