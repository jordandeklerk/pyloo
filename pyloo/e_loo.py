"""Functions for computing weighted expectations."""

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import xarray as xr
from arviz import InferenceData

from .base import ISMethod, compute_importance_weights
from .psis import _gpdfit
from .utils import _logsumexp, to_inference_data, wrap_xarray_ufunc

__all__ = ["e_loo", "ExpectationResult"]


@dataclass
class ExpectationResult:
    """Container for results from expectation calculations.

    Attributes
    ----------
    value : xr.DataArray
        The computed expectation value. For quantile calculations with multiple
        probabilities, this will include a 'quantile' dimension. Otherwise, it will
        have the same dimensions as the input data, excluding the sample dimension.
    pareto_k : xr.DataArray
        Function-specific Pareto k diagnostic values with the same dimensions as
        the value attribute, excluding any quantile dimension.
    """

    value: xr.DataArray
    pareto_k: xr.DataArray


def e_loo(
    data: InferenceData | Any,
    *,
    var_name: str | None = None,
    group: str = "posterior_predictive",
    log_likelihood_group: str = "log_likelihood",
    log_likelihood_var: str | None = None,
    weights: xr.DataArray | None = None,
    log_weights: xr.DataArray | None = None,
    type: str = "mean",
    probs: float | Sequence[float] | None = None,
    method: str | ISMethod = "psis",
    reff: float = 1.0,
) -> ExpectationResult:
    """Compute weighted expectations using importance sampling weights.

    This function computes expectations (mean, variance, standard deviation, or quantiles)
    of posterior or posterior predictive samples, weighted by importance sampling weights.
    The weights are typically derived from leave-one-out cross-validation.

    Parameters
    ----------
    data : InferenceData or convertible object
        An ArviZ InferenceData object or any object that can be converted to
        InferenceData containing posterior or posterior predictive samples.
    var_name : str, optional
        Name of the variable in the specified group to compute expectations for.
        If None and there is only one variable, that variable will be used.
    group : str, default "posterior_predictive"
        Name of the InferenceData group containing the variable to compute
        expectations for. Typically "posterior_predictive" or "posterior".
    log_likelihood_group : str, default "log_likelihood"
        Name of the group containing log likelihood values, used for computing
        importance weights if not provided directly.
    log_likelihood_var : str, optional
        Name of the variable in log_likelihood group to use. If None and var_name
        is provided, will try to use var_name. If None and var_name is None,
        will use the only variable if there is only one.
    weights : xr.DataArray, optional
        Pre-computed importance sampling weights (not log weights).
        If not provided, weights will be computed from log-likelihood values.
    log_weights : xr.DataArray, optional
        Pre-computed log importance sampling weights.
        If not provided, will be computed from log-likelihood values.
    type : str, default "mean"
        Type of expectation to compute:
        - "mean": weighted mean
        - "variance": weighted variance
        - "sd": weighted standard deviation
        - "quantile": weighted quantiles
    probs : float or sequence of floats, optional
        Probabilities for computing quantiles. Required if type="quantile".
    method : str or ISMethod, default "psis"
        Importance sampling method to use if weights need to be computed:
        - "psis": Pareto Smoothed Importance Sampling (recommended)
        - "sis": Standard Importance Sampling
        - "tis": Truncated Importance Sampling
    reff : float, default 1.0
        Relative MCMC efficiency, used for PSIS. Default is 1.0.

    Returns
    -------
    ExpectationResult
        Container with computed expectation value and diagnostics.

    Examples
    --------
    Compute weighted mean of posterior predictive samples:

    .. code-block:: python

        import arviz as az
        from pyloo import e_loo

        idata = az.load_arviz_data("centered_eight")

        # Compute weighted mean of posterior predictive samples
        result = e_loo(idata, var_name="obs", type="mean")
        print(result.value)
        print(result.pareto_k)

    Compute weighted quantiles with pre-computed weights:

    .. code-block:: python

        import arviz as az
        from pyloo import compute_importance_weights, e_loo

        idata = az.load_arviz_data("centered_eight")

        log_like = -idata.log_likelihood.obs
        log_weights, pareto_k = compute_importance_weights(log_like)

        # Compute weighted quantiles
        result = e_loo(
            idata,
            var_name="obs",
            log_weights=log_weights,
            type="quantile",
            probs=[0.025, 0.5, 0.975]
        )
        print(result.value.sel(quantile=0.5))  # Median
    """
    idata = to_inference_data(data)

    if type not in ["mean", "variance", "sd", "quantile"]:
        raise ValueError("type must be 'mean', 'variance', 'sd' or 'quantile'")

    if type == "quantile":
        if probs is None:
            raise ValueError("probs must be provided for quantile calculation")
        if np.isscalar(probs):
            probs_array = np.array([probs])
        else:
            probs_array = np.asarray(probs)
        if not np.all((probs_array > 0) & (probs_array < 1)):
            raise ValueError("probs must be between 0 and 1")
    else:
        probs_array = None

    if not hasattr(idata, group):
        raise ValueError(f"InferenceData object does not have a {group} group")

    data_group = getattr(idata, group)

    if var_name is None:
        var_names = list(data_group.data_vars)
        if len(var_names) == 1:
            var_name = var_names[0]
        else:
            raise ValueError(
                f"Multiple variables found in {group} group. Please specify var_name"
                f" from: {var_names}"
            )
    elif var_name not in data_group.data_vars:
        raise ValueError(
            f"Variable '{var_name}' not found in {group} group. Available variables:"
            f" {list(data_group.data_vars)}"
        )

    x_data = data_group[var_name]

    if "chain" in x_data.dims and "draw" in x_data.dims:
        x_data = x_data.stack(__sample__=("chain", "draw"))

    if weights is not None:
        log_weights = np.log(weights)

        if isinstance(log_weights, xr.DataArray):
            if (
                "__sample__" not in log_weights.dims
                and "chain" in log_weights.dims
                and "draw" in log_weights.dims
            ):
                log_weights = log_weights.stack(__sample__=("chain", "draw"))
    elif log_weights is None:
        if not hasattr(idata, log_likelihood_group):
            raise ValueError(
                f"InferenceData object does not have a {log_likelihood_group} group and"
                " no weights provided"
            )

        ll_group = getattr(idata, log_likelihood_group)
        if log_likelihood_var is None:
            log_likelihood_var = var_name

        if log_likelihood_var not in ll_group.data_vars:
            ll_vars = list(ll_group.data_vars)
            if len(ll_vars) == 1:
                log_likelihood_var = ll_vars[0]
            else:
                raise ValueError(
                    f"Multiple variables found in {log_likelihood_group} group. "
                    f"Please specify log_likelihood_var from: {ll_vars}"
                )

        log_like = ll_group[log_likelihood_var]

        if "chain" in log_like.dims and "draw" in log_like.dims:
            log_like = log_like.stack(__sample__=("chain", "draw"))

        log_weights, pareto_k_values = compute_importance_weights(
            -log_like, method=method, reff=reff
        )
    else:
        if isinstance(log_weights, xr.DataArray):
            if (
                "__sample__" not in log_weights.dims
                and "chain" in log_weights.dims
                and "draw" in log_weights.dims
            ):
                log_weights = log_weights.stack(__sample__=("chain", "draw"))

    if "__sample__" not in log_weights.dims:
        if "chain" in log_weights.dims and "draw" in log_weights.dims:
            log_weights = log_weights.stack(__sample__=("chain", "draw"))
        else:
            sample_dim = log_weights.dims[-1]
            log_weights = log_weights.rename({sample_dim: "__sample__"})

    if isinstance(log_weights, xr.DataArray) and isinstance(x_data, xr.DataArray):
        x_dims = [d for d in x_data.dims if d != "__sample__"]
        lw_dims = [d for d in log_weights.dims if d != "__sample__"]

        shared_dims = set(x_dims) & set(lw_dims)
        if shared_dims:
            x_data, log_weights = xr.align(
                x_data, log_weights, join="inner", exclude=["__sample__"]
            )

    if type == "mean":
        value = _compute_weighted_mean(x_data, log_weights)
    elif type == "variance":
        value = _compute_weighted_variance(x_data, log_weights)
    elif type == "sd":
        value = _compute_weighted_sd(x_data, log_weights)
    else:
        value = _compute_weighted_quantiles(x_data, log_weights, probs_array)

    if "pareto_k_values" not in locals():
        log_ratios = log_weights

        if type == "quantile":
            h = None
        elif type in ("variance", "sd"):
            h = x_data**2
        else:
            h = x_data

        pareto_k_values = _compute_pareto_k(h, log_ratios)

    return ExpectationResult(value=value, pareto_k=pareto_k_values)


def _normalize_log_weights(log_weights: xr.DataArray, sample_axis: int) -> xr.DataArray:
    """Normalize log weights to sum to 1 on the log scale."""
    return log_weights - _logsumexp(log_weights, axis=sample_axis, keepdims=True)


def _wvar_func(x_vals: np.ndarray, w_vals: np.ndarray) -> float:
    """Calculate weighted variance from arrays of values and weights."""
    if np.allclose(x_vals, x_vals[0]):
        return 0.0

    w_sum_sq = np.sum(w_vals**2)
    if np.isclose(w_sum_sq, 1.0):
        return 0.0

    mean = np.sum(w_vals * x_vals)
    mean_sq = np.sum(w_vals * x_vals**2)

    var = (mean_sq - mean**2) / (1 - w_sum_sq)
    return max(var, 0.0)


def _weighted_quantile(x_vals: np.ndarray, w_vals: np.ndarray, prob: float) -> float:
    """Calculate weighted quantile from arrays of values and weights."""
    if np.allclose(w_vals, w_vals[0]):
        return np.quantile(x_vals, prob)

    idx = np.argsort(x_vals)
    x_sorted, w_sorted = x_vals[idx], w_vals[idx]

    ww = np.cumsum(w_sorted) / np.sum(w_sorted)

    ids = np.where(ww >= prob)[0]
    if len(ids) == 0:
        return x_sorted[-1]
    else:
        wi = ids[0]
        if wi == 0:
            return x_sorted[0]
        else:
            w1 = ww[wi - 1]
            x1 = x_sorted[wi - 1]
            return x1 + (x_sorted[wi] - x1) * (prob - w1) / (ww[wi] - w1)


def _khat_func(
    x_vals: np.ndarray | None, log_ratios_vals: np.ndarray, tail_len: int = 20
) -> float:
    """Calculate Pareto k diagnostic from arrays of values and log ratios."""
    r_theta = np.exp(log_ratios_vals - np.max(log_ratios_vals))

    x_tail = -np.sort(-r_theta)[:tail_len]
    if len(x_tail) < 5 or np.allclose(x_tail, x_tail[0]):
        khat_r = np.inf
    else:
        exp_cutoff = np.exp(np.log(x_tail[-1]))
        khat_r, _ = _gpdfit(x_tail - exp_cutoff)

    if (
        x_vals is None
        or np.allclose(x_vals, x_vals[0])
        or len(np.unique(x_vals)) == 2
        or np.any(np.isnan(x_vals))
        or np.any(np.isinf(x_vals))
    ):
        return khat_r

    hr = x_vals * r_theta
    x_tail_left = np.sort(hr)[:tail_len]
    x_tail_right = -np.sort(-hr)[:tail_len]
    x_tail = np.concatenate([x_tail_left, x_tail_right])

    if len(x_tail) < 5 or np.allclose(x_tail, x_tail[0]):
        khat_hr = np.inf
    else:
        exp_cutoff = np.exp(np.log(x_tail[-1]))
        khat_hr, _ = _gpdfit(x_tail - exp_cutoff)

    if np.isnan(khat_hr) and np.isnan(khat_r):
        return np.nan
    return max(khat_hr, khat_r)


def _khat_wrapper(dummy: np.ndarray, log_ratios_vals: np.ndarray) -> float:
    """Wrapper for _khat_func when x values are None."""
    return _khat_func(None, log_ratios_vals)


def _compute_weighted_mean(x: xr.DataArray, log_weights: xr.DataArray) -> xr.DataArray:
    """Compute weighted mean for xarray inputs."""
    normalized_log_weights = _normalize_log_weights(
        log_weights, x.get_axis_num("__sample__")
    )
    weights = np.exp(normalized_log_weights)

    return (weights * x).sum(dim="__sample__")


def _compute_weighted_variance(
    x: xr.DataArray, log_weights: xr.DataArray
) -> xr.DataArray:
    """Compute weighted variance for xarray inputs."""
    normalized_log_weights = _normalize_log_weights(
        log_weights, x.get_axis_num("__sample__")
    )
    weights = np.exp(normalized_log_weights)

    result = wrap_xarray_ufunc(
        _wvar_func,
        x,
        weights,
        input_core_dims=[["__sample__"], ["__sample__"]],
        output_core_dims=[[]],
        vectorize=True,
    )

    return result


def _compute_weighted_sd(x: xr.DataArray, log_weights: xr.DataArray) -> xr.DataArray:
    """Compute weighted standard deviation for xarray inputs."""
    variance = _compute_weighted_variance(x, log_weights)
    return np.sqrt(variance)


def _compute_weighted_quantiles(
    x: xr.DataArray, log_weights: xr.DataArray, probs: np.ndarray
) -> xr.DataArray:
    """Compute weighted quantiles for xarray inputs."""
    normalized_log_weights = _normalize_log_weights(
        log_weights, x.get_axis_num("__sample__")
    )
    weights = np.exp(normalized_log_weights)

    if np.isscalar(probs):
        probs = np.array([probs])
    else:
        probs = np.asarray(probs)

    sample_axis = x.get_axis_num("__sample__")
    non_sample_dims = list(x.dims)
    non_sample_dims.pop(sample_axis)
    non_sample_shape = tuple(x.sizes[d] for d in non_sample_dims)

    result_shape = non_sample_shape + (len(probs),)
    result_data = np.zeros(result_shape)

    iter_coords = []
    for d in non_sample_dims:
        iter_coords.append(list(range(x.sizes[d])))

    if non_sample_dims:
        for idx in np.ndindex(non_sample_shape):
            idx_dict = dict(zip(non_sample_dims, idx))
            idx_dict["__sample__"] = slice(None)

            x_vals = x.isel(idx_dict).values
            w_vals = weights.isel(idx_dict).values

            for p_idx, prob in enumerate(probs):
                flat_idx = idx + (p_idx,)
                result_data[flat_idx] = _weighted_quantile(x_vals, w_vals, prob)
    else:
        x_vals = x.values
        w_vals = weights.values
        for p_idx, prob in enumerate(probs):
            result_data[p_idx] = _weighted_quantile(x_vals, w_vals, prob)

    output_dims = non_sample_dims + ["quantile"]
    coords = {}
    for d in non_sample_dims:
        coords[d] = x.coords[d]
    coords["quantile"] = probs

    return xr.DataArray(result_data, dims=output_dims, coords=coords)


def _compute_pareto_k(
    x: xr.DataArray | None, log_ratios: xr.DataArray, tail_len: int = 20
) -> xr.DataArray:
    """Compute Pareto k diagnostic for xarray inputs."""
    if x is None:
        dummy_x = xr.zeros_like(log_ratios)

        result = wrap_xarray_ufunc(
            _khat_wrapper,
            dummy_x,
            log_ratios,
            input_core_dims=[["__sample__"], ["__sample__"]],
            output_core_dims=[[]],
            vectorize=True,
        )
    else:
        result = wrap_xarray_ufunc(
            _khat_func,
            x,
            log_ratios,
            input_core_dims=[["__sample__"], ["__sample__"]],
            output_core_dims=[[]],
            vectorize=True,
        )

    return result
