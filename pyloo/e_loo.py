"""Functions for computing weighted expectations."""

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import xarray as xr
from arviz import InferenceData

from .psis import _gpdfit
from .utils import _logsumexp, to_inference_data, wrap_xarray_ufunc

__all__ = [
    "e_loo",
    "ExpectationResult",
    "compute_pareto_k",
    "k_hat",
    "_pareto_min_ss",
    "_pareto_khat_threshold",
    "_pareto_convergence_rate",
]


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
    min_ss : xr.DataArray, optional
        Minimum sample size for reliable Pareto smoothed estimate. If the actual
        sample size is greater than min_ss, then Pareto smoothed estimates can be
        considered reliable.
    khat_threshold : xr.DataArray, optional
        Threshold below which k-hat values result in reliable Pareto smoothed
        estimates. The threshold is lower for smaller effective sample sizes.
    convergence_rate : xr.DataArray, optional
        Relative convergence rate compared to the central limit theorem. Applicable
        only if the actual sample size is sufficiently large (greater than min_ss).
    """

    value: xr.DataArray
    pareto_k: xr.DataArray
    min_ss: xr.DataArray | None = None
    khat_threshold: xr.DataArray | None = None
    convergence_rate: xr.DataArray | None = None


def e_loo(
    data: InferenceData | Any,
    *,
    var_name: str | None = None,
    group: str = "posterior_predictive",
    weights: xr.DataArray | None = None,
    log_weights: xr.DataArray | None = None,
    log_ratios: xr.DataArray | None = None,
    type: str = "mean",
    probs: float | Sequence[float] | None = None,
) -> ExpectationResult:
    """Compute weighted expectations using importance sampling weights.

    This function computes expectations (mean, variance, standard deviation, or quantiles)
    of posterior or posterior predictive samples, weighted by importance sampling weights.
    The weights must be pre-computed, typically from leave-one-out cross-validation.

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
    weights : xr.DataArray, optional
        Pre-computed importance sampling weights (not log weights).
        Either weights or log_weights must be provided.
    log_weights : xr.DataArray, optional
        Pre-computed log importance sampling weights.
        Either weights or log_weights must be provided.
    log_ratios : xr.DataArray, optional
        Raw (not smoothed) log ratios for computing more accurate Pareto k diagnostics.
        If not provided, log_weights will be used instead, which may result in
        slightly optimistic Pareto k estimates.
    type : str, default "mean"
        Type of expectation to compute:
        - "mean": weighted mean
        - "variance": weighted variance
        - "sd": weighted standard deviation
        - "quantile": weighted quantiles
    probs : float or sequence of floats, optional
        Probabilities for computing quantiles. Required if type="quantile".

    Returns
    -------
    ExpectationResult
        Container with computed expectation value and diagnostics:
        - value: The computed expectation value
        - pareto_k: Function-specific Pareto k diagnostic values
        - min_ss: Minimum sample size for reliable Pareto smoothed estimate
        - khat_threshold: Threshold below which k-hat values result in reliable estimates
        - convergence_rate: Relative convergence rate compared to the central limit theorem

    Notes
    -----
    The expectations computed by ``e_loo`` assume that the PSIS approximation
    is reliable. The reliability of the PSIS approximation can be assessed using
    the Pareto k diagnostic.

    Examples
    --------
    Compute weighted mean with pre-computed weights:

    .. code-block:: python

        import arviz as az
        from pyloo import e_loo
        from pyloo import psis

        idata = az.load_arviz_data("centered_eight")

        log_like = idata.log_likelihood.obs
        if "chain" in log_like.dims and "draw" in log_like.dims:
            log_like = log_like.stack(__sample__=("chain", "draw"))

        # Compute PSIS weights
        log_weights, pareto_k = psis(-log_like)

        result = e_loo(
            idata,
            var_name="obs",
            log_weights=log_weights,
            log_ratios=-log_like,  # Optional: for more accurate diagnostics
            type="mean"
        )

        print(result.value)
        print(result.pareto_k)
        print(result.min_ss)  # Minimum sample size for reliable estimates
        print(result.khat_threshold)  # Threshold for reliable k-hat values
        print(result.convergence_rate)  # Relative convergence rate
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

    if weights is None and log_weights is None:
        raise ValueError("Either weights or log_weights must be provided")

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

    if log_weights is not None:
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
    else:  # type == "quantile"
        value = _compute_weighted_quantiles(x_data, log_weights, probs_array)

    if log_ratios is None:
        log_ratios = log_weights

    if type == "quantile":
        h = None
    elif type in ("variance", "sd"):
        # For variance and sd, h = x^2
        h = x_data**2
    else:
        # For mean, h = x
        h = x_data

    pareto_k_values = compute_pareto_k(h, log_ratios)

    if "__sample__" in x_data.dims:
        n_samples = x_data.sizes["__sample__"]
    else:
        n_samples = x_data.shape[0]

    min_ss = xr.apply_ufunc(_pareto_min_ss, pareto_k_values, vectorize=True)

    khat_threshold = xr.full_like(pareto_k_values, _pareto_khat_threshold(n_samples))

    convergence_rate = xr.apply_ufunc(
        lambda k: _pareto_convergence_rate(k, n_samples),
        pareto_k_values,
        vectorize=True,
    )

    return ExpectationResult(
        value=value,
        pareto_k=pareto_k_values,
        min_ss=min_ss,
        khat_threshold=khat_threshold,
        convergence_rate=convergence_rate,
    )


def compute_pareto_k(
    x: xr.DataArray | np.ndarray | None,
    log_ratios: xr.DataArray | np.ndarray,
    tail_len: int = 20,
) -> xr.DataArray | float:
    """Compute Pareto k diagnostic for expectation calculations.

    Parameters
    ----------
    x : xr.DataArray, np.ndarray, or None
        Values of function h(theta). The interpretation depends on the type of expectation:
        - For mean: the original values
        - For variance/sd: the squared values
        - For quantiles: None
        If an xarray DataArray is provided, it must have a '__sample__' dimension.
    log_ratios : xr.DataArray or np.ndarray
        Log ratios (log importance weights before normalization).
        If an xarray DataArray is provided, it must have a '__sample__' dimension.
        These should be the raw log ratios, not smoothed weights.
    tail_len : int, default 20
        Number of tail samples to use for fitting the generalized Pareto distribution.
        Must be at least 5 for reliable estimation.

    Returns
    -------
    xr.DataArray or float
        Estimated Pareto k shape parameter. If input is xarray, output will be xarray
        with the same dimensions as the input, excluding the sample dimension.
    """
    if tail_len < 5:
        raise ValueError("tail_len must be at least 5")

    if isinstance(log_ratios, xr.DataArray):
        if "__sample__" not in log_ratios.dims:
            raise ValueError("log_ratios must have '__sample__' dimension")

        if x is not None and isinstance(x, xr.DataArray):
            if "__sample__" not in x.dims:
                raise ValueError("x must have '__sample__' dimension")
            x_array = x
        else:
            x_array = xr.zeros_like(log_ratios)

        result = wrap_xarray_ufunc(
            k_hat,
            x_array,
            log_ratios,
            input_core_dims=[["__sample__"], ["__sample__"]],
            output_core_dims=[[]],
            vectorize=True,
            func_kwargs={"tail_len": tail_len},
        )
        return result

    if isinstance(log_ratios, np.ndarray):
        if x is not None and isinstance(x, np.ndarray):
            if x.shape != log_ratios.shape:
                raise ValueError("x and log_ratios must have the same shape")

    return k_hat(x, log_ratios, tail_len)


def k_hat(
    x_vals: np.ndarray | None, log_ratios_vals: np.ndarray, tail_len: int = 20
) -> float:
    """Calculate Pareto k diagnostic by fitting a Generalized Pareto Distribution
    to one or two tails of ``x``. This can be used to estimate the number of fractional
    moments that is useful for convergence diagnostics.

    Parameters
    ----------
    x_vals : np.ndarray or None
        Values of function h(theta). For mean, this is the original values.
        For variance/sd, this is the squared values. For quantiles, this is None.
    log_ratios_vals : np.ndarray
        Log ratios (log importance weights before normalization)
    tail_len : int, default 20
        Number of tail samples to use for fitting the generalized Pareto distribution

    Returns
    -------
    float
        Estimated Pareto k shape parameter
    """
    r_theta = np.exp(log_ratios_vals - np.max(log_ratios_vals))
    sorted_r = -np.sort(-r_theta)[:tail_len]

    if len(sorted_r) < 5 or np.allclose(sorted_r, sorted_r[0]):
        khat_r = np.inf
    else:
        cutoff = sorted_r[-1]
        khat_r, _ = _gpdfit(sorted_r - cutoff)

    if (
        x_vals is None
        or np.allclose(x_vals, x_vals[0])
        or len(np.unique(x_vals)) == 2
        or np.any(np.isnan(x_vals))
        or np.any(np.isinf(x_vals))
    ):
        return khat_r

    hr = x_vals * r_theta

    left_tail = np.sort(hr)[:tail_len]
    right_tail = -np.sort(-hr)[:tail_len]

    if len(left_tail) < 5 or np.allclose(left_tail, left_tail[0]):
        khat_hr_left = -np.inf
    else:
        cutoff_left = left_tail[-1]
        khat_hr_left, _ = _gpdfit(-(left_tail - cutoff_left))

    if len(right_tail) < 5 or np.allclose(right_tail, right_tail[0]):
        khat_hr_right = -np.inf
    else:
        cutoff_right = right_tail[-1]
        khat_hr_right, _ = _gpdfit(right_tail - cutoff_right)

    khat_hr = max(khat_hr_left, khat_hr_right)

    if np.isnan(khat_hr) and np.isnan(khat_r):
        return np.nan

    return max(khat_hr, khat_r)


def _pareto_min_ss(k: float) -> float:
    """Calculate minimum sample size for reliable Pareto smoothed estimate."""
    if k < 1:
        return 10 ** (1 / (1 - max(0, k)))
    else:
        return float("inf")


def _pareto_khat_threshold(n_samples: int) -> float:
    """Calculate k-hat threshold for reliable Pareto smoothed estimate."""
    return 1 - 1 / np.log10(n_samples)


def _pareto_convergence_rate(k: float, n_samples: int) -> float:
    """Calculate convergence rate for Pareto smoothed estimate."""
    # k < 0: bounded distribution
    if k < 0:
        return 1.0
    # k > 1: non-finite mean
    elif k > 1:
        return 0.0
    # k = 0.5: limit value
    elif k == 0.5:
        return 1 - 1 / np.log(n_samples)
    # 0 < k < 1 and k != 0.5: smooth approximation
    elif 0 < k < 1:
        n = n_samples
        return max(
            0,
            (2 * (k - 1) * n ** (2 * k + 1) + (1 - 2 * k) * n ** (2 * k) + n**2)
            / ((n - 1) * (n - n ** (2 * k))),
        )
    else:
        return 1.0


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


def _normalize_log_weights(log_weights: xr.DataArray, sample_axis: int) -> xr.DataArray:
    """Normalize log weights to sum to 1 on the log scale."""
    return log_weights - _logsumexp(log_weights, axis=sample_axis, keepdims=True)
