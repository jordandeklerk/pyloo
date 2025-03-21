"""Tests for the e_loo module."""

import numpy as np
import pytest
import xarray as xr
from arviz import InferenceData

from ...base import compute_importance_weights
from ...e_loo import (
    ExpectationResult,
    _compute_weighted_mean,
    _compute_weighted_quantiles,
    _compute_weighted_sd,
    _compute_weighted_variance,
    _normalize_log_weights,
    _weighted_quantile,
    _wvar_func,
    compute_pareto_k,
    e_loo,
    k_hat,
)
from ..helpers import assert_arrays_allclose, assert_finite, assert_positive


def test_expectation_result_dataclass():
    """Test the ExpectationResult dataclass."""
    value = xr.DataArray([1.0, 2.0, 3.0], dims=["dim1"])
    pareto_k = xr.DataArray([0.1, 0.2, 0.3], dims=["dim1"])

    result = ExpectationResult(value=value, pareto_k=pareto_k)

    assert isinstance(result, ExpectationResult)
    assert isinstance(result.value, xr.DataArray)
    assert isinstance(result.pareto_k, xr.DataArray)
    assert_arrays_allclose(result.value.values, [1.0, 2.0, 3.0])
    assert_arrays_allclose(result.pareto_k.values, [0.1, 0.2, 0.3])


def test_normalize_log_weights():
    """Test the _normalize_log_weights function."""
    log_weights = xr.DataArray(
        [[-1.0, -2.0], [-3.0, -4.0]],
        dims=("__sample__", "obs_dim"),
        coords={"obs_dim": [0, 1]},
    )

    normalized = _normalize_log_weights(
        log_weights, log_weights.get_axis_num("__sample__")
    )

    summed = np.exp(normalized).sum(dim="__sample__")
    assert_arrays_allclose(summed.values, [1.0, 1.0], rtol=1e-5)


def test_wvar_func():
    """Test the _wvar_func function."""
    x_const = np.array([5.0, 5.0, 5.0, 5.0])
    w_const = np.array([0.25, 0.25, 0.25, 0.25])
    assert_arrays_allclose(_wvar_func(x_const, w_const), 0.0)

    x_vary = np.array([1.0, 2.0, 3.0, 4.0])
    w_vary = np.array([0.1, 0.2, 0.3, 0.4])
    var = _wvar_func(x_vary, w_vary)
    assert var > 0

    w_extreme = np.array([0.97, 0.01, 0.01, 0.01])
    var_extreme = _wvar_func(x_vary, w_extreme)
    assert var_extreme >= 0


def test_weighted_quantile():
    """Test the _weighted_quantile function."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    w = np.array([0.1, 0.2, 0.4, 0.2, 0.1])

    median = _weighted_quantile(x, w, 0.5)
    assert_arrays_allclose(median, 2.5)

    q1 = _weighted_quantile(x, w, 0.25)
    q3 = _weighted_quantile(x, w, 0.75)
    assert q1 < median < q3

    w_equal = np.ones_like(x) / len(x)
    median_equal = _weighted_quantile(x, w_equal, 0.5)
    assert_arrays_allclose(median_equal, 3.0)

    assert _weighted_quantile(x, w, 0.0) == 1.0
    assert _weighted_quantile(x, w, 1.0) == 5.0


def test_khat_func():
    """Test the _khat_func function."""
    rng = np.random.default_rng(42)
    x = rng.normal(size=100)
    log_ratios = rng.normal(size=100)

    k = k_hat(x, log_ratios)
    assert isinstance(k, float)
    assert k >= 0 or np.isnan(k)

    k_none = k_hat(None, log_ratios)
    assert isinstance(k_none, float)
    assert k_none >= 0 or np.isnan(k_none)

    x_const = np.ones_like(x)
    k_const = k_hat(x_const, log_ratios)
    assert isinstance(k_const, float)
    assert k_const >= 0 or np.isnan(k_const)


def test_compute_weighted_mean(centered_eight):
    """Test the _compute_weighted_mean function."""
    x = centered_eight.posterior_predictive.obs.stack(__sample__=("chain", "draw"))
    log_weights = centered_eight.log_likelihood.obs.stack(__sample__=("chain", "draw"))

    result = _compute_weighted_mean(x, log_weights)

    assert isinstance(result, xr.DataArray)
    assert "school" in result.dims
    assert "__sample__" not in result.dims
    assert result.shape == (8,)
    assert_finite(result)


def test_compute_weighted_variance(centered_eight):
    """Test the _compute_weighted_variance function."""
    x = centered_eight.posterior_predictive.obs.stack(__sample__=("chain", "draw"))
    log_weights = centered_eight.log_likelihood.obs.stack(__sample__=("chain", "draw"))

    result = _compute_weighted_variance(x, log_weights)

    assert isinstance(result, xr.DataArray)
    assert "school" in result.dims
    assert "__sample__" not in result.dims
    assert result.shape == (8,)
    assert_positive(result)


def test_compute_weighted_sd(centered_eight):
    """Test the _compute_weighted_sd function."""
    x = centered_eight.posterior_predictive.obs.stack(__sample__=("chain", "draw"))
    log_weights = centered_eight.log_likelihood.obs.stack(__sample__=("chain", "draw"))

    result = _compute_weighted_sd(x, log_weights)

    assert isinstance(result, xr.DataArray)
    assert "school" in result.dims
    assert "__sample__" not in result.dims
    assert result.shape == (8,)
    assert_positive(result)

    variance = _compute_weighted_variance(x, log_weights)
    assert_arrays_allclose(result.values, np.sqrt(variance.values))


def test_compute_weighted_quantiles(centered_eight):
    """Test the _compute_weighted_quantiles function."""
    x = centered_eight.posterior_predictive.obs.stack(__sample__=("chain", "draw"))
    log_weights = centered_eight.log_likelihood.obs.stack(__sample__=("chain", "draw"))
    probs = np.array([0.25, 0.5, 0.75])

    result = _compute_weighted_quantiles(x, log_weights, probs)

    assert isinstance(result, xr.DataArray)
    assert "school" in result.dims
    assert "quantile" in result.dims
    assert "__sample__" not in result.dims
    assert result.shape == (8, 3)
    assert_finite(result)

    for school in result.coords["school"].values:
        assert np.all(np.diff(result.sel(school=school).values) >= 0)


def test_compute_pareto_k(centered_eight):
    """Test the _compute_pareto_k function."""
    x = centered_eight.posterior_predictive.obs.stack(__sample__=("chain", "draw"))
    log_weights = centered_eight.log_likelihood.obs.stack(__sample__=("chain", "draw"))

    result = compute_pareto_k(x, log_weights)

    assert isinstance(result, xr.DataArray)
    assert "school" in result.dims
    assert "__sample__" not in result.dims
    assert result.shape == (8,)

    result_none = compute_pareto_k(None, log_weights)

    assert isinstance(result_none, xr.DataArray)
    assert "school" in result_none.dims
    assert "__sample__" not in result_none.dims
    assert result_none.shape == (8,)


def test_e_loo_with_arrays(centered_eight):
    """Test e_loo with xarray DataArrays."""
    x = centered_eight.posterior_predictive.obs.stack(__sample__=("chain", "draw"))
    log_weights = centered_eight.log_likelihood.obs.stack(__sample__=("chain", "draw"))

    n_chains = 4
    n_draws = 25
    n_obs = 3

    rng = np.random.default_rng(42)

    pp_data = rng.normal(size=(n_chains, n_draws, n_obs))
    pp_da = xr.DataArray(
        pp_data, dims=("chain", "draw", "obs_id"), coords={"obs_id": range(n_obs)}
    )
    pp_ds = xr.Dataset({"y": pp_da})

    ll_data = rng.normal(size=(n_chains, n_draws, n_obs))
    ll_da = xr.DataArray(
        ll_data, dims=("chain", "draw", "obs_id"), coords={"obs_id": range(n_obs)}
    )
    ll_ds = xr.Dataset({"y": ll_da})

    idata = InferenceData(posterior_predictive=pp_ds, log_likelihood=ll_ds)

    mean_value = _compute_weighted_mean(x, log_weights)
    pareto_k = compute_pareto_k(x, log_weights)
    result_mean = ExpectationResult(value=mean_value, pareto_k=pareto_k)

    assert isinstance(result_mean, ExpectationResult)
    assert isinstance(result_mean.value, xr.DataArray)
    assert isinstance(result_mean.pareto_k, xr.DataArray)
    assert result_mean.value.shape == (8,)
    assert result_mean.pareto_k.shape == (8,)

    var_value = _compute_weighted_variance(x, log_weights)
    result_var = ExpectationResult(value=var_value, pareto_k=pareto_k)

    assert isinstance(result_var, ExpectationResult)
    assert result_var.value.shape == (8,)
    assert result_var.pareto_k.shape == (8,)
    assert_positive(result_var.value)

    sd_value = _compute_weighted_sd(x, log_weights)
    result_sd = ExpectationResult(value=sd_value, pareto_k=pareto_k)

    assert isinstance(result_sd, ExpectationResult)
    assert result_sd.value.shape == (8,)
    assert result_sd.pareto_k.shape == (8,)
    assert_positive(result_sd.value)
    assert_arrays_allclose(result_sd.value.values, np.sqrt(result_var.value.values))

    probs = [0.25, 0.5, 0.75]
    quant_value = _compute_weighted_quantiles(x, log_weights, np.array(probs))
    result_quant = ExpectationResult(value=quant_value, pareto_k=pareto_k)

    assert isinstance(result_quant, ExpectationResult)
    assert result_quant.value.shape == (8, 3)
    assert result_quant.pareto_k.shape == (8,)
    assert "quantile" in result_quant.value.dims
    assert_arrays_allclose(result_quant.value.coords["quantile"].values, probs)

    idata_log_weights = -idata.log_likelihood.y.stack(__sample__=("chain", "draw"))

    result_idata = e_loo(
        idata, log_weights=idata_log_weights, var_name="y", type="mean"
    )

    assert isinstance(result_idata, ExpectationResult)
    assert result_idata.value.shape == (n_obs,)
    assert result_idata.pareto_k.shape == (n_obs,)


def test_e_loo_with_weights(centered_eight):
    """Test e_loo with pre-computed weights."""
    log_like = -centered_eight.log_likelihood.obs.stack(__sample__=("chain", "draw"))

    log_weights, _ = compute_importance_weights(log_like)
    weights = np.exp(log_weights)

    result_weights = e_loo(centered_eight, weights=weights, var_name="obs", type="mean")

    assert isinstance(result_weights, ExpectationResult)
    assert result_weights.value.shape == (8,)
    assert result_weights.pareto_k.shape == (8,)

    result_log_weights = e_loo(
        centered_eight, weights=weights, var_name="obs", type="mean"
    )

    assert isinstance(result_log_weights, ExpectationResult)
    assert result_log_weights.value.shape == (8,)
    assert result_log_weights.pareto_k.shape == (8,)

    assert_arrays_allclose(result_weights.value.values, result_log_weights.value.values)


def test_e_loo_with_inference_data(centered_eight):
    """Test e_loo with InferenceData objects."""
    log_like = -centered_eight.log_likelihood.obs.stack(__sample__=("chain", "draw"))
    log_weights, _ = compute_importance_weights(log_like)

    result = e_loo(
        centered_eight,
        var_name="obs",
        log_weights=log_weights,
        log_ratios=log_like,
        type="mean",
    )

    assert isinstance(result, ExpectationResult)
    assert isinstance(result.value, xr.DataArray)
    assert isinstance(result.pareto_k, xr.DataArray)
    assert result.value.shape == (8,)
    assert result.pareto_k.shape == (8,)
    assert result.min_ss is not None
    assert result.khat_threshold is not None
    assert result.convergence_rate is not None

    result_quant = e_loo(
        centered_eight,
        var_name="obs",
        log_weights=log_weights,
        log_ratios=log_like,
        type="quantile",
        probs=[0.25, 0.5, 0.75],
    )

    assert isinstance(result_quant, ExpectationResult)
    assert result_quant.value.shape[0] * result_quant.value.shape[1] == 24
    assert result_quant.pareto_k.shape == (8,)
    assert "quantile" in result_quant.value.dims

    result_posterior = e_loo(
        centered_eight,
        var_name="theta",
        group="posterior",
        log_weights=log_weights,
        log_ratios=log_like,
        type="mean",
    )

    assert isinstance(result_posterior, ExpectationResult)
    assert result_posterior.value.shape == (8,)
    assert result_posterior.pareto_k.shape == (8,)


def test_e_loo_errors(centered_eight):
    """Test error handling in e_loo."""
    x, log_weights = (
        centered_eight.posterior_predictive.obs,
        centered_eight.log_likelihood.obs,
    )

    with pytest.raises(ValueError, match="type must be"):
        e_loo(x, log_weights=log_weights, type="invalid")

    with pytest.raises(ValueError, match="probs must be provided"):
        e_loo(x, log_weights=log_weights, type="quantile")

    with pytest.raises(ValueError, match="probs must be between"):
        e_loo(x, log_weights=log_weights, type="quantile", probs=[-0.1, 1.1])

    with pytest.raises(ValueError):
        e_loo(x)

    invalid_data = InferenceData()
    with pytest.raises(ValueError):
        e_loo(invalid_data, var_name="obs")


def test_e_loo_with_multidimensional_data():
    """Test e_loo with multidimensional data."""
    rng = np.random.default_rng(42)
    n_samples = 100
    n_dim1 = 3
    n_dim2 = 2

    x = rng.normal(size=(n_samples, n_dim1, n_dim2))
    log_weights = rng.normal(size=(n_samples, n_dim1, n_dim2))
    log_weights -= np.max(log_weights, axis=0)

    x_da = xr.DataArray(
        x,
        dims=("__sample__", "dim1", "dim2"),
        coords={"dim1": range(n_dim1), "dim2": range(n_dim2)},
    )
    log_weights_da = xr.DataArray(
        log_weights,
        dims=("__sample__", "dim1", "dim2"),
        coords={"dim1": range(n_dim1), "dim2": range(n_dim2)},
    )

    mean_value = _compute_weighted_mean(x_da, log_weights_da)

    pareto_k_values = xr.zeros_like(mean_value)
    result_mean = ExpectationResult(value=mean_value, pareto_k=pareto_k_values)

    assert isinstance(result_mean, ExpectationResult)
    assert result_mean.value.shape == (n_dim1, n_dim2)
    assert result_mean.pareto_k.shape == (n_dim1, n_dim2)
    assert "dim1" in result_mean.value.dims
    assert "dim2" in result_mean.value.dims

    probs = [0.25, 0.5, 0.75]
    quant_value = _compute_weighted_quantiles(x_da, log_weights_da, np.array(probs))
    result_quant = ExpectationResult(value=quant_value, pareto_k=pareto_k_values)

    assert isinstance(result_quant, ExpectationResult)
    assert result_quant.value.shape == (n_dim1, n_dim2, 3)
    assert result_quant.pareto_k.shape == (n_dim1, n_dim2)
    assert "quantile" in result_quant.value.dims


def test_e_loo_with_aligned_data():
    """Test e_loo with data that needs alignment."""
    rng = np.random.default_rng(42)
    n_samples = 100
    n_obs = 3

    x = rng.normal(size=(n_samples, n_obs))
    log_weights = rng.normal(size=(n_samples, n_obs))
    log_weights -= np.max(log_weights, axis=0)

    x_da = xr.DataArray(
        x, dims=("__sample__", "obs_dim"), coords={"obs_dim": ["a", "b", "c"]}
    )
    log_weights_da = xr.DataArray(
        log_weights, dims=("__sample__", "obs_dim"), coords={"obs_dim": ["a", "b", "c"]}
    )

    x_da = x_da.expand_dims({"extra_dim": ["x"]})

    x_aligned, log_weights_aligned = xr.align(
        x_da, log_weights_da, join="inner", exclude=["__sample__"]
    )

    mean_value = _compute_weighted_mean(x_aligned, log_weights_aligned)

    pareto_k_values = xr.zeros_like(mean_value)
    result = ExpectationResult(value=mean_value, pareto_k=pareto_k_values)

    assert isinstance(result, ExpectationResult)
    assert "obs_dim" in result.value.dims
    assert "extra_dim" in result.value.dims
    assert result.value.size == 3
    assert result.pareto_k.size == 3


def test_e_loo_with_constant_values():
    """Test e_loo with constant values."""
    n_samples = 100
    n_obs = 3

    x = np.ones((n_samples, n_obs))
    log_weights = np.random.default_rng(42).normal(size=(n_samples, n_obs))
    log_weights -= np.max(log_weights, axis=0)

    x_da = xr.DataArray(
        x, dims=("__sample__", "obs_dim"), coords={"obs_dim": range(n_obs)}
    )
    log_weights_da = xr.DataArray(
        log_weights, dims=("__sample__", "obs_dim"), coords={"obs_dim": range(n_obs)}
    )

    mean_value = _compute_weighted_mean(x_da, log_weights_da)

    pareto_k_values = xr.zeros_like(mean_value)
    result_mean = ExpectationResult(value=mean_value, pareto_k=pareto_k_values)

    assert isinstance(result_mean, ExpectationResult)
    assert_arrays_allclose(result_mean.value.values, np.ones(n_obs))

    var_value = _compute_weighted_variance(x_da, log_weights_da)
    result_var = ExpectationResult(value=var_value, pareto_k=pareto_k_values)

    assert isinstance(result_var, ExpectationResult)
    assert_arrays_allclose(result_var.value.values, np.zeros(n_obs), atol=1e-10)

    sd_value = _compute_weighted_sd(x_da, log_weights_da)
    result_sd = ExpectationResult(value=sd_value, pareto_k=pareto_k_values)

    assert isinstance(result_sd, ExpectationResult)
    assert_arrays_allclose(result_sd.value.values, np.zeros(n_obs), atol=1e-10)


def test_e_loo_with_extreme_weights():
    """Test e_loo with extreme weights."""
    n_samples = 100
    n_obs = 3

    x = np.random.default_rng(42).normal(size=(n_samples, n_obs))
    log_weights = np.ones((n_samples, n_obs)) * -1000
    log_weights[0, :] = 0

    x_da = xr.DataArray(
        x, dims=("__sample__", "obs_dim"), coords={"obs_dim": range(n_obs)}
    )
    log_weights_da = xr.DataArray(
        log_weights, dims=("__sample__", "obs_dim"), coords={"obs_dim": range(n_obs)}
    )

    mean_value = _compute_weighted_mean(x_da, log_weights_da)

    pareto_k_values = xr.zeros_like(mean_value)
    result = ExpectationResult(value=mean_value, pareto_k=pareto_k_values)

    assert isinstance(result, ExpectationResult)
    assert_finite(result.value)
    assert_arrays_allclose(result.value.values, x[0, :])


def test_e_loo_with_eight_schools(centered_eight, non_centered_eight):
    """Test e_loo with eight schools data from both parameterizations."""
    log_like_centered = -centered_eight.log_likelihood.obs.stack(
        __sample__=("chain", "draw")
    )
    log_weights_centered, _ = compute_importance_weights(log_like_centered)

    result_centered = e_loo(
        centered_eight,
        var_name="obs",
        log_weights=log_weights_centered,
        log_ratios=log_like_centered,
        type="mean",
    )

    assert isinstance(result_centered, ExpectationResult)
    assert result_centered.value.shape == (8,)
    assert result_centered.pareto_k.shape == (8,)
    assert result_centered.min_ss is not None
    assert result_centered.khat_threshold is not None
    assert result_centered.convergence_rate is not None

    log_like_non_centered = -non_centered_eight.log_likelihood.obs.stack(
        __sample__=("chain", "draw")
    )
    log_weights_non_centered, _ = compute_importance_weights(log_like_non_centered)

    result_non_centered = e_loo(
        non_centered_eight,
        var_name="obs",
        log_weights=log_weights_non_centered,
        log_ratios=log_like_non_centered,
        type="mean",
    )

    assert isinstance(result_non_centered, ExpectationResult)
    assert result_non_centered.value.shape == (8,)
    assert result_non_centered.pareto_k.shape == (8,)
    assert result_non_centered.min_ss is not None
    assert result_non_centered.khat_threshold is not None
    assert result_non_centered.convergence_rate is not None

    assert np.allclose(
        result_centered.value.values,
        result_non_centered.value.values,
        rtol=0.3,
        atol=0.5,
    )


def test_e_loo_with_custom_inference_data():
    """Test e_loo with a custom InferenceData object."""
    n_chains = 4
    n_draws = 100
    n_obs = 3

    rng = np.random.default_rng(42)

    pp_data = rng.normal(size=(n_chains, n_draws, n_obs))
    pp_da = xr.DataArray(
        pp_data, dims=("chain", "draw", "obs_id"), coords={"obs_id": range(n_obs)}
    )
    pp_ds = xr.Dataset({"y": pp_da})

    ll_data = rng.normal(size=(n_chains, n_draws, n_obs))
    ll_da = xr.DataArray(
        ll_data, dims=("chain", "draw", "obs_id"), coords={"obs_id": range(n_obs)}
    )
    ll_ds = xr.Dataset({"y": ll_da})

    idata = InferenceData(posterior_predictive=pp_ds, log_likelihood=ll_ds)

    result = e_loo(idata, log_weights=-ll_da, var_name="y", type="mean")

    assert isinstance(result, ExpectationResult)
    assert result.value.shape == (n_obs,)
    assert result.pareto_k.shape == (n_obs,)


def test_e_loo_numerical_stability():
    """Test numerical stability of e_loo with challenging data."""
    n_samples = 100
    n_obs = 3

    rng = np.random.default_rng(42)
    x = rng.normal(size=(n_samples, n_obs))

    log_weights = np.ones((n_samples, n_obs)) * -1000
    log_weights[0, :] = 0

    x[1, :] = 1e10

    x_da = xr.DataArray(
        x, dims=("__sample__", "obs_dim"), coords={"obs_dim": range(n_obs)}
    )
    log_weights_da = xr.DataArray(
        log_weights, dims=("__sample__", "obs_dim"), coords={"obs_dim": range(n_obs)}
    )

    mean_value = _compute_weighted_mean(x_da, log_weights_da)
    pareto_k = compute_pareto_k(x_da, log_weights_da)
    result_mean = ExpectationResult(value=mean_value, pareto_k=pareto_k)

    assert isinstance(result_mean, ExpectationResult)
    assert_finite(result_mean.value)

    var_value = _compute_weighted_variance(x_da, log_weights_da)
    result_var = ExpectationResult(value=var_value, pareto_k=pareto_k)

    assert isinstance(result_var, ExpectationResult)
    assert_finite(result_var.value)

    sd_value = _compute_weighted_sd(x_da, log_weights_da)
    result_sd = ExpectationResult(value=sd_value, pareto_k=pareto_k)

    assert isinstance(result_sd, ExpectationResult)
    assert_finite(result_sd.value)

    probs = [0.25, 0.5, 0.75]
    quant_value = _compute_weighted_quantiles(x_da, log_weights_da, np.array(probs))
    result_quant = ExpectationResult(value=quant_value, pareto_k=pareto_k)

    assert isinstance(result_quant, ExpectationResult)
    assert_finite(result_quant.value)
