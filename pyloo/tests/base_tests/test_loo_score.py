"""Tests for the loo_score module."""

import logging
import warnings

import numpy as np
import pytest
import xarray as xr
from arviz import InferenceData

from ...loo_score import (
    EXX_loo_compute,
    LooScoreResult,
    _crps,
    _get_data,
    loo_score,
    validate_crps_input,
)
from ..helpers import assert_arrays_allclose, assert_finite, assert_positive


def test_loo_score_basic(prepare_inference_data_for_crps):
    """Test basic functionality of loo_score."""
    idata = prepare_inference_data_for_crps

    result = loo_score(
        idata,
        x_group="posterior_predictive",
        x_var="obs",
        x2_group="posterior_predictive",
        x2_var="obs2",
        y_group="observed_data",
        y_var="obs",
    )

    logging.info(result)

    assert result is not None
    assert isinstance(result, LooScoreResult)
    assert hasattr(result, "estimates")
    assert hasattr(result, "pointwise")
    assert len(result.pointwise) == 8


def test_loo_score_scaled(prepare_inference_data_for_crps):
    """Test loo_score with scale=True (LOO-SCRPS)."""
    idata = prepare_inference_data_for_crps

    result = loo_score(
        idata,
        x_group="posterior_predictive",
        x_var="obs",
        x2_group="posterior_predictive",
        x2_var="obs2",
        y_group="observed_data",
        y_var="obs",
        scale=True,
    )

    logging.info(result)

    assert result is not None
    assert isinstance(result, LooScoreResult)
    assert hasattr(result, "estimates")
    assert hasattr(result, "pointwise")
    assert len(result.pointwise) == 8


def test_loo_score_pointwise(prepare_inference_data_for_crps):
    """Test loo_score with pointwise=True."""
    idata = prepare_inference_data_for_crps

    result = loo_score(
        idata,
        x_group="posterior_predictive",
        x_var="obs",
        x2_group="posterior_predictive",
        x2_var="obs2",
        y_group="observed_data",
        y_var="obs",
        pointwise=True,
    )

    assert result is not None
    assert isinstance(result, LooScoreResult)
    assert hasattr(result, "estimates")
    assert hasattr(result, "pointwise")
    assert hasattr(result, "pareto_k")
    assert hasattr(result, "good_k")
    assert len(result.pointwise) == 8
    assert len(result.pareto_k) == 8


def test_loo_score_with_reff(prepare_inference_data_for_crps):
    """Test loo_score with specified reff."""
    idata = prepare_inference_data_for_crps

    result = loo_score(
        idata,
        x_group="posterior_predictive",
        x_var="obs",
        x2_group="posterior_predictive",
        x2_var="obs2",
        y_group="observed_data",
        y_var="obs",
        reff=0.8,
    )

    assert result is not None
    assert isinstance(result, LooScoreResult)
    assert hasattr(result, "estimates")
    assert hasattr(result, "pointwise")


def test_loo_score_permutations(prepare_inference_data_for_crps):
    """Test loo_score with multiple permutations."""
    idata = prepare_inference_data_for_crps

    result = loo_score(
        idata,
        x_group="posterior_predictive",
        x_var="obs",
        x2_group="posterior_predictive",
        x2_var="obs2",
        y_group="observed_data",
        y_var="obs",
        permutations=5,
    )

    assert result is not None
    assert isinstance(result, LooScoreResult)
    assert hasattr(result, "estimates")
    assert hasattr(result, "pointwise")


def test_loo_score_missing_posterior(prepare_inference_data_for_crps):
    """Test loo_score with missing posterior group."""
    idata = prepare_inference_data_for_crps

    idata_no_posterior = InferenceData(
        posterior_predictive=idata.posterior_predictive,
        log_likelihood=idata.log_likelihood,
        observed_data=idata.observed_data,
    )

    with pytest.raises(TypeError, match="Must be able to extract a posterior group"):
        loo_score(
            idata_no_posterior,
            x_group="posterior_predictive",
            x_var="obs",
            x2_group="posterior_predictive",
            x2_var="obs2",
            y_group="observed_data",
            y_var="obs",
            reff=None,
        )

    result = loo_score(
        idata_no_posterior,
        x_group="posterior_predictive",
        x_var="obs",
        x2_group="posterior_predictive",
        x2_var="obs2",
        y_group="observed_data",
        y_var="obs",
        reff=0.8,
    )

    assert result is not None
    assert isinstance(result, LooScoreResult)
    assert hasattr(result, "estimates")
    assert hasattr(result, "pointwise")


def test_loo_score_missing_groups(centered_eight):
    """Test loo_score with missing required groups."""
    with pytest.raises(
        ValueError, match="Variable 'obs2' not found in posterior_predictive group"
    ):
        loo_score(
            centered_eight,
            x_group="posterior_predictive",
            x_var="obs",
            x2_group="posterior_predictive",
            x2_var="obs2",
            y_group="observed_data",
            y_var="obs",
        )

    idata_no_obs = InferenceData(
        posterior=centered_eight.posterior,
        posterior_predictive=centered_eight.posterior_predictive,
        log_likelihood=centered_eight.log_likelihood,
    )

    with pytest.raises(ValueError, match="does not have a observed_data group"):
        loo_score(
            idata_no_obs,
            x_group="posterior_predictive",
            x_var="obs",
            x2_group="posterior_predictive",
            x2_var="obs",
            y_group="observed_data",
            y_var="obs",
        )


def test_loo_score_missing_variables(prepare_inference_data_for_crps):
    """Test loo_score with missing variables."""
    idata = prepare_inference_data_for_crps

    with pytest.raises(ValueError, match="Variable 'nonexistent' not found"):
        loo_score(
            idata,
            x_group="posterior_predictive",
            x_var="nonexistent",
            x2_group="posterior_predictive",
            x2_var="obs2",
            y_group="observed_data",
            y_var="obs",
        )

    with pytest.raises(ValueError, match="Variable 'nonexistent' not found"):
        loo_score(
            idata,
            x_group="posterior_predictive",
            x_var="obs",
            x2_group="posterior_predictive",
            x2_var="nonexistent",
            y_group="observed_data",
            y_var="obs",
        )

    with pytest.raises(ValueError, match="Variable 'nonexistent' not found"):
        loo_score(
            idata,
            x_group="posterior_predictive",
            x_var="obs",
            x2_group="posterior_predictive",
            x2_var="obs2",
            y_group="observed_data",
            y_var="nonexistent",
        )


def test_loo_score_warning_high_k(prepare_inference_data_for_crps):
    """Test loo_score warning for high Pareto k values."""
    idata = prepare_inference_data_for_crps

    log_lik = idata.log_likelihood["obs"].values.copy()
    log_lik[:, :, 0] = 10

    ll_data = xr.Dataset(
        {"obs": (["chain", "draw", "school"], log_lik)},
        coords=idata.log_likelihood.coords,
    )

    modified_idata = InferenceData(
        posterior=idata.posterior,
        posterior_predictive=idata.posterior_predictive,
        log_likelihood=ll_data,
        observed_data=idata.observed_data,
    )

    with warnings.catch_warnings(record=True) as w:
        result = loo_score(
            modified_idata,
            x_group="posterior_predictive",
            x_var="obs",
            x2_group="posterior_predictive",
            x2_var="obs2",
            y_group="observed_data",
            y_var="obs",
            pointwise=True,
        )

        assert any(
            "shape parameter of Pareto distribution" in str(msg.message) for msg in w
        )
        assert result.warning is True
        assert np.any(result.pareto_k > result.good_k)


def test_validate_crps_input():
    """Test validate_crps_input function."""
    n_samples = 100
    n_obs = 5

    x = xr.DataArray(
        np.random.randn(n_samples, n_obs),
        dims=("__sample__", "obs_id"),
        coords={"obs_id": range(n_obs)},
    )

    x2 = xr.DataArray(
        np.random.randn(n_samples, n_obs),
        dims=("__sample__", "obs_id"),
        coords={"obs_id": range(n_obs)},
    )

    y = xr.DataArray(
        np.random.randn(n_obs),
        dims=("obs_id",),
        coords={"obs_id": range(n_obs)},
    )

    log_lik = xr.DataArray(
        np.random.randn(n_samples, n_obs),
        dims=("__sample__", "obs_id"),
        coords={"obs_id": range(n_obs)},
    )

    validate_crps_input(x, x2, y, log_lik)

    x2_wrong_dims = xr.DataArray(
        np.random.randn(n_samples, n_obs),
        dims=("__sample__", "wrong_dim"),
        coords={"wrong_dim": range(n_obs)},
    )

    with pytest.raises(ValueError, match="x and x2 must have the same dimensions"):
        validate_crps_input(x, x2_wrong_dims, y, log_lik)

    x2_wrong_shape = xr.DataArray(
        np.random.randn(n_samples, n_obs - 1),
        dims=("__sample__", "obs_id"),
        coords={"obs_id": range(n_obs - 1)},
    )

    with pytest.raises(ValueError, match="x and x2 must have the same shape"):
        validate_crps_input(x, x2_wrong_shape, y, log_lik)

    y_wrong_dims = xr.DataArray(
        np.random.randn(n_obs),
        dims=("wrong_dim",),
        coords={"wrong_dim": range(n_obs)},
    )

    with pytest.raises(ValueError, match="y dimensions .* are not compatible"):
        validate_crps_input(x, x2, y_wrong_dims, log_lik)

    log_lik_wrong_dims = xr.DataArray(
        np.random.randn(n_samples, n_obs),
        dims=("__sample__", "wrong_dim"),
        coords={"wrong_dim": range(n_obs)},
    )

    with pytest.raises(ValueError, match="log_lik dimensions .* are not compatible"):
        validate_crps_input(x, x2, y, log_lik_wrong_dims)

    log_lik_no_sample = xr.DataArray(
        np.random.randn(n_obs),
        dims=("obs_id",),
        coords={"obs_id": range(n_obs)},
    )

    with pytest.raises(ValueError, match="log_lik must have '__sample__' dimension"):
        validate_crps_input(x, x2, y, log_lik_no_sample)


def test_get_data(prepare_inference_data_for_crps):
    """Test _get_data function with real model data."""
    idata = prepare_inference_data_for_crps

    x_data, x2_data, y_data, log_lik = _get_data(
        idata,
        x_group="posterior_predictive",
        x_var="obs",
        x2_group="posterior_predictive",
        x2_var="obs2",
        y_group="observed_data",
        y_var="obs",
        log_likelihood=idata.log_likelihood.obs,
    )

    assert isinstance(x_data, xr.DataArray)
    assert isinstance(x2_data, xr.DataArray)
    assert isinstance(y_data, xr.DataArray)
    assert isinstance(log_lik, xr.DataArray)

    assert "__sample__" in x_data.dims
    assert "__sample__" in x2_data.dims
    assert "school" in x_data.dims
    assert "school" in x2_data.dims
    assert "school" in y_data.dims

    with pytest.raises(ValueError, match="does not have a nonexistent group"):
        _get_data(
            idata,
            x_group="nonexistent",
            x_var="obs",
            x2_group="posterior_predictive",
            x2_var="obs2",
            y_group="observed_data",
            y_var="obs",
        )

    with pytest.raises(ValueError, match="does not have a nonexistent group"):
        _get_data(
            idata,
            x_group="posterior_predictive",
            x_var="obs",
            x2_group="nonexistent",
            x2_var="obs",
            y_group="observed_data",
            y_var="obs",
        )

    with pytest.raises(ValueError, match="does not have a nonexistent group"):
        _get_data(
            idata,
            x_group="posterior_predictive",
            x_var="obs",
            x2_group="posterior_predictive",
            x2_var="obs2",
            y_group="nonexistent",
            y_var="obs",
        )

    with pytest.raises(ValueError, match="Variable 'nonexistent' not found"):
        _get_data(
            idata,
            x_group="posterior_predictive",
            x_var="nonexistent",
            x2_group="posterior_predictive",
            x2_var="obs2",
            y_group="observed_data",
            y_var="obs",
        )

    with pytest.raises(ValueError, match="Variable 'nonexistent' not found"):
        _get_data(
            idata,
            x_group="posterior_predictive",
            x_var="obs",
            x2_group="posterior_predictive",
            x2_var="nonexistent",
            y_group="observed_data",
            y_var="obs",
        )

    with pytest.raises(ValueError, match="Variable 'nonexistent' not found"):
        _get_data(
            idata,
            x_group="posterior_predictive",
            x_var="obs",
            x2_group="posterior_predictive",
            x2_var="obs2",
            y_group="observed_data",
            y_var="nonexistent",
        )


def test_EXX_loo_compute(prepare_inference_data_for_crps):
    """Test EXX_loo_compute function with real model data."""
    idata = prepare_inference_data_for_crps

    x_data = idata.posterior_predictive.obs.stack(__sample__=("chain", "draw"))
    x2_data = idata.posterior_predictive.obs2.stack(__sample__=("chain", "draw"))
    log_lik = idata.log_likelihood.obs.stack(__sample__=("chain", "draw"))

    result = EXX_loo_compute(x_data, x2_data, log_lik, r_eff=0.8)

    assert isinstance(result, xr.DataArray)
    assert result.dims == ("school",)
    assert result.shape == (8,)
    assert_finite(result)
    assert_positive(result)


def test_crps_fun():
    """Test _crps_fun function."""
    n_obs = 8

    rng = np.random.default_rng(42)

    EXX = xr.DataArray(
        rng.uniform(0.5, 1.5, size=n_obs),
        dims=("school",),
        coords={"school": [f"school_{i}" for i in range(n_obs)]},
    )

    EXy = xr.DataArray(
        rng.uniform(0.1, 0.5, size=n_obs),
        dims=("school",),
        coords={"school": [f"school_{i}" for i in range(n_obs)]},
    )

    crps = _crps(EXX, EXy, scale=False)

    assert isinstance(crps, xr.DataArray)
    assert crps.dims == ("school",)
    assert crps.shape == (n_obs,)
    assert_finite(crps)

    expected_crps = 0.5 * EXX - EXy
    assert_arrays_allclose(crps, expected_crps)

    scrps = _crps(EXX, EXy, scale=True)

    assert isinstance(scrps, xr.DataArray)
    assert scrps.dims == ("school",)
    assert scrps.shape == (n_obs,)
    assert_finite(scrps)

    expected_scrps = -EXy / EXX - 0.5 * np.log(EXX)
    assert_arrays_allclose(scrps, expected_scrps)


def test_loo_score_nan_handling(prepare_inference_data_for_crps):
    """Test loo_score with NaN values."""
    idata = prepare_inference_data_for_crps

    pp_data = idata.posterior_predictive.copy()
    pp_values = pp_data.obs.values.copy()
    pp_values[0, 0, 0] = np.nan
    pp_data["obs"] = (pp_data.obs.dims, pp_values)

    modified_idata = InferenceData(
        posterior=idata.posterior,
        posterior_predictive=pp_data,
        log_likelihood=idata.log_likelihood,
        observed_data=idata.observed_data,
    )

    with warnings.catch_warnings(record=True) as w:
        result = loo_score(
            modified_idata,
            x_group="posterior_predictive",
            x_var="obs",
            x2_group="posterior_predictive",
            x2_var="obs2",
            y_group="observed_data",
            y_var="obs",
        )

        assert any("NaN values detected" in str(msg.message) for msg in w)
        assert result is not None
        assert isinstance(result, LooScoreResult)
        assert hasattr(result, "estimates")
        assert not np.isnan(result.estimates["Estimate"])


def test_loo_score_inf_handling(prepare_inference_data_for_crps):
    """Test loo_score with infinite values."""
    idata = prepare_inference_data_for_crps

    pp_data = idata.posterior_predictive.copy()
    pp_values = pp_data.obs.values.copy()
    pp_values[0, 0, 1] = np.inf
    pp_data["obs"] = (pp_data.obs.dims, pp_values)

    modified_idata = InferenceData(
        posterior=idata.posterior,
        posterior_predictive=pp_data,
        log_likelihood=idata.log_likelihood,
        observed_data=idata.observed_data,
    )

    with warnings.catch_warnings(record=True) as w:
        result = loo_score(
            modified_idata,
            x_group="posterior_predictive",
            x_var="obs",
            x2_group="posterior_predictive",
            x2_var="obs2",
            y_group="observed_data",
            y_var="obs",
        )

        assert any("Infinite values detected" in str(msg.message) for msg in w)
        assert result is not None
        assert isinstance(result, LooScoreResult)
        assert hasattr(result, "estimates")
        assert not np.isinf(result.estimates["Estimate"])


def test_loo_score_with_var_name(prepare_inference_data_for_crps):
    """Test loo_score with var_name parameter for log_likelihood."""
    idata = prepare_inference_data_for_crps

    ll_data = idata.log_likelihood.copy()
    ll_data["obs2"] = ll_data["obs"] * 0.9

    modified_idata = InferenceData(
        posterior=idata.posterior,
        posterior_predictive=idata.posterior_predictive,
        log_likelihood=ll_data,
        observed_data=idata.observed_data,
    )

    with pytest.raises(TypeError, match="Found several log likelihood arrays"):
        loo_score(
            modified_idata,
            x_group="posterior_predictive",
            x_var="obs",
            x2_group="posterior_predictive",
            x2_var="obs2",
            y_group="observed_data",
            y_var="obs",
        )

    result = loo_score(
        modified_idata,
        x_group="posterior_predictive",
        x_var="obs",
        x2_group="posterior_predictive",
        x2_var="obs2",
        y_group="observed_data",
        y_var="obs",
        var_name="obs",
    )

    assert result is not None
    assert isinstance(result, LooScoreResult)
    assert hasattr(result, "estimates")
    assert hasattr(result, "pointwise")
