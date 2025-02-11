"""Tests for the LOO-CV subsampling module."""
import time
from copy import deepcopy

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_array_almost_equal

from pyloo.loo import loo
from pyloo.loo_subsample import EstimatorMethod, LooApproximationMethod, loo_subsample

from ..helpers import create_large_model


@pytest.fixture(scope="session")
def large_model():
    """Create a large model for testing subsampling."""
    return create_large_model(n_obs=10000)


def test_loo_subsample_performance(large_model):
    """Test that subsampling is faster than full LOO for large datasets."""
    start_time = time.time()
    full_loo = loo(large_model)
    full_time = time.time() - start_time

    start_time = time.time()
    sub_loo = loo_subsample(large_model, observations=1000)
    sub_time = time.time() - start_time

    assert sub_time < full_time / 2
    assert np.abs(sub_loo["elpd_loo"] - full_loo["elpd_loo"]) < 3 * sub_loo["se"]


@pytest.mark.parametrize("method", [m.value for m in LooApproximationMethod])
def test_loo_subsample_approximations(large_model, method):
    """Test different LOO approximation methods."""
    result = loo_subsample(
        large_model,
        observations=1000,
        loo_approximation=method,
    )
    assert result is not None
    assert "elpd_loo" in result
    assert result["p_loo"] >= 0


@pytest.mark.parametrize("estimator", [m.value for m in EstimatorMethod])
def test_loo_subsample_estimators(large_model, estimator):
    """Test different estimator methods."""
    result = loo_subsample(
        large_model,
        observations=1000,
        estimator=estimator,
    )
    assert result is not None
    assert "elpd_loo" in result
    assert hasattr(result.estimates, "subsampling_SE")


def test_loo_subsample_pointwise(large_model):
    """Test LOO subsampling with pointwise=True."""
    result = loo_subsample(large_model, observations=1000, pointwise=True)
    assert result is not None
    assert "loo_i" in result
    assert "pareto_k" in result

    # Check that non-sampled points are NaN
    assert np.any(np.isnan(result["loo_i"]))
    assert np.sum(~np.isnan(result["loo_i"])) == 1000


def test_loo_subsample_observations_validation(large_model):
    """Test validation of observations parameter."""
    n_obs = large_model.log_likelihood["obs"].shape[2]

    with pytest.raises(ValueError):
        loo_subsample(large_model, observations=n_obs + 1)

    with pytest.raises(ValueError):
        loo_subsample(large_model, observations=0)

    with pytest.raises(ValueError):
        loo_subsample(large_model, observations=np.array([n_obs + 1]))

    with pytest.raises(ValueError):
        loo_subsample(large_model, observations=np.array([-1]))


def test_loo_subsample_approximation_draws(large_model):
    """Test subsampling with different numbers of approximation draws."""
    n_draws = large_model.posterior.dims["draw"]

    result = loo_subsample(
        large_model,
        observations=1000,
        loo_approximation_draws=n_draws // 2,
    )
    assert result is not None

    with pytest.raises(ValueError):
        loo_subsample(
            large_model,
            observations=1000,
            loo_approximation_draws=n_draws * len(large_model.posterior.chain) + 1,
        )


def test_loo_subsample_nan_handling(large_model):
    """Test LOO subsampling with NaN values."""
    large_model = deepcopy(large_model)
    log_like = large_model.log_likelihood["obs"].values
    log_like[0, 0, 0] = np.nan

    large_model.log_likelihood["obs"] = xr.DataArray(
        log_like,
        dims=large_model.log_likelihood["obs"].dims,
        coords=large_model.log_likelihood["obs"].coords,
    )

    with pytest.warns(UserWarning):
        result = loo_subsample(large_model, observations=1000)
        assert result is not None
        assert not np.isnan(result["elpd_loo"])


def test_loo_subsample_inf_handling(large_model):
    """Test LOO subsampling with infinite values."""
    large_model = deepcopy(large_model)
    log_like = large_model.log_likelihood["obs"].values
    log_like[0, 0, 0] = np.inf

    large_model.log_likelihood["obs"] = xr.DataArray(
        log_like,
        dims=large_model.log_likelihood["obs"].dims,
        coords=large_model.log_likelihood["obs"].coords,
    )

    with pytest.warns(UserWarning):
        result = loo_subsample(large_model, observations=1000)
        assert result is not None
        assert not np.isinf(result["elpd_loo"])


def test_loo_subsample_warning(large_model):
    """Test warning for high Pareto k values in subsampling."""
    large_model = deepcopy(large_model)
    # Make one observation extremely influential by setting a very large log-likelihood
    # This creates a highly influential point that should trigger high Pareto k values
    log_like = large_model.log_likelihood["obs"].values
    # Set observation 1 to have extremely large log-likelihood values
    log_like[:, :, 1] = 10000.0  # Much larger value to ensure high Pareto k
    # Also make nearby values very different to create instability
    log_like[:, :, 0] = -10000.0
    log_like[:, :, 2] = -10000.0

    large_model.log_likelihood["obs"] = xr.DataArray(
        log_like,
        dims=large_model.log_likelihood["obs"].dims,
        coords=large_model.log_likelihood["obs"].coords,
    )

    with pytest.warns(UserWarning):
        result = loo_subsample(large_model, observations=1000, pointwise=True)
        assert result is not None
        assert any(k > result["good_k"] for k in result["pareto_k"][~np.isnan(result["pareto_k"])])


def test_loo_subsample_multiple_groups(large_model):
    """Test LOO subsampling with multiple log_likelihood groups."""
    large_model = deepcopy(large_model)
    large_model.log_likelihood["obs2"] = large_model.log_likelihood["obs"]

    with pytest.raises(TypeError):
        loo_subsample(large_model, observations=1000)

    result = loo_subsample(large_model, observations=1000, var_name="obs")
    assert result is not None


def test_loo_subsample_consistency(large_model):
    """Test consistency of LOO subsampling with different sample sizes."""
    n_obs = [500, 1000, 2000]
    results = []

    for n in n_obs:
        result = loo_subsample(large_model, observations=n)
        results.append(result["elpd_loo"])

    ses = []
    for n in n_obs:
        result = loo_subsample(large_model, observations=n)
        ses.append(result["se"])

    assert ses[0] > ses[-1], "Standard error should decrease with more observations"


def test_loo_subsample_exact_indices(large_model):
    """Test LOO subsampling with exact indices provided."""
    indices = np.array([0, 100, 200, 300])
    result = loo_subsample(large_model, observations=indices, pointwise=True)
    assert result is not None

    pointwise = result["loo_i"]
    non_nan_idx = np.where(~np.isnan(pointwise))[0]
    assert_array_almost_equal(non_nan_idx, indices)


def test_loo_subsample_default_parameters(large_model):
    """Test that default parameters produce valid results for large datasets."""
    result = loo_subsample(large_model, pointwise=True)

    # Check that Pareto k values are reasonable
    pareto_k = result["pareto_k"][~np.isnan(result["pareto_k"])]
    assert np.all(pareto_k <= result["good_k"]), "All Pareto k values should be below good_k threshold"
    assert result["se"] < np.abs(result["elpd_loo"]) * 0.1, "Standard error should be less than 10% of elpd_loo"

    # Compare with full LOO
    full_loo = loo(large_model)
    rel_diff = np.abs(result["elpd_loo"] - full_loo["elpd_loo"]) / np.abs(full_loo["elpd_loo"])
    assert rel_diff < 0.1, "Subsampled LOO should be within 10% of full LOO"
