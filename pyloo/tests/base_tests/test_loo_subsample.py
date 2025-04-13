"""Tests for the LOO-CV subsampling module."""

import logging
import time
from copy import deepcopy

import numpy as np
import pytest
import xarray as xr

from ...loo import loo
from ...loo_approximate_posterior import loo_approximate_posterior
from ...loo_subsample import (
    EstimatorMethod,
    LooApproximationMethod,
    loo_subsample,
    update_subsample,
)
from ...wrapper.pymc.laplace import Laplace
from ..helpers import (
    assert_array_almost_equal,
    assert_arrays_allclose,
    create_large_model,
)

logger = logging.getLogger(__name__)


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

    rel_diff = np.abs(sub_loo["elpd_loo"] - full_loo["elpd_loo"]) / np.abs(
        full_loo["elpd_loo"]
    )
    assert rel_diff < 0.1, "Subsampled LOO should be within 10% of full LOO"


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
        ses.append(result["subsampling_SE"])

    assert (
        ses[0] > ses[-1]
    ), "Subsampling standard error should decrease with more observations"


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

    pareto_k = result["pareto_k"][~np.isnan(result["pareto_k"])]
    assert np.all(
        pareto_k <= result["good_k"]
    ), "All Pareto k values should be below good_k threshold"

    full_loo = loo(large_model)
    rel_diff = np.abs(result["elpd_loo"] - full_loo["elpd_loo"]) / np.abs(
        full_loo["elpd_loo"]
    )
    assert rel_diff < 0.1, "Subsampled LOO should be within 10% of full LOO"


def test_update_subsample_basic(large_model):
    """Test basic update functionality."""
    result = loo_subsample(large_model, observations=1000)
    updated = update_subsample(result, observations=2000)

    assert updated is not None
    assert updated["subsample_size"] == 2000
    assert updated["subsampling_SE"] <= result["subsampling_SE"]


def test_update_subsample_validation():
    """Test validation in update_subsample."""
    with pytest.raises(TypeError, match="must be an ELPDData object"):
        update_subsample(None, observations=1000)

    with pytest.raises(TypeError, match="must be an ELPDData object"):
        update_subsample({}, observations=1000)


def test_update_subsample_consistency(large_model):
    """Test consistency of results after update."""
    result = loo_subsample(large_model, observations=1000)
    updated = update_subsample(result, observations=1000)
    rel_diff = np.abs(updated["elpd_loo"] - result["elpd_loo"]) / np.abs(
        result["elpd_loo"]
    )
    rel_diff = np.abs(updated["elpd_loo"] - result["elpd_loo"]) / np.abs(
        result["elpd_loo"]
    )
    assert (
        rel_diff < 0.1
    ), "Updated results should be similar to original with same observations"


def test_update_subsample_parameter_inheritance(large_model):
    """Test that update inherits parameters correctly."""
    result = loo_subsample(
        large_model,
        observations=1000,
        loo_approximation="plpd",
        estimator="diff_srs",
        pointwise=True,
    )

    updated = update_subsample(result, observations=2000)

    assert hasattr(updated.estimates, "loo_approximation")
    assert updated.estimates.loo_approximation == "plpd"
    assert hasattr(updated.estimates, "estimator")
    assert updated.estimates.estimator == "diff_srs"
    assert "loo_i" in updated  # pointwise was True


def test_update_subsample_parameter_override(large_model):
    """Test that update allows parameter overrides."""
    result = loo_subsample(
        large_model,
        observations=1000,
        loo_approximation="plpd",
        estimator="diff_srs",
    )
    updated = update_subsample(
        result,
        observations=2000,
        loo_approximation="lpd",
        estimator="srs",
    )

    assert hasattr(updated.estimates, "loo_approximation")
    assert updated.estimates.loo_approximation == "lpd"
    assert hasattr(updated.estimates, "estimator")
    assert updated.estimates.estimator == "srs"


def test_update_subsample_exact_indices(large_model):
    """Test update with exact indices."""
    initial_indices = np.array([0, 100, 200, 300])
    result = loo_subsample(large_model, observations=initial_indices, pointwise=True)

    new_indices = np.array([0, 100, 200, 300, 400, 500])
    updated = update_subsample(result, observations=new_indices, pointwise=True)

    non_nan_idx = np.where(~np.isnan(updated["loo_i"]))[0]
    assert_array_almost_equal(non_nan_idx, new_indices)


def test_loo_subsample_with_posterior_correction(simple_model):
    """Test loo_subsample with posterior correction."""
    model, _ = simple_model

    wrapper = Laplace(model)
    laplace_result = wrapper.fit()

    log_p = wrapper.compute_logp().flatten()
    log_q = wrapper.compute_logq().flatten()

    n_obs = laplace_result.idata.log_likelihood["y"].shape[2]
    subsample_size = max(2, n_obs // 2)

    reg_loo_subsample = loo_subsample(
        laplace_result.idata,
        observations=subsample_size,
        pointwise=True,
    )

    result = loo_subsample(
        laplace_result.idata,
        observations=subsample_size,
        log_p=log_p,
        log_q=log_q,
        pointwise=True,
    )

    loo_approx = loo_approximate_posterior(
        laplace_result.idata, log_p, log_q, pointwise=True
    )

    assert result is not None
    assert "elpd_loo" in result
    assert "loo_i" in result
    assert hasattr(result, "log_p")
    assert hasattr(result, "log_q")
    assert_arrays_allclose(result.log_p, log_p)
    assert_arrays_allclose(result.log_q, log_q)

    logger.info(result)
    logger.info(loo_approx)
    logger.info(reg_loo_subsample)


def test_update_subsample_with_posterior_correction(simple_model):
    """Test update_subsample with posterior correction."""
    model, _ = simple_model

    wrapper = Laplace(model)
    laplace_result = wrapper.fit()

    log_p = wrapper.compute_logp().flatten()
    log_q = wrapper.compute_logq().flatten()

    n_obs = laplace_result.idata.log_likelihood["y"].shape[2]
    initial_size = max(2, n_obs // 3)

    initial_result = loo_subsample(
        laplace_result.idata,
        observations=initial_size,
        log_p=log_p,
        log_q=log_q,
    )

    updated_size = max(3, n_obs // 2)
    updated_result = update_subsample(initial_result, observations=updated_size)

    assert updated_result is not None
    assert updated_result["subsample_size"] == updated_size
    assert hasattr(updated_result, "log_p")
    assert hasattr(updated_result, "log_q")
    assert_arrays_allclose(updated_result.log_p, log_p)
    assert_arrays_allclose(updated_result.log_q, log_q)

    assert updated_result["subsampling_SE"] <= initial_result["subsampling_SE"]

    logger.info(initial_result)
    logger.info(updated_result)
