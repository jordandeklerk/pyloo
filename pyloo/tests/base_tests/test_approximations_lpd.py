"""Tests for Log Predictive Density (LPD) approximation."""

import numpy as np
import xarray as xr

from ...approximations.lpd import LPDApproximation


def test_lpd_approximation_basic(log_likelihood_data):
    approx = LPDApproximation()

    result = approx.compute_approximation(log_likelihood_data)

    assert isinstance(result, np.ndarray)
    assert result.shape == (8,)
    assert np.all(np.isfinite(result))


def test_lpd_approximation_with_n_draws(log_likelihood_data):
    approx = LPDApproximation()

    n_samples = log_likelihood_data.sizes["__sample__"]
    n_draws = n_samples // 2

    result_full = approx.compute_approximation(log_likelihood_data)
    result_subset = approx.compute_approximation(log_likelihood_data, n_draws=n_draws)

    assert isinstance(result_subset, np.ndarray)
    assert result_subset.shape == (8,)
    assert np.all(np.isfinite(result_subset))
    assert not np.allclose(result_full, result_subset)


def test_lpd_approximation_with_extreme_values(log_likelihood_data):
    approx = LPDApproximation()

    log_likelihood_extreme = log_likelihood_data.copy(deep=True)
    log_likelihood_extreme.values[0, 0] = 1e10
    log_likelihood_extreme.values[0, 1] = -1e10

    result = approx.compute_approximation(log_likelihood_extreme)
    assert isinstance(result, np.ndarray)
    assert result.shape == (8,)
    assert np.all(np.isfinite(result))


def test_lpd_approximation_with_constant_values(log_likelihood_data):
    approx = LPDApproximation()

    log_likelihood_constant = log_likelihood_data.copy(deep=True)
    log_likelihood_constant.values[0, :] = 1.0

    result = approx.compute_approximation(log_likelihood_constant)
    assert isinstance(result, np.ndarray)
    assert result.shape == (8,)
    assert np.all(np.isfinite(result))


def test_lpd_approximation_formula():
    approx = LPDApproximation()

    n_obs = 2
    n_samples = 3
    log_likelihood = xr.DataArray(
        np.array([
            [0.0, -1.0, -2.0],
            [-3.0, -4.0, -5.0],
        ]),
        dims=["obs_id", "__sample__"],
        coords={"obs_id": range(n_obs), "__sample__": range(n_samples)},
    )

    result = approx.compute_approximation(log_likelihood)

    expected_1 = np.log(np.mean(np.exp([0.0, -1.0, -2.0])))
    expected_2 = np.log(np.mean(np.exp([-3.0, -4.0, -5.0])))

    assert np.isclose(result[0], expected_1)
    assert np.isclose(result[1], expected_2)


def test_lpd_approximation_with_multidimensional_data(multidim_data):
    approx = LPDApproximation()

    log_likelihood = xr.DataArray(
        multidim_data["llm"],
        dims=["chain", "draw", "dim1", "dim2"],
        coords={
            "chain": range(multidim_data["llm"].shape[0]),
            "draw": range(multidim_data["llm"].shape[1]),
            "dim1": range(multidim_data["llm"].shape[2]),
            "dim2": range(multidim_data["llm"].shape[3]),
        },
    )

    log_likelihood = log_likelihood.stack(__sample__=("chain", "draw"))

    result = approx.compute_approximation(log_likelihood)

    assert isinstance(result, np.ndarray)
    assert result.shape == (15, 2)
    assert np.all(np.isfinite(result))


def test_lpd_approximation_with_extreme_data(extreme_data):
    log_likelihood = xr.DataArray(
        extreme_data.T,
        dims=["obs_id", "__sample__"],
        coords={
            "obs_id": range(extreme_data.shape[1]),
            "__sample__": range(extreme_data.shape[0]),
        },
    )

    approx = LPDApproximation()
    result = approx.compute_approximation(log_likelihood)

    assert isinstance(result, np.ndarray)
    assert result.shape == (extreme_data.shape[1],)
    assert np.all(np.isfinite(result))


def test_lpd_approximation_with_centered_eight(centered_eight):
    log_likelihood = centered_eight.log_likelihood.obs.stack(
        __sample__=("chain", "draw")
    )

    approx = LPDApproximation()
    result = approx.compute_approximation(log_likelihood)

    assert isinstance(result, np.ndarray)
    assert result.shape == (8,)
    assert np.all(np.isfinite(result))
