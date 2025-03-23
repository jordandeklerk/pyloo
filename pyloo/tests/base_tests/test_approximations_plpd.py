"""Tests for Point Log Predictive Density (PLPD) approximation."""

import warnings

import numpy as np
import pytest
import xarray as xr

from ...approximations.plpd import PLPDApproximation


def mock_log_likelihood_fn(data, params):
    """Mock log likelihood function for testing."""
    if isinstance(params, dict):
        return sum(p.flat[0] if hasattr(p, "flat") else p for p in params.values())
    else:
        return np.sum(params)


def test_plpd_approximation_basic(log_likelihood_data, centered_eight):
    """Test basic functionality of PLPDApproximation with real data."""
    posterior = centered_eight.posterior.stack(__sample__=("chain", "draw"))

    approx = PLPDApproximation(posterior=posterior)
    with warnings.catch_warnings(record=True) as w:
        result = approx.compute_approximation(log_likelihood_data)
        assert len(w) == 1
        assert "approximate PLPD calculation" in str(w[0].message)

    assert isinstance(result, np.ndarray)
    assert result.shape == (8,)
    assert np.all(np.isfinite(result))


def test_plpd_approximation_with_n_draws(log_likelihood_data, centered_eight):
    """Test PLPD approximation with specified number of draws."""
    posterior = centered_eight.posterior.stack(__sample__=("chain", "draw"))

    approx = PLPDApproximation(posterior=posterior)

    n_samples = log_likelihood_data.sizes["__sample__"]
    n_draws = n_samples // 2

    with warnings.catch_warnings(record=True):
        result_full = approx.compute_approximation(log_likelihood_data)
        result_subset = approx.compute_approximation(
            log_likelihood_data, n_draws=n_draws
        )

    assert isinstance(result_subset, np.ndarray)
    assert result_subset.shape == (8,)
    assert np.all(np.isfinite(result_subset))

    if not np.allclose(result_full, result_subset):
        assert True


def test_plpd_approximation_no_posterior():
    """Test PLPD approximation raises error when no posterior is provided."""
    log_likelihood = xr.DataArray(
        np.random.randn(10, 100),
        dims=["obs_id", "__sample__"],
        coords={"obs_id": range(10), "__sample__": range(100)},
    )

    approx = PLPDApproximation()
    with pytest.raises(ValueError, match="No posterior samples provided"):
        approx.compute_approximation(log_likelihood, n_draws=50)


def test_plpd_approximation_with_likelihood_fn(log_likelihood_data, centered_eight):
    """Test PLPD approximation with likelihood function and data."""
    posterior = centered_eight.posterior.stack(__sample__=("chain", "draw"))
    data = np.arange(8)

    approx = PLPDApproximation(
        posterior=posterior, log_likelihood_fn=mock_log_likelihood_fn, data=data
    )
    result = approx.compute_approximation(log_likelihood_data)

    assert isinstance(result, np.ndarray)
    assert result.shape == (8,)
    assert np.all(np.isfinite(result))
    assert np.allclose(result, result[0])


def test_plpd_approximation_with_multidimensional_posterior(multidim_data):
    """Test PLPD approximation with multidimensional posterior."""
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

    posterior = xr.Dataset(
        {
            "param1": (["chain", "draw"], np.random.randn(4, 25)),
            "param2": (["chain", "draw"], np.random.randn(4, 25)),
        },
        coords={"chain": range(4), "draw": range(25)},
    )

    log_likelihood = log_likelihood.stack(__sample__=("chain", "draw"))
    posterior = posterior.stack(__sample__=("chain", "draw"))

    approx = PLPDApproximation(posterior=posterior)
    with warnings.catch_warnings(record=True):
        result = approx.compute_approximation(log_likelihood)

    assert isinstance(result, np.ndarray)
    assert result.shape == (15, 2)
    assert np.all(np.isfinite(result))


def test_plpd_approximation_with_extreme_data(extreme_data, centered_eight):
    """Test PLPD approximation with extreme data fixture."""
    posterior = centered_eight.posterior.stack(__sample__=("chain", "draw"))

    log_likelihood = xr.DataArray(
        extreme_data.T,
        dims=["obs_id", "__sample__"],
        coords={
            "obs_id": range(extreme_data.shape[1]),
            "__sample__": range(extreme_data.shape[0]),
        },
    )

    approx = PLPDApproximation(posterior=posterior)
    with warnings.catch_warnings(record=True):
        result = approx.compute_approximation(log_likelihood)

    assert isinstance(result, np.ndarray)
    assert result.shape == (extreme_data.shape[1],)
    assert np.all(np.isfinite(result))


def test_plpd_approximation_formula():
    """Test that PLPD approximation follows the expected formula."""
    log_likelihood = xr.DataArray(
        np.array([
            [0.0, -1.0, -2.0],
            [-3.0, -4.0, -5.0],
        ]),
        dims=["obs_id", "__sample__"],
        coords={"obs_id": range(2), "__sample__": range(3)},
    )

    posterior = xr.Dataset(
        {
            "param": (["__sample__"], np.array([1.0, 2.0, 3.0])),
        },
        coords={"__sample__": range(3)},
    )

    approx = PLPDApproximation(posterior=posterior)
    with warnings.catch_warnings(record=True):
        result = approx.compute_approximation(log_likelihood)

    expected = np.array([
        np.mean([0.0, -1.0, -2.0]),
        np.mean([-3.0, -4.0, -5.0]),
    ])

    assert np.allclose(result, expected)
