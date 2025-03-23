"""Tests for importance sampling based LOO approximations."""

import numpy as np
import xarray as xr

from ...approximations.importance_sampling import (
    ImportanceSamplingApproximation,
    SISApproximation,
    TISApproximation,
)
from ...base import ISMethod


def test_importance_sampling_approximation_base(log_likelihood_data):
    """Test the base ImportanceSamplingApproximation class with real data."""
    approx = ImportanceSamplingApproximation(method=ISMethod.PSIS)
    assert approx.method == ISMethod.PSIS

    result = approx.compute_approximation(log_likelihood_data)

    assert isinstance(result, np.ndarray)
    assert result.shape == (8,)
    assert np.all(np.isfinite(result))


def test_sis_approximation(log_likelihood_data):
    """Test the SISApproximation class with real data."""
    approx = SISApproximation()
    assert approx.method == ISMethod.SIS

    result = approx.compute_approximation(log_likelihood_data)

    assert isinstance(result, np.ndarray)
    assert result.shape == (8,)
    assert np.all(np.isfinite(result))


def test_tis_approximation(log_likelihood_data):
    """Test the TISApproximation class with real data."""
    approx = TISApproximation()
    assert approx.method == ISMethod.TIS

    result = approx.compute_approximation(log_likelihood_data)

    assert isinstance(result, np.ndarray)
    assert result.shape == (8,)
    assert np.all(np.isfinite(result))


def test_importance_sampling_with_n_draws(log_likelihood_data):
    """Test importance sampling with specified number of draws."""
    approx = ImportanceSamplingApproximation(method=ISMethod.PSIS)

    n_samples = log_likelihood_data.sizes["__sample__"]
    n_draws = n_samples // 2

    result_full = approx.compute_approximation(log_likelihood_data)
    result_subset = approx.compute_approximation(log_likelihood_data, n_draws=n_draws)

    assert isinstance(result_subset, np.ndarray)
    assert result_subset.shape == (8,)
    assert np.all(np.isfinite(result_subset))
    assert not np.allclose(result_full, result_subset)


def test_importance_sampling_with_extreme_values(log_likelihood_data):
    """Test importance sampling with extreme log-likelihood values."""
    approx = ImportanceSamplingApproximation(method=ISMethod.PSIS)

    log_likelihood_extreme = log_likelihood_data.copy(deep=True)
    log_likelihood_extreme.values[0, 0] = 1e10
    log_likelihood_extreme.values[0, 1] = -1e10

    result = approx.compute_approximation(log_likelihood_extreme)
    assert isinstance(result, np.ndarray)
    assert result.shape == (8,)
    assert np.all(np.isfinite(result))


def test_importance_sampling_with_constant_values(log_likelihood_data):
    """Test importance sampling with constant log-likelihood values."""
    approx = ImportanceSamplingApproximation(method=ISMethod.PSIS)

    log_likelihood_constant = log_likelihood_data.copy(deep=True)
    log_likelihood_constant.values[0, :] = 1.0

    result = approx.compute_approximation(log_likelihood_constant)
    assert isinstance(result, np.ndarray)
    assert result.shape == (8,)
    assert np.all(np.isfinite(result))


def test_importance_sampling_methods_comparison(log_likelihood_data):
    """Compare results from different importance sampling methods."""
    psis_approx = ImportanceSamplingApproximation(method=ISMethod.PSIS)
    sis_approx = SISApproximation()
    tis_approx = TISApproximation()

    psis_result = psis_approx.compute_approximation(log_likelihood_data)
    sis_result = sis_approx.compute_approximation(log_likelihood_data)
    tis_result = tis_approx.compute_approximation(log_likelihood_data)

    assert np.all(np.isfinite(psis_result))
    assert np.all(np.isfinite(sis_result))
    assert np.all(np.isfinite(tis_result))

    assert np.max(np.abs(psis_result - sis_result)) < 5.0
    assert np.max(np.abs(psis_result - tis_result)) < 5.0
    assert np.max(np.abs(sis_result - tis_result)) < 5.0


def test_importance_sampling_with_multidimensional_data(multidim_data):
    """Test importance sampling with multidimensional data."""
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

    for approx_class in [
        lambda: ImportanceSamplingApproximation(method=ISMethod.PSIS),
        SISApproximation,
        TISApproximation,
    ]:
        approx = approx_class()
        result = approx.compute_approximation(log_likelihood)

        assert isinstance(result, np.ndarray)
        assert result.shape == (15, 2)
        assert np.all(np.isfinite(result))


def test_importance_sampling_with_extreme_data(extreme_data):
    """Test importance sampling with extreme data fixture."""
    log_likelihood = xr.DataArray(
        extreme_data.T,
        dims=["obs_id", "__sample__"],
        coords={
            "obs_id": range(extreme_data.shape[1]),
            "__sample__": range(extreme_data.shape[0]),
        },
    )

    for approx_class in [
        lambda: ImportanceSamplingApproximation(method=ISMethod.PSIS),
        SISApproximation,
        TISApproximation,
    ]:
        approx = approx_class()
        result = approx.compute_approximation(log_likelihood)

        assert isinstance(result, np.ndarray)
        assert result.shape == (extreme_data.shape[1],)
        assert np.all(np.isfinite(result))


def test_importance_sampling_with_centered_eight(centered_eight):
    """Test importance sampling with the centered_eight dataset."""
    log_likelihood = centered_eight.log_likelihood.obs.stack(
        __sample__=("chain", "draw")
    )

    for approx_class in [
        lambda: ImportanceSamplingApproximation(method=ISMethod.PSIS),
        SISApproximation,
        TISApproximation,
    ]:
        approx = approx_class()
        result = approx.compute_approximation(log_likelihood)

        assert isinstance(result, np.ndarray)
        assert result.shape == (8,)
        assert np.all(np.isfinite(result))
