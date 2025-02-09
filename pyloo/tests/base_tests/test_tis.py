"""Tests for Truncated Importance Sampling (TIS) implementation."""

import numpy as np
import xarray as xr

from ...tis import _tislw, tislw
from ..helpers import (
    assert_arrays_allclose,
    assert_finite,
    assert_positive,
    assert_shape_equal,
)


def test_tislw_numpy(log_likelihood_data):
    """Test TIS with numpy array input."""
    log_weights = log_likelihood_data.values
    smoothed_log_weights, ess = tislw(log_weights)
    assert_shape_equal(smoothed_log_weights, log_weights)
    assert isinstance(ess, np.ndarray)
    assert ess.shape == log_weights.shape[:-1]
    assert_arrays_allclose(np.exp(smoothed_log_weights).sum(axis=-1), 1.0, rtol=1e-6)


def test_tislw_xarray(log_likelihood_data):
    """Test TIS with xarray DataArray input."""
    smoothed_log_weights, ess = tislw(log_likelihood_data)
    assert isinstance(smoothed_log_weights, xr.DataArray)
    assert isinstance(ess, xr.DataArray)
    assert smoothed_log_weights.dims == log_likelihood_data.dims
    assert ess.dims == tuple(d for d in log_likelihood_data.dims if d != "__sample__")
    assert_arrays_allclose(np.exp(smoothed_log_weights).sum("__sample__"), 1.0, rtol=1e-6)


def test_tislw_1d_input(rng):
    """Test tislw with 1D input."""
    log_ratios = rng.normal(size=1000)
    log_weights, ess = tislw(log_ratios)

    assert log_weights.shape == (1000,)
    assert isinstance(ess, np.ndarray)  # ESS is always a numpy array
    assert ess.shape == ()
    assert_finite(log_weights)
    assert_positive(ess)
    assert_arrays_allclose(np.exp(log_weights).sum(), 1.0, rtol=1e-6)


def test_tislw_input_validation():
    """Test input validation in tislw."""
    log_ratios = np.array([1.0, 2.0, 3.0])
    log_weights, ess = tislw(log_ratios)
    assert isinstance(log_weights, np.ndarray)
    assert_arrays_allclose(np.exp(log_weights).sum(), 1.0, rtol=1e-6)


def test_tislw_weight_normalization(numpy_arrays):
    """Test that weights are properly normalized."""
    log_ratios = numpy_arrays["random_ratios"]
    log_weights, _ = tislw(log_ratios)

    weights = np.exp(log_weights)
    sums = np.sum(weights, axis=-1)
    assert_arrays_allclose(sums, np.ones_like(sums), rtol=1e-6)


def test_tislw_with_real_data(log_likelihood_data):
    """Test tislw with real log likelihood data."""
    log_ratios = log_likelihood_data.values
    log_weights, ess = tislw(log_ratios)

    assert_finite(log_weights)
    assert_positive(ess)
    assert ess.shape == log_ratios.shape[:-1]
    assert_arrays_allclose(np.exp(log_weights).sum(axis=-1), 1.0, rtol=1e-6)


def test_tislw_extreme_values(extreme_data):
    """Test tislw with extreme values."""
    log_weights, ess = tislw(extreme_data)

    assert_finite(log_weights)
    assert_positive(ess)

    weights = np.exp(log_weights)
    sums = np.sum(weights, axis=-1)
    assert_arrays_allclose(sums, np.ones_like(sums), rtol=1e-6)


def test_internal_tislw():
    """Test the internal _tislw function directly."""
    log_weights = np.array([1.0, 2.0, 3.0, 4.0])
    n_samples = len(log_weights)
    smoothed_lw, ess = _tislw(log_weights, n_samples)

    assert_finite(smoothed_lw)
    assert_positive(ess)
    assert_arrays_allclose(np.exp(smoothed_lw).sum(), 1.0, rtol=1e-6)

    # Test truncation
    max_weight = np.max(np.exp(smoothed_lw))
    cutpoint = np.exp(np.log(n_samples) * 0.5 / n_samples)
    assert max_weight <= cutpoint


def test_tislw_truncation_bound():
    """Test that TIS weights are properly bounded."""
    rng = np.random.default_rng(42)
    log_ratios = rng.normal(size=(1000, 5))
    log_weights, _ = tislw(log_ratios)

    weights = np.exp(log_weights)
    sums = np.sum(weights, axis=-1)
    assert_arrays_allclose(sums, np.ones_like(sums), rtol=1e-6)
    assert_finite(weights)
    assert np.all(weights >= 0)


def test_tislw_consistency():
    """Test consistency of TIS weights with different sample sizes."""
    rng = np.random.default_rng(42)
    sizes = [100, 1000, 10000]

    for S in sizes:
        log_ratios = rng.normal(size=S)
        log_weights, _ = tislw(log_ratios)
        weights = np.exp(log_weights)

        assert_arrays_allclose(np.sum(weights), 1.0, rtol=1e-6)
        assert_finite(weights)
        assert np.all(weights >= 0)

        # Test truncation bound
        max_weight = np.max(weights)
        cutpoint = np.exp(np.log(S) * 0.5 / S)
        assert max_weight <= cutpoint


def test_tislw_constant_weights():
    """Test tislw with constant log-weights."""
    log_weights = np.ones(100)
    smoothed_lw, ess = tislw(log_weights)
    assert_arrays_allclose(smoothed_lw, -np.log(len(log_weights)), rtol=1e-6)
    assert_positive(ess)
