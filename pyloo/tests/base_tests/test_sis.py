"""Tests for Standard Importance Sampling (SIS) implementation."""

import numpy as np
import xarray as xr

from ...sis import sislw
from ..helpers import (
    assert_arrays_allclose,
    assert_arrays_almost_equal,
    assert_finite,
    assert_positive,
    assert_shape_equal,
)


def test_sislw_numpy(log_likelihood_data):
    """Test SIS with numpy array input."""
    log_weights = log_likelihood_data.values
    smoothed_log_weights, ess = sislw(log_weights)
    assert_shape_equal(smoothed_log_weights, log_weights)
    assert isinstance(ess, np.ndarray)
    assert ess.shape == log_weights.shape[:-1]
    assert_arrays_allclose(np.exp(smoothed_log_weights).sum(axis=-1), 1.0, rtol=1e-6)


def test_sislw_xarray(log_likelihood_data):
    """Test SIS with xarray DataArray input."""
    smoothed_log_weights, ess = sislw(log_likelihood_data)
    assert isinstance(smoothed_log_weights, xr.DataArray)
    assert isinstance(ess, xr.DataArray)
    assert smoothed_log_weights.dims == log_likelihood_data.dims
    assert ess.dims == tuple(d for d in log_likelihood_data.dims if d != "__sample__")
    assert_arrays_allclose(
        np.exp(smoothed_log_weights).sum("__sample__"), 1.0, rtol=1e-6
    )


def test_sislw_1d_input(rng):
    """Test sislw with 1D input."""
    log_ratios = rng.normal(size=1000)
    log_weights, ess = sislw(log_ratios)

    assert log_weights.shape == (1000,)
    assert_finite(log_weights)
    assert_positive(ess)
    assert_arrays_allclose(np.exp(log_weights).sum(), 1.0, rtol=1e-6)


def test_sislw_input_validation():
    """Test input validation in sislw."""
    log_ratios = np.array([1.0, 2.0, 3.0])
    log_weights, ess = sislw(log_ratios)
    assert isinstance(log_weights, np.ndarray)
    assert_arrays_allclose(np.exp(log_weights).sum(), 1.0, rtol=1e-6)


def test_sislw_weight_normalization(numpy_arrays):
    """Test that weights are properly normalized."""
    log_ratios = numpy_arrays["random_ratios"]
    log_weights, _ = sislw(log_ratios)

    weights = np.exp(log_weights)
    sums = np.sum(weights, axis=-1)
    assert_arrays_almost_equal(sums, np.ones_like(sums))


def test_sislw_with_real_data(log_likelihood_data):
    """Test sislw with real log likelihood data."""
    log_ratios = log_likelihood_data.values
    log_weights, ess = sislw(log_ratios)

    assert_finite(log_weights)
    assert_positive(ess)
    assert ess.shape == log_ratios.shape[:-1]
    assert_arrays_allclose(np.exp(log_weights).sum(axis=-1), 1.0, rtol=1e-6)


def test_sislw_extreme_values(extreme_data):
    """Test sislw with extreme values."""
    log_weights, ess = sislw(extreme_data)

    assert_finite(log_weights)
    assert_positive(ess)

    weights = np.exp(log_weights)
    sums = np.sum(weights, axis=-1)
    assert_arrays_almost_equal(sums, np.ones_like(sums))


def test_sislw_constant_weights():
    """Test sislw with constant log-weights."""
    log_weights = np.ones(100)
    smoothed_lw, ess = sislw(log_weights)
    assert_arrays_allclose(smoothed_lw, -np.log(len(log_weights)), rtol=1e-6)
    assert_positive(ess)
