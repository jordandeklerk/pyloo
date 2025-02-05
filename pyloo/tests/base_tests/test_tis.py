"""Tests for Truncated Importance Sampling (TIS) implementation."""

import numpy as np
import pytest

from ...tis import _truncate, tislw
from ..helpers import (
    assert_arrays_allclose,
    assert_arrays_almost_equal,
    assert_arrays_equal,
    assert_finite,
    assert_positive,
    assert_shape_equal,
)


def test_tislw_basic(rng):
    """Test basic functionality of tislw."""
    log_ratios = rng.normal(size=(1000, 8))
    log_weights, pareto_k, ess = tislw(log_ratios)

    assert_shape_equal(log_weights, log_ratios)
    assert_finite(log_weights)
    assert_arrays_equal(pareto_k, np.zeros(8))
    assert_positive(ess)
    assert ess.shape == (8,)


def test_tislw_1d_input(rng):
    """Test tislw with 1D input."""
    log_ratios = rng.normal(size=1000)
    log_weights, pareto_k, ess = tislw(log_ratios)

    assert log_weights.shape == (1000,)
    assert pareto_k.shape == ()
    assert ess.shape == ()
    assert_finite(log_weights)
    assert_positive(ess)


def test_tislw_r_eff_validation():
    """Test r_eff parameter validation in tislw."""
    log_ratios = np.random.normal(size=(1000, 8))
    _, _, ess1 = tislw(log_ratios, r_eff=0.7)
    assert_positive(ess1)

    r_eff_array = np.full(8, 0.7)
    _, _, ess2 = tislw(log_ratios, r_eff=r_eff_array)
    assert_arrays_allclose(ess1, ess2)

    with pytest.raises(ValueError):
        tislw(log_ratios, r_eff=np.ones(10))


def test_tislw_input_validation():
    """Test input validation in tislw."""
    with pytest.raises(ValueError):
        tislw(np.random.normal(size=(10, 10, 10)))

    log_ratios = [1.0, 2.0, 3.0]
    log_weights, _, _ = tislw(log_ratios)
    assert isinstance(log_weights, np.ndarray)


def test_tislw_weight_normalization(numpy_arrays):
    """Test that weights are properly normalized."""
    log_ratios = numpy_arrays["random_ratios"]
    log_weights, _, _ = tislw(log_ratios)

    weights = np.exp(log_weights)
    sums = np.sum(weights, axis=0)
    assert_arrays_almost_equal(sums, np.ones_like(sums))


def test_tislw_with_real_data(log_likelihood_data):
    """Test tislw with real log likelihood data."""
    log_ratios = log_likelihood_data.values.T
    log_weights, pareto_k, ess = tislw(log_ratios)

    assert_finite(log_weights)
    assert_arrays_equal(pareto_k, np.zeros(log_ratios.shape[1]))
    assert_positive(ess)
    assert ess.shape == (log_ratios.shape[1],)


def test_tislw_extreme_values(extreme_data):
    """Test tislw with extreme values."""
    log_weights, pareto_k, ess = tislw(extreme_data)

    assert_finite(log_weights)
    assert_arrays_equal(pareto_k, np.zeros(extreme_data.shape[1]))
    assert_positive(ess)

    weights = np.exp(log_weights)
    sums = np.sum(weights, axis=0)
    assert_arrays_almost_equal(sums, np.ones_like(sums))


def test_truncate_function():
    """Test the _truncate helper function."""
    log_ratios = np.array([1.0, 2.0, 3.0, 4.0])
    log_weights = _truncate(log_ratios)

    weights = np.exp(log_weights)
    assert_arrays_almost_equal(np.sum(weights), 1.0)
    assert_finite(weights)
    assert np.all(weights <= np.exp(np.max(log_weights)))
    order = np.argsort(log_ratios)
    weights_sorted = weights[order]
    assert np.all(np.diff(weights_sorted[:-1]) >= -1e-10)


def test_tislw_truncation_bound():
    """Test that TIS weights are properly bounded."""
    rng = np.random.default_rng(42)
    log_ratios = rng.normal(size=(1000, 5))
    log_weights, _, _ = tislw(log_ratios)

    weights = np.exp(log_weights)
    sums = np.sum(weights, axis=0)
    assert_arrays_almost_equal(sums, np.ones_like(sums))
    assert_finite(weights)
    assert np.all(weights >= 0)


def test_tislw_consistency():
    """Test consistency of TIS weights with different sample sizes."""
    rng = np.random.default_rng(42)
    sizes = [100, 1000, 10000]

    for S in sizes:
        log_ratios = rng.normal(size=S)
        log_weights, _, _ = tislw(log_ratios)
        weights = np.exp(log_weights)

        assert_arrays_almost_equal(np.sum(weights), 1.0)

        assert_finite(weights)
        assert np.all(weights >= 0)

        order = np.argsort(log_ratios)
        weights_sorted = weights[order]
        assert np.all(np.diff(weights_sorted[:-1]) >= -1e-10)
