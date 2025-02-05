"""Tests for Standard Importance Sampling (SIS) implementation."""

import numpy as np
import pytest

from ...sis import sislw
from ..helpers import (
    assert_arrays_allclose,
    assert_arrays_almost_equal,
    assert_arrays_equal,
    assert_finite,
    assert_positive,
    assert_shape_equal,
    generate_psis_data,
)


def test_sislw_basic(rng):
    """Test basic functionality of sislw."""
    log_ratios = rng.normal(size=(1000, 8))
    log_weights, pareto_k, ess = sislw(log_ratios)

    assert_shape_equal(log_weights, log_ratios)
    assert_finite(log_weights)
    assert_arrays_equal(pareto_k, np.zeros(8))
    assert_positive(ess)
    assert ess.shape == (8,)


def test_sislw_1d_input(rng):
    """Test sislw with 1D input."""
    log_ratios = rng.normal(size=1000)
    log_weights, pareto_k, ess = sislw(log_ratios)

    assert log_weights.shape == (1000,)
    assert pareto_k.shape == ()
    assert ess.shape == ()
    assert_finite(log_weights)
    assert_positive(ess)


def test_sislw_r_eff_validation():
    """Test r_eff parameter validation in sislw."""
    log_ratios = np.random.normal(size=(1000, 8))
    _, _, ess1 = sislw(log_ratios, r_eff=0.7)
    assert_positive(ess1)

    r_eff_array = np.full(8, 0.7)
    _, _, ess2 = sislw(log_ratios, r_eff=r_eff_array)
    assert_arrays_allclose(ess1, ess2)

    with pytest.raises(ValueError):
        sislw(log_ratios, r_eff=np.ones(10))


def test_sislw_input_validation():
    """Test input validation in sislw."""
    with pytest.raises(ValueError):
        sislw(np.random.normal(size=(10, 10, 10)))

    log_ratios = [1.0, 2.0, 3.0]
    log_weights, _, _ = sislw(log_ratios)
    assert isinstance(log_weights, np.ndarray)


def test_sislw_weight_normalization(numpy_arrays):
    """Test that weights are properly normalized."""
    log_ratios = numpy_arrays["random_ratios"]
    log_weights, _, _ = sislw(log_ratios)

    weights = np.exp(log_weights)
    sums = np.sum(weights, axis=0)
    assert_arrays_almost_equal(sums, np.ones_like(sums))


def test_sislw_with_real_data(log_likelihood_data):
    """Test sislw with real log likelihood data."""
    log_ratios = log_likelihood_data.values.T
    log_weights, pareto_k, ess = sislw(log_ratios)

    assert_finite(log_weights)
    assert_arrays_equal(pareto_k, np.zeros(log_ratios.shape[1]))
    assert_positive(ess)
    assert ess.shape == (log_ratios.shape[1],)


def test_standardimportancesampling_class():
    """Test StandardImportanceSampling class functionality."""
    data = generate_psis_data(np.random.default_rng(42))
    log_ratios = data["log_ratios"]
    r_eff = data["r_eff"]

    log_weights1, pareto_k1, ess1 = sislw(log_ratios)
    assert_finite(log_weights1)
    assert_arrays_equal(pareto_k1, np.zeros(log_ratios.shape[1]))
    assert_positive(ess1)

    log_weights2, pareto_k2, ess2 = sislw(log_ratios, r_eff=r_eff)
    assert_finite(log_weights2)
    assert_arrays_equal(pareto_k2, np.zeros(log_ratios.shape[1]))
    assert_positive(ess2)

    assert_arrays_allclose(log_weights1, log_weights2)
    assert_arrays_allclose(ess1, ess2)


def test_sislw_extreme_values(extreme_data):
    """Test sislw with extreme values."""
    log_weights, pareto_k, ess = sislw(extreme_data)

    assert_finite(log_weights)
    assert_arrays_equal(pareto_k, np.zeros(extreme_data.shape[1]))
    assert_positive(ess)

    weights = np.exp(log_weights)
    sums = np.sum(weights, axis=0)
    assert_arrays_almost_equal(sums, np.ones_like(sums))
