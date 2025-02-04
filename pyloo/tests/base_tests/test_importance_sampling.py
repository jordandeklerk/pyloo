"""Tests for importance sampling module."""

import numpy as np
import pytest

from ...importance_sampling import ImportanceSampling, do_importance_sampling
from ..helpers import (
    assert_arrays_allclose,
    assert_arrays_equal,
    assert_finite,
    assert_shape_equal,
    does_not_warn,
    generate_psis_data,
)


class MockImportanceSampling(ImportanceSampling):
    """Mock implementation for testing abstract base class."""

    def compute_weights(self, log_ratios, r_eff=None, **kwargs):
        """Mock implementation that returns identity weights."""
        return log_ratios, np.zeros(log_ratios.shape[1]), None


def test_abstract_base_class():
    """Test that abstract base class cannot be instantiated directly."""
    with pytest.raises(TypeError):
        ImportanceSampling()


def test_mock_importance_sampling(numpy_arrays):
    """Test mock implementation of ImportanceSampling."""
    mock_is = MockImportanceSampling()
    log_ratios = numpy_arrays["random_ratios"]
    weights, diagnostics, ess = mock_is.compute_weights(log_ratios)
    assert_shape_equal(weights, log_ratios)
    assert len(diagnostics) == log_ratios.shape[1]
    assert ess is None


def test_do_importance_sampling_psis(log_likelihood_data):
    """Test PSIS importance sampling with real log likelihood data."""
    log_ratios = -log_likelihood_data.values.T
    r_eff = np.ones(log_ratios.shape[1])

    with does_not_warn():
        weights, diagnostics, ess = do_importance_sampling(log_ratios, r_eff=r_eff, method="psis")

    assert_shape_equal(weights, log_ratios)
    assert len(diagnostics) == log_ratios.shape[1]
    assert_finite(weights)
    assert_finite(diagnostics)
    assert ess is not None
    assert_finite(ess)


@pytest.mark.skip(reason="TODO: Implement TIS and SIS methods")
def test_do_importance_sampling_methods(log_likelihood_data):
    """Test different importance sampling methods."""
    log_ratios = -log_likelihood_data.values.T
    r_eff = np.ones(log_ratios.shape[1])
    methods = ["psis", "tis", "sis"]

    for method in methods:
        weights, diagnostics, ess = do_importance_sampling(log_ratios, r_eff=r_eff, method=method)

        assert_shape_equal(weights, log_ratios)
        assert len(diagnostics) == log_ratios.shape[1]
        assert_finite(weights)

        if method == "psis":
            assert ess is not None
            assert_finite(diagnostics)
        elif method in ["tis", "sis"]:
            assert np.all(diagnostics == 0)


def test_do_importance_sampling_extreme_data(extreme_data):
    """Test importance sampling with extreme values."""
    r_eff = np.ones(extreme_data.shape[1])

    weights, diagnostics, ess = do_importance_sampling(extreme_data, r_eff=r_eff, method="psis")
    assert_finite(weights)
    assert len(diagnostics) == extreme_data.shape[1]
    assert ess is not None
    assert_finite(ess)


def test_do_importance_sampling_invalid_method(numpy_arrays):
    """Test error handling for invalid method."""
    log_ratios = numpy_arrays["random_ratios"]
    with pytest.raises(ValueError, match="not implemented"):
        do_importance_sampling(log_ratios, r_eff=1.0, method="invalid")


def test_do_importance_sampling_invalid_inputs(numpy_arrays, multidim_data):
    """Test error handling for invalid inputs."""
    with pytest.raises(ValueError, match="must be 1D or 2D array"):
        do_importance_sampling(multidim_data["llm"], r_eff=1.0)

    log_ratios = numpy_arrays["random_ratios"]
    with pytest.raises(ValueError, match="must be a scalar or have length equal"):
        do_importance_sampling(log_ratios, r_eff=np.ones(3))


def test_importance_sampling_weight_properties(log_likelihood_data):
    """Test properties of importance sampling weights."""
    log_ratios = -log_likelihood_data.values.T
    r_eff = np.ones(log_ratios.shape[1])

    weights, _, _ = do_importance_sampling(log_ratios, r_eff=r_eff, method="psis")

    exp_weights = np.exp(weights)
    weight_sums = np.sum(exp_weights, axis=0)
    assert_arrays_allclose(weight_sums, np.ones_like(weight_sums))

    assert np.all(exp_weights >= 0)


def test_importance_sampling_reproducibility(rng, log_likelihood_data):
    """Test that importance sampling gives reproducible results."""
    log_ratios = -log_likelihood_data.values.T
    r_eff = np.ones(log_ratios.shape[1])

    weights1, diag1, ess1 = do_importance_sampling(log_ratios, r_eff=r_eff, method="psis")
    weights2, diag2, ess2 = do_importance_sampling(log_ratios, r_eff=r_eff, method="psis")

    assert_arrays_equal(weights1, weights2)
    assert_arrays_equal(diag1, diag2)
    assert_arrays_equal(ess1, ess2)


def test_importance_sampling_generated_data(rng):
    """Test importance sampling with generated test data."""
    data = generate_psis_data(rng)

    weights, diagnostics, ess = do_importance_sampling(data["log_ratios"], r_eff=data["r_eff"], method="psis")

    assert_shape_equal(weights, data["log_ratios"])
    assert len(diagnostics) == data["log_ratios"].shape[1]
    assert_finite(weights)
    assert_finite(diagnostics)
    assert ess is not None
    assert_finite(ess)
