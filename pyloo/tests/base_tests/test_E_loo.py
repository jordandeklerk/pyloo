"""Tests for expectation calculations."""

import numpy as np
import pytest
from ...E_loo import e_loo, ExpectationResult, _wmean, _wvar, _wsd, _wquant
from ...psis import PSISObject
from ..helpers import (
    assert_arrays_allclose,
    assert_arrays_almost_equal,
    assert_positive,
)


def make_test_data(n_samples=1000, n_obs=1):
    """Create test data for expectations."""
    rng = np.random.default_rng(42)
    x = rng.normal(size=(n_samples, n_obs))
    log_weights = rng.normal(size=(n_samples, n_obs))
    log_weights -= np.max(log_weights, axis=0)  # stabilize
    pareto_k = np.zeros(n_obs)
    tail_len = np.minimum(20, int(0.2 * n_samples))

    psis_obj = PSISObject(log_weights=log_weights, pareto_k=pareto_k, tail_len=tail_len)

    return x, psis_obj


def test_e_loo_vector():
    """Test e_loo with vector inputs."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    log_weights = np.array([-1.0, -1.0, -1.0, -1.0, -1.0])
    psis_obj = PSISObject(log_weights=log_weights, pareto_k=np.array([0.0]), tail_len=2)

    result = e_loo(x, psis_obj, type="mean")
    assert isinstance(result, ExpectationResult)
    assert_arrays_allclose(result.value, 3.0)
    assert_positive(result.pareto_k)

    result = e_loo(x, psis_obj, type="variance")
    assert_arrays_allclose(result.value, 2.5)

    result = e_loo(x, psis_obj, type="sd")
    assert_arrays_allclose(result.value, np.sqrt(2.5))

    result = e_loo(x, psis_obj, type="quantile", probs=[0.25, 0.5, 0.75])
    assert_arrays_almost_equal(result.value, [2.0, 3.0, 4.0])


def test_e_loo_matrix():
    """Test e_loo with matrix inputs."""
    x = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]])
    log_weights = np.array(
        [[-1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0]]
    )
    psis_obj = PSISObject(
        log_weights=log_weights, pareto_k=np.array([0.0, 0.0]), tail_len=2
    )

    result = e_loo(x, psis_obj, type="mean")
    assert isinstance(result, ExpectationResult)
    assert_arrays_almost_equal(result.value, [3.0, 4.0])
    assert len(result.pareto_k) == 2
    assert_positive(result.pareto_k)

    result = e_loo(x, psis_obj, type="variance")
    assert_arrays_almost_equal(result.value, [2.5, 2.5])

    result = e_loo(x, psis_obj, type="sd")
    assert_arrays_almost_equal(result.value, [np.sqrt(2.5), np.sqrt(2.5)])

    result = e_loo(x, psis_obj, type="quantile", probs=[0.25, 0.5, 0.75])
    expected = np.array([[2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
    assert_arrays_almost_equal(result.value, expected)


def test_e_loo_errors():
    """Test error handling in e_loo."""
    x = np.array([1.0, 2.0, 3.0])
    log_weights = np.array([-1.0, -1.0, -1.0])
    psis_obj = PSISObject(log_weights=log_weights, pareto_k=np.array([0.0]), tail_len=2)

    with pytest.raises(ValueError, match="type must be"):
        e_loo(x, psis_obj, type="invalid")

    with pytest.raises(ValueError, match="probs must be provided"):
        e_loo(x, psis_obj, type="quantile")

    with pytest.raises(ValueError, match="probs must be between"):
        e_loo(x, psis_obj, type="quantile", probs=[-0.1, 1.1])

    with pytest.raises(ValueError, match="x and psis_object must have same"):
        e_loo(x[:-1], psis_obj)

    with pytest.raises(ValueError):
        e_loo(np.array([]), psis_obj)

    bad_psis = PSISObject(
        log_weights=np.array([np.inf, -np.inf, np.nan]),
        pareto_k=np.array([0.0]),
        tail_len=2,
    )
    with pytest.raises(ValueError):
        e_loo(x, bad_psis)


def test_e_loo_with_real_data():
    """Test e_loo with more realistic data."""
    x, psis_obj = make_test_data(n_samples=1000, n_obs=5)

    for type in ["mean", "variance", "sd"]:
        result = e_loo(x, psis_obj, type=type)
        assert isinstance(result, ExpectationResult)
        assert result.value.shape == (5,)
        assert result.pareto_k.shape == (5,)

    probs = [0.1, 0.5, 0.9]
    result = e_loo(x, psis_obj, type="quantile", probs=probs)
    assert result.value.shape == (len(probs), 5)
    assert result.pareto_k.shape == (5,)

    assert np.all(np.diff(result.value, axis=0) >= -1e-10)


def test_pareto_k_estimation():
    """Test Pareto k estimation with different input types."""
    n_samples = 1000
    x = np.random.normal(size=n_samples)

    log_weights = np.random.pareto(3, size=n_samples)
    log_weights = np.log(log_weights) - np.max(np.log(log_weights))

    psis_obj = PSISObject(
        log_weights=log_weights, pareto_k=np.array([0.0]), tail_len=int(0.2 * n_samples)
    )

    test_cases = [
        x,
        np.ones_like(x),
        np.where(x > 0, 1, 0),
        np.full_like(x, np.inf),
        np.full_like(x, np.nan),
    ]

    for test_x in test_cases:
        result = e_loo(test_x, psis_obj)
        assert isinstance(result.pareto_k, float)
        assert result.pareto_k >= 0


def test_weighted_calculations():
    """Test individual weighted calculation functions."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    w = np.array([0.1, 0.2, 0.4, 0.2, 0.1])

    mean = _wmean(x, w)
    assert_arrays_allclose(mean, 3.0)

    var = _wvar(x, w)
    assert_positive(var)

    sd = _wsd(x, w)
    assert_arrays_allclose(sd, np.sqrt(var))

    quants = _wquant(x, w, np.array([0.25, 0.5, 0.75]))
    assert_positive(np.diff(quants))


def test_numerical_stability():
    """Test numerical stability with extreme weights."""
    n_samples = 1000
    x = np.random.normal(size=n_samples)

    log_weights = np.array([-1000.0] * (n_samples - 1) + [0.0])
    psis_obj = PSISObject(
        log_weights=log_weights, pareto_k=np.array([0.0]), tail_len=int(0.2 * n_samples)
    )

    for type in ["mean", "variance", "sd"]:
        result = e_loo(x, psis_obj, type=type)
        assert np.isfinite(result.value)

    result = e_loo(x, psis_obj, type="quantile", probs=[0.1, 0.5, 0.9])
    assert np.all(np.isfinite(result.value))


def test_weighted_quantiles_edge_cases():
    """Test weighted quantile function with edge cases."""
    x = np.array([1.0, 2.0, 3.0])
    w = np.array([0.5, 0.0, 0.5])

    probs = [0.0, 0.25, 0.5, 0.75, 1.0]
    quants = _wquant(x, w, np.array(probs))
    assert len(quants) == len(probs)
    assert quants[0] == 1.0
    assert quants[-1] == 3.0

    w_const = np.array([1 / 3, 1 / 3, 1 / 3])
    quants_const = _wquant(x, w_const, np.array([0.5]))
    assert_arrays_allclose(quants_const, [2.0])

    w_single = np.array([1.0, 0.0, 0.0])
    quants_single = _wquant(x, w_single, np.array([0.5]))
    assert_arrays_allclose(quants_single, [1.0])


def test_constant_values():
    """Test behavior with constant values."""
    x = np.ones(100)
    log_weights = np.random.normal(size=100)
    log_weights -= np.max(log_weights)

    psis_obj = PSISObject(
        log_weights=log_weights, pareto_k=np.array([0.0]), tail_len=20
    )

    result_mean = e_loo(x, psis_obj, type="mean")
    assert_arrays_allclose(result_mean.value, 1.0)

    result_var = e_loo(x, psis_obj, type="variance")
    assert_arrays_allclose(result_var.value, 0.0, atol=1e-10)

    result_sd = e_loo(x, psis_obj, type="sd")
    assert_arrays_allclose(result_sd.value, 0.0, atol=1e-10)

    result_quant = e_loo(x, psis_obj, type="quantile", probs=[0.1, 0.5, 0.9])
    assert_arrays_allclose(result_quant.value, 1.0)
