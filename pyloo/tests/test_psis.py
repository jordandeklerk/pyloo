"""Tests for PSIS functionality."""
import copy
import numpy as np
import pytest
from scipy.special import logsumexp
import arviz as az

from pyloo.psis import PSISObject, psislw, _gpdfit, _gpinv, _logsumexp
from pyloo.tests.helpers import (
    assert_arrays_allclose,
    assert_arrays_almost_equal,
    assert_arrays_equal,
    assert_finite,
    assert_positive,
    assert_bounded,
)


def test_psislw(centered_eight):
    """Test PSIS-LOO against ArviZ implementation."""
    log_like = centered_eight.log_likelihood.obs.stack(__sample__=("chain", "draw"))
    log_like = log_like.values.T  
    log_weights, pareto_k = psislw(-log_like)
    _, arviz_k = az.stats.psislw(-centered_eight.log_likelihood.obs.stack(__sample__=("chain", "draw")))
    assert_arrays_allclose(pareto_k, arviz_k.values)


def test_psislw_r_eff(centered_eight):
    """Test PSIS-LOO with relative effective sample sizes."""
    log_like = centered_eight.log_likelihood.obs.stack(__sample__=("chain", "draw"))
    log_like = log_like.values.T  
    r_eff = np.full(log_like.shape[1], 0.7)
    log_weights, pareto_k = psislw(-log_like, r_eff)
    _, arviz_k = az.stats.psislw(-centered_eight.log_likelihood.obs.stack(__sample__=("chain", "draw")), reff=0.7)
    assert_arrays_allclose(pareto_k, arviz_k.values)


def test_psislw_bad_r_eff(numpy_arrays):
    """Test PSIS-LOO with mismatched r_eff dimensions."""
    log_ratios = numpy_arrays["random_ratios"]
    r_eff = np.array([0.7, 0.8])
    with pytest.raises(ValueError):
        psislw(log_ratios, r_eff)


def test_psis_object(numpy_arrays):
    """Test PSISObject creation and attributes."""
    log_weights = numpy_arrays["random_weights"]
    pareto_k = np.random.uniform(size=8)
    n_eff = np.random.uniform(size=8) * 1000
    r_eff = n_eff / 1000
    
    psis = PSISObject(log_weights, pareto_k, n_eff, r_eff)
    assert_arrays_equal(psis.log_weights, log_weights)
    assert_arrays_equal(psis.pareto_k, pareto_k)
    assert_arrays_equal(psis.n_eff, n_eff)
    assert_arrays_equal(psis.r_eff, r_eff)


def test_gpdfit(numpy_arrays):
    """Test generalized Pareto distribution fitting."""
    x = np.sort(np.abs(numpy_arrays["normal"]))
    k, sigma = _gpdfit(x)
    assert_finite(k)
    assert_positive(sigma)


def test_gpinv():
    """Test inverse generalized Pareto distribution."""
    probs = np.array([0.1, 0.5, 0.9])
    result = _gpinv(probs, 0.1, 1.0)
    assert_finite(result)
    assert_bounded(result, lower=0)
    assert_arrays_almost_equal(result, az.stats.stats._gpinv(probs, 0.1, 1.0))


def test_logsumexp():
    """Test log sum exp implementation."""
    x = np.array([-1000, -1.0, 0.0, 1000])
    assert_arrays_allclose(_logsumexp(x), logsumexp(x))


def test_psislw_smooths_for_low_k():
    """Test PSIS-LOO smoothing for low k values."""
    rng = np.random.default_rng(44)
    x = rng.normal(size=100)
    x_smoothed, k = psislw(x.copy())
    assert k < 1/3
    assert not np.allclose(x - logsumexp(x), x_smoothed)


def test_psislw_extreme_values(centered_eight):
    """Test PSIS-LOO with extreme values."""
    log_like = centered_eight.log_likelihood.obs.stack(__sample__=("chain", "draw"))
    log_like = log_like.values.T
    # Modify one school to have extreme values
    log_like[:, 1] = 10
    _, pareto_k = psislw(-log_like)
    assert pareto_k[1] > 0.7


def test_psislw_multidimensional(centered_eight):
    """Test PSIS-LOO with multidimensional data."""
    log_like = centered_eight.log_likelihood.obs.stack(__sample__=("chain", "draw"))
    log_like = log_like.values.T
    
    llm = log_like.reshape(-1, 4, 2).copy()  
    ll1 = log_like.copy()  
    
    log_weights_m, pareto_k_m = psislw(-llm.reshape(-1, llm.shape[-2] * llm.shape[-1]))
    log_weights_1, pareto_k_1 = psislw(-ll1)
    
    assert_arrays_allclose(pareto_k_m, pareto_k_1)
    assert_arrays_allclose(log_weights_m, log_weights_1)


def test_psislw_all_k_high(centered_eight):
    """Test PSIS-LOO with high k values."""
    centered_eight = copy.deepcopy(centered_eight)
    # Make one of the khats high by setting extreme values
    centered_eight.log_likelihood["obs"][:, :, 1] = 10
    log_like = centered_eight.log_likelihood.obs.stack(__sample__=("chain", "draw"))
    log_like = log_like.values.T
    _, pareto_k = psislw(-log_like)
    # Check that non-modified schools have finite k values
    assert np.all(np.isfinite(pareto_k[np.arange(len(pareto_k)) != 1]))
    # The modified school should have k > 0.7 (unreliable) or infinite
    assert pareto_k[1] > 0.7 or np.isinf(pareto_k[1])


@pytest.mark.parametrize("probs", [True, False])
@pytest.mark.parametrize("kappa", [-1, -0.5, 1e-30, 0.5, 1])
@pytest.mark.parametrize("sigma", [0, 2])
def test_gpinv_parametrized(probs, kappa, sigma):
    """Test inverse GPD with various parameters."""
    if probs:
        probs = np.array([0.1, 0.1, 0.1, 0.2, 0.3])
    else:
        probs = np.array([-0.1, 0.1, 0.1, 0.2, 0.3])
    result = _gpinv(probs, kappa, sigma)
    assert len(result) == len(probs)
    if sigma <= 0:
        assert np.all(np.isnan(result))
    else:
        assert_finite(result[probs > 0])
