import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal
from scipy.special import logsumexp
import arviz as az

from pyloo.psis import PSISObject, psislw, _gpdfit, _gpinv, _logsumexp


@pytest.fixture(scope="session")
def centered_eight():
    return az.load_arviz_data("centered_eight")


def test_psislw(centered_eight):
    log_like = centered_eight.log_likelihood.obs.stack(__sample__=("chain", "draw"))
    log_like = log_like.values.T  
    log_weights, pareto_k = psislw(-log_like)
    _, arviz_k = az.stats.psislw(-centered_eight.log_likelihood.obs.stack(__sample__=("chain", "draw")))
    assert_allclose(pareto_k, arviz_k.values)


def test_psislw_r_eff(centered_eight):
    log_like = centered_eight.log_likelihood.obs.stack(__sample__=("chain", "draw"))
    log_like = log_like.values.T  
    r_eff = np.full(log_like.shape[1], 0.7)
    log_weights, pareto_k = psislw(-log_like, r_eff)
    _, arviz_k = az.stats.psislw(-centered_eight.log_likelihood.obs.stack(__sample__=("chain", "draw")), reff=0.7)
    assert_allclose(pareto_k, arviz_k.values)


def test_psislw_bad_r_eff():
    log_ratios = np.random.normal(size=(1000, 8))
    r_eff = np.array([0.7, 0.8])
    with pytest.raises(ValueError):
        psislw(log_ratios, r_eff)


def test_psis_object():
    log_weights = np.random.normal(size=(1000, 8))
    pareto_k = np.random.uniform(size=8)
    n_eff = np.random.uniform(size=8) * 1000
    r_eff = n_eff / 1000
    
    psis = PSISObject(log_weights, pareto_k, n_eff, r_eff)
    assert np.array_equal(psis.log_weights, log_weights)
    assert np.array_equal(psis.pareto_k, pareto_k)
    assert np.array_equal(psis.n_eff, n_eff)
    assert np.array_equal(psis.r_eff, r_eff)


def test_gpdfit():
    x = np.sort(np.abs(np.random.normal(size=1000)))
    k, sigma = _gpdfit(x)
    assert np.isfinite(k)
    assert sigma > 0


def test_gpinv():
    probs = np.array([0.1, 0.5, 0.9])
    result = _gpinv(probs, 0.1, 1.0)
    assert np.all(np.isfinite(result))
    assert np.all(result >= 0)
    assert_array_almost_equal(result, az.stats.stats._gpinv(probs, 0.1, 1.0))


def test_logsumexp():
    x = np.array([-1000, -1.0, 0.0, 1000])
    assert_allclose(_logsumexp(x), logsumexp(x))


def test_psislw_smooths_for_low_k():
    rng = np.random.default_rng(44)
    x = rng.normal(size=100)
    x_smoothed, k = psislw(x.copy())
    assert k[0] < 1/3
    assert not np.allclose(x - logsumexp(x), x_smoothed)


def test_psislw_extreme_values(centered_eight):
    log_like = centered_eight.log_likelihood.obs.stack(__sample__=("chain", "draw"))
    log_like = log_like.values.T 
    log_like[:, 1] = 10  # Make one observation have extreme values
    log_weights, pareto_k = psislw(-log_like)
    assert pareto_k[1] > 0.7  # Should trigger high k warning


def test_psislw_multidimensional():
    # Test with multidimensional log-likelihood array
    llm = np.random.rand(4, 23, 15, 2)  # chain, draw, dim1, dim2
    ll1 = llm.reshape(4, 23, 15 * 2)    # chain, draw, combined_dims
    
    # Test that reshaping preserves PSIS results
    log_weights_m, pareto_k_m = psislw(-llm.reshape(-1, llm.shape[-2] * llm.shape[-1]).T)
    log_weights_1, pareto_k_1 = psislw(-ll1.reshape(-1, ll1.shape[-1]).T)
    
    assert_allclose(pareto_k_m, pareto_k_1)
    assert_allclose(log_weights_m, log_weights_1)


def test_psislw_all_k_high():
    # Test case where all k values are high (> 0.7)
    n_samples = 2000
    n_obs = 5
    # Generate data that will produce high k values
    log_ratios = np.random.normal(0, 10, size=(n_samples, n_obs))
    log_ratios[:, 0] = 1000  # Extreme values
    log_weights, pareto_k = psislw(-log_ratios)
    assert np.all(pareto_k > 0.7)


@pytest.mark.parametrize("probs", [True, False])
@pytest.mark.parametrize("kappa", [-1, -0.5, 1e-30, 0.5, 1])
@pytest.mark.parametrize("sigma", [0, 2])
def test_gpinv_parametrized(probs, kappa, sigma):
    if probs:
        probs = np.array([0.1, 0.1, 0.1, 0.2, 0.3])
    else:
        probs = np.array([-0.1, 0.1, 0.1, 0.2, 0.3])
    result = _gpinv(probs, kappa, sigma)
    assert len(result) == len(probs)
    if sigma <= 0:
        assert np.all(np.isnan(result))
    else:
        assert np.all(np.isfinite(result[probs > 0]))
        