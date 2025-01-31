import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal
from scipy.special import logsumexp
import arviz as az

from ..psis import PSISObject, psislw, _gpdfit, _gpinv, _logsumexp


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
    _, arviz_k = az.stats.psislw(-centered_eight.log_likelihood.obs.stack(__sample__=("chain", "draw")), r_eff=0.7)
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
    log_like[:, 1] = 10
    log_weights, pareto_k = psislw(-log_like)
    assert pareto_k[1] > 0.7
    