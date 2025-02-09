"""Tests for PSIS functionality."""
import numpy as np
import pytest
import xarray as xr
from arviz.data import load_arviz_data
from numpy.testing import assert_allclose
from scipy.special import logsumexp

from ...loo import loo
from ...psis import _gpdfit, _gpinv, _psislw, psislw
from ...utils import get_log_likelihood


@pytest.fixture(scope="module")
def log_weights_data():
    """Generate test data for PSIS calculations."""
    rng = np.random.default_rng(42)
    return rng.normal(size=(4, 100))


def test_psislw_numpy(log_weights_data):
    """Test PSIS with numpy array input."""
    log_weights = log_weights_data.copy()
    smoothed_log_weights, pareto_k = psislw(log_weights)
    assert smoothed_log_weights.shape == log_weights.shape
    assert isinstance(pareto_k, np.ndarray)
    assert pareto_k.shape == log_weights.shape[:-1]
    assert_allclose(np.exp(smoothed_log_weights).sum(axis=-1), 1.0, rtol=1e-6)


def test_psislw_xarray(log_weights_data):
    """Test PSIS with xarray DataArray input."""
    log_weights = xr.DataArray(
        log_weights_data,
        dims=["chain", "__sample__"],
        coords={"chain": range(4), "__sample__": range(100)},
    )
    smoothed_log_weights, pareto_k = psislw(log_weights)
    assert isinstance(smoothed_log_weights, xr.DataArray)
    assert isinstance(pareto_k, xr.DataArray)
    assert smoothed_log_weights.dims == log_weights.dims
    assert pareto_k.dims == ("chain",)
    assert_allclose(np.exp(smoothed_log_weights).sum("__sample__"), 1.0, rtol=1e-6)


def test_psislw_smooths_for_low_k(log_weights_data):
    """Check that log-weights are smoothed regardless of k value."""
    x = log_weights_data[0]  # Take first chain
    x_smoothed, k = psislw(x.copy())
    assert not np.allclose(x - logsumexp(x), x_smoothed)


def test_psislw_reff():
    """Test PSIS with different relative efficiency values."""
    rng = np.random.default_rng(42)
    log_weights = rng.normal(size=(100,))

    for reff in [0.5, 1.0, 2.0]:
        smoothed_lw, k = psislw(log_weights, reff=reff)
        assert smoothed_lw.shape == log_weights.shape
        assert isinstance(k, np.ndarray)  # k should be an array
        assert_allclose(np.exp(smoothed_lw).sum(), 1.0, rtol=1e-6)


def test_gpdfit():
    """Test Generalized Pareto Distribution parameter estimation."""
    rng = np.random.default_rng(42)
    data = np.sort(rng.pareto(a=3, size=1000))
    k, sigma = _gpdfit(data)
    assert np.isfinite(k)
    assert np.isfinite(sigma)
    assert sigma > 0


@pytest.mark.parametrize(
    "probs",
    [
        np.array([0.1, 0.5, 0.9]),  # Valid probabilities
        np.array([0, 0.5, 1]),  # Edge cases
        np.array([-0.1, 0.5, 1.1]),  # Invalid probabilities
    ],
)
@pytest.mark.parametrize("kappa", [-1, -0.5, 0, 0.5, 1])
@pytest.mark.parametrize("sigma", [0, 1, 2])
def test_gpinv(probs, kappa, sigma):
    """Test inverse Generalized Pareto Distribution function."""
    result = _gpinv(probs, kappa, sigma)
    assert len(result) == len(probs)

    valid_mask = (probs > 0) & (probs < 1)
    if sigma > 0:
        assert not np.any(np.isnan(result[valid_mask]))
        if kappa >= 0:
            assert np.all(result[probs == 1] == np.inf)
    else:
        assert np.all(np.isnan(result))


def test_psislw_insufficient_tail_samples():
    """Test PSIS behavior with insufficient tail samples."""
    # Create data with very few tail samples
    log_weights = np.array([1.0, 1.1, 1.2, 1.3])
    smoothed_lw, k = psislw(log_weights)
    assert k == np.inf  # Should get inf when not enough tail samples
    assert_allclose(np.exp(smoothed_lw).sum(), 1.0, rtol=1e-6)


def test_internal_psislw():
    """Test the internal _psislw function directly."""
    rng = np.random.default_rng(42)
    log_weights = rng.normal(size=100)
    cutoff_ind = -20
    cutoffmin = np.log(np.finfo(float).tiny)

    smoothed_lw, k = _psislw(log_weights, cutoff_ind, cutoffmin)
    assert smoothed_lw.shape == log_weights.shape
    assert np.isscalar(k)
    assert_allclose(np.exp(smoothed_lw).sum(), 1.0, rtol=1e-6)


def test_psislw_extreme_values():
    """Test PSIS with extreme values in log-weights."""
    log_weights = np.array([1e3, 1e2, 10, 1, 0.1, 0.01])
    smoothed_lw, k = psislw(log_weights)
    assert_allclose(np.exp(smoothed_lw).sum(), 1.0, rtol=1e-6)
    assert k == np.inf


def test_psislw_constant_weights():
    """Test PSIS with constant log-weights."""
    log_weights = np.ones(100)
    smoothed_lw, k = psislw(log_weights)
    assert_allclose(smoothed_lw, -np.log(len(log_weights)), rtol=1e-6)
    assert k == np.inf  # Should get inf for constant weights (no tail)


def test_psislw_arviz_match():
    """Test that our PSIS implementation matches ArviZ's results."""
    centered_eight = load_arviz_data("centered_eight")
    pareto_k = loo(centered_eight, pointwise=True, reff=0.7)["pareto_k"]
    log_likelihood = get_log_likelihood(centered_eight)
    log_likelihood = log_likelihood.stack(__sample__=("chain", "draw"))
    assert_allclose(pareto_k, psislw(-log_likelihood, 0.7)[1])
