"""Tests for expectation calculations."""

import numpy as np
import pytest
import arviz as az

from ...E_loo import (
    ExpectationResult,
    e_loo,
    normalize_weights,
    validate_psis_inputs,
    validate_weights,
    _wmean,
    _wvar,
    _wsd,
    _wquantile,
    _is_degenerate,
    _e_loo_khat,
    PARETO_K_WARN,
    PARETO_K_GOOD,
    MIN_TAIL_LENGTH,
)
from ...psis import PSISObject
from ..helpers import (
    assert_arrays_allclose,
    assert_finite,
    assert_positive,
    assert_bounded,
    assert_shape_equal,
    generate_psis_data,
)


@pytest.fixture(params=["centered_eight", "non_centered_eight"])
def schools_data(request):
    """Load both centered and non-centered eight schools data from ArviZ."""
    data = az.load_arviz_data(request.param)
    return data


@pytest.fixture
def schools_psis(schools_data):
    """Create PSIS object from schools data."""
    log_lik = schools_data.log_likelihood.y.values
    n_obs = log_lik.shape[1]
    psis_obj = PSISObject(log_weights=-log_lik, pareto_k=np.zeros(n_obs))
    return psis_obj


def test_validate_psis_inputs(schools_data, schools_psis):
    """Test validation of PSIS inputs."""
    theta = schools_data.posterior.theta.values
    
    # Valid inputs should not raise
    validate_psis_inputs(theta, schools_psis)
    validate_psis_inputs(theta, schools_psis, schools_psis.log_weights)
    
    # Invalid inputs should raise
    with pytest.raises(ValueError, match="must be a PSISObject"):
        validate_psis_inputs(theta, None)
        
    with pytest.raises(ValueError, match="same first dimension"):
        validate_psis_inputs(theta[:-1], schools_psis)
        
    with pytest.raises(ValueError, match="same shape"):
        validate_psis_inputs(theta, schools_psis, theta[:-1])


def test_validate_weights():
    """Test validation of importance sampling weights."""
    # Valid weights should not raise
    weights = np.array([[0.5, 0.3], [0.5, 0.7]])
    validate_weights(weights)
    
    # Invalid weights should raise
    with pytest.raises(ValueError, match="must be finite"):
        validate_weights(np.array([[np.inf, 0.5], [0.5, 0.5]]))
        
    with pytest.raises(ValueError, match="must be non-negative"):
        validate_weights(np.array([[-0.1, 0.5], [1.1, 0.5]]))
        
    with pytest.raises(ValueError, match="must sum to 1"):
        validate_weights(np.array([[0.1, 0.5], [0.1, 0.5]]))


def test_normalize_weights(schools_psis):
    """Test weight normalization with real data."""
    # Test with real log weights
    norm_weights = normalize_weights(schools_psis.log_weights)
    assert_finite(norm_weights)
    assert_positive(norm_weights)
    assert_arrays_allclose(np.sum(norm_weights, axis=0), 1.0)
    
    # Test numerical stability with extreme values
    extreme_log = np.array([[1000.0, -1000.0], [-1000.0, 1000.0]])
    norm_extreme = normalize_weights(extreme_log, log=True)
    assert_arrays_allclose(np.sum(norm_extreme, axis=0), 1.0)
    assert_finite(norm_extreme)
    assert_positive(norm_extreme)


def test_weighted_statistics(schools_data, schools_psis):
    """Test weighted statistical calculations with real data."""
    theta = schools_data.posterior.theta.values
    weights = normalize_weights(schools_psis.log_weights)
    
    # Test mean
    wmean = _wmean(theta, weights)
    assert_finite(wmean)
    assert_shape_equal(wmean, np.zeros(theta.shape[1]))
    
    # Test variance
    wvar = _wvar(theta, weights)
    assert_finite(wvar)
    assert_positive(wvar)
    assert_shape_equal(wvar, np.zeros(theta.shape[1]))
    
    # Test standard deviation
    wsd = _wsd(theta, weights)
    assert_finite(wsd)
    assert_positive(wsd)
    assert_arrays_allclose(wsd, np.sqrt(wvar))
    
    # Test quantiles
    probs = np.array([0.25, 0.5, 0.75])
    wq = _wquantile(theta, weights, probs)
    assert_finite(wq)
    assert_shape_equal(wq, np.zeros((len(probs), theta.shape[1])))
    assert np.all(np.diff(wq, axis=0) >= 0)  # Check monotonicity


def test_is_degenerate():
    """Test degenerate array detection."""
    # Non-degenerate cases
    assert not _is_degenerate(np.array([1.0, 2.0, 3.0]))
    assert not _is_degenerate(np.array([1.0, 1.5, 2.0]))
    
    # Degenerate cases
    assert _is_degenerate(np.array([1.0, 1.0, 1.0]))  # constant
    assert _is_degenerate(np.array([0.0, 1.0, 0.0]))  # binary
    assert _is_degenerate(np.array([1.0, np.nan, 3.0]))  # non-finite
    assert _is_degenerate(np.array([1.0, np.inf, 3.0]))  # non-finite


def test_e_loo_with_schools(schools_data, schools_psis):
    """Test E_loo with schools data."""
    theta = schools_data.posterior.theta.values
    
    # Test mean
    result = e_loo(theta, schools_psis, type="mean")
    assert isinstance(result, ExpectationResult)
    assert_shape_equal(result.value, np.zeros(theta.shape[1]))
    assert_shape_equal(result.pareto_k, np.zeros(theta.shape[1]))
    assert_finite(result.value)
    assert_finite(result.pareto_k)
    
    # Test variance
    result = e_loo(theta, schools_psis, type="variance")
    assert_shape_equal(result.value, np.zeros(theta.shape[1]))
    assert_positive(result.value)
    
    # Test standard deviation
    result = e_loo(theta, schools_psis, type="sd")
    assert_shape_equal(result.value, np.zeros(theta.shape[1]))
    assert_positive(result.value)
    
    # Test quantiles
    probs = np.array([0.25, 0.5, 0.75])
    result = e_loo(theta, schools_psis, type="quantile", probs=probs)
    assert_shape_equal(result.value, np.zeros((len(probs), theta.shape[1])))
    assert np.all(np.diff(result.value, axis=0) >= 0)  # Check monotonicity


def test_e_loo_with_log_ratios(schools_data, schools_psis):
    """Test E_loo with explicit log ratios."""
    theta = schools_data.posterior.theta.values
    log_lik = schools_data.log_likelihood.y.values
    
    # Should work with valid log ratios
    result = e_loo(theta, schools_psis, log_ratios=-log_lik)
    assert isinstance(result, ExpectationResult)
    assert_finite(result.value)
    assert_finite(result.pareto_k)
    
    # Should fail with invalid log ratios
    with pytest.raises(ValueError, match="same shape"):
        e_loo(theta, schools_psis, log_ratios=log_lik[:-1])


def test_e_loo_khat(schools_data, schools_psis):
    """Test Pareto k diagnostic calculation."""
    theta = schools_data.posterior.theta.values
    
    # Test with different function types
    result_mean = e_loo(theta, schools_psis, type="mean")
    result_var = e_loo(theta, schools_psis, type="variance")
    result_quant = e_loo(theta, schools_psis, type="quantile", probs=[0.5])
    
    # k values should be finite
    assert_finite(result_mean.pareto_k)
    assert_finite(result_var.pareto_k)
    assert_finite(result_quant.pareto_k)
    
    # Test diagnostic thresholds
    assert_bounded(result_mean.pareto_k, upper=PARETO_K_WARN)
    k_good = result_mean.pareto_k <= PARETO_K_GOOD
    k_ok = (result_mean.pareto_k > PARETO_K_GOOD) & (result_mean.pareto_k <= PARETO_K_WARN)
    assert np.all(k_good | k_ok)


def test_e_loo_khat_direct(schools_data, schools_psis):
    """Test _e_loo_khat function directly with real data."""
    theta = schools_data.posterior.theta.values
    log_lik = schools_data.log_likelihood.y.values
    
    # Test with mean function (h = x)
    k_mean = _e_loo_khat(theta, -log_lik, schools_psis)
    assert_finite(k_mean)
    assert_bounded(k_mean, lower=0)  # k values should be non-negative
    
    # Test with variance function (h = x^2)
    k_var = _e_loo_khat(theta**2, -log_lik, schools_psis)
    assert_finite(k_var)
    assert_bounded(k_var, lower=0)
    
    # Test with degenerate h
    k_degen = _e_loo_khat(np.ones_like(theta), -log_lik, schools_psis)
    assert_finite(k_degen)
    
    # Test with None h (should use raw ratios)
    k_none = _e_loo_khat(None, -log_lik, schools_psis)
    assert_finite(k_none)
    assert_bounded(k_none, lower=0)
    
    # Test tail length effects
    schools_psis_small = PSISObject(
        log_weights=schools_psis.log_weights,
        pareto_k=schools_psis.pareto_k,
        tail_len=MIN_TAIL_LENGTH-1
    )
    k_small = _e_loo_khat(theta, -log_lik, schools_psis_small)
    assert np.any(np.isinf(k_small))  # Should have some infinite k values


def test_numerical_stability():
    """Test numerical stability with extreme values."""
    # Generate data with extreme values
    rng = np.random.default_rng(1)
    data = generate_psis_data(rng, n_samples=1000, n_obs=3)
    x = np.exp(rng.normal(size=(1000, 3)) * 10)  # Very large values
    psis_obj = PSISObject(log_weights=data["log_ratios"] * 100,  # Extreme weights
                         pareto_k=np.zeros(3))
    
    # Test mean
    result = e_loo(x, psis_obj, type="mean")
    assert_finite(result.value)
    assert_finite(result.pareto_k)
    
    # Test variance
    result = e_loo(x, psis_obj, type="variance")
    assert_finite(result.value)
    assert_positive(result.value)
    
    # Test quantiles
    result = e_loo(x, psis_obj, type="quantile", probs=[0.1, 0.5, 0.9])
    assert_finite(result.value)
    assert np.all(np.diff(result.value, axis=0) >= 0)


def test_tail_length_edge_cases():
    """Test edge cases for tail length in k-hat calculation."""
    # Generate minimal test data
    rng = np.random.default_rng(2)
    data = generate_psis_data(rng, n_samples=100, n_obs=1)
    x = rng.normal(size=(100, 1))
    
    # Test with very small tail length
    psis_obj = PSISObject(log_weights=data["log_ratios"],
                         pareto_k=np.zeros(1),
                         tail_len=MIN_TAIL_LENGTH-1)
    result = e_loo(x, psis_obj, type="mean")
    assert np.isinf(result.pareto_k[0])
    
    # Test with default tail length
    psis_obj = PSISObject(log_weights=data["log_ratios"],
                         pareto_k=np.zeros(1))
    result = e_loo(x, psis_obj, type="mean")
    assert_finite(result.pareto_k)


def test_input_validation(schools_data, schools_psis):
    """Test input validation and error messages."""
    theta = schools_data.posterior.theta.values
    
    # Test invalid type
    with pytest.raises(ValueError, match="type must be"):
        e_loo(theta, schools_psis, type="invalid")
        
    # Test missing probs for quantile
    with pytest.raises(ValueError, match="probs must be provided"):
        e_loo(theta, schools_psis, type="quantile")
        
    # Test invalid probs
    with pytest.raises(ValueError, match="between 0 and 1"):
        e_loo(theta, schools_psis, type="quantile", probs=[-0.1])
    with pytest.raises(ValueError, match="between 0 and 1"):
        e_loo(theta, schools_psis, type="quantile", probs=[1.1])


def test_compare_parameterizations(schools_data, schools_psis):
    """Compare results between centered and non-centered parameterizations."""
    theta = schools_data.posterior.theta.values
    mu = schools_data.posterior.mu.values
    
    # Results should be similar for both parameterizations
    result_theta = e_loo(theta, schools_psis, type="mean")
    result_mu = e_loo(mu, schools_psis, type="mean")
    
    # Both should give finite, reasonable results
    assert_finite(result_theta.value)
    assert_finite(result_mu.value)
    assert_finite(result_theta.pareto_k)
    assert_finite(result_mu.pareto_k)