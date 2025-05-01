"""Tests for PSIS functionality."""

import arviz as az
import numpy as np
import pytest
import xarray as xr
from scipy.special import logsumexp

from ...psis import _gpdfit, _gpinv, _psislw, psislw
from ...utils import get_log_likelihood
from ..helpers import (
    assert_arrays_allclose,
    assert_finite,
    assert_positive,
    assert_shape_equal,
)


def test_psislw_numpy(log_likelihood_data):
    log_weights = log_likelihood_data.values
    smoothed_log_weights, pareto_k = psislw(log_weights)
    assert_shape_equal(smoothed_log_weights, log_weights)
    assert isinstance(pareto_k, np.ndarray)
    assert pareto_k.shape == log_weights.shape[:-1]
    assert_arrays_allclose(np.exp(smoothed_log_weights).sum(axis=-1), 1.0, rtol=1e-6)


def test_psislw_xarray(log_likelihood_data):
    smoothed_log_weights, pareto_k = psislw(log_likelihood_data)
    assert isinstance(smoothed_log_weights, xr.DataArray)
    assert isinstance(pareto_k, xr.DataArray)
    assert smoothed_log_weights.dims == log_likelihood_data.dims
    assert pareto_k.dims == tuple(
        d for d in log_likelihood_data.dims if d != "__sample__"
    )
    assert_arrays_allclose(
        np.exp(smoothed_log_weights).sum("__sample__"), 1.0, rtol=1e-6
    )


def test_psislw_smooths_for_low_k(log_likelihood_data):
    x = log_likelihood_data.isel(
        {d: 0 for d in log_likelihood_data.dims if d != "__sample__"}
    )
    x_smoothed, k = psislw(x.copy())
    assert not np.allclose(x - logsumexp(x), x_smoothed)


def test_psislw_reff(log_likelihood_data):
    x = log_likelihood_data.isel(
        {d: 0 for d in log_likelihood_data.dims if d != "__sample__"}
    )

    for reff in [0.5, 1.0, 2.0]:
        smoothed_lw, k = psislw(x.values, reff=reff)  # Use numpy array input
        assert_shape_equal(smoothed_lw, x)
        assert isinstance(k, np.ndarray)
        assert_arrays_allclose(np.exp(smoothed_lw).sum(), 1.0, rtol=1e-6)


def test_gpdfit(log_likelihood_data):
    x = log_likelihood_data.isel(
        {d: 0 for d in log_likelihood_data.dims if d != "__sample__"}
    )
    x = np.sort(np.exp(x.values) - np.exp(x.values.min()))
    k, sigma = _gpdfit(x)
    assert_finite(k)
    assert_finite(sigma)
    assert_positive(sigma)


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
    log_weights = np.array([1.0, 1.1, 1.2, 1.3])
    smoothed_lw, k = psislw(log_weights)
    assert k == np.inf  # Should get inf when not enough tail samples
    assert_arrays_allclose(np.exp(smoothed_lw).sum(), 1.0, rtol=1e-6)


def test_internal_psislw(log_likelihood_data):
    x = log_likelihood_data.isel(
        {d: 0 for d in log_likelihood_data.dims if d != "__sample__"}
    ).values
    cutoff_ind = -20
    cutoffmin = np.log(np.finfo(float).tiny)

    smoothed_lw, k = _psislw(x, cutoff_ind, cutoffmin)
    assert_shape_equal(smoothed_lw, x)
    assert np.isscalar(k)
    assert_arrays_allclose(np.exp(smoothed_lw).sum(), 1.0, rtol=1e-6)


def test_psislw_extreme_values(extreme_data):
    smoothed_lw, k = psislw(extreme_data)
    assert_arrays_allclose(np.exp(smoothed_lw).sum(axis=-1), 1.0, rtol=1e-6)
    assert np.all(k == np.inf)  # All k values should be infinite


def test_psislw_constant_weights():
    log_weights = np.ones(100)
    smoothed_lw, k = psislw(log_weights)
    assert_arrays_allclose(smoothed_lw, -np.log(len(log_weights)), rtol=1e-6)
    assert k == np.inf  # Should get inf for constant weights (no tail)


@pytest.mark.parametrize("data_fixture", ["centered_eight", "non_centered_eight"])
def test_psislw_arviz_match(data_fixture, request):
    data = request.getfixturevalue(data_fixture)

    log_likelihood = get_log_likelihood(data)
    log_likelihood = log_likelihood.stack(__sample__=("chain", "draw"))

    our_smoothed_weights, our_pareto_k = psislw(-log_likelihood, reff=0.7)
    arviz_smoothed_weights, arviz_pareto_k = az.stats.psislw(-log_likelihood, reff=0.7)

    assert_arrays_allclose(our_pareto_k, arviz_pareto_k)
    assert_arrays_allclose(our_smoothed_weights, arviz_smoothed_weights)

    our_weights_sum = np.exp(our_smoothed_weights).sum("__sample__")
    arviz_weights_sum = np.exp(arviz_smoothed_weights).sum("__sample__")
    assert_arrays_allclose(our_weights_sum, 1.0, rtol=1e-6)
    assert_arrays_allclose(arviz_weights_sum, 1.0, rtol=1e-6)


def test_psislw_multidimensional(multidim_data):
    data = xr.DataArray(
        multidim_data["llm"],
        dims=["chain", "draw", "dim1", "dim2"],
        coords={
            "chain": range(multidim_data["llm"].shape[0]),
            "draw": range(multidim_data["llm"].shape[1]),
            "dim1": range(multidim_data["llm"].shape[2]),
            "dim2": range(multidim_data["llm"].shape[3]),
        },
    )
    data = data.stack(__sample__=("chain", "draw"))

    our_smoothed_weights, our_pareto_k = psislw(-data, reff=0.7)
    arviz_smoothed_weights, arviz_pareto_k = az.stats.psislw(-data, reff=0.7)

    assert our_smoothed_weights.dims == arviz_smoothed_weights.dims
    assert our_pareto_k.dims == arviz_pareto_k.dims

    assert_arrays_allclose(our_smoothed_weights, arviz_smoothed_weights)
    assert_arrays_allclose(our_pareto_k, arviz_pareto_k)

    our_weights_sum = np.exp(our_smoothed_weights).sum("__sample__")
    arviz_weights_sum = np.exp(arviz_smoothed_weights).sum("__sample__")
    assert_arrays_allclose(our_weights_sum, 1.0, rtol=1e-6)
    assert_arrays_allclose(arviz_weights_sum, 1.0, rtol=1e-6)
