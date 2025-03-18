"""Tests for Laplace Variational Inference wrapper."""

import numpy as np
import pytest
from arviz import InferenceData

from ...wrapper.laplace import PYMC_EXTRAS_AVAILABLE, Laplace, LaplaceVIResult
from ..helpers import assert_bounded, assert_finite


def test_laplace_wrapper_initialization(simple_model):
    """Test wrapper initialization and validation."""
    model, idata = simple_model

    wrapper = Laplace(model, idata)
    assert wrapper.model == model
    assert wrapper.idata == idata


@pytest.mark.skipif(
    not PYMC_EXTRAS_AVAILABLE,
    reason="pymc-extras not installed",
)
def test_laplace_fit(simple_model):
    """Test fitting the model using Laplace approximation."""
    model, idata = simple_model

    wrapper = Laplace(model, idata)
    result = wrapper.fit(
        optimize_method="BFGS",
        chains=2,
        draws=1000,
        progressbar=False,
    )

    assert isinstance(result, LaplaceVIResult)
    assert result.model == model
    assert isinstance(result.idata, InferenceData)
    assert hasattr(result.idata, "posterior")
    assert hasattr(result.idata, "fit")

    posterior = result.idata.posterior
    assert "alpha" in posterior
    assert "beta" in posterior
    assert "sigma" in posterior

    assert posterior.dims["chain"] == 2
    assert posterior.dims["draw"] == 1000

    fit = result.idata.fit
    assert "mean_vector" in fit
    assert "covariance_matrix" in fit

    assert wrapper.result is result
    assert wrapper.idata is result.idata


@pytest.mark.skipif(
    not PYMC_EXTRAS_AVAILABLE,
    reason="pymc-extras not installed",
)
def test_compute_log_prob_target(simple_model):
    """Test computation of log probability under the target distribution."""
    model, idata = simple_model

    wrapper = Laplace(model, idata)
    wrapper.fit(
        optimize_method="BFGS",
        chains=2,
        draws=100,
        progressbar=False,
    )

    logP = wrapper.compute_logp()

    assert isinstance(logP, np.ndarray)
    assert logP.shape == (200,)

    assert_finite(logP)
    assert_bounded(logP, upper=0)


@pytest.mark.skipif(
    not PYMC_EXTRAS_AVAILABLE,
    reason="pymc-extras not installed",
)
def test_compute_log_prob_proposal(simple_model):
    """Test computation of log probability under the proposal distribution."""
    model, idata = simple_model

    wrapper = Laplace(model, idata)
    wrapper.fit(
        optimize_method="BFGS",
        chains=2,
        draws=100,
        progressbar=False,
    )

    logQ = wrapper.compute_logq()

    assert isinstance(logQ, np.ndarray)
    assert logQ.shape == (200,)

    assert_finite(logQ)


@pytest.mark.skipif(
    not PYMC_EXTRAS_AVAILABLE,
    reason="pymc-extras not installed",
)
def test_log_probability_comparison(simple_model):
    """Test that log probabilities from target and proposal are reasonably close."""
    model, idata = simple_model

    wrapper = Laplace(model, idata)
    wrapper.fit(
        optimize_method="BFGS",
        chains=2,
        draws=1000,
        progressbar=False,
    )

    logP = wrapper.compute_logp()
    logQ = wrapper.compute_logq()

    assert_finite(logP)
    assert_finite(logQ)
    assert_bounded(logP, upper=0)

    log_weights = logP - logQ
    print(log_weights)

    mean_log_weight = np.mean(log_weights)
    print(mean_log_weight)

    stddev_log_weight = np.std(log_weights)
    print(stddev_log_weight)

    max_log_weight = np.max(log_weights)
    print(max_log_weight)

    min_log_weight = np.min(log_weights)
    print(min_log_weight)


@pytest.mark.skipif(
    not PYMC_EXTRAS_AVAILABLE,
    reason="pymc-extras not installed",
)
def test_laplace_wrapper_with_hierarchical_model(hierarchical_model):
    """Test Laplace with a hierarchical model."""
    model, idata = hierarchical_model

    wrapper = Laplace(model, idata)
    result = wrapper.fit(
        optimize_method="BFGS",
        chains=2,
        draws=1000,
        progressbar=False,
    )

    assert isinstance(result, LaplaceVIResult)
    assert result.model == model
    assert isinstance(result.idata, InferenceData)
    assert hasattr(result.idata, "posterior")

    posterior = result.idata.posterior
    assert "alpha" in posterior
    assert "beta" in posterior
    assert "group_sigma" in posterior
    assert "group_effects_raw" in posterior
    assert "sigma_y" in posterior

    assert posterior.sizes["chain"] == 2
    assert posterior.sizes["draw"] == 1000
    assert "group" in posterior["group_effects_raw"].dims
    assert posterior["group_effects_raw"].sizes["group"] == 8
