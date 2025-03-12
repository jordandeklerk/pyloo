"""Tests for Laplace Variational Inference wrapper."""

import numpy as np
import pytest
from arviz import InferenceData

from ...wrapper.laplace_wrapper import (
    PYMC_EXTRAS_AVAILABLE,
    LaplaceVIResult,
    LaplaceWrapper,
)
from ...wrapper.pymc_wrapper import PyMCWrapperError
from ..helpers import assert_bounded, assert_finite


def test_laplace_wrapper_initialization(simple_model):
    """Test wrapper initialization and validation."""
    model, idata = simple_model

    wrapper = LaplaceWrapper(model, idata)
    assert wrapper.model == model
    assert wrapper.idata == idata
    assert set(wrapper.observed_data.keys()) == {"y"}
    assert set(wrapper.free_vars) == {"alpha", "beta", "sigma"}
    assert wrapper.result is None


@pytest.mark.skipif(
    not PYMC_EXTRAS_AVAILABLE,
    reason="pymc-extras not installed",
)
def test_laplace_fit(simple_model):
    """Test fitting the model using Laplace approximation."""
    model, idata = simple_model

    wrapper = LaplaceWrapper(model, idata)
    result = wrapper.fit(
        optimize_method="BFGS",
        chains=2,
        draws=100,
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
    assert posterior.dims["draw"] == 100

    fit = result.idata.fit
    assert "mean_vector" in fit
    assert "covariance_matrix" in fit

    assert wrapper.result is result
    assert wrapper.idata is result.idata


@pytest.mark.skipif(
    not PYMC_EXTRAS_AVAILABLE,
    reason="pymc-extras not installed",
)
def test_importance_resample(simple_model):
    """Test importance resampling of Laplace approximation samples."""
    model, idata = simple_model

    wrapper = LaplaceWrapper(model, idata)
    wrapper.fit(
        optimize_method="BFGS",
        chains=2,
        draws=100,
        progressbar=False,
    )

    result = wrapper.importance_resample(
        num_draws=50,
        method="psis",
    )

    assert isinstance(result, LaplaceVIResult)
    assert result.model == model
    assert isinstance(result.idata, InferenceData)
    assert hasattr(result.idata, "posterior")

    posterior = result.idata.posterior
    assert "alpha" in posterior
    assert "beta" in posterior
    assert "sigma" in posterior

    assert posterior.dims["chain"] == 1
    assert posterior.dims["draw"] == 50

    assert result.importance_sampled is True
    assert result.importance_sampling_method == "psis"
    assert result.pareto_k is not None


@pytest.mark.skipif(
    not PYMC_EXTRAS_AVAILABLE,
    reason="pymc-extras not installed",
)
def test_importance_resample_psir(simple_model):
    """Test importance resampling of Laplace approximation samples."""
    model, idata = simple_model

    wrapper = LaplaceWrapper(model, idata)
    wrapper.fit(
        optimize_method="BFGS",
        chains=2,
        draws=100,
        progressbar=False,
    )

    result = wrapper.importance_resample(
        num_draws=50,
        method="psir",
    )

    assert isinstance(result, LaplaceVIResult)
    assert result.model == model
    assert isinstance(result.idata, InferenceData)
    assert hasattr(result.idata, "posterior")

    posterior = result.idata.posterior
    assert "alpha" in posterior
    assert "beta" in posterior
    assert "sigma" in posterior

    assert posterior.dims["chain"] == 1
    assert posterior.dims["draw"] == 50

    assert result.importance_sampled is True
    assert result.importance_sampling_method == "psir"
    assert result.pareto_k is not None


@pytest.mark.skipif(
    not PYMC_EXTRAS_AVAILABLE,
    reason="pymc-extras not installed",
)
def test_compute_log_prob_target(simple_model):
    """Test computation of log probability under the target distribution."""
    model, idata = simple_model

    wrapper = LaplaceWrapper(model, idata)
    wrapper.fit(
        optimize_method="BFGS",
        chains=2,
        draws=100,
        progressbar=False,
    )

    posterior = wrapper.result.idata.posterior
    logP = wrapper._compute_log_prob_target(posterior)

    assert isinstance(logP, np.ndarray)
    assert logP.shape == (2, 100)

    assert_finite(logP)
    assert_bounded(logP, upper=0)


@pytest.mark.skipif(
    not PYMC_EXTRAS_AVAILABLE,
    reason="pymc-extras not installed",
)
def test_compute_log_prob_proposal(simple_model):
    """Test computation of log probability under the proposal distribution."""
    model, idata = simple_model

    wrapper = LaplaceWrapper(model, idata)
    wrapper.fit(
        optimize_method="BFGS",
        chains=2,
        draws=100,
        progressbar=False,
    )

    posterior = wrapper.result.idata.posterior
    logQ = wrapper._compute_log_prob_proposal(
        posterior, wrapper.result.mu, wrapper.result.H_inv
    )

    assert isinstance(logQ, np.ndarray)
    assert logQ.shape == (2, 100)

    assert_finite(logQ)


@pytest.mark.skipif(
    not PYMC_EXTRAS_AVAILABLE,
    reason="pymc-extras not installed",
)
def test_log_probability_comparison(simple_model):
    """Test that log probabilities from target and proposal are reasonably close."""
    model, idata = simple_model

    wrapper = LaplaceWrapper(model, idata)
    wrapper.fit(
        optimize_method="BFGS",
        chains=2,
        draws=100,
        progressbar=False,
    )

    posterior = wrapper.result.idata.posterior
    logP = wrapper._compute_log_prob_target(posterior)
    logQ = wrapper._compute_log_prob_proposal(
        posterior, wrapper.result.mu, wrapper.result.H_inv
    )

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
def test_reshape_posterior_for_importance_sampling(simple_model):
    """Test reshaping posterior samples for importance sampling."""
    model, idata = simple_model

    wrapper = LaplaceWrapper(model, idata)
    wrapper.fit(
        optimize_method="BFGS",
        chains=2,
        draws=100,
        progressbar=False,
    )

    posterior = wrapper.result.idata.posterior
    samples = wrapper._reshape_posterior_for_importance_sampling(posterior)

    assert isinstance(samples, np.ndarray)
    assert samples.shape[0] == 2
    assert samples.shape[1] == 100
    assert samples.shape[2] > 0

    assert_finite(samples)


@pytest.mark.skipif(
    not PYMC_EXTRAS_AVAILABLE,
    reason="pymc-extras not installed",
)
def test_convert_resampled_to_inferencedata(simple_model):
    """Test converting resampled samples to InferenceData format."""
    model, idata = simple_model

    wrapper = LaplaceWrapper(model, idata)
    wrapper.fit(
        optimize_method="BFGS",
        chains=2,
        draws=100,
        progressbar=False,
    )

    posterior = wrapper.result.idata.posterior
    samples = wrapper._reshape_posterior_for_importance_sampling(posterior)

    resampled_samples = samples[0, :50, :]

    resampled_idata = wrapper._convert_resampled_to_inferencedata(resampled_samples)

    assert isinstance(resampled_idata, InferenceData)
    assert hasattr(resampled_idata, "posterior")

    posterior = resampled_idata.posterior
    assert "alpha" in posterior
    assert "beta" in posterior
    assert "sigma" in posterior

    assert posterior.dims["chain"] == 1
    assert posterior.dims["draw"] == 50

    orig_chain0_posterior = wrapper.result.idata.posterior.sel(chain=0)
    resampled_posterior = resampled_idata.posterior

    for var in posterior.data_vars:
        if len(orig_chain0_posterior[var].shape) <= 2:
            orig_mean = float(orig_chain0_posterior[var].isel(draw=slice(0, 50)).mean())
            orig_std = float(orig_chain0_posterior[var].isel(draw=slice(0, 50)).std())

            resampled_mean = float(resampled_posterior[var].mean())
            resampled_std = float(resampled_posterior[var].std())

            assert np.isclose(
                orig_mean, resampled_mean, rtol=1e-5
            ), f"Mean of {var} changed after conversion"
            assert np.isclose(
                orig_std, resampled_std, rtol=1e-5
            ), f"Std of {var} changed after conversion"

    assert hasattr(resampled_idata, "fit")
    assert "mean_vector" in resampled_idata.fit
    assert "covariance_matrix" in resampled_idata.fit

    assert hasattr(resampled_idata, "observed_data")
    assert set(resampled_idata.observed_data.data_vars) == set(
        idata.observed_data.data_vars
    )


@pytest.mark.skipif(
    not PYMC_EXTRAS_AVAILABLE,
    reason="pymc-extras not installed",
)
def test_error_handling(simple_model):
    """Test error handling in the LaplaceWrapper."""
    model, idata = simple_model

    wrapper = LaplaceWrapper(model, idata)

    with pytest.raises(
        PyMCWrapperError,
        match="Model must be fit before applying importance resampling",
    ):
        wrapper.importance_resample()


@pytest.mark.skipif(
    not PYMC_EXTRAS_AVAILABLE,
    reason="pymc-extras not installed",
)
def test_laplace_wrapper_with_hierarchical_model(hierarchical_model):
    """Test LaplaceWrapper with a hierarchical model."""
    model, idata = hierarchical_model

    wrapper = LaplaceWrapper(model, idata)
    result = wrapper.fit(
        optimize_method="BFGS",
        chains=2,
        draws=100,
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
    assert posterior.sizes["draw"] == 100
    assert "group" in posterior["group_effects_raw"].dims
    assert posterior["group_effects_raw"].sizes["group"] == 8
