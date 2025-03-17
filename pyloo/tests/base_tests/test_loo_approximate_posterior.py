"""Tests for the LOO approximate posterior module."""

import numpy as np
import pymc as pm
import pytest
import xarray as xr
from arviz.data import InferenceData

from ...log_density import compute_log_weights
from ...loo import loo
from ...loo_approximate_posterior import loo_approximate_posterior
from ...wrapper.laplace import Laplace
from ..helpers import assert_arrays_allclose


@pytest.fixture
def simple_model_with_approximation(simple_model):
    """Create a simple model with Laplace approximation."""
    model, _ = simple_model
    wrapper = Laplace(model)
    result = wrapper.fit()

    log_p = wrapper.compute_logp().flatten()
    log_g = wrapper.compute_logq().flatten()

    return model, result.idata, log_p, log_g


def test_loo_approximate_posterior_basic(simple_model_with_approximation):
    """Test basic functionality of loo_approximate_posterior."""
    _, idata, log_p, log_g = simple_model_with_approximation

    result = loo_approximate_posterior(idata, log_p, log_g)

    print(result)

    assert result is not None
    assert "elpd_loo" in result
    assert "p_loo" in result
    assert "se" in result
    assert hasattr(result, "approximate_posterior")
    assert "log_p" in result.approximate_posterior
    assert "log_g" in result.approximate_posterior
    assert_arrays_allclose(result.approximate_posterior["log_p"], log_p)
    assert_arrays_allclose(result.approximate_posterior["log_g"], log_g)


@pytest.mark.parametrize("scale", ["log", "negative_log", "deviance"])
def test_loo_approximate_posterior_scales(simple_model_with_approximation, scale):
    """Test loo_approximate_posterior with different scales."""
    _, idata, log_p, log_g = simple_model_with_approximation

    result = loo_approximate_posterior(idata, log_p, log_g, scale=scale)

    assert result["scale"] == scale

    standard_loo = loo(idata, scale=scale)

    print(result)
    print(standard_loo)

    assert np.sign(result["elpd_loo"]) == np.sign(standard_loo["elpd_loo"])


def test_loo_approximate_posterior_pointwise(simple_model_with_approximation):
    """Test loo_approximate_posterior with pointwise=True."""
    _, idata, log_p, log_g = simple_model_with_approximation

    result = loo_approximate_posterior(idata, log_p, log_g, pointwise=True)

    assert result is not None
    assert "loo_i" in result
    assert "pareto_k" in result
    assert hasattr(result, "approximate_posterior")


def test_loo_approximate_posterior_methods(simple_model_with_approximation):
    """Test loo_approximate_posterior with different importance sampling methods."""
    _, idata, log_p, log_g = simple_model_with_approximation

    result_psis = loo_approximate_posterior(idata, log_p, log_g, pointwise=True)
    assert "pareto_k" in result_psis

    result_sis = loo_approximate_posterior(
        idata, log_p, log_g, pointwise=True, method="sis"
    )
    assert "ess" in result_sis

    result_tis = loo_approximate_posterior(
        idata, log_p, log_g, pointwise=True, method="tis"
    )
    assert "ess" in result_tis


def test_loo_approximate_posterior_invalid_method(simple_model_with_approximation):
    """Test loo_approximate_posterior with invalid method."""
    _, idata, log_p, log_g = simple_model_with_approximation

    with pytest.raises(ValueError, match="Invalid method"):
        loo_approximate_posterior(idata, log_p, log_g, method="invalid")


def test_loo_approximate_posterior_invalid_scale(simple_model_with_approximation):
    """Test loo_approximate_posterior with invalid scale."""
    _, idata, log_p, log_g = simple_model_with_approximation

    with pytest.raises(TypeError, match="Valid scale values are"):
        loo_approximate_posterior(idata, log_p, log_g, scale="invalid")


def test_loo_approximate_posterior_missing_loglik(simple_model_with_approximation):
    """Test loo_approximate_posterior with missing log_likelihood."""
    _, idata, log_p, log_g = simple_model_with_approximation

    idata_no_loglik = InferenceData(posterior=idata.posterior)

    with pytest.raises(TypeError, match="log likelihood not found"):
        loo_approximate_posterior(idata_no_loglik, log_p, log_g)


def test_loo_approximate_posterior_missing_posterior(simple_model_with_approximation):
    """Test loo_approximate_posterior with missing posterior and no reff provided."""
    _, idata, log_p, log_g = simple_model_with_approximation

    idata_no_posterior = InferenceData(log_likelihood=idata.log_likelihood)

    with pytest.raises(TypeError, match="Must be able to extract a posterior group"):
        loo_approximate_posterior(idata_no_posterior, log_p, log_g, reff=None)

    result = loo_approximate_posterior(idata_no_posterior, log_p, log_g, reff=0.7)
    assert result is not None


def test_loo_approximate_posterior_nan_handling(simple_model_with_approximation):
    """Test loo_approximate_posterior with NaN values in log-likelihood."""
    _, idata, log_p, log_g = simple_model_with_approximation

    idata_with_nan = idata.copy()
    log_like = idata_with_nan.log_likelihood["y"].values
    log_like[0, 0, 0] = np.nan

    idata_with_nan.log_likelihood["y"] = xr.DataArray(
        log_like,
        dims=idata_with_nan.log_likelihood["y"].dims,
        coords=idata_with_nan.log_likelihood["y"].coords,
    )

    with pytest.warns(UserWarning, match="NaN values detected"):
        result = loo_approximate_posterior(idata_with_nan, log_p, log_g)
        assert result is not None
        assert not np.isnan(result["elpd_loo"])


def test_loo_approximate_posterior_length_mismatch(simple_model_with_approximation):
    """Test loo_approximate_posterior with mismatched log_p and log_g lengths."""
    _, idata, log_p, log_g = simple_model_with_approximation

    short_log_p = log_p[:-10]

    with pytest.raises(ValueError, match="log_p and log_g must have the same length"):
        loo_approximate_posterior(idata, short_log_p, log_g)


def test_loo_approximate_posterior_multiple_groups(simple_model_with_approximation):
    """Test loo_approximate_posterior with multiple log_likelihood groups."""
    _, idata, log_p, log_g = simple_model_with_approximation

    idata_multi = idata.copy()
    idata_multi.log_likelihood["y2"] = idata_multi.log_likelihood["y"]

    with pytest.raises(TypeError, match="several log likelihood arrays"):
        loo_approximate_posterior(idata_multi, log_p, log_g)

    result = loo_approximate_posterior(idata_multi, log_p, log_g, var_name="y")
    assert result is not None


def test_loo_approximate_posterior_numerical_stability(simple_model_with_approximation):
    """Test numerical stability of loo_approximate_posterior with extreme values."""
    _, idata, log_p, log_g = simple_model_with_approximation

    extreme_log_p = log_p.copy() * 1e3
    extreme_log_g = log_g.copy() * 1e3

    result = loo_approximate_posterior(idata, extreme_log_p, extreme_log_g)
    assert result is not None
    assert np.isfinite(result["elpd_loo"])
    assert np.isfinite(result["p_loo"])
    assert np.isfinite(result["se"])


def test_loo_approximate_posterior_with_laplace_wrapper(simple_model):
    """Test integration with Laplace."""
    model, idata = simple_model

    wrapper = Laplace(model, idata)
    result = wrapper.fit()

    log_p = wrapper.compute_logp().flatten()
    log_g = wrapper.compute_logq().flatten()

    loo_result = loo_approximate_posterior(result.idata, log_p, log_g, pointwise=True)

    assert loo_result is not None
    assert "elpd_loo" in loo_result
    assert "pareto_k" in loo_result
    assert hasattr(loo_result, "approximate_posterior")

    standard_loo = loo(result.idata, pointwise=True)

    assert np.sign(loo_result["elpd_loo"]) == np.sign(standard_loo["elpd_loo"])
    assert np.all(np.isfinite(loo_result.pareto_k))


def test_loo_approximate_posterior_multidimensional(simple_model):
    """Test loo_approximate_posterior with multidimensional log-likelihood."""
    _, idata = simple_model

    log_likelihood = idata.log_likelihood["y"].values
    multidim_ll = log_likelihood.reshape(
        log_likelihood.shape[0], log_likelihood.shape[1], -1, 1
    )

    multidim_idata = idata.copy()
    multidim_idata.log_likelihood["y"] = xr.DataArray(
        multidim_ll,
        dims=["chain", "draw", "obs_id", "extra_dim"],
        coords={
            "chain": idata.log_likelihood.chain,
            "draw": idata.log_likelihood.draw,
            "obs_id": np.arange(multidim_ll.shape[2]),
            "extra_dim": [0],
        },
    )

    n_samples = (
        multidim_idata.log_likelihood["y"].stack(__sample__=("chain", "draw")).shape[-1]
    )
    log_p = np.random.normal(size=n_samples)
    log_g = np.random.normal(size=n_samples)

    result = loo_approximate_posterior(multidim_idata, log_p, log_g)

    assert result is not None
    assert "elpd_loo" in result
    assert np.isfinite(result["elpd_loo"])
    assert np.isfinite(result["p_loo"])
    assert np.isfinite(result["se"])


def test_loo_approximate_posterior_variational_with_laplace(simple_model):
    """Test loo_approximate_posterior with variational inference using Laplace."""
    model, idata = simple_model

    wrapper = Laplace(model)
    result = wrapper.fit()

    log_p = wrapper.compute_logp().flatten()
    log_g = wrapper.compute_logq().flatten()

    loo_result_psis = loo_approximate_posterior(
        result.idata,
        log_p,
        log_g,
        method="psis",
        pointwise=True,
    )

    assert loo_result_psis is not None
    assert "elpd_loo" in loo_result_psis
    assert "pareto_k" in loo_result_psis
    assert np.isfinite(loo_result_psis["elpd_loo"])

    standard_loo = loo(idata, pointwise=True)

    print(standard_loo)
    print(loo_result_psis)

    assert np.sign(loo_result_psis["elpd_loo"]) == np.sign(standard_loo["elpd_loo"])


def test_loo_approximate_posterior_wells(wells_model):
    """Test loo_approximate_posterior with variational inference using Laplace."""
    model, idata = wells_model

    wrapper = Laplace(model)
    result = wrapper.fit(chains=4, draws=2000, seed=42)

    log_p = wrapper.compute_logp().flatten()
    log_g = wrapper.compute_logq().flatten()

    standard_loo = loo(idata, pointwise=True)

    if hasattr(standard_loo, "pareto_k"):
        print(f"Standard LOO pareto_k shape: {standard_loo.pareto_k.shape}")
        print(
            f"Standard LOO pareto_k min: {np.min(standard_loo.pareto_k)}, max:"
            f" {np.max(standard_loo.pareto_k)}"
        )
        print(f"Standard LOO pareto_k > 0.7: {np.sum(standard_loo.pareto_k > 0.7)}")
        print(f"Standard LOO pareto_k > 1.0: {np.sum(standard_loo.pareto_k > 1.0)}")

    loo_result_psis = loo_approximate_posterior(
        result.idata,
        log_p,
        log_g,
        method="psis",
        pointwise=True,
    )

    assert loo_result_psis is not None
    assert "elpd_loo" in loo_result_psis
    assert "pareto_k" in loo_result_psis
    assert np.isfinite(loo_result_psis["elpd_loo"])

    print(standard_loo)
    print(loo_result_psis)

    if hasattr(loo_result_psis, "pareto_k"):
        print(f"Approximate LOO pareto_k shape: {loo_result_psis.pareto_k.shape}")
        print(
            f"Approximate LOO pareto_k min: {np.min(loo_result_psis.pareto_k)}, max:"
            f" {np.max(loo_result_psis.pareto_k)}"
        )
        print(
            f"Approximate LOO pareto_k > 0.7: {np.sum(loo_result_psis.pareto_k > 0.7)}"
        )
        print(
            f"Approximate LOO pareto_k > 1.0: {np.sum(loo_result_psis.pareto_k > 1.0)}"
        )

        if np.sum(loo_result_psis.pareto_k > 1.0) > 0:
            print(f"Bad pareto_k value: {loo_result_psis.pareto_k}")

    assert np.sign(loo_result_psis["elpd_loo"]) == np.sign(standard_loo["elpd_loo"])


def test_loo_approximate_posterior_constant_values(simple_model_with_approximation):
    """Test loo_approximate_posterior with constant log-likelihood values."""
    _, idata, log_p, log_g = simple_model_with_approximation

    idata_const = idata.copy()
    log_like = idata_const.log_likelihood["y"].values
    log_like[:] = 1.0

    idata_const.log_likelihood["y"] = xr.DataArray(
        log_like,
        dims=idata_const.log_likelihood["y"].dims,
        coords=idata_const.log_likelihood["y"].coords,
    )

    with pytest.warns(UserWarning, match="The point-wise LOO is the same"):
        result = loo_approximate_posterior(idata_const, log_p, log_g, pointwise=True)
        assert result is not None
        assert np.allclose(result["loo_i"].values, result["loo_i"].values[0])


def test_loo_approximate_posterior_wells_advi(wells_model):
    """Test loo_approximate_posterior with variational inference using Laplace."""
    model, idata = wells_model

    with model:
        mean_field = pm.fit(method="advi")

    log_p, log_q, _ = compute_log_weights(mean_field, 1000)
    trace = mean_field.sample(1000)
    pm.compute_log_likelihood(trace, model=model, extend_inferencedata=True)

    standard_loo = loo(idata, pointwise=True)

    if hasattr(standard_loo, "pareto_k"):
        print(f"Standard LOO pareto_k shape: {standard_loo.pareto_k.shape}")
        print(
            f"Standard LOO pareto_k min: {np.min(standard_loo.pareto_k)}, max:"
            f" {np.max(standard_loo.pareto_k)}"
        )
        print(f"Standard LOO pareto_k > 0.7: {np.sum(standard_loo.pareto_k > 0.7)}")
        print(f"Standard LOO pareto_k > 1.0: {np.sum(standard_loo.pareto_k > 1.0)}")

    loo_approx = loo_approximate_posterior(
        data=trace,
        log_p=log_p,
        log_q=log_q,
        pointwise=True,
        resample_method="psis",
    )
    loo_approx

    assert loo_approx is not None
    assert "elpd_loo" in loo_approx
    assert "pareto_k" in loo_approx
    assert np.isfinite(loo_approx["elpd_loo"])

    print(standard_loo)
    print(loo_approx)

    if hasattr(loo_approx, "pareto_k"):
        print(f"Approximate LOO pareto_k shape: {loo_approx.pareto_k.shape}")
        print(
            f"Approximate LOO pareto_k min: {np.min(loo_approx.pareto_k)}, max:"
            f" {np.max(loo_approx.pareto_k)}"
        )
        print(f"Approximate LOO pareto_k > 0.7: {np.sum(loo_approx.pareto_k > 0.7)}")
        print(f"Approximate LOO pareto_k > 1.0: {np.sum(loo_approx.pareto_k > 1.0)}")

        if np.sum(loo_approx.pareto_k > 1.0) > 0:
            print(f"Bad pareto_k value: {loo_approx.pareto_k}")

    assert np.sign(loo_approx["elpd_loo"]) == np.sign(standard_loo["elpd_loo"])


def test_loo_approximate_posterior_wells_full_rank_advi(wells_model):
    """Test loo_approximate_posterior with variational inference using Laplace."""
    model, idata = wells_model

    with model:
        mean_field = pm.fit(method="fullrank_advi")

    log_p, log_q, _ = compute_log_weights(mean_field, 1000)
    trace = mean_field.sample(1000)
    pm.compute_log_likelihood(trace, model=model, extend_inferencedata=True)

    standard_loo = loo(idata, pointwise=True)

    if hasattr(standard_loo, "pareto_k"):
        print(f"Standard LOO pareto_k shape: {standard_loo.pareto_k.shape}")
        print(
            f"Standard LOO pareto_k min: {np.min(standard_loo.pareto_k)}, max:"
            f" {np.max(standard_loo.pareto_k)}"
        )
        print(f"Standard LOO pareto_k > 0.7: {np.sum(standard_loo.pareto_k > 0.7)}")
        print(f"Standard LOO pareto_k > 1.0: {np.sum(standard_loo.pareto_k > 1.0)}")

    loo_approx = loo_approximate_posterior(
        data=trace,
        log_p=log_p,
        log_q=log_q,
        pointwise=True,
        resample_method="psis",
    )
    loo_approx

    assert loo_approx is not None
    assert "elpd_loo" in loo_approx
    assert "pareto_k" in loo_approx
    assert np.isfinite(loo_approx["elpd_loo"])

    print(standard_loo)
    print(loo_approx)

    if hasattr(loo_approx, "pareto_k"):
        print(f"Approximate LOO pareto_k shape: {loo_approx.pareto_k.shape}")
        print(
            f"Approximate LOO pareto_k min: {np.min(loo_approx.pareto_k)}, max:"
            f" {np.max(loo_approx.pareto_k)}"
        )
        print(f"Approximate LOO pareto_k > 0.7: {np.sum(loo_approx.pareto_k > 0.7)}")
        print(f"Approximate LOO pareto_k > 1.0: {np.sum(loo_approx.pareto_k > 1.0)}")

        if np.sum(loo_approx.pareto_k > 1.0) > 0:
            print(f"Bad pareto_k value: {loo_approx.pareto_k}")

    assert np.sign(loo_approx["elpd_loo"]) == np.sign(standard_loo["elpd_loo"])
