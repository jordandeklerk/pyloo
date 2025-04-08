# -*- coding: utf-8 -*-
"""Tests for loo_nonfactor module."""

import logging

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest
import xarray as xr
from arviz import InferenceData

from pyloo import loo_mvn
from pyloo.elpd import ELPDData


def test_loo_mvn_basic():
    """Test basic functionality of loo_mvn."""
    n_obs = 10
    n_samples = 100
    n_chains = 2

    mu_samples = np.random.randn(n_chains, n_samples, n_obs)
    cov_samples = np.empty((n_chains, n_samples, n_obs, n_obs))
    for c in range(n_chains):
        for s in range(n_samples):
            A = np.random.randn(n_obs, n_obs) * 0.1
            cov_samples[c, s, :, :] = A @ A.T + np.eye(n_obs) * 0.5

    y_obs_vals = np.random.randn(n_obs)

    idata = InferenceData(
        posterior=xr.Dataset(
            {
                "mu": (("chain", "draw", "obs_dim"), mu_samples),
                "cov": (("chain", "draw", "obs_dim", "obs_dim_bis"), cov_samples),
            },
            coords={
                "chain": np.arange(n_chains),
                "draw": np.arange(n_samples),
                "obs_dim": np.arange(n_obs),
                "obs_dim_bis": np.arange(n_obs),
            },
        ),
        observed_data=xr.Dataset(
            {"y": (("obs_dim",), y_obs_vals)},
            coords={"obs_dim": np.arange(n_obs)},
        ),
    )

    try:
        loo_results = loo_mvn(idata, var_name="y", pointwise=True)

        assert isinstance(loo_results, ELPDData)
        assert "elpd_loo" in loo_results
        assert "p_loo" in loo_results
        assert "loo_i" in loo_results
        assert "pareto_k" in loo_results
        assert loo_results.loo_i.shape == (n_obs,)
        assert loo_results.pareto_k.shape == (n_obs,)
        assert not np.isnan(loo_results.elpd_loo)
        assert not np.isnan(loo_results.p_loo)

    except Exception as e:
        pytest.fail(f"loo_mvn raised an unexpected exception: {e}")


def test_loo_mvn_precision_input():
    """Test loo_mvn using precision matrix input."""
    n_obs = 8
    n_samples = 120
    n_chains = 2

    mu_samples = np.random.randn(n_chains, n_samples, n_obs)
    prec_samples = np.empty((n_chains, n_samples, n_obs, n_obs))
    for c in range(n_chains):
        for s in range(n_samples):
            A = np.random.randn(n_obs, n_obs) * 0.1
            cov_s = A @ A.T + np.eye(n_obs) * 0.5
            try:
                prec_samples[c, s, :, :] = np.linalg.inv(cov_s)
            except np.linalg.LinAlgError:
                prec_samples[c, s, :, :] = np.eye(n_obs)

    y_obs_vals = np.random.randn(n_obs)

    idata = InferenceData(
        posterior=xr.Dataset(
            {
                "mu": (("chain", "draw", "obs_dim"), mu_samples),
                "prec": (("chain", "draw", "obs_dim", "obs_dim_bis"), prec_samples),
            },
            coords={
                "chain": np.arange(n_chains),
                "draw": np.arange(n_samples),
                "obs_dim": np.arange(n_obs),
                "obs_dim_bis": np.arange(n_obs),
            },
        ),
        observed_data=xr.Dataset(
            {"y": (("obs_dim",), y_obs_vals)},
            coords={"obs_dim": np.arange(n_obs)},
        ),
    )

    try:
        loo_results = loo_mvn(idata, var_name="y", prec_var_name="prec", pointwise=True)

        assert isinstance(loo_results, ELPDData)
        assert "elpd_loo" in loo_results
        assert "p_loo" in loo_results
        assert "loo_i" in loo_results
        assert "pareto_k" in loo_results
        assert loo_results.loo_i.shape == (n_obs,)
        assert loo_results.pareto_k.shape == (n_obs,)
        assert not np.isnan(loo_results.elpd_loo)
        assert not np.isnan(loo_results.p_loo)

    except Exception as e:
        pytest.fail(
            f"loo_mvn with precision matrix raised an unexpected exception: {e}"
        )


@pytest.mark.skipif(pm is None, reason="PyMC not installed")
def test_loo_mvn_pymc_model():
    """Test loo_mvn with data generated from a PyMC model using a joint MVN likelihood."""
    n_obs = 200
    y_obs = np.random.randn(n_obs)

    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0, sigma=2, shape=n_obs)
        sd = pm.HalfNormal("sd", sigma=1, shape=n_obs)

        cov_values = pt.diag(sd**2)
        pm.Deterministic("cov", cov_values)

        pm.MvNormal("y_obs", mu=mu, cov=cov_values, observed=y_obs)

    with model:
        idata = pm.sample(
            draws=500, tune=500, chains=2, return_inferencedata=True, target_accept=0.9
        )

        loo_results = loo_mvn(
            idata,
            var_name="y_obs",
            mu_var_name="mu",
            cov_var_name="cov",
            pointwise=True,
        )

        assert isinstance(loo_results, ELPDData)

        assert "elpd_loo" in loo_results
        assert "p_loo" in loo_results
        assert "loo_i" in loo_results
        assert "pareto_k" in loo_results
        assert loo_results.loo_i.shape == (n_obs,)
        assert loo_results.pareto_k.shape == (n_obs,)

        assert not np.isnan(loo_results.elpd_loo)
        assert not np.isnan(loo_results.p_loo)
        assert not np.any(np.isnan(loo_results.loo_i))
        assert not np.any(np.isnan(loo_results.pareto_k))

        logging.info(f"loo_mvn successful: elpd_loo={loo_results.elpd_loo}")
        logging.info(f"p_loo={loo_results.p_loo}")
        logging.info(f"p_loo_se={loo_results.p_loo_se}")
