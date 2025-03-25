"""Tests for the leave-one-group-out cross-validation module."""

import logging

import arviz as az
import numpy as np
import pytest

import pyloo as pl
from pyloo.loo_group import loo_group

try:
    import pymc as pm

    HAS_PYMC = True
except ImportError:
    HAS_PYMC = False


@pytest.mark.skipif(not HAS_PYMC, reason="PyMC not installed")
def test_loo_group_radon():
    """Test leave-one-group-out cross-validation with radon data."""
    data = az.load_arviz_data("radon")
    group_ids = data.constant_data.county_idx.values

    logo_results = loo_group(data, group_ids)

    assert hasattr(logo_results, "elpd_logo")
    assert hasattr(logo_results, "se")
    assert hasattr(logo_results, "p_logo")

    assert logo_results.n_groups == len(np.unique(group_ids))

    logo_results_pw = loo_group(data, group_ids, pointwise=True)
    logging.info(logo_results_pw)

    assert hasattr(logo_results_pw, "logo_i")
    assert hasattr(logo_results_pw, "pareto_k")
    assert len(logo_results_pw.logo_i) == len(np.unique(group_ids))
    assert len(logo_results_pw.pareto_k) == len(np.unique(group_ids))

    assert np.isclose(logo_results_pw.logo_i.values.sum(), logo_results_pw.elpd_logo)

    loo_results = pl.loo(data)
    assert not np.isclose(logo_results.elpd_logo, loo_results.elpd_loo)

    diff = abs(logo_results.elpd_logo - loo_results.elpd_loo)
    assert diff < abs(loo_results.elpd_loo) * 0.5

    loo_results_pw = pl.loo(data, pointwise=True)
    logging.info(loo_results_pw)
    assert logo_results.p_logo < loo_results.p_loo


@pytest.mark.skipif(not HAS_PYMC, reason="PyMC not installed")
def test_loo_group_synthetic():
    """Test leave-one-group-out cross-validation with synthetic data."""
    np.random.seed(42)
    n_groups = 10
    n_per_group = 20
    n_samples = n_groups * n_per_group

    group_means = np.random.normal(0, 1, n_groups)
    group_ids = np.repeat(np.arange(n_groups), n_per_group)

    x = np.random.normal(0, 1, n_samples)
    y = x + group_means[group_ids] + np.random.normal(0, 0.5, n_samples)

    with pm.Model():
        mu_a = pm.Normal("mu_a", mu=0, sigma=1)
        sigma_a = pm.HalfNormal("sigma_a", sigma=1)

        a = pm.Normal("a", mu=mu_a, sigma=sigma_a, shape=n_groups)

        beta = pm.Normal("beta", mu=0, sigma=1)
        sigma = pm.HalfNormal("sigma", sigma=1)

        mu = a[group_ids] + beta * x

        pm.Normal("y", mu=mu, sigma=sigma, observed=y)

        idata = pm.sample(
            1000,
            tune=1000,
            chains=2,
            random_seed=42,
            progressbar=False,
            idata_kwargs={"log_likelihood": True},
        )

    logo_results = loo_group(idata, group_ids)

    assert hasattr(logo_results, "elpd_logo")
    assert hasattr(logo_results, "se")
    assert hasattr(logo_results, "p_logo")

    assert logo_results.n_groups == n_groups

    logo_results_pw = loo_group(idata, group_ids, pointwise=True)

    assert hasattr(logo_results_pw, "logo_i")
    assert hasattr(logo_results_pw, "pareto_k")
    assert len(logo_results_pw.logo_i) == n_groups
    assert len(logo_results_pw.pareto_k) == n_groups

    assert np.isclose(logo_results_pw.logo_i.values.sum(), logo_results_pw.elpd_logo)

    loo_results = pl.loo(idata)

    assert not np.isclose(logo_results.elpd_logo, loo_results.elpd_loo)

    diff = abs(logo_results.elpd_logo - loo_results.elpd_loo)
    assert diff < abs(loo_results.elpd_loo) * 0.5
