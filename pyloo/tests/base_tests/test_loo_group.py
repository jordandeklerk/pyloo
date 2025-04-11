"""Tests for the leave-one-group-out cross-validation module."""

import logging
import warnings

import arviz as az
import numpy as np
import pymc as pm
import pytest

from ...loo import loo
from ...loo_group import loo_group
from ...loo_kfold import loo_kfold
from ...wrapper.pymc import PyMCWrapper

# TODO: Why is this giving HTTPError now?
# def test_loo_group_radon():
#     """Test leave-one-group-out cross-validation with radon data."""
#     data = az.load_arviz_data("radon")
#     group_ids = data.constant_data.county_idx.values

#     logo_results = loo_group(data, group_ids)

#     assert hasattr(logo_results, "elpd_logo")
#     assert hasattr(logo_results, "se")
#     assert hasattr(logo_results, "p_logo")

#     assert logo_results.n_groups == len(np.unique(group_ids))

#     logo_results_pw = loo_group(data, group_ids, pointwise=True)
#     logging.info(logo_results_pw)

#     assert hasattr(logo_results_pw, "logo_i")
#     assert hasattr(logo_results_pw, "pareto_k")
#     assert len(logo_results_pw.logo_i) == len(np.unique(group_ids))
#     assert len(logo_results_pw.pareto_k) == len(np.unique(group_ids))

#     assert np.isclose(logo_results_pw.logo_i.values.sum(), logo_results_pw.elpd_logo)

#     loo_results = loo(data)
#     assert not np.isclose(logo_results.elpd_logo, loo_results.elpd_loo)

#     diff = abs(logo_results.elpd_logo - loo_results.elpd_loo)
#     assert diff < abs(loo_results.elpd_loo) * 0.5

#     loo_results_pw = loo(data, pointwise=True)
#     logging.info(loo_results_pw)
#     assert logo_results.p_logo < loo_results.p_loo


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

    loo_results = loo(idata)

    assert not np.isclose(logo_results.elpd_logo, loo_results.elpd_loo)

    diff = abs(logo_results.elpd_logo - loo_results.elpd_loo)
    assert diff < abs(loo_results.elpd_loo) * 0.5


def test_loo_group_different_methods(simple_model):
    """Test loo_group with different importance sampling methods."""
    _, idata = simple_model

    n_obs = len(idata.log_likelihood.y.obs_id)
    n_groups = 5

    group_ids = np.array([i // (n_obs // n_groups) for i in range(n_obs)])

    logo_psis = loo_group(idata, group_ids, var_name="y")
    assert hasattr(logo_psis, "elpd_logo")
    assert logo_psis.n_groups == n_groups

    with warnings.catch_warnings(record=True) as w:
        logo_sis = loo_group(idata, group_ids, method="sis", var_name="y")
        assert any(
            ("Using SIS for LOGO computation" in str(warn.message) for warn in w)
        )

    assert hasattr(logo_sis, "elpd_logo")
    assert logo_sis.n_groups == n_groups

    with warnings.catch_warnings(record=True) as w:
        logo_tis = loo_group(idata, group_ids, method="tis", var_name="y")
        assert any(
            ("Using TIS for LOGO computation" in str(warn.message) for warn in w)
        )

    assert hasattr(logo_tis, "elpd_logo")
    assert logo_tis.n_groups == n_groups


def test_loo_group_different_scales(simple_model):
    """Test loo_group with different scale options."""
    _, idata = simple_model

    n_obs = len(idata.log_likelihood.y.obs_id)
    n_groups = 5

    group_ids = np.array([i // (n_obs // n_groups) for i in range(n_obs)])

    logo_log = loo_group(idata, group_ids, scale="log", var_name="y")

    logo_neg_log = loo_group(idata, group_ids, scale="negative_log", var_name="y")
    assert np.isclose(logo_log.elpd_logo, -logo_neg_log.elpd_logo)

    logo_deviance = loo_group(idata, group_ids, scale="deviance", var_name="y")
    assert np.isclose(logo_log.elpd_logo * -2, logo_deviance.elpd_logo)

    assert logo_log.scale == "log"
    assert logo_neg_log.scale == "negative_log"
    assert logo_deviance.scale == "deviance"


def test_loo_group_var_name(multi_observed_model):
    """Test loo_group with specified var_name parameter."""
    _, idata = multi_observed_model

    n_obs = len(idata.log_likelihood.y1.obs_id)
    group_ids = np.repeat(np.arange(5), n_obs // 5)

    logo_y1 = loo_group(idata, group_ids, var_name="y1")
    assert hasattr(logo_y1, "elpd_logo")

    logo_y2 = loo_group(idata, group_ids, var_name="y2")
    assert hasattr(logo_y2, "elpd_logo")

    assert not np.isclose(logo_y1.elpd_logo, logo_y2.elpd_logo)


def test_loo_group_reff_parameter(simple_model):
    """Test loo_group with specified reff parameter."""
    _, idata = simple_model

    n_obs = len(idata.log_likelihood.y.obs_id)
    n_groups = 5

    group_ids = np.array([i // (n_obs // n_groups) for i in range(n_obs)])

    logo_default_reff = loo_group(idata, group_ids, pointwise=True, var_name="y")
    logo_fixed_reff = loo_group(
        idata, group_ids, reff=0.7, pointwise=True, var_name="y"
    )

    assert not np.isclose(logo_default_reff.elpd_logo, logo_fixed_reff.elpd_logo)

    logo_reff_1 = loo_group(idata, group_ids, reff=1.0, pointwise=True, var_name="y")

    assert not np.all(np.isclose(logo_default_reff.pareto_k, logo_reff_1.pareto_k))


def test_loo_group_nan_values(simple_model):
    """Test loo_group with NaN values in log_likelihood."""
    _, idata = simple_model

    log_like = idata.log_likelihood.y.values.copy()

    log_like[0, 0:10, 5] = np.nan
    log_like[1, 20:30, 15] = np.nan

    modified_log_like = idata.log_likelihood.copy()
    modified_log_like["y"] = (["chain", "draw", "obs_id"], log_like)

    modified_idata = az.InferenceData(
        **{
            group: getattr(idata, group)
            for group in idata._groups
            if group != "log_likelihood"
        },
        log_likelihood=modified_log_like,
    )

    n_obs = modified_idata.log_likelihood.dims["obs_id"]
    n_groups = 5
    group_ids = np.array([i // (n_obs // n_groups) for i in range(n_obs)])

    with warnings.catch_warnings(record=True) as w:
        logo_with_nans = loo_group(modified_idata, group_ids, var_name="y")
        assert any((
            "NaN values detected in log-likelihood" in str(warning.message)
            for warning in w
        ))

    assert hasattr(logo_with_nans, "elpd_logo")
    assert logo_with_nans.n_groups == n_groups


def test_loo_group_input_validation(simple_model):
    """Test loo_group error handling for invalid inputs."""
    _, idata = simple_model

    n_obs = idata.log_likelihood.dims["obs_id"]
    n_groups = 5
    group_size = n_obs // n_groups

    group_ids = np.array([i // group_size for i in range(n_obs)])

    invalid_group_ids = np.repeat(np.arange(n_groups), group_size - 1)[: n_obs - 5]
    with pytest.raises(
        ValueError, match="Length of group_ids .* must match the number of observations"
    ):
        loo_group(idata, invalid_group_ids)

    with pytest.raises(
        TypeError, match='Valid scale values are "deviance", "log", "negative_log"'
    ):
        loo_group(idata, group_ids, scale="invalid_scale")

    with pytest.raises(ValueError, match="Invalid method"):
        loo_group(idata, group_ids, method="invalid_method")


def test_loo_group_problematic_k(problematic_k_model):
    """Test loo_group behavior with problematic Pareto k values."""
    _, idata = problematic_k_model

    n_groups = 5
    group_size = 400
    group_ids = np.repeat(np.arange(n_groups), group_size)

    with warnings.catch_warnings(record=True) as w:
        logo_results = loo_group(idata, group_ids, pointwise=True)
        assert any((
            "Estimated shape parameter of Pareto distribution is greater than"
            in str(warning.message)
            for warning in w
        ))

    assert np.any(logo_results.pareto_k > 0.7)
    assert logo_results.warning


def test_loo_group_compared_to_loo_kfold(simple_model):
    """Compare loo_group results with loo_kfold."""
    model, idata = simple_model
    wrapper = PyMCWrapper(model, idata)

    n_obs = len(idata.log_likelihood.y.obs_id)
    n_groups = 5

    group_ids = np.array([i // (n_obs // n_groups) for i in range(n_obs)])

    logo_results = loo_group(idata, group_ids, var_name="y")

    kfold_indices = []
    for i in range(n_groups):
        fold_idx = np.where(group_ids == i)[0]
        kfold_indices.append(fold_idx)

    kfold_results = loo_kfold(wrapper, kfold_indices=kfold_indices)

    relative_diff = abs(logo_results.elpd_logo - kfold_results.elpd_kfold) / abs(
        logo_results.elpd_logo
    )

    assert relative_diff < 0.3
    assert logo_results.n_groups == len(kfold_indices)

    logging.info(logo_results)
    logging.info(kfold_results)


def test_loo_group_with_custom_groups(shared_variable_model):
    """Test loo_group with custom group definitions."""
    _, idata = shared_variable_model

    n_obs = idata.log_likelihood.dims["obs_id"]
    n_groups = 3

    custom_group_ids1 = np.zeros(n_obs, dtype=int)
    custom_group_ids1[0 : n_obs // 3] = 0
    custom_group_ids1[n_obs // 3 : 2 * n_obs // 3] = 1
    custom_group_ids1[2 * n_obs // 3 :] = 2

    custom_group_ids2 = np.zeros(n_obs, dtype=int)
    for i in range(n_obs):
        custom_group_ids2[i] = i % n_groups

    logo_custom1 = loo_group(idata, custom_group_ids1)

    logo_custom2 = loo_group(idata, custom_group_ids2)

    assert not np.isclose(logo_custom1.elpd_logo, logo_custom2.elpd_logo)

    assert logo_custom1.n_groups == n_groups
    assert logo_custom2.n_groups == n_groups
