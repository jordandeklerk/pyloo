"""Tests for loo_nonfactor module."""

import logging

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest
import xarray as xr
from arviz import InferenceData

from pyloo import loo_nonfactor
from pyloo.elpd import ELPDData


def test_loo_nonfactor_basic():
    """Test basic functionality of loo_nonfactor."""
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
        loo_results = loo_nonfactor(idata, var_name="y", pointwise=True)

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
        pytest.fail(f"loo_nonfactor raised an unexpected exception: {e}")


def test_loo_nonfactor_precision_input():
    """Test loo_nonfactor using precision matrix input."""
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
        loo_results = loo_nonfactor(
            idata, var_name="y", prec_var_name="prec", pointwise=True
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

    except Exception as e:
        pytest.fail(
            f"loo_nonfactor with precision matrix raised an unexpected exception: {e}"
        )


def test_verify_mvn_structure():
    """Test the model structure verification helper function."""
    n_obs = 5
    n_samples = 10
    n_chains = 2

    mu_samples = np.random.randn(n_chains, n_samples, n_obs)
    cov_samples = np.random.randn(n_chains, n_samples, n_obs, n_obs)
    for c in range(n_chains):
        for s in range(n_samples):
            cov_samples[c, s] = cov_samples[c, s] @ cov_samples[c, s].T + np.eye(n_obs)

    idata_valid = InferenceData(
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
        )
    )

    idata_no_mu = InferenceData(
        posterior=xr.Dataset(
            {
                "cov": (("chain", "draw", "obs_dim", "obs_dim_bis"), cov_samples),
            },
            coords={
                "chain": np.arange(n_chains),
                "draw": np.arange(n_samples),
                "obs_dim": np.arange(n_obs),
                "obs_dim_bis": np.arange(n_obs),
            },
        )
    )

    idata_no_cov_prec = InferenceData(
        posterior=xr.Dataset(
            {
                "mu": (("chain", "draw", "obs_dim"), mu_samples),
            },
            coords={
                "chain": np.arange(n_chains),
                "draw": np.arange(n_samples),
                "obs_dim": np.arange(n_obs),
            },
        )
    )

    idata_no_posterior = InferenceData(
        observed_data=xr.Dataset(
            {"y": (("obs_dim",), np.random.randn(n_obs))},
            coords={"obs_dim": np.arange(n_obs)},
        )
    )

    from pyloo.loo_nonfactor import _validate_mvn_structure

    assert _validate_mvn_structure(idata_valid, "mu", None, None) is True

    with pytest.warns(UserWarning, match="Mean vector .* not found"):
        assert (
            _validate_mvn_structure(idata_no_mu, "wrong_mu_name", None, None) is False
        )

    with pytest.warns(
        UserWarning, match="Neither covariance nor precision matrix found"
    ):
        assert (
            _validate_mvn_structure(idata_no_cov_prec, "mu", "wrong_cov", "wrong_prec")
            is False
        )

    assert _validate_mvn_structure(idata_no_posterior, "mu", None, None) is False


def test_loo_nonfactor_warnings():
    """Test that appropriate warnings are raised for incorrect model structures."""
    n_obs = 5
    n_samples = 10
    n_chains = 2

    mu_samples = np.random.randn(n_chains, n_samples, n_obs)
    y_obs_vals = np.random.randn(n_obs)

    idata_missing_cov = InferenceData(
        posterior=xr.Dataset(
            {
                "mu": (("chain", "draw", "obs_dim"), mu_samples),
            },
            coords={
                "chain": np.arange(n_chains),
                "draw": np.arange(n_samples),
                "obs_dim": np.arange(n_obs),
            },
        ),
        observed_data=xr.Dataset(
            {"y": (("obs_dim",), y_obs_vals)},
            coords={"obs_dim": np.arange(n_obs)},
        ),
    )

    with pytest.warns(
        UserWarning,
        match="specifically designed for non-factorized multivariate normal models",
    ):
        with pytest.warns(
            UserWarning, match="Neither covariance nor precision matrix found"
        ):
            with pytest.raises(
                ValueError, match="Could not find posterior samples for covariance"
            ):
                loo_nonfactor(idata_missing_cov, var_name="y")


def test_loo_nonfactor_both_cov_prec():
    """Test that loo_nonfactor works when both covariance and precision matrices are provided."""
    n_obs = 5
    n_samples = 10
    n_chains = 2

    mu_samples = np.random.randn(n_chains, n_samples, n_obs)
    cov_samples = np.empty((n_chains, n_samples, n_obs, n_obs))
    prec_samples = np.empty((n_chains, n_samples, n_obs, n_obs))

    for c in range(n_chains):
        for s in range(n_samples):
            A = np.random.randn(n_obs, n_obs) * 0.1
            cov_samples[c, s, :, :] = A @ A.T + np.eye(n_obs) * 0.5
            prec_samples[c, s, :, :] = np.linalg.inv(cov_samples[c, s, :, :])

    y_obs_vals = np.random.randn(n_obs)

    idata = InferenceData(
        posterior=xr.Dataset(
            {
                "mu": (("chain", "draw", "obs_dim"), mu_samples),
                "cov": (("chain", "draw", "obs_dim", "obs_dim_bis"), cov_samples),
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

    loo_results = loo_nonfactor(idata, var_name="y")
    assert isinstance(loo_results, ELPDData)

    loo_results_prec = loo_nonfactor(
        idata, var_name="y", prec_var_name="prec", cov_var_name=None
    )
    assert isinstance(loo_results_prec, ELPDData)


def test_loo_nonfactor_custom_names():
    """Test loo_nonfactor with custom variable names."""
    n_obs = 5
    n_samples = 10
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
                "mean_vector": (("chain", "draw", "obs_dim"), mu_samples),
                "covariance_matrix": (
                    ("chain", "draw", "obs_dim", "obs_dim_bis"),
                    cov_samples,
                ),
            },
            coords={
                "chain": np.arange(n_chains),
                "draw": np.arange(n_samples),
                "obs_dim": np.arange(n_obs),
                "obs_dim_bis": np.arange(n_obs),
            },
        ),
        observed_data=xr.Dataset(
            {"observations": (("obs_dim",), y_obs_vals)},
            coords={"obs_dim": np.arange(n_obs)},
        ),
    )

    loo_results = loo_nonfactor(
        idata,
        var_name="observations",
        mu_var_name="mean_vector",
        cov_var_name="covariance_matrix",
    )

    assert isinstance(loo_results, ELPDData)
    assert not np.isnan(loo_results.elpd_loo)
    assert not np.isnan(loo_results.p_loo)

    with pytest.warns(UserWarning, match="Mean vector 'wrong_mu' not found"):
        with pytest.raises(ValueError, match="Posterior variable 'wrong_mu' not found"):
            loo_nonfactor(
                idata,
                var_name="observations",
                mu_var_name="wrong_mu",
                cov_var_name="covariance_matrix",
            )


def test_loo_nonfactor_singular_matrices():
    """Test loo_nonfactor handling of singular matrices."""
    n_obs = 5
    n_samples = 10
    n_chains = 2

    mu_samples = np.random.randn(n_chains, n_samples, n_obs)
    cov_samples = np.empty((n_chains, n_samples, n_obs, n_obs))

    for c in range(n_chains):
        for s in range(n_samples):
            if c == 0 and s == 0:
                cov_samples[c, s, :, :] = np.ones((n_obs, n_obs)) * 0.1
            else:
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

    with pytest.warns(UserWarning, match="Covariance matrix is singular"):
        loo_results = loo_nonfactor(idata, var_name="y")
        assert isinstance(loo_results, ELPDData)


@pytest.mark.skipif(pm is None, reason="PyMC not installed")
def test_loo_nonfactor_pymc_model():
    """Test loo_nonfactor with a realistic spatial model using a joint MVN likelihood."""
    np.random.seed(42)
    n_obs = 25

    coords_x = np.random.uniform(0, 10, size=n_obs)
    coords_y = np.random.uniform(0, 10, size=n_obs)

    coords = np.vstack([coords_x, coords_y]).T
    dists = np.zeros((n_obs, n_obs))
    for i in range(n_obs):
        for j in range(n_obs):
            dists[i, j] = np.sqrt(np.sum((coords[i] - coords[j]) ** 2))

    true_sigma = 1.0
    true_ls = 2.0
    true_cov = true_sigma**2 * np.exp(-dists / true_ls)
    true_cov += np.eye(n_obs) * 0.01

    true_mean = 2 + 0.5 * coords_x - 0.3 * coords_y
    y_obs = np.random.multivariate_normal(true_mean, true_cov)

    with pm.Model() as model:
        intercept = pm.Normal("intercept", mu=0, sigma=2)
        beta_x = pm.Normal("beta_x", mu=0, sigma=1)
        beta_y = pm.Normal("beta_y", mu=0, sigma=1)

        mean_values = intercept + beta_x * coords_x + beta_y * coords_y

        pm.Deterministic("mu", mean_values)

        sigma = pm.HalfNormal("sigma", sigma=2)
        length_scale = pm.Gamma("length_scale", alpha=2, beta=1)

        cov_values = sigma**2 * pt.exp(-dists / length_scale)
        cov_values = cov_values + pt.eye(n_obs) * 0.01

        pm.Deterministic("cov", cov_values)
        pm.MvNormal("y_obs", mu=mean_values, cov=cov_values, observed=y_obs)

    with model:
        idata = pm.sample(
            draws=500,
            tune=1000,
            chains=2,
            return_inferencedata=True,
            target_accept=0.95,
        )

        loo_results = loo_nonfactor(
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

        logging.info(f"loo_nonfactor successful: elpd_loo={loo_results.elpd_loo}")
        logging.info(f"p_loo={loo_results.p_loo}")
        logging.info(f"p_loo_se={loo_results.p_loo_se}")

        logging.info(f"{loo_results}")
