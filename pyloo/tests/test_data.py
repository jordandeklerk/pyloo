"""Data fixtures for pyloo tests."""

import numpy as np
import pytest
import xarray as xr
from arviz import InferenceData


@pytest.fixture(scope="session")
def loo_predictive_metric_data():
    """Create test data for loo_predictive_metric tests."""
    n_chains = 4
    n_draws = 250
    n_obs = 10

    rng = np.random.default_rng(42)
    y = rng.normal(0, 1, n_obs)

    mu_samples = rng.normal(y, 0.1, (n_chains, n_draws, n_obs))
    log_lik = -0.5 * np.log(2 * np.pi) - 0.5 * ((y - mu_samples) / 1.0) ** 2

    pp_data = xr.Dataset(
        {"predictions": (["chain", "draw", "observation"], mu_samples)},
        coords={
            "chain": np.arange(n_chains),
            "draw": np.arange(n_draws),
            "observation": np.arange(n_obs),
        },
    )

    ll_data = xr.Dataset(
        {"obs": (["chain", "draw", "observation"], log_lik)},
        coords={
            "chain": np.arange(n_chains),
            "draw": np.arange(n_draws),
            "observation": np.arange(n_obs),
        },
    )

    obs_data = xr.Dataset(
        {"obs": (["observation"], y)}, coords={"observation": np.arange(n_obs)}
    )

    idata = InferenceData(
        posterior_predictive=pp_data, log_likelihood=ll_data, observed_data=obs_data
    )

    pp_samples = idata.posterior_predictive.predictions.stack(
        __sample__=("chain", "draw")
    )
    log_lik_samples = idata.log_likelihood.obs.stack(__sample__=("chain", "draw"))

    return {
        "idata": idata,
        "y": y,
        "pp_samples": pp_samples,
        "log_lik_samples": log_lik_samples,
    }


@pytest.fixture(scope="session")
def loo_predictive_metric_binary_data():
    """Create binary test data for loo_predictive_metric tests."""
    n_chains = 4
    n_draws = 250
    n_obs = 100

    rng = np.random.default_rng(42)
    true_probs = rng.uniform(0.1, 0.9, n_obs)
    y_binary = rng.binomial(1, true_probs, n_obs)

    y_binary_reshaped = y_binary.reshape(1, 1, -1)

    prob_samples = np.zeros((n_chains, n_draws, n_obs))
    for i in range(n_chains):
        for j in range(n_draws):
            prob_samples[i, j] = rng.beta(y_binary + 1, 2 - y_binary)

    log_lik_binary = y_binary_reshaped * np.log(prob_samples) + (
        1 - y_binary_reshaped
    ) * np.log(1 - prob_samples)

    pp_data = xr.Dataset(
        {"predictions": (["chain", "draw", "observation"], prob_samples)},
        coords={
            "chain": np.arange(n_chains),
            "draw": np.arange(n_draws),
            "observation": np.arange(n_obs),
        },
    )

    ll_data = xr.Dataset(
        {"obs": (["chain", "draw", "observation"], log_lik_binary)},
        coords={
            "chain": np.arange(n_chains),
            "draw": np.arange(n_draws),
            "observation": np.arange(n_obs),
        },
    )

    obs_data = xr.Dataset(
        {"obs": (["observation"], y_binary)}, coords={"observation": np.arange(n_obs)}
    )

    idata = InferenceData(
        posterior_predictive=pp_data, log_likelihood=ll_data, observed_data=obs_data
    )

    pp_samples = idata.posterior_predictive.predictions.stack(
        __sample__=("chain", "draw")
    )
    log_lik_samples = idata.log_likelihood.obs.stack(__sample__=("chain", "draw"))

    return {
        "idata": idata,
        "y_binary": y_binary,
        "pp_samples": pp_samples,
        "log_lik_samples": log_lik_samples,
    }


@pytest.fixture(scope="session")
def prepare_inference_data_for_crps(centered_eight):
    """Prepare inference data for CRPS calculations using real model data."""
    idata = centered_eight
    pp_orig = idata.posterior_predictive.obs.values

    rng = np.random.default_rng(42)
    pp_new = pp_orig + rng.normal(0, 0.1, size=pp_orig.shape)

    pp_combined = xr.Dataset(
        {
            "obs": (["chain", "draw", "school"], pp_orig),
            "obs2": (["chain", "draw", "school"], pp_new),
        },
        coords=idata.posterior_predictive.coords,
        attrs=idata.posterior_predictive.attrs,
    )

    new_idata = InferenceData(
        posterior=idata.posterior,
        posterior_predictive=pp_combined,
        log_likelihood=idata.log_likelihood,
        observed_data=idata.observed_data,
    )

    assert hasattr(new_idata, "posterior_predictive")
    assert "obs" in new_idata.posterior_predictive.data_vars
    assert "obs2" in new_idata.posterior_predictive.data_vars

    return new_idata


@pytest.fixture(scope="session")
def mvn_inference_data():
    """Create InferenceData with multivariate normal model data."""
    rng = np.random.default_rng(42)
    n_obs = 10
    n_samples = 100
    n_chains = 2

    mu_samples = rng.normal(0, 1, size=(n_chains, n_samples, n_obs))
    cov_samples = np.empty((n_chains, n_samples, n_obs, n_obs))

    for c in range(n_chains):
        for s in range(n_samples):
            A = rng.normal(0, 0.1, size=(n_obs, n_obs))
            cov_samples[c, s, :, :] = A @ A.T + np.eye(n_obs) * 0.5

    y_obs_vals = rng.normal(0, 1, size=n_obs)

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

    return idata


@pytest.fixture(scope="session")
def mvt_inference_data():
    """Create InferenceData with multivariate Student-t model data."""
    rng = np.random.default_rng(42)
    n_obs = 10
    n_samples = 100
    n_chains = 2

    mu_samples = rng.normal(0, 1, size=(n_chains, n_samples, n_obs))
    cov_samples = np.empty((n_chains, n_samples, n_obs, n_obs))
    df_samples = np.abs(rng.gamma(5, 1, size=(n_chains, n_samples))) + 2.0

    for c in range(n_chains):
        for s in range(n_samples):
            A = rng.normal(0, 0.1, size=(n_obs, n_obs))
            cov_samples[c, s, :, :] = A @ A.T + np.eye(n_obs) * 0.5

    y_obs_vals = rng.normal(0, 1, size=n_obs)

    idata = InferenceData(
        posterior=xr.Dataset(
            {
                "mu": (("chain", "draw", "obs_dim"), mu_samples),
                "cov": (("chain", "draw", "obs_dim", "obs_dim_bis"), cov_samples),
                "df": (("chain", "draw"), df_samples),
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

    return idata


@pytest.fixture(scope="session")
def mvn_precision_data():
    """Create InferenceData with precision matrix instead of covariance."""
    rng = np.random.default_rng(42)
    n_obs = 8
    n_samples = 120
    n_chains = 2

    mu_samples = rng.normal(0, 1, size=(n_chains, n_samples, n_obs))
    prec_samples = np.empty((n_chains, n_samples, n_obs, n_obs))

    for c in range(n_chains):
        for s in range(n_samples):
            A = rng.normal(0, 0.1, size=(n_obs, n_obs))
            cov_s = A @ A.T + np.eye(n_obs) * 0.5
            try:
                prec_samples[c, s, :, :] = np.linalg.inv(cov_s)
            except np.linalg.LinAlgError:
                prec_samples[c, s, :, :] = np.eye(n_obs)

    y_obs_vals = rng.normal(0, 1, size=n_obs)

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

    return idata


@pytest.fixture(scope="session")
def mvt_precision_data():
    """Create InferenceData with Student-t model using precision matrix."""
    rng = np.random.default_rng(42)
    n_obs = 8
    n_samples = 120
    n_chains = 2

    mu_samples = rng.normal(0, 1, size=(n_chains, n_samples, n_obs))
    prec_samples = np.empty((n_chains, n_samples, n_obs, n_obs))
    df_samples = np.abs(rng.gamma(5, 1, size=(n_chains, n_samples))) + 2.0

    for c in range(n_chains):
        for s in range(n_samples):
            A = rng.normal(0, 0.1, size=(n_obs, n_obs))
            cov_s = A @ A.T + np.eye(n_obs) * 0.5
            try:
                prec_samples[c, s, :, :] = np.linalg.inv(cov_s)
            except np.linalg.LinAlgError:
                prec_samples[c, s, :, :] = np.eye(n_obs)

    y_obs_vals = rng.normal(0, 1, size=n_obs)

    idata = InferenceData(
        posterior=xr.Dataset(
            {
                "mu": (("chain", "draw", "obs_dim"), mu_samples),
                "prec": (("chain", "draw", "obs_dim", "obs_dim_bis"), prec_samples),
                "df": (("chain", "draw"), df_samples),
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

    return idata


@pytest.fixture(scope="session")
def mvn_custom_names_data():
    """Create InferenceData with custom variable names."""
    rng = np.random.default_rng(42)
    n_obs = 5
    n_samples = 10
    n_chains = 2

    mu_samples = rng.normal(0, 1, size=(n_chains, n_samples, n_obs))
    cov_samples = np.empty((n_chains, n_samples, n_obs, n_obs))

    for c in range(n_chains):
        for s in range(n_samples):
            A = rng.normal(0, 0.1, size=(n_obs, n_obs))
            cov_samples[c, s, :, :] = A @ A.T + np.eye(n_obs) * 0.5

    y_obs_vals = rng.normal(0, 1, size=n_obs)

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

    return idata


@pytest.fixture(scope="session")
def mvt_custom_names_data():
    """Create InferenceData with Student-t model using custom variable names."""
    rng = np.random.default_rng(42)
    n_obs = 5
    n_samples = 10
    n_chains = 2

    mu_samples = rng.normal(0, 1, size=(n_chains, n_samples, n_obs))
    cov_samples = np.empty((n_chains, n_samples, n_obs, n_obs))
    df_samples = np.abs(rng.gamma(5, 1, size=(n_chains, n_samples))) + 2.0

    for c in range(n_chains):
        for s in range(n_samples):
            A = rng.normal(0, 0.1, size=(n_obs, n_obs))
            cov_samples[c, s, :, :] = A @ A.T + np.eye(n_obs) * 0.5

    y_obs_vals = rng.normal(0, 1, size=n_obs)

    idata = InferenceData(
        posterior=xr.Dataset(
            {
                "location": (("chain", "draw", "obs_dim"), mu_samples),
                "scale_matrix": (
                    ("chain", "draw", "obs_dim", "obs_dim_bis"),
                    cov_samples,
                ),
                "nu": (("chain", "draw"), df_samples),
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

    return idata


@pytest.fixture(scope="session")
def mvt_negative_df_data():
    """Create InferenceData with Student-t model including negative degrees of freedom."""
    rng = np.random.default_rng(42)
    n_obs = 5
    n_samples = 10
    n_chains = 2

    mu_samples = rng.normal(0, 1, size=(n_chains, n_samples, n_obs))
    cov_samples = np.empty((n_chains, n_samples, n_obs, n_obs))

    df_samples = np.abs(rng.gamma(5, 1, size=(n_chains, n_samples))) + 2.0
    df_samples[0, 0] = -1.0

    for c in range(n_chains):
        for s in range(n_samples):
            A = rng.normal(0, 0.1, size=(n_obs, n_obs))
            cov_samples[c, s, :, :] = A @ A.T + np.eye(n_obs) * 0.5

    y_obs_vals = rng.normal(0, 1, size=n_obs)

    idata = InferenceData(
        posterior=xr.Dataset(
            {
                "mu": (("chain", "draw", "obs_dim"), mu_samples),
                "cov": (("chain", "draw", "obs_dim", "obs_dim_bis"), cov_samples),
                "df": (("chain", "draw"), df_samples),
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

    return idata


@pytest.fixture(scope="session")
def singular_matrix_data():
    """Create InferenceData with singular covariance matrix."""
    rng = np.random.default_rng(42)
    n_obs = 5
    n_samples = 10
    n_chains = 2

    mu_samples = rng.normal(0, 1, size=(n_chains, n_samples, n_obs))
    cov_samples = np.empty((n_chains, n_samples, n_obs, n_obs))

    for c in range(n_chains):
        for s in range(n_samples):
            if c == 0 and s == 0:
                cov_samples[c, s, :, :] = np.ones((n_obs, n_obs)) * 0.1
            else:
                A = rng.normal(0, 0.1, size=(n_obs, n_obs))
                cov_samples[c, s, :, :] = A @ A.T + np.eye(n_obs) * 0.5

    y_obs_vals = rng.normal(0, 1, size=n_obs)

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

    return idata


@pytest.fixture(scope="session")
def both_cov_prec_data():
    """Create InferenceData with both covariance and precision matrices."""
    rng = np.random.default_rng(42)
    n_obs = 5
    n_samples = 10
    n_chains = 2

    mu_samples = rng.normal(0, 1, size=(n_chains, n_samples, n_obs))
    cov_samples = np.empty((n_chains, n_samples, n_obs, n_obs))
    prec_samples = np.empty((n_chains, n_samples, n_obs, n_obs))

    for c in range(n_chains):
        for s in range(n_samples):
            A = rng.normal(0, 0.1, size=(n_obs, n_obs))
            cov_samples[c, s, :, :] = A @ A.T + np.eye(n_obs) * 0.5
            prec_samples[c, s, :, :] = np.linalg.inv(cov_samples[c, s, :, :])

    y_obs_vals = rng.normal(0, 1, size=n_obs)

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

    return idata


@pytest.fixture(scope="session")
def missing_cov_data():
    """Create InferenceData with missing covariance matrix."""
    rng = np.random.default_rng(42)
    n_obs = 5
    n_samples = 10
    n_chains = 2

    mu_samples = rng.normal(0, 1, size=(n_chains, n_samples, n_obs))
    y_obs_vals = rng.normal(0, 1, size=n_obs)

    idata = InferenceData(
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

    return idata


@pytest.fixture(scope="session")
def mvn_validation_data():
    """Create data for testing model structure validation."""
    rng = np.random.default_rng(42)
    n_obs = 5
    n_samples = 10
    n_chains = 2

    mu_samples = rng.normal(0, 1, size=(n_chains, n_samples, n_obs))
    cov_samples = rng.normal(0, 1, size=(n_chains, n_samples, n_obs, n_obs))

    for c in range(n_chains):
        for s in range(n_samples):
            cov_samples[c, s] = cov_samples[c, s] @ cov_samples[c, s].T + np.eye(n_obs)

    valid_idata = InferenceData(
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

    no_mu_idata = InferenceData(
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

    no_cov_prec_idata = InferenceData(
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

    no_posterior_idata = InferenceData(
        observed_data=xr.Dataset(
            {"y": (("obs_dim",), rng.normal(0, 1, size=n_obs))},
            coords={"obs_dim": np.arange(n_obs)},
        )
    )

    return {
        "valid": valid_idata,
        "no_mu": no_mu_idata,
        "no_cov_prec": no_cov_prec_idata,
        "no_posterior": no_posterior_idata,
    }


@pytest.fixture(scope="session")
def mvt_validation_data():
    """Create data for testing Student-t model structure validation."""
    rng = np.random.default_rng(42)
    n_obs = 5
    n_samples = 10
    n_chains = 2

    mu_samples = rng.normal(0, 1, size=(n_chains, n_samples, n_obs))
    cov_samples = rng.normal(0, 1, size=(n_chains, n_samples, n_obs, n_obs))
    df_samples = np.abs(rng.gamma(5, 1, size=(n_chains, n_samples))) + 2.0

    for c in range(n_chains):
        for s in range(n_samples):
            cov_samples[c, s] = cov_samples[c, s] @ cov_samples[c, s].T + np.eye(n_obs)

    valid_idata = InferenceData(
        posterior=xr.Dataset(
            {
                "mu": (("chain", "draw", "obs_dim"), mu_samples),
                "cov": (("chain", "draw", "obs_dim", "obs_dim_bis"), cov_samples),
                "df": (("chain", "draw"), df_samples),
            },
            coords={
                "chain": np.arange(n_chains),
                "draw": np.arange(n_samples),
                "obs_dim": np.arange(n_obs),
                "obs_dim_bis": np.arange(n_obs),
            },
        )
    )

    missing_df_idata = InferenceData(
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

    return {"valid": valid_idata, "missing_df": missing_df_idata}
