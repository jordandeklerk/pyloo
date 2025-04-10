"""Model fixtures for pyloo tests."""

import os

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import pytest
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation
from pymc_marketing.prior import Prior


@pytest.fixture(scope="session")
def hierarchical_model():
    """Create a hierarchical model with multiple observations for testing."""
    rng = np.random.default_rng(42)

    n_groups = 8
    n_points = 20

    alpha = 0.8
    beta = 1.2
    group_effects = rng.normal(0, 0.5, size=n_groups)

    X = rng.normal(0, 1, size=(n_groups, n_points))

    Y = (
        alpha
        + group_effects[:, None]
        + beta * X
        + rng.normal(0, 0.2, size=(n_groups, n_points))
    )

    coords = {"group": range(n_groups), "obs_id": range(n_points)}

    with pm.Model(coords=coords) as model:
        alpha = pm.Normal("alpha", mu=0, sigma=2)
        beta = pm.Normal("beta", mu=0, sigma=2)
        group_sigma = pm.HalfNormal("group_sigma", sigma=0.5)

        group_effects_raw = pm.Normal("group_effects_raw", mu=0, sigma=1, dims="group")
        group_effects = pm.Deterministic(
            "group_effects", group_effects_raw * group_sigma, dims="group"
        )

        mu = alpha + group_effects[:, None] + beta * X

        sigma_y = pm.HalfNormal("sigma_y", sigma=0.5)
        pm.Normal("Y", mu=mu, sigma=sigma_y, observed=Y, dims=("group", "obs_id"))

        idata = pm.sample(
            1000,
            tune=2000,
            target_accept=0.95,
            random_seed=42,
            idata_kwargs={"log_likelihood": True},
        )

    return model, idata


@pytest.fixture(scope="session")
def hierarchical_model_no_coords():
    """Create a hierarchical model without explicit coordinates."""
    rng = np.random.default_rng(42)

    n_groups = 8
    n_points = 20

    alpha = 0.8
    beta = 1.2
    group_effects = rng.normal(0, 0.5, size=n_groups)

    X = rng.normal(0, 1, size=(n_groups, n_points))
    Y = (
        alpha
        + group_effects[:, None]
        + beta * X
        + rng.normal(0, 0.2, size=(n_groups, n_points))
    )

    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0, sigma=2)
        beta = pm.Normal("beta", mu=0, sigma=2)
        group_sigma = pm.HalfNormal("group_sigma", sigma=0.5)

        group_effects_raw = pm.Normal(
            "group_effects_raw", mu=0, sigma=1, shape=n_groups
        )
        group_effects = pm.Deterministic(
            "group_effects", group_effects_raw * group_sigma
        )

        mu = alpha + group_effects[:, None] + beta * X

        sigma_y = pm.HalfNormal("sigma_y", sigma=0.5)
        pm.Normal("Y", mu=mu, sigma=sigma_y, observed=Y)

        idata = pm.sample(
            1000,
            tune=2000,
            target_accept=0.95,
            random_seed=42,
            idata_kwargs={"log_likelihood": True},
        )

    return model, idata


@pytest.fixture(scope="session")
def simple_model():
    """Create a simple linear regression model for testing."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, size=100)
    true_alpha = 1.0
    true_beta = 2.0
    true_sigma = 1.0
    y = true_alpha + true_beta * X + rng.normal(0, true_sigma, size=100)

    with pm.Model(coords={"obs_id": range(len(X))}) as model:
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=10)
        sigma = pm.HalfNormal("sigma", sigma=10)

        mu = alpha + beta * X

        pm.Normal("y", mu=mu, sigma=sigma, observed=y, dims="obs_id")

        idata = pm.sample(
            1000, tune=1000, random_seed=42, idata_kwargs={"log_likelihood": True}
        )

    return model, idata


@pytest.fixture(scope="session")
def poisson_model():
    """Create a Poisson regression model for testing different likelihood types."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, size=100)
    true_alpha = 0.5
    true_beta = 0.3

    lambda_rate = np.exp(true_alpha + true_beta * X)
    y = rng.poisson(lambda_rate)

    with pm.Model(coords={"obs_id": range(len(X))}) as model:
        alpha = pm.Normal("alpha", mu=0, sigma=1)
        beta = pm.Normal("beta", mu=0, sigma=1)

        lambda_rate = pm.math.exp(alpha + beta * X)
        pm.Poisson("y", mu=lambda_rate, observed=y, dims="obs_id")

        idata = pm.sample(
            1000, tune=1000, random_seed=42, idata_kwargs={"log_likelihood": True}
        )

    return model, idata


@pytest.fixture(scope="session")
def multi_observed_model():
    """Create a model with multiple observed variables."""
    rng = np.random.default_rng(42)
    n_samples = 100

    X = rng.normal(0, 1, size=n_samples)
    true_alpha = 1.0
    true_beta = 2.0
    true_sigma1 = 0.5
    true_sigma2 = 0.8

    y1 = true_alpha + true_beta * X + rng.normal(0, true_sigma1, size=n_samples)
    y2 = true_alpha - true_beta * X + rng.normal(0, true_sigma2, size=n_samples)

    with pm.Model(coords={"obs_id": range(n_samples)}) as model:
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=10)
        sigma1 = pm.HalfNormal("sigma1", sigma=10)
        sigma2 = pm.HalfNormal("sigma2", sigma=10)

        mu1 = alpha + beta * X
        mu2 = alpha - beta * X

        pm.Normal("y1", mu=mu1, sigma=sigma1, observed=y1, dims="obs_id")
        pm.Normal("y2", mu=mu2, sigma=sigma2, observed=y2, dims="obs_id")

        idata = pm.sample(
            1000, tune=1000, random_seed=42, idata_kwargs={"log_likelihood": True}
        )

    return model, idata


@pytest.fixture(scope="session")
def shared_variable_model():
    """Create a model with shared variables across observations."""
    rng = np.random.default_rng(42)
    n_groups = 3
    n_per_group = 50

    shared_effect = 0.5
    group_effects = rng.normal(0, 0.3, size=n_groups)

    X = []
    y = []
    groups = []

    for i in range(n_groups):
        X_group = rng.normal(0, 1, size=n_per_group)
        y_group = (
            shared_effect
            + group_effects[i]
            + 0.2 * X_group
            + rng.normal(0, 0.1, size=n_per_group)
        )

        X.extend(X_group)
        y.extend(y_group)
        groups.extend([i] * n_per_group)

    X = np.array(X)
    y = np.array(y)
    groups = np.array(groups)

    coords = {"obs_id": range(len(X)), "group": range(n_groups)}

    with pm.Model(coords=coords) as model:
        shared_effect = pm.Normal("shared_effect", mu=0, sigma=1)
        group_effects = pm.Normal("group_effects", mu=0, sigma=0.5, dims="group")
        beta = pm.Normal("beta", mu=0, sigma=1)
        sigma = pm.HalfNormal("sigma", sigma=0.5)

        mu = shared_effect + group_effects[groups] + beta * X
        pm.Normal("y", mu=mu, sigma=sigma, observed=y, dims="obs_id")

        idata = pm.sample(
            1000, tune=1000, random_seed=42, idata_kwargs={"log_likelihood": True}
        )

    return model, idata


@pytest.fixture(scope="session")
def bernoulli_model():
    """Create a Bernoulli model for testing different likelihood types."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, size=100)
    true_alpha = 0.5
    true_beta = 0.3

    p = 1 / (1 + np.exp(-(true_alpha + true_beta * X)))
    y = rng.binomial(1, p)

    with pm.Model(coords={"obs_id": range(len(X))}) as model:
        alpha = pm.Normal("alpha", mu=0, sigma=1)
        beta = pm.Normal("beta", mu=0, sigma=1)

        p = pm.math.invlogit(alpha + beta * X)
        pm.Bernoulli("y", p=p, observed=y, dims="obs_id")

        idata = pm.sample(
            1000, tune=1000, random_seed=42, idata_kwargs={"log_likelihood": True}
        )

    return model, idata


@pytest.fixture(scope="session")
def large_regression_model():
    """Create a large linear regression model with many observations for K-fold testing."""
    rng = np.random.default_rng(42)
    n_obs = 1000
    n_predictors = 3

    X = rng.normal(0, 1, size=(n_obs, n_predictors))

    true_alpha = 1.5
    true_betas = np.array([0.8, -0.5, 1.2])
    true_sigma = 0.7

    y = true_alpha + X @ true_betas + rng.normal(0, true_sigma, size=n_obs)

    with pm.Model(
        coords={"obs_id": range(n_obs), "predictor": range(n_predictors)}
    ) as model:
        alpha = pm.Normal("alpha", mu=0, sigma=5)
        betas = pm.Normal("betas", mu=0, sigma=2, dims="predictor")
        sigma = pm.HalfNormal("sigma", sigma=3)

        mu = alpha + pm.math.dot(X, betas)

        pm.Normal("y", mu=mu, sigma=sigma, observed=y, dims="obs_id")

        idata = pm.sample(
            draws=1000, tune=500, random_seed=42, idata_kwargs={"log_likelihood": True}
        )

    return model, idata


@pytest.fixture(scope="session")
def student_t_model():
    """Create a Student's T model for testing different likelihood types."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, size=100)
    true_alpha = 1.0
    true_beta = 2.0
    true_sigma = 1.0
    true_nu = 4.0

    y = true_alpha + true_beta * X + rng.standard_t(true_nu) * true_sigma

    with pm.Model(coords={"obs_id": range(len(X))}) as model:
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=10)
        sigma = pm.HalfNormal("sigma", sigma=10)
        nu = pm.Exponential("nu", lam=1 / 10)

        mu = alpha + beta * X
        pm.StudentT("y", nu=nu, mu=mu, sigma=sigma, observed=y, dims="obs_id")

        idata = pm.sample(
            1000, tune=1000, random_seed=42, idata_kwargs={"log_likelihood": True}
        )

    return model, idata


@pytest.fixture(scope="session")
def mixture_model():
    """Create a mixture model for testing mixture likelihoods."""
    rng = np.random.default_rng(42)
    n_samples = 200

    true_w = 0.7
    true_mu1 = -2.0
    true_mu2 = 3.0
    true_sigma1 = 0.5
    true_sigma2 = 1.0

    component = rng.binomial(1, true_w, size=n_samples)
    y = np.zeros(n_samples)
    y[component == 1] = rng.normal(true_mu1, true_sigma1, size=np.sum(component == 1))
    y[component == 0] = rng.normal(true_mu2, true_sigma2, size=np.sum(component == 0))

    with pm.Model(coords={"obs_id": range(n_samples)}) as model:
        w = pm.Beta("w", alpha=2, beta=2)

        mu1 = pm.Normal("mu1", mu=0, sigma=5)
        mu2 = pm.Normal("mu2", mu=0, sigma=5)

        sigma1 = pm.HalfNormal("sigma1", sigma=2)
        sigma2 = pm.HalfNormal("sigma2", sigma=2)

        comp1 = pm.Normal.dist(mu=mu1, sigma=sigma1)
        comp2 = pm.Normal.dist(mu=mu2, sigma=sigma2)

        pm.Mixture(
            "y", w=[w, 1 - w], comp_dists=[comp1, comp2], observed=y, dims="obs_id"
        )

        idata = pm.sample(
            1000,
            tune=1000,
            target_accept=0.9,
            random_seed=42,
            idata_kwargs={"log_likelihood": True},
        )

    return model, idata


@pytest.fixture(scope="session")
def problematic_k_model():
    """Create a model that generates problematic pareto k values for LOO-CV testing."""
    rng = np.random.default_rng(42)

    n_obs = 2000
    true_alpha = 1.0
    true_beta = 2.0
    true_sigma = 0.1
    df = 2

    X = rng.normal(0, 1, size=n_obs)

    outlier_indices = np.arange(0, n_obs, 10)
    X[outlier_indices] = rng.normal(5, 2, size=len(outlier_indices))

    noise = np.zeros(n_obs)
    regular_indices = np.ones(n_obs, dtype=bool)
    regular_indices[outlier_indices] = False

    noise[regular_indices] = rng.normal(0, true_sigma, size=regular_indices.sum())
    noise[outlier_indices] = (
        rng.standard_t(df, size=len(outlier_indices)) * true_sigma * 10
    )

    extreme_indices = np.arange(5, n_obs, 50)
    noise[extreme_indices] = (
        rng.standard_t(df, size=len(extreme_indices)) * true_sigma * 20
    )

    y = true_alpha + true_beta * X + noise

    with pm.Model(coords={"obs_id": range(n_obs)}) as model:
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=10)
        sigma = pm.HalfNormal("sigma", sigma=10)

        mu = alpha + beta * X
        pm.Normal("y", mu=mu, sigma=sigma, observed=y, dims="obs_id")

        idata = pm.sample(
            1000,
            tune=2000,
            target_accept=0.95,
            random_seed=42,
            idata_kwargs={"log_likelihood": True},
        )

    return model, idata


@pytest.fixture(scope="session")
def roaches_model():
    """Create a model for the roaches dataset."""
    file_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "data",
        "roaches.csv",
    )
    data = pd.read_csv(file_path)
    data["roach1"] = np.sqrt(data["roach1"])

    X = data[["roach1", "treatment", "senior"]].values
    y = data["y"].values
    offset = np.log(data["exposure2"]).values
    K = X.shape[1]

    beta_prior_scale = 2.5
    alpha_prior_scale = 5.0

    with pm.Model() as model:
        beta = pm.Normal("beta", mu=0, sigma=beta_prior_scale, shape=K)
        intercept = pm.Normal("intercept", mu=0, sigma=alpha_prior_scale)
        eta = pm.math.dot(X, beta) + intercept + offset
        pm.Poisson("y", mu=pm.math.exp(eta), observed=y)

        idata = pm.sample(
            chains=4,
            draws=1000,
            tune=1000,
            target_accept=0.9,
            random_seed=42,
            idata_kwargs={"log_likelihood": True},
        )

    return model, idata


@pytest.fixture(scope="session")
def approximate_posterior_model():
    """Create a model with an approximate posterior (ADVI) for testing."""
    rng = np.random.default_rng(42)
    n_samples = 5000

    X = rng.normal(0, 1, size=n_samples)
    true_alpha = 0.5
    true_beta = 1.5
    true_sigma = 0.8
    y = true_alpha + true_beta * X + rng.normal(0, true_sigma, size=n_samples)

    with pm.Model(coords={"obs_id": range(n_samples)}) as model:
        alpha = pm.Normal("alpha", mu=0, sigma=2)
        beta = pm.Normal("beta", mu=0, sigma=2)
        sigma = pm.HalfNormal("sigma", sigma=2)

        mu = alpha + beta * X
        pm.Normal("y", mu=mu, sigma=sigma, observed=y, dims="obs_id")

        approx = pm.fit(
            method="advi",
            n=1000,
            random_seed=42,
        )

        idata = approx.sample(1000, return_inferencedata=True)
        pm.compute_log_likelihood(idata, model=model, extend_inferencedata=True)

    return model, idata, approx


@pytest.fixture(scope="session")
def wells_model():
    """Create a logistic regression model for the arsenic wells dataset."""
    file_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "wells.csv"
    )
    data = pd.read_csv(file_path)

    y = data["switch"].values

    data["dist100"] = data["dist"] / 100

    X = np.column_stack([np.ones(len(data)), data[["dist100", "arsenic"]].values])

    P = X.shape[1]
    N = len(y)

    with pm.Model(coords={"obs_id": range(N), "predictor": range(P)}) as model:
        beta = pm.Normal("beta", mu=0, sigma=1, dims="predictor")

        eta = pm.math.dot(X, beta)

        pm.Bernoulli("y", logit_p=eta, observed=y, dims="obs_id")

        idata = pm.sample(
            chains=4,
            draws=1000,
            tune=1000,
            target_accept=0.9,
            random_seed=42,
            idata_kwargs={"log_likelihood": True},
        )

    return model, idata


@pytest.fixture(scope="session")
def high_dimensional_regression_model():
    """Create a high-dimensional linear regression model with many predictors."""
    rng = np.random.default_rng(42)
    n_obs = 50
    n_predictors = 60

    X = rng.normal(0, 1, size=(n_obs, n_predictors))

    true_alpha = 1.5
    true_betas = rng.normal(0, 1, size=n_predictors)
    true_sigma = 0.7

    y = true_alpha + X @ true_betas + rng.normal(0, true_sigma, size=n_obs)

    with pm.Model(
        coords={"obs_id": range(n_obs), "predictor": range(n_predictors)}
    ) as model:
        alpha = pm.Normal("alpha", mu=0, sigma=5)
        betas = pm.Normal("betas", mu=0, sigma=1, dims="predictor")
        sigma = pm.HalfNormal("sigma", sigma=3)

        mu = alpha + pm.math.dot(X, betas)

        pm.Normal("y", mu=mu, sigma=sigma, observed=y, dims="obs_id")

        idata = pm.sample(
            draws=1000, tune=500, random_seed=42, idata_kwargs={"log_likelihood": True}
        )

    return model, idata


@pytest.fixture(scope="session")
def mvn_spatial_model():
    """Create a PyMC spatial model with multivariate normal likelihood."""
    if pm is None:
        pytest.skip("PyMC not installed")

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

        idata = pm.sample(
            draws=500,
            tune=1000,
            chains=2,
            return_inferencedata=True,
            target_accept=0.95,
        )

    return model, idata, coords_x, coords_y, dists


@pytest.fixture(scope="session")
def mvt_spatial_model():
    """Create a PyMC spatial model with multivariate Student-t likelihood."""
    if pm is None:
        pytest.skip("PyMC not installed")

    np.random.seed(42)
    n_obs = 20

    coords_x = np.random.uniform(0, 10, size=n_obs)
    coords_y = np.random.uniform(0, 10, size=n_obs)

    coords = np.vstack([coords_x, coords_y]).T
    dists = np.zeros((n_obs, n_obs))
    for i in range(n_obs):
        for j in range(n_obs):
            dists[i, j] = np.sqrt(np.sum((coords[i] - coords[j]) ** 2))

    true_df = 4.0
    true_sigma = 1.0
    true_ls = 2.0
    true_cov = true_sigma**2 * np.exp(-dists / true_ls)
    true_cov += np.eye(n_obs) * 0.01

    true_mean = 2 + 0.5 * coords_x - 0.3 * coords_y

    z = np.random.multivariate_normal(np.zeros(n_obs), true_cov)
    chi2 = np.random.chisquare(true_df) / true_df
    y_obs = true_mean + z / np.sqrt(chi2)

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

        df = pm.Gamma("df", alpha=2, beta=0.5)
        pm.Deterministic("df_det", df)

        pm.MvStudentT("y_obs", nu=df, mu=mean_values, scale=cov_values, observed=y_obs)

        idata = pm.sample(
            draws=500,
            tune=1000,
            chains=2,
            return_inferencedata=True,
            target_accept=0.95,
        )

    return model, idata, coords_x, coords_y, dists


@pytest.fixture(scope="session")
def mmm_model():
    """Create a media mix model for testing."""
    rng = np.random.default_rng(42)

    n_weeks = 52 * 3

    x1 = rng.uniform(0, 1, size=n_weeks)
    x2 = rng.uniform(0, 1, size=n_weeks)
    x1[x1 > 0.9] *= 2
    x2[x2 < 0.2] = 0

    t = np.arange(n_weeks)
    dayofyear = np.linspace(0, 365, n_weeks)
    seasonality = 0.5 * (
        np.sin(2 * np.pi * dayofyear / 365) + np.cos(2 * np.pi * dayofyear / 365)
    )

    event_1 = np.zeros(n_weeks)
    event_1[20] = 1

    beta_1, beta_2 = 2.0, 1.5
    trend_coef = 0.02
    event_effect = 2.0

    y = (
        1.0
        + trend_coef * t
        + seasonality
        + event_effect * event_1
        + beta_1 * x1
        + beta_2 * x2
        + rng.normal(0, 0.2, size=n_weeks)
    )

    data = pd.DataFrame({
        "date_week": pd.date_range(start="2023-01-01", periods=n_weeks, freq="W-MON"),
        "x1": x1,
        "x2": x2,
        "t": t,
        "event_1": event_1,
        "y": y,
    })

    model_config = {
        "intercept": Prior("Normal", mu=0.0, sigma=1.0),
        "saturation_beta": Prior("HalfNormal", sigma=[1.0, 1.0]),
        "gamma_control": Prior("Normal", mu=0, sigma=0.5),
        "gamma_fourier": Prior("Laplace", mu=0, b=0.2),
        "likelihood": Prior("Normal", sigma=Prior("HalfNormal", sigma=1.0)),
    }

    mmm = MMM(
        model_config=model_config,
        date_column="date_week",
        adstock=GeometricAdstock(l_max=4),
        saturation=LogisticSaturation(),
        channel_columns=["x1", "x2"],
        control_columns=["event_1", "t"],
        yearly_seasonality=1,
    )

    X = data.drop("y", axis=1)
    y = data["y"]

    idata = mmm.fit(
        X=X,
        y=y,
        chains=2,
        draws=500,
        tune=500,
        target_accept=0.9,
        random_seed=42,
        idata_kwargs={"log_likelihood": True},
    )

    return mmm.model, idata
