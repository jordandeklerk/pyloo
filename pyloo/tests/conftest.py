"""Shared fixtures for pyloo tests."""

import os

import arviz as az
import numpy as np
import pymc as pm
import pytest


@pytest.fixture(scope="session")
def rng():
    """Return a numpy random number generator with fixed seed."""
    return np.random.default_rng(44)


@pytest.fixture(scope="session")
def centered_eight():
    """Return the centered_eight dataset from ArviZ."""
    return az.load_arviz_data("centered_eight")


@pytest.fixture(scope="session")
def non_centered_eight():
    """Return the non_centered_eight dataset from ArviZ."""
    return az.load_arviz_data("non_centered_eight")


def pytest_addoption(parser):
    """Add command line options to pytest."""
    parser.addoption(
        "--save",
        action="store_true",
        default=False,
        help="Save output from tests",
    )


@pytest.fixture(scope="session")
def save_dir(request):
    """Return directory for saving test output."""
    save = request.config.getoption("--save")
    if save:
        directory = "test_output"
        os.makedirs(directory, exist_ok=True)
        return directory
    return None


@pytest.fixture(scope="session")
def numpy_arrays(rng):
    """Return common numpy arrays used in tests."""
    return {
        "normal": rng.normal(size=1000),
        "uniform": rng.uniform(size=1000),
        "random_weights": rng.normal(size=(1000, 8)),
        "random_ratios": rng.normal(size=(2000, 5)),
    }


@pytest.fixture(scope="session")
def log_likelihood_data(centered_eight):
    """Return log likelihood data for PSIS tests."""
    return centered_eight.log_likelihood.obs.stack(__sample__=("chain", "draw"))


@pytest.fixture(scope="session")
def multidim_data(rng):
    """Return multidimensional data for testing."""
    return {
        "llm": rng.normal(size=(4, 23, 15, 2)),  # chain, draw, dim1, dim2
        "ll1": rng.normal(size=(4, 23, 30)),  # chain, draw, combined_dims
    }


@pytest.fixture(scope="session")
def extreme_data(log_likelihood_data):
    """Return data with extreme values for testing edge cases."""
    data = log_likelihood_data.values.T.copy()
    data[:, 1] = 10  # Add extreme values
    return data


@pytest.fixture(scope="module")
def eight_schools_params():
    """Share setup for eight schools."""
    return {
        "J": 8,
        "y": np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]),
        "sigma": np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]),
    }


@pytest.fixture(scope="module")
def draws():
    """Share default draw count."""
    return 500


@pytest.fixture(scope="module")
def chains():
    """Share default chain count."""
    return 2


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
def student_t_model():
    """Create a Student's T model for testing different likelihood types."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, size=100)
    true_alpha = 1.0
    true_beta = 2.0
    true_sigma = 1.0
    true_nu = 4.0  # Degrees of freedom

    y = true_alpha + true_beta * X + rng.standard_t(true_nu) * true_sigma

    with pm.Model(coords={"obs_id": range(len(X))}) as model:
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=10)
        sigma = pm.HalfNormal("sigma", sigma=10)
        nu = pm.Exponential("nu", lam=1 / 10)  # Prior for degrees of freedom

        mu = alpha + beta * X
        pm.StudentT("y", nu=nu, mu=mu, sigma=sigma, observed=y, dims="obs_id")

        idata = pm.sample(
            1000, tune=1000, random_seed=42, idata_kwargs={"log_likelihood": True}
        )

    return model, idata
