"""Test helper functions for pyloo."""

import warnings
from contextlib import contextmanager

import numpy as np
import pytest
from arviz import from_dict
from numpy.testing import assert_allclose, assert_array_almost_equal


@contextmanager
def does_not_warn(warning=Warning):
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        yield
        for w in caught_warnings:
            if issubclass(w.category, warning):
                warning_msg = (
                    f"Expected no {warning.__name__} but caught warning with message: "
                    f"{w.message}"
                )
                raise AssertionError(warning_msg)


def create_model(seed=10, transpose=False):
    """Create model with fake data."""
    np.random.seed(seed)
    nchains = 4
    ndraws = 500
    data = {
        "J": 8,
        "y": np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]),
        "sigma": np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]),
    }
    posterior = {
        "mu": np.random.randn(nchains, ndraws),
        "tau": abs(np.random.randn(nchains, ndraws)),
        "eta": np.random.randn(nchains, ndraws, data["J"]),
        "theta": np.random.randn(nchains, ndraws, data["J"]),
    }
    posterior_predictive = {"y": np.random.randn(nchains, ndraws, len(data["y"]))}
    sample_stats = {
        "energy": np.random.randn(nchains, ndraws),
        "diverging": np.random.randn(nchains, ndraws) > 0.90,
        "max_depth": np.random.randn(nchains, ndraws) > 0.90,
    }
    log_likelihood = {
        "y": np.random.randn(nchains, ndraws, data["J"]),
    }
    prior = {
        "mu": np.random.randn(nchains, ndraws) / 2,
        "tau": abs(np.random.randn(nchains, ndraws)) / 2,
        "eta": np.random.randn(nchains, ndraws, data["J"]) / 2,
        "theta": np.random.randn(nchains, ndraws, data["J"]) / 2,
    }
    prior_predictive = {"y": np.random.randn(nchains, ndraws, len(data["y"])) / 2}
    sample_stats_prior = {
        "energy": np.random.randn(nchains, ndraws),
        "diverging": (np.random.randn(nchains, ndraws) > 0.95).astype(int),
    }
    model = from_dict(
        posterior=posterior,
        posterior_predictive=posterior_predictive,
        sample_stats=sample_stats,
        log_likelihood=log_likelihood,
        prior=prior,
        prior_predictive=prior_predictive,
        sample_stats_prior=sample_stats_prior,
        observed_data={"y": data["y"]},
        dims={
            "y": ["obs_dim"],
            "log_likelihood": ["obs_dim"],
            "theta": ["school"],
            "eta": ["school"],
        },
        coords={"obs_dim": range(data["J"])},
    )
    if transpose:
        for group in model._groups:
            group_dataset = getattr(model, group)
            if all(dim in group_dataset.dims for dim in ("draw", "chain")):
                setattr(model, group, group_dataset.transpose(*["draw", "chain"], ...))
    return model


def create_multidimensional_model(seed=10, transpose=False):
    """Create model with fake data."""
    np.random.seed(seed)
    nchains = 4
    ndraws = 500
    ndim1 = 5
    ndim2 = 7
    data = {
        "y": np.random.normal(size=(ndim1, ndim2)),
        "sigma": np.random.normal(size=(ndim1, ndim2)),
    }
    posterior = {
        "mu": np.random.randn(nchains, ndraws),
        "tau": abs(np.random.randn(nchains, ndraws)),
        "eta": np.random.randn(nchains, ndraws, ndim1, ndim2),
        "theta": np.random.randn(nchains, ndraws, ndim1, ndim2),
    }
    posterior_predictive = {"y": np.random.randn(nchains, ndraws, ndim1, ndim2)}
    sample_stats = {
        "energy": np.random.randn(nchains, ndraws),
        "diverging": np.random.randn(nchains, ndraws) > 0.90,
    }
    log_likelihood = {
        "y": np.random.randn(nchains, ndraws, ndim1, ndim2),
    }
    prior = {
        "mu": np.random.randn(nchains, ndraws) / 2,
        "tau": abs(np.random.randn(nchains, ndraws)) / 2,
        "eta": np.random.randn(nchains, ndraws, ndim1, ndim2) / 2,
        "theta": np.random.randn(nchains, ndraws, ndim1, ndim2) / 2,
    }
    prior_predictive = {"y": np.random.randn(nchains, ndraws, ndim1, ndim2) / 2}
    sample_stats_prior = {
        "energy": np.random.randn(nchains, ndraws),
        "diverging": (np.random.randn(nchains, ndraws) > 0.95).astype(int),
    }
    model = from_dict(
        posterior=posterior,
        posterior_predictive=posterior_predictive,
        sample_stats=sample_stats,
        log_likelihood=log_likelihood,
        prior=prior,
        prior_predictive=prior_predictive,
        sample_stats_prior=sample_stats_prior,
        observed_data={"y": data["y"]},
        dims={"y": ["dim1", "dim2"], "log_likelihood": ["dim1", "dim2"]},
        coords={"dim1": range(ndim1), "dim2": range(ndim2)},
    )
    if transpose:
        for group in model._groups:
            group_dataset = getattr(model, group)
            if all(dim in group_dataset.dims for dim in ("draw", "chain")):
                setattr(model, group, group_dataset.transpose(*["draw", "chain"], ...))
    return model


def create_data_random(groups=None, seed=10):
    """Create InferenceData object using random data."""
    if groups is None:
        groups = ["posterior", "sample_stats", "observed_data", "posterior_predictive"]
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(4, 500, 8))
    idata_dict = {
        "posterior": {"a": data[..., 0], "b": data},
        "sample_stats": {"a": data[..., 0], "b": data},
        "observed_data": {"b": data[0, 0, :]},
        "posterior_predictive": {"a": data[..., 0], "b": data},
        "prior": {"a": data[..., 0], "b": data},
        "prior_predictive": {"a": data[..., 0], "b": data},
        "warmup_posterior": {"a": data[..., 0], "b": data},
        "warmup_posterior_predictive": {"a": data[..., 0], "b": data},
        "warmup_prior": {"a": data[..., 0], "b": data},
    }
    idata = from_dict(
        **{group: ary for group, ary in idata_dict.items() if group in groups},
        save_warmup=True,
    )
    return idata


@pytest.fixture()
def data_random():
    """Fixture containing InferenceData object using random data."""
    idata = create_data_random()
    return idata


@pytest.fixture(scope="module")
def models():
    """Fixture containing 2 mock inference data instances for testing."""

    class Models:
        model_1 = create_model(seed=10)
        model_2 = create_model(seed=11, transpose=True)

    return Models()


@pytest.fixture(scope="module")
def multidim_models():
    """Fixture containing 2 mock inference data instances with multidimensional data for testing."""
    # blank line to keep black and pydocstyle happy

    class Models:
        model_1 = create_multidimensional_model(seed=10)
        model_2 = create_multidimensional_model(seed=11, transpose=True)

    return Models()


def create_large_model(seed=10, n_obs=10000):
    """Create model with large number of observations for testing subsampling.

    This creates a model with a large number of observations to demonstrate
    the benefits of subsampling in LOO-CV computation. The model simulates
    a hierarchical normal model with:

    y_i ~ Normal(theta_i, sigma)
    theta_i ~ Normal(mu, tau)
    """
    np.random.seed(seed)
    nchains = 4
    ndraws = 500

    true_mu = 1.0
    true_tau = 2.0
    true_sigma = 1.0

    true_theta = np.random.normal(true_mu, true_tau, size=n_obs)
    y = np.random.normal(true_theta, true_sigma)

    posterior = {
        "mu": np.random.normal(y.mean(), 0.1, size=(nchains, ndraws)),
        "tau": np.abs(np.random.normal(2, 0.1, size=(nchains, ndraws))),
        "theta": np.random.normal(y[None, None, :], 0.1, size=(nchains, ndraws, n_obs)),
    }

    theta = posterior["theta"]
    log_likelihood = {
        "obs": (
            -0.5 * np.log(2 * np.pi)
            - 0.5 * np.log(true_sigma**2)
            - 0.5 * ((y[None, None, :] - theta) / true_sigma) ** 2
        )
    }

    posterior_predictive = {
        "y": np.random.normal(theta, true_sigma, size=(nchains, ndraws, n_obs))
    }

    sample_stats = {
        "energy": np.random.randn(nchains, ndraws),
        "diverging": np.random.randn(nchains, ndraws) > 0.95,
    }

    return from_dict(
        posterior=posterior,
        posterior_predictive=posterior_predictive,
        sample_stats=sample_stats,
        log_likelihood=log_likelihood,
        observed_data={"y": y},
        dims={
            "theta": ["obs_id"],
            "y": ["obs_id"],
            "log_likelihood": ["obs_id"],
        },
        coords={
            "obs_id": range(n_obs),
        },
    )


def generate_psis_data(rng, n_samples=1000, n_obs=8):
    """Generate random data for PSIS tests."""
    return {
        "log_ratios": rng.normal(size=(n_samples, n_obs)),
        "r_eff": np.full(n_obs, 0.7),
    }


def assert_arrays_equal(x, y, **kwargs):
    """Assert that two arrays are equal."""
    np.testing.assert_array_equal(x, y, **kwargs)


def assert_arrays_almost_equal(x, y, decimal=7, **kwargs):
    """Assert that two arrays are almost equal."""
    assert_array_almost_equal(x, y, decimal=decimal, **kwargs)


def assert_arrays_allclose(actual, desired, rtol=1e-7, atol=0, **kwargs):
    """Assert that two arrays are almost equal within tolerances."""
    assert_allclose(actual, desired, rtol=rtol, atol=atol, **kwargs)


def assert_finite(x):
    """Assert that all array elements are finite."""
    assert np.all(np.isfinite(x))


def assert_positive(x):
    """Assert that all array elements are positive."""
    assert np.all(x > 0)


def assert_bounded(x, lower=None, upper=None):
    """Assert that all array elements are within bounds."""
    if lower is not None:
        assert np.all(x >= lower)
    if upper is not None:
        assert np.all(x <= upper)


def assert_shape_equal(x, y):
    """Assert that two arrays have the same shape."""
    assert x.shape == y.shape


def assert_dtype(x, dtype):
    """Assert that array has expected dtype."""
    assert x.dtype == dtype
