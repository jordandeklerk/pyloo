"""Test helper functions for pyloo."""

import logging
import warnings
from contextlib import contextmanager

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal

_log = logging.getLogger(__name__)


@contextmanager
def does_not_warn(warning=Warning):
    """Context manager to ensure no warnings are raised."""
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        yield
        for w in caught_warnings:
            if issubclass(w.category, warning):
                raise AssertionError(f"Expected no {warning.__name__} but caught warning with message: {w.message}")


@pytest.fixture(scope="module")
def eight_schools_params():
    """Share setup for eight schools."""
    return {
        "J": 8,
        "y": np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]),
        "sigma": np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]),
    }


def create_eight_schools_model(seed=10):
    """Create model with fake data for eight schools example."""
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
        "obs": np.random.randn(nchains, ndraws, data["J"]),
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

    observed_data = {"y": data["y"]}

    class ModelData:
        def __init__(self, groups):
            for group_name, group_data in groups.items():
                setattr(self, group_name, DataGroup(group_data))

    class DataGroup:
        def __init__(self, data_dict):
            for key, value in data_dict.items():
                setattr(self, key, value)

    groups = {
        "posterior": posterior,
        "posterior_predictive": posterior_predictive,
        "sample_stats": sample_stats,
        "log_likelihood": log_likelihood,
        "prior": prior,
        "prior_predictive": prior_predictive,
        "sample_stats_prior": sample_stats_prior,
        "observed_data": observed_data,
    }

    return ModelData(groups)


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
