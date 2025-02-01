"""Test helper functions for pyloo."""

import logging
import warnings
from contextlib import contextmanager
import numpy as np
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
                raise AssertionError(
                    f"Expected no {warning.__name__} but caught warning with message: {w.message}"
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
