"""Shared fixtures for pyloo tests."""

import os

import arviz as az
import numpy as np
import pytest

from .helpers import extreme_data as _extreme_data
from .helpers import log_likelihood_data as _log_likelihood_data
from .helpers import multidim_data as _multidim_data
from .helpers import numpy_arrays as _numpy_arrays

# Re-export fixtures from helpers
extreme_data = _extreme_data
log_likelihood_data = _log_likelihood_data
multidim_data = _multidim_data
numpy_arrays = _numpy_arrays


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
