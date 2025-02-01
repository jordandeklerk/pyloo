"""Shared fixtures for pyloo tests."""

import numpy as np
import os
import pytest
import arviz as az


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
