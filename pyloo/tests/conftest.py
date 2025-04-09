"""Shared fixtures for pyloo tests."""

import os

import arviz as az
import numpy as np
import pytest

from .helpers import data_random, models, multidim_models
from .models import (
    approximate_posterior_model,
    bernoulli_model,
    hierarchical_model,
    hierarchical_model_no_coords,
    high_dimensional_regression_model,
    large_regression_model,
    mixture_model,
    mmm_model,
    multi_observed_model,
    mvn_spatial_model,
    mvt_spatial_model,
    poisson_model,
    problematic_k_model,
    roaches_model,
    shared_variable_model,
    simple_model,
    student_t_model,
    wells_model,
)
from .test_data import (
    both_cov_prec_data,
    loo_predictive_metric_binary_data,
    loo_predictive_metric_data,
    missing_cov_data,
    mvn_custom_names_data,
    mvn_inference_data,
    mvn_precision_data,
    mvn_validation_data,
    mvt_custom_names_data,
    mvt_inference_data,
    mvt_negative_df_data,
    mvt_precision_data,
    mvt_validation_data,
    prepare_inference_data_for_crps,
    singular_matrix_data,
)

__all__ = [
    "data_random",
    "models",
    "multidim_models",
    "mmm_model",
    "hierarchical_model",
    "hierarchical_model_no_coords",
    "simple_model",
    "poisson_model",
    "multi_observed_model",
    "shared_variable_model",
    "bernoulli_model",
    "large_regression_model",
    "student_t_model",
    "mixture_model",
    "problematic_k_model",
    "roaches_model",
    "approximate_posterior_model",
    "wells_model",
    "high_dimensional_regression_model",
    "mvn_spatial_model",
    "mvt_spatial_model",
    "loo_predictive_metric_data",
    "loo_predictive_metric_binary_data",
    "prepare_inference_data_for_crps",
    "mvn_inference_data",
    "mvt_inference_data",
    "mvn_precision_data",
    "mvt_precision_data",
    "mvn_custom_names_data",
    "mvt_custom_names_data",
    "mvt_negative_df_data",
    "singular_matrix_data",
    "both_cov_prec_data",
    "missing_cov_data",
    "mvn_validation_data",
    "mvt_validation_data",
    "rng",
    "centered_eight",
    "non_centered_eight",
    "pytest_addoption",
    "save_dir",
    "numpy_arrays",
    "log_likelihood_data",
    "multidim_data",
    "extreme_data",
    "eight_schools_params",
    "draws",
    "chains",
]


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
