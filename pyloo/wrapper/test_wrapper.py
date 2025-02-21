"""Tests for PyMC model wrapper."""

import numpy as np
import pymc as pm
import pytest
from arviz import InferenceData

from .pymc_wrapper import PyMCWrapper


@pytest.fixture
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


def test_wrapper_initialization(simple_model):
    """Test wrapper initialization and validation."""
    model, idata = simple_model

    wrapper = PyMCWrapper(model, idata)
    assert wrapper.model == model
    assert wrapper.idata == idata
    assert set(wrapper.observed_data.keys()) == {"y"}
    assert set(wrapper.free_vars) == {"alpha", "beta", "sigma"}

    idata_no_posterior = InferenceData()
    with pytest.raises(ValueError, match="must contain posterior samples"):
        PyMCWrapper(model, idata_no_posterior)


def test_get_log_likelihood(simple_model):
    """Test log likelihood computation."""
    model, idata = simple_model
    wrapper = PyMCWrapper(model, idata)

    log_like = wrapper.get_log_likelihood()
    assert isinstance(log_like, dict)
    assert "y" in log_like
    assert log_like["y"].shape[0] == len(idata.posterior.chain)
    assert log_like["y"].shape[1] == len(idata.posterior.draw)
    assert log_like["y"].shape[2] == 100

    indices = {"y": slice(0, 50)}
    log_like_subset = wrapper.get_log_likelihood(indices=indices)
    assert log_like_subset["y"].shape[2] == 50


def test_select_observations(simple_model):
    """Test observation selection functionality."""
    model, idata = simple_model
    wrapper = PyMCWrapper(model, idata)

    data, coords = wrapper.select_observations("y", slice(0, 50))
    assert data.shape[0] == 50
    assert coords is not None
    assert "obs_id" in coords
    assert len(coords["obs_id"]) == 50

    indices = np.array([0, 10, 20])
    data, coords = wrapper.select_observations("y", indices)
    assert data.shape[0] == 3
    assert coords is not None
    assert "obs_id" in coords
    assert len(coords["obs_id"]) == 3

    with pytest.raises(ValueError, match="not found in observed data"):
        wrapper.select_observations("invalid_var", slice(0, 50))


def test_set_data(simple_model):
    """Test data updating functionality."""
    model, idata = simple_model
    wrapper = PyMCWrapper(model, idata)

    new_data = np.random.normal(0, 1, size=100)
    wrapper.set_data({"y": new_data})
    np.testing.assert_array_equal(wrapper.observed_data["y"], new_data)

    new_coords = {"obs_id": list(range(50))}
    new_data = np.random.normal(0, 1, size=50)
    wrapper.set_data({"y": new_data}, coords=new_coords)
    np.testing.assert_array_equal(wrapper.observed_data["y"], new_data)

    with pytest.raises(ValueError, match="Incompatible dimensions"):
        wrapper.set_data({"y": np.random.normal(0, 1, size=(100, 2))})

    with pytest.raises(ValueError, match="not found in model"):
        wrapper.set_data({"invalid_var": np.random.normal(0, 1, size=100)})


def test_get_missing_mask(simple_model):
    """Test missing value mask functionality."""
    model, idata = simple_model
    wrapper = PyMCWrapper(model, idata)

    data = wrapper.observed_data["y"].copy()
    data[::2] = np.nan
    wrapper.set_data({"y": data})

    mask = wrapper.get_missing_mask("y")
    assert mask.shape == (100,)
    assert np.all(mask[::2])
    assert not np.any(mask[1::2])

    with pytest.raises(ValueError, match="not found in observed data"):
        wrapper.get_missing_mask("invalid_var")


def test_model_validation(simple_model):
    """Test model state validation."""
    model, idata = simple_model

    idata_wrong_shape = idata.copy()
    idata_wrong_shape.posterior["alpha"] = idata_wrong_shape.posterior[
        "alpha"
    ].expand_dims("new_dim")
    with pytest.raises(ValueError, match="Shape mismatch"):
        PyMCWrapper(model, idata_wrong_shape)

    idata_missing_var = idata.copy()
    del idata_missing_var.posterior["alpha"]
    with pytest.raises(ValueError, match="Missing posterior samples"):
        PyMCWrapper(model, idata_missing_var)


def test_coordinate_handling(simple_model):
    """Test coordinate validation and handling."""
    model, idata = simple_model
    wrapper = PyMCWrapper(model, idata)

    new_data = np.random.normal(0, 1, size=50)
    invalid_coords = {"invalid_dim": list(range(50))}
    with pytest.raises(ValueError, match="Missing coordinates"):
        wrapper.set_data({"y": new_data}, coords=invalid_coords)

    invalid_coords = {"obs_id": list(range(10))}
    with pytest.raises(ValueError, match="Coordinate length"):
        wrapper.set_data({"y": new_data}, coords=invalid_coords)
