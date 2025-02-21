"""Tests for PyMC model wrapper."""

import numpy as np
import pymc as pm
import pytest
import xarray as xr
from arviz import InferenceData

from ...loo import loo
from ...wrapper.pymc_wrapper import PyMCWrapper


@pytest.fixture
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


@pytest.fixture
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


def test_log_likelihood(simple_model):
    """Test log likelihood computation."""
    model, idata = simple_model
    wrapper = PyMCWrapper(model, idata)

    log_like = wrapper.log_likelihood()
    assert isinstance(log_like, dict)
    assert "y" in log_like

    n_samples = len(idata.posterior.chain) * len(idata.posterior.draw)
    assert log_like["y"].shape == (n_samples, 100)

    indices = {"y": slice(0, 50)}
    log_like_subset = wrapper.log_likelihood(indices=indices)
    assert log_like_subset["y"].shape == (n_samples, 50)


def test_select_observations(simple_model):
    """Test observation selection functionality."""
    model, idata = simple_model
    wrapper = PyMCWrapper(model, idata)

    selected, remaining = wrapper.select_observations(slice(0, 50))
    assert selected.shape[0] == 50
    assert remaining.shape[0] == 50

    all_data = np.concatenate([selected, remaining])
    original_data = wrapper.observed_data["y"].copy()
    all_data.sort()
    original_data.sort()
    np.testing.assert_array_almost_equal(all_data, original_data)

    selected, remaining = wrapper.select_observations(slice(0, 50), var_name="y")
    assert selected.shape[0] == 50
    assert remaining.shape[0] == 50

    all_data = np.concatenate([selected, remaining])
    original_data = wrapper.observed_data["y"].copy()
    all_data.sort()
    original_data.sort()
    np.testing.assert_array_almost_equal(all_data, original_data)

    indices = np.array([0, 10, 20])
    selected, remaining = wrapper.select_observations(indices)
    assert selected.shape[0] == 3
    assert remaining.shape[0] == 97

    all_data = np.concatenate([selected, remaining])
    original_data = wrapper.observed_data["y"].copy()
    all_data.sort()
    original_data.sort()
    np.testing.assert_array_almost_equal(all_data, original_data)

    with pytest.raises(ValueError, match="not found in observed data"):
        wrapper.select_observations(slice(0, 50), var_name="invalid_var")


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


def test_hierarchical_model_no_coords(hierarchical_model_no_coords):
    """Test wrapper functionality with a hierarchical model without coordinates."""
    model, idata = hierarchical_model_no_coords
    wrapper = PyMCWrapper(model, idata)

    assert set(wrapper.observed_data.keys()) == {"Y"}
    assert set(wrapper.free_vars) == {
        "alpha",
        "beta",
        "group_sigma",
        "group_effects_raw",
        "sigma_y",
    }

    dims_y = wrapper.get_dims("Y")
    assert dims_y is None or dims_y == []

    log_like = wrapper.log_likelihood()
    assert set(log_like.keys()) == {"Y"}
    n_samples = len(idata.posterior.chain) * len(idata.posterior.draw)
    assert log_like["Y"].shape == (n_samples, 8, 20)

    selected_y, remaining_y = wrapper.select_observations(
        indices=np.array([0, 2]), axis=0
    )
    assert selected_y.shape == (2, 20)
    assert remaining_y.shape == (6, 20)  # Original had 8 groups

    new_y = np.random.normal(0, 1, size=(8, 20))
    wrapper.set_data({"Y": new_y})
    np.testing.assert_array_equal(wrapper.observed_data["Y"], new_y)

    with pytest.warns(UserWarning, match="Automatically enabling log likelihood"):
        new_idata = wrapper.sample_posterior(draws=200, chains=2, random_seed=42)

    assert hasattr(new_idata, "posterior")
    assert hasattr(new_idata, "log_likelihood")
    posterior_vars = set(new_idata.posterior.data_vars.keys())
    expected_vars = {
        "alpha",
        "beta",
        "group_sigma",
        "sigma_y",
        "group_effects_raw",
        "group_effects",
    }
    assert posterior_vars == expected_vars

    assert new_idata.posterior["group_effects"].shape == (2, 200, 8)


def test_hierarchical_model_wrapper(hierarchical_model):
    """Test wrapper functionality with a hierarchical model."""
    model, idata = hierarchical_model
    wrapper = PyMCWrapper(model, idata)

    assert set(wrapper.observed_data.keys()) == {"Y"}
    assert set(wrapper.free_vars) == {
        "alpha",
        "beta",
        "group_sigma",
        "group_effects_raw",
        "sigma_y",
    }

    dims_y = wrapper.get_dims("Y")
    assert list(dims_y) == ["group", "obs_id"]

    log_like = wrapper.log_likelihood()
    assert set(log_like.keys()) == {"Y"}
    n_samples = len(idata.posterior.chain) * len(idata.posterior.draw)
    assert log_like["Y"].shape == (n_samples, 8, 20)

    selected_y, remaining_y = wrapper.select_observations(
        indices=np.array([0, 2]), axis=0
    )
    assert selected_y.shape == (2, 20)
    assert remaining_y.shape == (6, 20)  # Original had 8 groups

    # Verify that selected and remaining data together make up the complete dataset
    all_data = np.concatenate([selected_y, remaining_y], axis=0)
    original_data = wrapper.observed_data["Y"].copy()
    all_data.sort(axis=0)
    all_data.sort(axis=1)
    original_data.sort(axis=0)
    original_data.sort(axis=1)
    np.testing.assert_array_almost_equal(all_data, original_data)

    selected_y, remaining_y = wrapper.select_observations(
        indices=np.array([0, 2]), var_name="Y", axis=0
    )
    assert selected_y.shape == (2, 20)
    assert remaining_y.shape == (6, 20)

    all_data = np.concatenate([selected_y, remaining_y], axis=0)
    original_data = wrapper.observed_data["Y"].copy()
    all_data.sort(axis=0)
    all_data.sort(axis=1)
    original_data.sort(axis=0)
    original_data.sort(axis=1)
    np.testing.assert_array_almost_equal(all_data, original_data)

    new_y = np.random.normal(0, 1, size=(8, 20))
    wrapper.set_data({"Y": new_y}, coords={"group": range(8), "obs_id": range(20)})
    np.testing.assert_array_equal(wrapper.observed_data["Y"], new_y)

    with pytest.warns(UserWarning, match="Automatically enabling log likelihood"):
        new_idata = wrapper.sample_posterior(draws=200, chains=2, random_seed=42)

        assert hasattr(new_idata, "posterior")
        assert hasattr(new_idata, "log_likelihood")
        posterior_vars = set(new_idata.posterior.data_vars.keys())
        expected_vars = {
            "alpha",
            "beta",
            "group_sigma",
            "sigma_y",
            "group_effects_raw",
            "group_effects",
        }
        assert posterior_vars == expected_vars

    assert new_idata.posterior["group_effects"].shape == (2, 200, 8)

    subset_indices = {"Y": slice(0, 4)}
    subset_axis = {"Y": 0}
    subset_log_like = wrapper.log_likelihood(indices=subset_indices, axis=subset_axis)
    assert subset_log_like["Y"].shape == (n_samples, 4, 20)


def test_sample_posterior(simple_model):
    """Test resampling from the fitted model."""
    model, idata = simple_model
    wrapper = PyMCWrapper(model, idata)

    with pytest.warns(UserWarning, match="Automatically enabling log likelihood"):
        new_idata = wrapper.sample_posterior(draws=500, random_seed=42)

    assert isinstance(new_idata, InferenceData)
    assert hasattr(new_idata, "posterior")
    assert hasattr(new_idata, "log_likelihood")
    assert set(new_idata.posterior.data_vars.keys()) == {"alpha", "beta", "sigma"}

    assert new_idata.posterior.dims["chain"] == 4
    assert new_idata.posterior.dims["draw"] == 500

    custom_idata = wrapper.sample_posterior(
        draws=200,
        tune=500,
        chains=2,
        target_accept=0.9,
        random_seed=42,
        idata_kwargs={"log_likelihood": True},
    )

    assert custom_idata.posterior.dims["chain"] == 2
    assert custom_idata.posterior.dims["draw"] == 200
    assert hasattr(custom_idata, "log_likelihood")

    alpha_samples = custom_idata.posterior["alpha"].values
    beta_samples = custom_idata.posterior["beta"].values

    assert np.abs(np.mean(alpha_samples) - 1.0) < 0.5
    assert np.abs(np.mean(beta_samples) - 2.0) < 0.5


def test_log_likelihood__i_workflow(simple_model):
    """Test the full LOO-CV workflow including exact computation for problematic observations."""
    model, idata = simple_model
    wrapper = PyMCWrapper(model, idata)

    loo_results = loo(idata, pointwise=True)

    # Force a problematic observation by artificially setting its pareto k high
    loo_results.pareto_k[0] = 0.8
    problematic_idx = 0

    original_data = wrapper.get_observed_data()
    holdout_data, training_data = wrapper.select_observations(
        np.array([problematic_idx])
    )

    wrapper.set_data({wrapper.get_observed_name(): training_data})
    refitted_idata = wrapper.sample_posterior(
        draws=1000, tune=1000, chains=2, random_seed=42
    )

    log_like = wrapper.log_likelihood__i(
        wrapper.get_observed_name(), problematic_idx, refitted_idata
    )

    assert isinstance(log_like, xr.DataArray)

    assert "chain" in log_like.dims
    assert "draw" in log_like.dims
    assert log_like.sizes["chain"] == 2
    assert log_like.sizes["draw"] == 1000

    if "obs_id" in log_like.dims:
        assert log_like.sizes["obs_id"] == 1

    assert np.all(np.isfinite(log_like))
    assert np.all(log_like < 0)

    wrapper.set_data({wrapper.get_observed_name(): original_data})
