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


@pytest.fixture
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


@pytest.fixture
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


@pytest.fixture
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
    assert isinstance(log_like, xr.DataArray)
    assert set(log_like.dims) == {"chain", "draw", "obs_id"}

    n_chains = len(idata.posterior.chain)
    n_draws = len(idata.posterior.draw)
    assert log_like.sizes["chain"] == n_chains
    assert log_like.sizes["draw"] == n_draws
    assert log_like.sizes["obs_id"] == 100

    log_like = wrapper.log_likelihood(var_name="y")
    assert isinstance(log_like, xr.DataArray)
    assert set(log_like.dims) == {"chain", "draw", "obs_id"}

    log_like_subset = wrapper.log_likelihood(indices=slice(0, 50))
    assert log_like_subset.sizes["obs_id"] == 50


def test_hierarchical_log_likelihood(hierarchical_model):
    """Test log likelihood computation for hierarchical model."""
    model, idata = hierarchical_model
    wrapper = PyMCWrapper(model, idata)

    log_like = wrapper.log_likelihood()
    assert isinstance(log_like, xr.DataArray)
    assert set(log_like.dims) == {"chain", "draw", "group", "obs_id"}

    n_chains = len(idata.posterior.chain)
    n_draws = len(idata.posterior.draw)
    assert log_like.sizes["chain"] == n_chains
    assert log_like.sizes["draw"] == n_draws
    assert log_like.sizes["group"] == 8
    assert log_like.sizes["obs_id"] == 20

    log_like = wrapper.log_likelihood(var_name="Y")
    assert isinstance(log_like, xr.DataArray)
    assert set(log_like.dims) == {"chain", "draw", "group", "obs_id"}

    log_like_subset = wrapper.log_likelihood(indices=slice(0, 4), axis=0)
    assert log_like_subset.sizes["group"] == 4
    assert log_like_subset.sizes["obs_id"] == 20


def test_hierarchical_model_no_coords_log_likelihood(hierarchical_model_no_coords):
    """Test log likelihood computation for hierarchical model without coordinates."""
    model, idata = hierarchical_model_no_coords
    wrapper = PyMCWrapper(model, idata)

    log_like = wrapper.log_likelihood()
    assert isinstance(log_like, xr.DataArray)
    assert set(log_like.dims) == {"chain", "draw", "dim_0", "dim_1"}

    n_chains = len(idata.posterior.chain)
    n_draws = len(idata.posterior.draw)
    assert log_like.sizes["chain"] == n_chains
    assert log_like.sizes["draw"] == n_draws
    assert log_like.sizes["dim_0"] == 8
    assert log_like.sizes["dim_1"] == 20

    log_like = wrapper.log_likelihood(var_name="Y")
    assert isinstance(log_like, xr.DataArray)
    assert set(log_like.dims) == {"chain", "draw", "dim_0", "dim_1"}

    log_like_subset = wrapper.log_likelihood(indices=slice(0, 4), axis=0)
    assert log_like_subset.sizes["dim_0"] == 4
    assert log_like_subset.sizes["dim_1"] == 20


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


def test_log_likelihood__i(simple_model):
    """Test single observation log likelihood computation."""
    model, idata = simple_model
    wrapper = PyMCWrapper(model, idata)

    log_like_i = wrapper.log_likelihood__i("y", 0, idata)
    assert isinstance(log_like_i, xr.DataArray)
    assert set(log_like_i.dims) == {"chain", "draw"}

    n_chains = len(idata.posterior.chain)
    n_draws = len(idata.posterior.draw)
    assert log_like_i.sizes["chain"] == n_chains
    assert log_like_i.sizes["draw"] == n_draws

    assert np.all(np.isfinite(log_like_i))
    assert np.all(log_like_i < 0)


def test_log_likelihood__i_workflow(simple_model, poisson_model, multi_observed_model):
    """Test the full LOO-CV workflow including exact computation for problematic observations."""
    model, idata = simple_model
    wrapper = PyMCWrapper(model, idata)

    loo_results = loo(idata, pointwise=True)

    loo_results.pareto_k[0] = 0.8
    problematic_idx = 0

    original_data = wrapper.get_observed_data()
    _, training_data = wrapper.select_observations(
        np.array([problematic_idx], dtype=int)
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
    assert np.all(np.isfinite(log_like))
    assert np.all(log_like < 0)

    wrapper.set_data({wrapper.get_observed_name(): original_data})

    model, idata = poisson_model
    wrapper = PyMCWrapper(model, idata)

    original_data = wrapper.get_observed_data()
    _, training_data = wrapper.select_observations(np.array([0], dtype=int))

    wrapper.set_data({"y": training_data})
    refitted_idata = wrapper.sample_posterior(
        draws=1000, tune=1000, chains=2, random_seed=42
    )

    log_like = wrapper.log_likelihood__i("y", 0, refitted_idata)
    assert isinstance(log_like, xr.DataArray)
    assert set(log_like.dims) == {"chain", "draw"}
    assert np.all(np.isfinite(log_like))

    wrapper.set_data({"y": original_data})

    model, idata = multi_observed_model
    wrapper = PyMCWrapper(model, idata)

    log_like_y1 = wrapper.log_likelihood__i("y1", 0, idata)
    log_like_y2 = wrapper.log_likelihood__i("y2", 0, idata)

    assert isinstance(log_like_y1, xr.DataArray)
    assert isinstance(log_like_y2, xr.DataArray)
    assert set(log_like_y1.dims) == {"chain", "draw"}
    assert set(log_like_y2.dims) == {"chain", "draw"}
    assert np.all(np.isfinite(log_like_y1))
    assert np.all(np.isfinite(log_like_y2))

    with pytest.raises(ValueError, match="Variable invalid_var not found in model"):
        wrapper.log_likelihood__i("invalid_var", 0, idata)

    with pytest.raises(IndexError):
        wrapper.log_likelihood__i("y1", 1000, idata)


def test_sample_posterior_options(simple_model):
    """Test sample_posterior method with different sampling options."""
    model, idata = simple_model
    wrapper = PyMCWrapper(model, idata)

    idata_min = wrapper.sample_posterior(draws=100, chains=1, random_seed=42)
    assert len(idata_min.posterior.draw) == 100
    assert len(idata_min.posterior.chain) == 1

    idata_1 = wrapper.sample_posterior(draws=100, chains=1, random_seed=42)
    idata_2 = wrapper.sample_posterior(draws=100, chains=1, random_seed=43)
    assert not np.allclose(
        idata_1.posterior["alpha"].values, idata_2.posterior["alpha"].values
    )

    idata_multi = wrapper.sample_posterior(draws=100, chains=4, random_seed=42)
    assert len(idata_multi.posterior.chain) == 4

    with pytest.raises(ValueError, match="Number of draws must be positive"):
        wrapper.sample_posterior(draws=-100)

    with pytest.raises(ValueError, match="Number of chains must be positive"):
        wrapper.sample_posterior(chains=0)


def test_different_likelihood_models(poisson_model):
    """Test wrapper functionality with different likelihood models."""
    model, idata = poisson_model
    wrapper = PyMCWrapper(model, idata)

    log_like = wrapper.log_likelihood()
    assert isinstance(log_like, xr.DataArray)
    assert set(log_like.dims) == {"chain", "draw", "obs_id"}
    assert np.all(np.isfinite(log_like))

    selected, remaining = wrapper.select_observations(slice(0, 50))
    assert selected.shape[0] == 50
    assert remaining.shape[0] == 50
    assert np.all(selected >= 0)
    assert np.all(remaining >= 0)

    new_data = np.random.poisson(5, size=100)
    wrapper.set_data({"y": new_data})
    np.testing.assert_array_equal(wrapper.observed_data["y"], new_data)


def test_edge_cases_data_handling(simple_model):
    """Test edge cases in data handling."""
    model, idata = simple_model
    wrapper = PyMCWrapper(model, idata)

    selected, remaining = wrapper.select_observations(np.array([], dtype=int))
    assert selected.shape[0] == 0
    assert remaining.shape[0] == 100

    selected, remaining = wrapper.select_observations(np.arange(100, dtype=int))
    assert selected.shape[0] == 100
    assert remaining.shape[0] == 0

    mask = np.zeros(100, dtype=bool)
    mask[::2] = True
    selected, remaining = wrapper.select_observations(mask)
    assert selected.shape[0] == 50
    assert remaining.shape[0] == 50

    with pytest.raises(IndexError, match="Index out of bounds"):
        wrapper.select_observations(np.array([100], dtype=int))

    with pytest.raises(IndexError, match="Negative indices"):
        wrapper.select_observations(np.array([-1], dtype=int))


def test_multi_observed_handling(multi_observed_model):
    """Test handling of models with multiple observed variables."""
    model, idata = multi_observed_model
    wrapper = PyMCWrapper(model, idata)

    assert set(wrapper.observed_data.keys()) == {"y1", "y2"}

    log_like_y1 = wrapper.log_likelihood(var_name="y1")
    log_like_y2 = wrapper.log_likelihood(var_name="y2")

    assert isinstance(log_like_y1, xr.DataArray)
    assert isinstance(log_like_y2, xr.DataArray)
    assert set(log_like_y1.dims) == {"chain", "draw", "obs_id"}
    assert set(log_like_y2.dims) == {"chain", "draw", "obs_id"}

    selected_y1, remaining_y1 = wrapper.select_observations(slice(0, 50), var_name="y1")
    selected_y2, remaining_y2 = wrapper.select_observations(slice(0, 50), var_name="y2")

    assert selected_y1.shape[0] == 50
    assert selected_y2.shape[0] == 50

    new_data = {
        "y1": np.random.normal(0, 1, size=100),
        "y2": np.random.normal(0, 1, size=100),
    }
    wrapper.set_data(new_data)
    np.testing.assert_array_equal(wrapper.observed_data["y1"], new_data["y1"])
    np.testing.assert_array_equal(wrapper.observed_data["y2"], new_data["y2"])


def test_shared_variable_handling(shared_variable_model):
    """Test handling of models with shared variables."""
    model, idata = shared_variable_model
    wrapper = PyMCWrapper(model, idata)

    assert "shared_effect" in wrapper.free_vars
    assert "group_effects" in wrapper.free_vars

    log_like = wrapper.log_likelihood()
    assert isinstance(log_like, xr.DataArray)
    assert set(log_like.dims) == {"chain", "draw", "obs_id"}

    selected, remaining = wrapper.select_observations(slice(0, 75))
    assert selected.shape[0] == 75
    assert remaining.shape[0] == 75

    refitted_idata = wrapper.sample_posterior(draws=100, chains=2, random_seed=42)
    assert "shared_effect" in refitted_idata.posterior
    assert "group_effects" in refitted_idata.posterior
    assert refitted_idata.posterior["group_effects"].sizes["group"] == 3
