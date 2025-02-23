"""Tests for PyMC model wrapper."""

import logging

import numpy as np
import pytest
import xarray as xr
from arviz import InferenceData

from ...loo import loo
from ...wrapper.pymc_wrapper import PyMCWrapper, PyMCWrapperError
from ..helpers import (
    assert_arrays_allclose,
    assert_arrays_almost_equal,
    assert_arrays_equal,
    assert_bounded,
    assert_finite,
    assert_positive,
    assert_shape_equal,
)


def test_wrapper_initialization(simple_model):
    """Test wrapper initialization and validation."""
    model, idata = simple_model

    wrapper = PyMCWrapper(model, idata)
    assert wrapper.model == model
    assert wrapper.idata == idata
    assert set(wrapper.observed_data.keys()) == {"y"}
    assert set(wrapper.free_vars) == {"alpha", "beta", "sigma"}

    idata_no_posterior = InferenceData()
    with pytest.raises(PyMCWrapperError, match="must contain posterior samples"):
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
    assert_arrays_almost_equal(all_data, original_data)

    selected, remaining = wrapper.select_observations(slice(0, 50), var_name="y")
    assert selected.shape[0] == 50
    assert remaining.shape[0] == 50

    all_data = np.concatenate([selected, remaining])
    original_data = wrapper.observed_data["y"].copy()
    all_data.sort()
    original_data.sort()
    assert_arrays_almost_equal(all_data, original_data)

    indices = np.array([0, 10, 20])
    selected, remaining = wrapper.select_observations(indices)
    assert selected.shape[0] == 3
    assert remaining.shape[0] == 97

    all_data = np.concatenate([selected, remaining])
    original_data = wrapper.observed_data["y"].copy()
    all_data.sort()
    original_data.sort()
    assert_arrays_almost_equal(all_data, original_data)

    with pytest.raises(PyMCWrapperError, match="not found in observed data"):
        wrapper.select_observations(slice(0, 50), var_name="invalid_var")


def test_set_data(simple_model):
    """Test data updating functionality."""
    model, idata = simple_model
    wrapper = PyMCWrapper(model, idata)

    new_data = np.random.normal(0, 1, size=100)
    wrapper.set_data({"y": new_data})
    assert_arrays_equal(wrapper.observed_data["y"], new_data)

    new_coords = {"obs_id": list(range(50))}
    new_data = np.random.normal(0, 1, size=50)
    wrapper.set_data({"y": new_data}, coords=new_coords)
    assert_arrays_equal(wrapper.observed_data["y"], new_data)

    with pytest.raises(PyMCWrapperError, match="Incompatible dimensions"):
        wrapper.set_data({"y": np.random.normal(0, 1, size=(100, 2))})

    with pytest.raises(PyMCWrapperError, match="not found in model"):
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

    with pytest.raises(PyMCWrapperError, match="not found in observed data"):
        wrapper.get_missing_mask("invalid_var")


def test_model_validation(simple_model):
    """Test model state validation."""
    model, idata = simple_model

    idata_wrong_shape = idata.copy()
    idata_wrong_shape.posterior["alpha"] = idata_wrong_shape.posterior[
        "alpha"
    ].expand_dims("new_dim")
    with pytest.raises(PyMCWrapperError, match="Shape mismatch"):
        PyMCWrapper(model, idata_wrong_shape)

    idata_missing_var = idata.copy()
    del idata_missing_var.posterior["alpha"]
    with pytest.raises(PyMCWrapperError, match="Missing posterior samples"):
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


def test_log_likelihood_i(simple_model):
    """Test single observation log likelihood computation."""
    model, idata = simple_model
    wrapper = PyMCWrapper(model, idata)

    log_like_i = wrapper.log_likelihood_i("y", 0, idata)
    assert isinstance(log_like_i, xr.DataArray)
    assert set(log_like_i.dims) == {"chain", "draw"}

    n_chains = len(idata.posterior.chain)
    n_draws = len(idata.posterior.draw)
    assert log_like_i.sizes["chain"] == n_chains
    assert log_like_i.sizes["draw"] == n_draws

    assert_finite(log_like_i)
    assert_bounded(log_like_i, upper=0)


def test_log_likelihood_i_workflow(simple_model, poisson_model, multi_observed_model):
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

    log_like = wrapper.log_likelihood_i(
        wrapper.get_observed_name(), problematic_idx, refitted_idata
    )

    assert isinstance(log_like, xr.DataArray)
    assert "chain" in log_like.dims
    assert "draw" in log_like.dims
    assert log_like.sizes["chain"] == 2
    assert log_like.sizes["draw"] == 1000
    assert_finite(log_like)
    assert_bounded(log_like, upper=0)

    wrapper.set_data({wrapper.get_observed_name(): original_data})

    model, idata = poisson_model
    wrapper = PyMCWrapper(model, idata)

    original_data = wrapper.get_observed_data()
    _, training_data = wrapper.select_observations(np.array([0], dtype=int))

    wrapper.set_data({"y": training_data})
    refitted_idata = wrapper.sample_posterior(
        draws=1000, tune=1000, chains=2, random_seed=42
    )

    log_like = wrapper.log_likelihood_i("y", 0, refitted_idata)
    assert isinstance(log_like, xr.DataArray)
    assert set(log_like.dims) == {"chain", "draw"}
    assert_finite(log_like)

    wrapper.set_data({"y": original_data})

    model, idata = multi_observed_model
    wrapper = PyMCWrapper(model, idata)

    log_like_y1 = wrapper.log_likelihood_i("y1", 0, idata)
    log_like_y2 = wrapper.log_likelihood_i("y2", 0, idata)

    assert isinstance(log_like_y1, xr.DataArray)
    assert isinstance(log_like_y2, xr.DataArray)
    assert set(log_like_y1.dims) == {"chain", "draw"}
    assert set(log_like_y2.dims) == {"chain", "draw"}
    assert_finite(log_like_y1)
    assert_finite(log_like_y2)

    with pytest.raises(
        PyMCWrapperError, match="Variable 'invalid_var' not found in model"
    ):
        wrapper.log_likelihood_i("invalid_var", 0, idata)

    with pytest.raises(IndexError):
        wrapper.log_likelihood_i("y1", 1000, idata)


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

    with pytest.raises(PyMCWrapperError, match="Number of draws must be positive"):
        wrapper.sample_posterior(draws=-100)

    with pytest.raises(PyMCWrapperError, match="Number of chains must be positive"):
        wrapper.sample_posterior(chains=0)


def test_different_likelihood_models(poisson_model):
    """Test wrapper functionality with different likelihood models."""
    model, idata = poisson_model
    wrapper = PyMCWrapper(model, idata)

    log_like = wrapper.log_likelihood()
    assert isinstance(log_like, xr.DataArray)
    assert set(log_like.dims) == {"chain", "draw", "obs_id"}
    assert_finite(log_like)

    selected, remaining = wrapper.select_observations(slice(0, 50))
    assert selected.shape[0] == 50
    assert remaining.shape[0] == 50
    assert np.all(selected >= 0)
    assert np.all(remaining >= 0)

    new_data = np.random.poisson(5, size=100)
    wrapper.set_data({"y": new_data})
    assert_arrays_equal(wrapper.observed_data["y"], new_data)


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

    with pytest.raises(IndexError, match="Index .* is out of bounds"):
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
    assert_arrays_equal(wrapper.observed_data["y1"], new_data["y1"])
    assert_arrays_equal(wrapper.observed_data["y2"], new_data["y2"])


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


def test_parameter_transformations(simple_model):
    """Test parameter transformation between constrained and unconstrained spaces."""
    model, idata = simple_model
    wrapper = PyMCWrapper(model, idata)
    unconstrained = wrapper.get_unconstrained_parameters()

    assert set(unconstrained.keys()) == {"alpha", "beta", "sigma"}

    # Check dimensions match original posterior
    assert unconstrained["alpha"].dims == ("chain", "draw")
    assert unconstrained["beta"].dims == ("chain", "draw")
    assert unconstrained["sigma"].dims == ("chain", "draw")

    # Check shapes match original posterior
    assert unconstrained["alpha"].shape == (
        len(idata.posterior.chain),
        len(idata.posterior.draw),
    )
    assert unconstrained["beta"].shape == (
        len(idata.posterior.chain),
        len(idata.posterior.draw),
    )
    assert unconstrained["sigma"].shape == (
        len(idata.posterior.chain),
        len(idata.posterior.draw),
    )

    # Check coordinates are preserved
    assert (
        unconstrained["alpha"].coords["chain"].equals(idata.posterior.coords["chain"])
    )
    assert unconstrained["alpha"].coords["draw"].equals(idata.posterior.coords["draw"])

    constrained = wrapper.constrain_parameters(unconstrained)
    assert set(constrained.keys()) == {"alpha", "beta", "sigma"}

    assert_shape_equal(constrained["alpha"], unconstrained["alpha"])
    assert_shape_equal(constrained["beta"], unconstrained["beta"])
    assert_shape_equal(constrained["sigma"], unconstrained["sigma"])

    # Check that sigma is positive in constrained space
    assert_positive(constrained["sigma"])

    # Check that transformations approximately invert each other
    assert_arrays_allclose(
        constrained["alpha"], wrapper.idata.posterior.alpha.values, rtol=1e-5
    )
    assert_arrays_allclose(
        constrained["beta"], wrapper.idata.posterior.beta.values, rtol=1e-5
    )
    assert_arrays_allclose(
        constrained["sigma"], wrapper.idata.posterior.sigma.values, rtol=1e-5
    )


def test_hierarchical_parameter_transformations(hierarchical_model):
    """Test parameter transformations with hierarchical model structure."""
    model, idata = hierarchical_model
    wrapper = PyMCWrapper(model, idata)
    unconstrained = wrapper.get_unconstrained_parameters()

    expected_params = {"alpha", "beta", "group_sigma", "group_effects_raw", "sigma_y"}
    assert set(unconstrained.keys()) == expected_params

    # Check dimensions match original posterior
    assert unconstrained["alpha"].dims == ("chain", "draw")
    assert unconstrained["beta"].dims == ("chain", "draw")
    assert unconstrained["group_sigma"].dims == ("chain", "draw")
    assert unconstrained["group_effects_raw"].dims == ("chain", "draw", "group")
    assert unconstrained["sigma_y"].dims == ("chain", "draw")

    # Check shapes match original posterior
    assert unconstrained["alpha"].shape == (
        len(idata.posterior.chain),
        len(idata.posterior.draw),
    )
    assert unconstrained["beta"].shape == (
        len(idata.posterior.chain),
        len(idata.posterior.draw),
    )
    assert unconstrained["group_sigma"].shape == (
        len(idata.posterior.chain),
        len(idata.posterior.draw),
    )
    assert unconstrained["group_effects_raw"].shape == (
        len(idata.posterior.chain),
        len(idata.posterior.draw),
        8,
    )
    assert unconstrained["sigma_y"].shape == (
        len(idata.posterior.chain),
        len(idata.posterior.draw),
    )

    # Check coordinates are preserved
    assert (
        unconstrained["alpha"].coords["chain"].equals(idata.posterior.coords["chain"])
    )
    assert unconstrained["alpha"].coords["draw"].equals(idata.posterior.coords["draw"])
    assert (
        unconstrained["group_effects_raw"]
        .coords["group"]
        .equals(idata.posterior.coords["group"])
    )

    constrained = wrapper.constrain_parameters(unconstrained)

    assert set(constrained.keys()) == expected_params

    # Check that constrained parameters match original posterior
    for param in expected_params:
        if param in idata.posterior:
            assert_arrays_allclose(
                constrained[param], idata.posterior[param].values, rtol=1e-5
            )

    # Check that sigma parameters are positive in constrained space
    assert_positive(constrained["group_sigma"])
    assert_positive(constrained["sigma_y"])


def test_logging_functionality(simple_model, caplog):
    """Test that logging messages are properly emitted."""
    model, idata = simple_model
    wrapper = PyMCWrapper(model, idata)

    data = wrapper.observed_data["y"].copy()
    data[::2] = np.nan
    wrapper.set_data({"y": data})

    with caplog.at_level(logging.WARNING):
        wrapper.select_observations(slice(0, 50))
        assert "Missing values detected in y" in caplog.text

    wrapper.set_data({"y": wrapper.idata.observed_data["y"].values})

    with caplog.at_level(logging.INFO):
        wrapper.sample_posterior(draws=10, chains=1)
        assert "Automatically enabling log likelihood computation" in caplog.text

    model_no_dims = model.copy()
    model_no_dims.named_vars_to_dims = {}
    wrapper_no_dims = PyMCWrapper(model_no_dims, idata)

    with caplog.at_level(logging.WARNING):
        wrapper_no_dims.log_likelihood_i("y", 0, idata)
        assert "Could not determine dimensions" in caplog.text


def test_error_messages(simple_model):
    """Test that error messages provide detailed context."""
    model, idata = simple_model
    wrapper = PyMCWrapper(model, idata)

    with pytest.raises(PyMCWrapperError) as exc_info:
        wrapper.set_data({"invalid_var": np.zeros(100)})
    assert "Available variables:" in str(exc_info.value)

    with pytest.raises(PyMCWrapperError) as exc_info:
        wrapper.set_data({"y": np.zeros((100, 2))})
    assert "Expected shape:" in str(exc_info.value)
    assert "got:" in str(exc_info.value)

    idata_missing = idata.copy()
    del idata_missing.posterior["alpha"]
    with pytest.raises(PyMCWrapperError) as exc_info:
        PyMCWrapper(model, idata_missing)
    assert "Missing posterior samples for variables:" in str(exc_info.value)
