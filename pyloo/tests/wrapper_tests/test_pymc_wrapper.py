"""Tests for PyMC model wrapper."""

import copy
import logging

import numpy as np
import pymc as pm
import pytest
import xarray as xr
from arviz import InferenceData

from ...loo import loo
from ...wrapper.pymc import PyMCWrapper, PyMCWrapperError
from ..helpers import (
    assert_arrays_allclose,
    assert_arrays_almost_equal,
    assert_arrays_equal,
    assert_bounded,
    assert_finite,
    assert_positive,
    assert_shape_equal,
)

logger = logging.getLogger(__name__)


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


def test_coordinate_handling_and_data_immutability(hierarchical_model):
    """Test coordinate validation, handling, and data immutability."""
    model, idata = hierarchical_model
    wrapper = PyMCWrapper(model, idata)

    new_data = np.random.normal(0, 1, size=(8, 20))
    coords = {"group": list(range(8)), "obs_id": list(range(20))}

    wrapper.set_data({"Y": new_data}, coords=coords)
    assert_arrays_equal(wrapper.observed_data["Y"], new_data)

    with pytest.raises((ValueError, RuntimeError)):
        wrapper.observed_data["Y"][0, 0] = 100.0

    new_data = np.random.normal(0, 1, size=(10, 25))
    with pytest.warns(UserWarning) as record:
        wrapper.set_data(
            {"Y": new_data},
            coords={"group": list(range(8)), "obs_id": list(range(20))},
            update_coords=True,
        )
    assert len(record) >= 1
    assert_arrays_equal(wrapper.observed_data["Y"], new_data)

    with pytest.raises((ValueError, RuntimeError)):
        wrapper.observed_data["Y"][0, 0] = 100.0

    new_data = np.random.normal(0, 1, size=(12, 30))
    with pytest.raises(ValueError, match="Missing coordinates for dimensions"):
        wrapper.set_data({"Y": new_data}, update_coords=False)

    new_data = np.random.normal(0, 1, size=(15, 35))
    with pytest.warns(UserWarning) as record:
        wrapper.set_data({"Y": new_data}, update_coords=True)

    assert len(record) >= 1
    assert any(
        "Automatically created coordinates" in str(w.message)
        or "length changed" in str(w.message)
        for w in record
    )
    assert_arrays_equal(wrapper.observed_data["Y"], new_data)


def test_dimension_validation(hierarchical_model):
    """Test dimension validation in set_data."""
    model, idata = hierarchical_model
    wrapper = PyMCWrapper(model, idata)

    new_data = np.random.normal(0, 1, size=(8, 20, 5))
    with pytest.raises(
        PyMCWrapperError, match="New data .* has .* dimensions but model expects"
    ):
        wrapper.set_data({"Y": new_data})

    new_data = np.random.normal(0, 1, size=(8, 20))
    coords = {"group": list(range(8))}
    with pytest.raises(ValueError, match="Missing coordinates for dimensions"):
        wrapper.set_data({"Y": new_data}, coords=coords, update_coords=False)

    coords = {"group": list(range(8)), "obs_id": list(range(10))}
    with pytest.raises(ValueError, match="Coordinate length .* does not match"):
        wrapper.set_data({"Y": new_data}, coords=coords, update_coords=False)


def test_set_data_return_value(simple_model):
    """Test that set_data returns None."""
    model, idata = simple_model
    wrapper = PyMCWrapper(model, idata)

    new_data = np.random.normal(0, 1, size=100)
    result = wrapper.set_data({"y": new_data})
    assert result is None


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


def test_model_validation_and_deep_copy(simple_model):
    """Test model state validation and deep copy independence."""
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

    wrapper1 = PyMCWrapper(model, idata)
    wrapper2 = PyMCWrapper(model, idata)

    wrapper1._untransformed_model.name = "modified_model"
    assert wrapper1._untransformed_model.name != wrapper2._untransformed_model.name

    original_data = wrapper1.get_observed_data()
    new_data = original_data.copy()
    new_data[0] = 999.0
    wrapper1.set_data({wrapper1.get_observed_name(): new_data})
    assert not np.array_equal(
        wrapper1.get_observed_data(), wrapper2.get_observed_data()
    )


def test_log_likelihood_i(simple_model):
    """Test single observation log likelihood computation."""
    model, idata = simple_model
    wrapper = PyMCWrapper(model, idata)

    log_like_i = wrapper.log_likelihood_i(0, idata)
    assert isinstance(log_like_i, xr.DataArray)
    assert set(log_like_i.dims) == {"chain", "draw"}

    n_chains = len(idata.posterior.chain)
    n_draws = len(idata.posterior.draw)
    assert log_like_i.sizes["chain"] == n_chains
    assert log_like_i.sizes["draw"] == n_draws

    assert_finite(log_like_i)
    assert_bounded(log_like_i, upper=0)


def test_log_likelihood_i_multiple_indices(simple_model):
    model, idata = simple_model
    wrapper = PyMCWrapper(model, idata)

    indices = np.array([0, 10, 20])
    log_like_i = wrapper.log_likelihood_i(indices, idata)

    assert isinstance(log_like_i, xr.DataArray)
    assert "chain" in log_like_i.dims
    assert "draw" in log_like_i.dims

    obs_dims = [dim for dim in log_like_i.dims if dim not in ("chain", "draw")]
    assert len(obs_dims) > 0

    obs_dim = obs_dims[0]
    assert log_like_i.sizes[obs_dim] == len(indices)

    if obs_dim in log_like_i.coords:
        assert np.array_equal(log_like_i.coords[obs_dim].values, indices)

    assert "observation_indices" in log_like_i.attrs
    assert log_like_i.attrs["observation_indices"] == indices.tolist()

    assert_finite(log_like_i)
    assert_bounded(log_like_i, upper=0)

    slice_idx = slice(0, 5)
    log_like_i = wrapper.log_likelihood_i(slice_idx, idata)

    assert isinstance(log_like_i, xr.DataArray)
    assert "chain" in log_like_i.dims
    assert "draw" in log_like_i.dims

    obs_dims = [dim for dim in log_like_i.dims if dim not in ("chain", "draw")]
    assert len(obs_dims) > 0

    obs_dim = obs_dims[0]
    assert log_like_i.sizes[obs_dim] == 5

    expected_indices = np.arange(0, 5)
    if obs_dim in log_like_i.coords:
        assert np.array_equal(log_like_i.coords[obs_dim].values, expected_indices)

    assert_finite(log_like_i)
    assert_bounded(log_like_i, upper=0)

    mask = np.zeros(100, dtype=bool)
    mask[[0, 10, 20]] = True
    log_like_i = wrapper.log_likelihood_i(mask, idata)

    assert isinstance(log_like_i, xr.DataArray)
    assert "chain" in log_like_i.dims
    assert "draw" in log_like_i.dims

    obs_dims = [dim for dim in log_like_i.dims if dim not in ("chain", "draw")]
    assert len(obs_dims) > 0

    obs_dim = obs_dims[0]
    assert log_like_i.sizes[obs_dim] == 3

    expected_indices = np.array([0, 10, 20])
    if obs_dim in log_like_i.coords:
        assert np.array_equal(log_like_i.coords[obs_dim].values, expected_indices)

    assert_finite(log_like_i)
    assert_bounded(log_like_i, upper=0)

    log_like_i = wrapper.log_likelihood_i(0, idata)

    assert isinstance(log_like_i, xr.DataArray)
    assert "chain" in log_like_i.dims
    assert "draw" in log_like_i.dims

    obs_dims = [dim for dim in log_like_i.dims if dim not in ("chain", "draw")]
    assert len(obs_dims) == 0

    assert "observation_index" in log_like_i.attrs
    assert log_like_i.attrs["observation_index"] == 0

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

    log_like = wrapper.log_likelihood_i(problematic_idx, refitted_idata)

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

    log_like = wrapper.log_likelihood_i(0, refitted_idata)
    assert isinstance(log_like, xr.DataArray)
    assert set(log_like.dims) == {"chain", "draw"}
    assert_finite(log_like)

    wrapper.set_data({"y": original_data})

    model, idata = multi_observed_model
    wrapper = PyMCWrapper(model, idata)

    log_like_y1 = wrapper.log_likelihood_i(0, idata, "y1")
    log_like_y2 = wrapper.log_likelihood_i(0, idata, "y2")

    assert isinstance(log_like_y1, xr.DataArray)
    assert isinstance(log_like_y2, xr.DataArray)
    assert set(log_like_y1.dims) == {"chain", "draw"}
    assert set(log_like_y2.dims) == {"chain", "draw"}
    assert_finite(log_like_y1)
    assert_finite(log_like_y2)

    with pytest.raises(
        PyMCWrapperError, match="Variable 'invalid_var' not found in model"
    ):
        wrapper.log_likelihood_i(0, idata, "invalid_var")

    with pytest.raises(IndexError):
        wrapper.log_likelihood_i(200, idata, "y1")


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


def test_edge_cases_data_handling(simple_model):
    """Test edge cases in data handling."""
    model, idata = simple_model
    wrapper = PyMCWrapper(model, idata)

    with pytest.raises(IndexError, match="Empty index array provided"):
        wrapper.select_observations(np.array([], dtype=int))

    selected, remaining = wrapper.select_observations(np.arange(100, dtype=int))
    assert selected.shape[0] == 100
    assert remaining.shape[0] == 0

    mask = np.zeros(100, dtype=bool)
    mask[::2] = True
    selected, remaining = wrapper.select_observations(mask)
    assert selected.shape[0] == 50
    assert remaining.shape[0] == 50

    with pytest.raises(IndexError, match="All indices are out of bounds for axis"):
        wrapper.select_observations(np.array([100], dtype=int))

    with pytest.raises(IndexError, match="All indices are out of bounds for axis"):
        wrapper.select_observations(np.array([-1], dtype=int))


def test_multi_observed_handling(multi_observed_model):
    """Test handling of models with multiple observed variables."""
    model, idata = multi_observed_model
    wrapper = PyMCWrapper(model, idata)

    assert set(wrapper.observed_data.keys()) == {"y1", "y2"}

    selected_y1, _ = wrapper.select_observations(slice(0, 50), var_name="y1")
    selected_y2, _ = wrapper.select_observations(slice(0, 50), var_name="y2")

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

    selected, remaining = wrapper.select_observations(slice(0, 75))
    assert selected.shape[0] == 75
    assert remaining.shape[0] == 75

    refitted_idata = wrapper.sample_posterior(draws=100, chains=2, random_seed=42)
    assert "shared_effect" in refitted_idata.posterior
    assert "group_effects" in refitted_idata.posterior
    assert refitted_idata.posterior["group_effects"].sizes["group"] == 3


def test_parameter_transformations(simple_model, caplog):
    """Test parameter transformation between constrained and unconstrained spaces."""
    model, idata = simple_model
    wrapper = PyMCWrapper(model, idata)
    unconstrained = wrapper.get_unconstrained_parameters()
    assert set(unconstrained.keys()) == {"alpha", "beta", "sigma"}

    assert unconstrained["alpha"].dims == ("chain", "draw")
    assert unconstrained["beta"].dims == ("chain", "draw")
    assert unconstrained["sigma"].dims == ("chain", "draw")

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

    assert (
        unconstrained["alpha"].coords["chain"].equals(idata.posterior.coords["chain"])
    )
    assert unconstrained["alpha"].coords["draw"].equals(idata.posterior.coords["draw"])

    posterior_sigma = idata.posterior["sigma"].values
    unconstrained_sigma = unconstrained["sigma"].values

    logger.info("\nTransformation visualization for sigma parameter:")
    logger.info("Chain 0, Draws 0-4:")
    logger.info(
        f"{'Original posterior':20} | {'Unconstrained space':20} |"
        f" {'Log of posterior':20}"
    )
    logger.info("-" * 65)

    for i in range(5):
        orig = posterior_sigma[0, i]
        uncon = unconstrained_sigma[0, i]
        log_orig = np.log(orig)
        logger.info(f"{orig:20.6f} | {uncon:20.6f} | {log_orig:20.6f}")

    sigma_var = None
    for var in model.free_RVs:
        if var.name == "sigma":
            sigma_var = var
            break

    if sigma_var is not None:
        transform = model.rvs_to_transforms.get(sigma_var)
        if transform is not None:
            logger.info("\nVerifying PyMC's direct transformation vs wrapper:")
            sample_values = posterior_sigma[0, :5]

            try:
                direct_transform = transform.backward(sample_values).eval()
                logger.info("\nDirect transform application:")
                logger.info(
                    f"{'Original sigma':20} | {'Direct transform':20} |"
                    f" {'Wrapper transform':20}"
                )
                logger.info("-" * 65)

                for i in range(5):
                    orig = sample_values[i]
                    direct = direct_transform[i]
                    wrapper_result = unconstrained_sigma[0, i]
                    logger.info(
                        f"{orig:20.6f} | {direct:20.6f} | {wrapper_result:20.6f}"
                    )

                direct_all = transform.backward(posterior_sigma).eval()
                transform_match = np.allclose(
                    direct_all, unconstrained_sigma, rtol=1e-5
                )
                logger.info(
                    f"\nDirect transform matches wrapper transform: {transform_match}"
                )

            except Exception as e:
                logger.info(f"Couldn't apply transform directly: {e}")

            logger.info(
                f"\nTransform details:\nType: {type(transform)}\nAttributes:"
                f" {dir(transform)}"
            )

    constrained = wrapper.constrain_parameters(unconstrained)
    assert set(constrained.keys()) == {"alpha", "beta", "sigma"}

    logger.info("\nReconstituted values:")
    logger.info(f"{'Original posterior':20} | {'Recalculated constrained':20}")
    logger.info("-" * 45)

    for i in range(5):
        orig = posterior_sigma[0, i]
        recon = constrained["sigma"].values[0, i]
        logger.info(f"{orig:20.6f} | {recon:20.6f}")

    assert_shape_equal(constrained["alpha"], unconstrained["alpha"])
    assert_shape_equal(constrained["beta"], unconstrained["beta"])
    assert_shape_equal(constrained["sigma"], unconstrained["sigma"])

    assert_positive(constrained["sigma"])

    assert_arrays_allclose(
        constrained["alpha"], wrapper.idata.posterior.alpha.values, rtol=1e-5
    )
    assert_arrays_allclose(
        constrained["beta"], wrapper.idata.posterior.beta.values, rtol=1e-5
    )
    assert_arrays_allclose(
        constrained["sigma"], wrapper.idata.posterior.sigma.values, rtol=1e-5
    )

    with caplog.at_level(logging.WARNING):

        class MockTransform:
            def backward(self, value, *args):
                raise ValueError("Test error")

            def forward(self, value, *args):
                raise ValueError("Test error")

        var = wrapper.model.free_RVs[0]
        original_transform = wrapper.model.rvs_to_transforms.get(var)

        wrapper.model.rvs_to_transforms[var] = MockTransform()
        unconstrained = wrapper.get_unconstrained_parameters()
        assert "Failed to transform" in caplog.text

        if original_transform is not None:
            wrapper.model.rvs_to_transforms[var] = original_transform
        else:
            wrapper.model.rvs_to_transforms.pop(var, None)


def test_hierarchical_parameter_transformations(hierarchical_model):
    """Test parameter transformations with hierarchical model structure."""
    model, idata = hierarchical_model
    wrapper = PyMCWrapper(model, idata)
    unconstrained = wrapper.get_unconstrained_parameters()

    expected_params = {"alpha", "beta", "group_sigma", "group_effects_raw", "sigma_y"}
    assert set(unconstrained.keys()) == expected_params

    assert unconstrained["alpha"].dims == ("chain", "draw")
    assert unconstrained["beta"].dims == ("chain", "draw")
    assert unconstrained["group_sigma"].dims == ("chain", "draw")
    assert unconstrained["group_effects_raw"].dims == ("chain", "draw", "group")
    assert unconstrained["sigma_y"].dims == ("chain", "draw")

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

    assert (
        unconstrained["alpha"].coords["chain"].equals(idata.posterior.coords["chain"])
    )
    assert unconstrained["alpha"].coords["draw"].equals(idata.posterior.coords["draw"])
    assert (
        unconstrained["group_effects_raw"]
        .coords["group"]
        .equals(idata.posterior.coords["group"])
    )

    posterior_sigma_y = idata.posterior["sigma_y"].values
    unconstrained_sigma_y = unconstrained["sigma_y"].values

    logger.info("\nTransformation visualization for sigma_y parameter:")
    logger.info("Chain 0, Draws 0-4:")
    logger.info(
        f"{'Original posterior':20} | {'Unconstrained space':20} |"
        f" {'Log of posterior':20}"
    )
    logger.info("-" * 65)

    for i in range(5):
        orig = posterior_sigma_y[0, i]
        uncon = unconstrained_sigma_y[0, i]
        log_orig = np.log(orig)
        logger.info(f"{orig:20.6f} | {uncon:20.6f} | {log_orig:20.6f}")

    sigma_y_var = None
    for var in model.free_RVs:
        if var.name == "sigma_y":
            sigma_y_var = var
            break

    if sigma_y_var is not None:
        transform = model.rvs_to_transforms.get(sigma_y_var)
        if transform is not None:
            logger.info("\nVerifying PyMC's direct transformation vs wrapper:")
            sample_values = posterior_sigma_y[0, :5]

            try:
                direct_transform = transform.backward(sample_values).eval()
                logger.info("\nDirect transform application:")
                logger.info(
                    f"{'Original sigma_y':20} | {'Direct transform':20} |"
                    f" {'Wrapper transform':20}"
                )
                logger.info("-" * 65)

                for i in range(5):
                    orig = sample_values[i]
                    direct = direct_transform[i]
                    wrapper_result = unconstrained_sigma_y[0, i]
                    logger.info(
                        f"{orig:20.6f} | {direct:20.6f} | {wrapper_result:20.6f}"
                    )

                direct_all = transform.backward(posterior_sigma_y).eval()
                transform_match = np.allclose(
                    direct_all, unconstrained_sigma_y, rtol=1e-5
                )
                logger.info(
                    f"\nDirect transform matches wrapper transform: {transform_match}"
                )

            except Exception as e:
                logger.info(f"Couldn't apply transform directly: {e}")

            logger.info(
                f"\nTransform details:\nType: {type(transform)}\nAttributes:"
                f" {dir(transform)}"
            )

    constrained = wrapper.constrain_parameters(unconstrained)

    assert set(constrained.keys()) == expected_params

    logger.info("\nReconstituted values:")
    logger.info(f"{'Original posterior':20} | {'Recalculated constrained':20}")
    logger.info("-" * 45)

    for i in range(5):
        orig = posterior_sigma_y[0, i]
        recon = constrained["sigma_y"].values[0, i]
        logger.info(f"{orig:20.6f} | {recon:20.6f}")

    for param in expected_params:
        if param in idata.posterior:
            assert_arrays_allclose(
                constrained[param], idata.posterior[param].values, rtol=1e-5
            )

    assert_positive(constrained["group_sigma"])
    assert_positive(constrained["sigma_y"])


def test_mixture_model_log_likelihood_i(mixture_model):
    """Test log_likelihood_i method with mixture model."""
    model, idata = mixture_model
    wrapper = PyMCWrapper(model, idata)

    log_like_i = wrapper.log_likelihood_i(0, idata)

    assert isinstance(log_like_i, xr.DataArray)
    assert set(log_like_i.dims) == {"chain", "draw"}
    assert_finite(log_like_i)
    assert_bounded(log_like_i, upper=0)


def test_mixture_model_parameter_transformations(mixture_model, caplog):
    """Test parameter transformations for mixture model."""
    model, idata = mixture_model
    wrapper = PyMCWrapper(model, idata)

    unconstrained = wrapper.get_unconstrained_parameters()

    expected_params = {"w", "mu1", "mu2", "sigma1", "sigma2"}
    assert set(unconstrained.keys()) == expected_params

    for param in expected_params:
        assert unconstrained[param].shape == (
            len(idata.posterior.chain),
            len(idata.posterior.draw),
        )

    with caplog.at_level(logging.WARNING):
        constrained = wrapper.constrain_parameters(unconstrained)

    assert set(constrained.keys()) == expected_params

    for param in expected_params:
        assert_arrays_allclose(
            constrained[param], idata.posterior[param].values, rtol=1e-5
        )

    assert_bounded(constrained["w"], lower=0, upper=1)
    assert_positive(constrained["sigma1"])
    assert_positive(constrained["sigma2"])

    for param in expected_params:
        assert unconstrained[param].shape == constrained[param].shape
        assert_finite(unconstrained[param])
        assert_finite(constrained[param])


def test_mixture_model_log_likelihood_i_workflow(mixture_model):
    """Test the full LOO-CV workflow for mixture model."""
    model, idata = mixture_model
    wrapper = PyMCWrapper(model, idata)

    test_idx = 10
    original_data = wrapper.get_observed_data()

    _, training_data = wrapper.select_observations(np.array([test_idx], dtype=int))

    wrapper.set_data({wrapper.get_observed_name(): training_data})
    refitted_idata = wrapper.sample_posterior(
        draws=100, tune=100, chains=2, random_seed=42
    )

    log_like = wrapper.log_likelihood_i(test_idx, refitted_idata)

    assert isinstance(log_like, xr.DataArray)
    assert set(log_like.dims) == {"chain", "draw"}
    assert log_like.sizes["chain"] == 2
    assert log_like.sizes["draw"] == 100
    assert_finite(log_like)

    wrapper.set_data({wrapper.get_observed_name(): original_data})


def test_check_implemented_methods():
    """Test the check_implemented_methods functionality."""

    class MockWrapper:
        def __init__(self):
            pass

        def implemented_method(self):
            return "This method is implemented"

    mock_wrapper = MockWrapper()
    mock_wrapper.check_implemented_methods = (
        PyMCWrapper.check_implemented_methods.__get__(mock_wrapper, MockWrapper)
    )

    required_methods = ["set_data", "log_likelihood_i", "implemented_method"]
    missing_methods = mock_wrapper.check_implemented_methods(required_methods)
    assert set(missing_methods) == {"set_data", "log_likelihood_i"}
    assert "implemented_method" not in missing_methods

    required_methods = ["implemented_method"]
    missing_methods = mock_wrapper.check_implemented_methods(required_methods)
    assert missing_methods == []

    missing_methods = mock_wrapper.check_implemented_methods([])
    assert missing_methods == []


def test_get_dims(simple_model, hierarchical_model):
    """Test the get_dims method for retrieving variable dimensions."""
    model, idata = simple_model
    wrapper = PyMCWrapper(model, idata)

    dims = wrapper.get_dims("y")
    assert dims is not None
    assert len(dims) > 0

    dims_implicit = wrapper.get_dims()
    assert dims_implicit is not None
    assert dims_implicit == dims

    dims_param = wrapper.get_dims("alpha")
    assert dims_param is None or len(dims_param) == 0

    model, idata = hierarchical_model
    wrapper = PyMCWrapper(model, idata)

    dims_hier = wrapper.get_dims("Y")
    assert dims_hier is not None
    assert len(dims_hier) == 2

    dims_nonexistent = wrapper.get_dims("nonexistent_var")
    assert dims_nonexistent is None


def test_get_draws(simple_model):
    """Test the get_draws method for retrieving posterior samples."""
    model, idata = simple_model
    wrapper = PyMCWrapper(model, idata)

    draws = wrapper.get_draws()

    assert isinstance(draws, xr.Dataset)

    for var_name in ["alpha", "beta", "sigma"]:
        assert var_name in draws

    assert "chain" in draws.dims
    assert "draw" in draws.dims

    n_chains = len(idata.posterior.chain)
    n_draws = len(idata.posterior.draw)
    assert draws.sizes["chain"] == n_chains
    assert draws.sizes["draw"] == n_draws

    idata_copy = copy.deepcopy(idata)
    delattr(idata_copy, "posterior")

    with pytest.raises(PyMCWrapperError, match="must contain posterior samples"):
        temp_wrapper = wrapper
        temp_wrapper.idata = idata_copy
        temp_wrapper.get_draws()


def test_get_variable(simple_model, hierarchical_model):
    """Test the get_variable method for retrieving model variables."""
    model, idata = simple_model
    wrapper = PyMCWrapper(model, idata)

    var = wrapper.get_variable("alpha")
    assert var is not None
    assert var.name == "alpha"

    var_missing = wrapper.get_variable("nonexistent_var")
    assert var_missing is None

    model, idata = hierarchical_model
    wrapper = PyMCWrapper(model, idata)

    var_obs = wrapper.get_variable("Y")
    assert var_obs is not None
    assert var_obs.name == "Y"

    var_hier = wrapper.get_variable("group_effects_raw")
    assert var_hier is not None
    assert var_hier.name == "group_effects_raw"


def test_variational_approximation_handling(simple_model, monkeypatch):
    """Test handling of variational approximations in the wrapper."""
    model, idata = simple_model

    class MockApproximation:
        def __init__(self):
            self.called = False

        def sample(self, draws=1000, random_seed=None):
            self.called = True
            return {
                "alpha": np.random.normal(0, 1, size=draws),
                "beta": np.random.normal(0, 1, size=draws),
                "sigma": np.abs(np.random.normal(0, 1, size=draws)),
            }

    mock_approx = MockApproximation()

    wrapper = PyMCWrapper(model, idata, approximation=mock_approx)

    assert wrapper.approximation is mock_approx

    assert wrapper.approximation is not None
    assert isinstance(wrapper.approximation, MockApproximation)


def test_parameter_transformation_edge_cases(simple_model, caplog):
    """Test edge cases in parameter transformations."""
    model, idata = simple_model
    wrapper = PyMCWrapper(model, idata)

    unconstrained = wrapper.get_unconstrained_parameters()

    invalid_unconstrained = {
        "alpha": "not an array",
    }

    with pytest.raises((TypeError, PyMCWrapperError)):
        wrapper.constrain_parameters(invalid_unconstrained)

    partial_unconstrained = {k: v for k, v in unconstrained.items() if k != "alpha"}
    constrained = wrapper.constrain_parameters(partial_unconstrained)

    assert "alpha" not in constrained
    assert set(constrained.keys()) == set(partial_unconstrained.keys())

    with caplog.at_level(logging.WARNING):

        class BrokenTransform:
            def forward(self, *args, **kwargs):
                raise ValueError("Test forward error")

            def backward(self, *args, **kwargs):
                raise ValueError("Test backward error")

        var = None
        for v in wrapper.model.free_RVs:
            if v.name == "sigma":
                var = v
                break

        if var is not None:
            original_transform = wrapper.model.rvs_to_transforms.get(var)

            try:
                wrapper.model.rvs_to_transforms[var] = BrokenTransform()

                broken_unconstrained = wrapper.get_unconstrained_parameters()
                assert (
                    "Failed to transform" in caplog.text
                    or "transform failed" in caplog.text
                )

                assert "sigma" in broken_unconstrained

                caplog.clear()
                broken_constrained = wrapper.constrain_parameters(broken_unconstrained)
                assert (
                    "Failed to transform" in caplog.text
                    or "transform failed" in caplog.text
                )

                assert "sigma" in broken_constrained

            finally:
                if original_transform is not None:
                    wrapper.model.rvs_to_transforms[var] = original_transform
                else:
                    wrapper.model.rvs_to_transforms.pop(var, None)


def test_missing_value_handling(simple_model):
    """Test handling of missing values in observed data."""
    model, idata = simple_model
    wrapper = PyMCWrapper(model, idata)

    original_data = wrapper.get_observed_data()
    data_with_missing = original_data.copy()
    data_with_missing[::3] = np.nan

    wrapper.set_data({"y": data_with_missing})

    mask = wrapper.get_missing_mask("y")
    assert np.all(mask[::3])
    assert not np.any(mask[1::3])
    assert not np.any(mask[2::3])

    mask_axis = wrapper.get_missing_mask("y", axis=0)
    assert np.sum(mask_axis) == np.sum(mask)

    with pytest.raises(PyMCWrapperError, match="Missing values found"):
        wrapper.log_likelihood_i(1, idata)

    selected, remaining = wrapper.select_observations(slice(0, 10))
    assert len(selected) == 10
    assert len(remaining) == 90

    wrapper.set_data({"y": original_data})


def test_sample_posterior_errors(simple_model):
    """Test comprehensive error cases for sample_posterior method."""
    model, idata = simple_model
    wrapper = PyMCWrapper(model, idata)

    with pytest.raises(PyMCWrapperError, match="Number of draws must be positive"):
        wrapper.sample_posterior(draws=0)

    with pytest.raises(PyMCWrapperError, match="Number of draws must be positive"):
        wrapper.sample_posterior(draws=-100)

    with pytest.raises(PyMCWrapperError, match="Number of chains must be positive"):
        wrapper.sample_posterior(chains=0)

    with pytest.raises(PyMCWrapperError, match="Number of chains must be positive"):
        wrapper.sample_posterior(chains=-2)

    try:
        wrapper.sample_posterior(draws=10, chains=1, target_accept=0.8)
    except Exception as e:
        raise e

    with pytest.raises(PyMCWrapperError):
        original_sample = pm.sample
        try:

            def mock_sample(*args, **kwargs):
                raise ValueError("Test error in sampling")

            pm.sample = mock_sample
            wrapper.sample_posterior(draws=10, chains=1)
        finally:
            pm.sample = original_sample


def test_coordinate_handling(hierarchical_model):
    """Test comprehensive coordinate handling in different scenarios."""
    model, idata = hierarchical_model
    wrapper = PyMCWrapper(model, idata)

    new_data = np.random.normal(0, 1, size=(8, 20))
    coords = {"group": list(range(8)), "obs_id": list(range(20))}
    wrapper.set_data({"Y": new_data}, coords=coords)
    assert_arrays_equal(wrapper.observed_data["Y"], new_data)

    new_data = np.random.normal(0, 1, size=(10, 25))
    with pytest.warns(UserWarning):
        wrapper.set_data(
            {"Y": new_data},
            coords={"group": list(range(8)), "obs_id": list(range(20))},
            update_coords=True,
        )
    assert_arrays_equal(wrapper.observed_data["Y"], new_data)

    new_data = np.random.normal(0, 1, size=(12, 30))
    with pytest.raises(
        ValueError, match="Coordinate length .* does not match variable shape"
    ):
        wrapper.set_data(
            {"Y": new_data},
            coords={"group": list(range(8)), "obs_id": list(range(20))},
            update_coords=False,
        )

    new_data = np.random.normal(0, 1, size=(8, 20))
    partial_coords = {"group": list(range(8))}
    with pytest.raises(ValueError, match="Missing coordinates for dimensions"):
        wrapper.set_data({"Y": new_data}, coords=partial_coords, update_coords=False)

    new_data = np.random.normal(0, 1, size=(8, 20))
    wrong_length_coords = {"group": list(range(8)), "obs_id": list(range(15))}
    with pytest.raises(ValueError, match="Coordinate length .* does not match"):
        wrapper.set_data(
            {"Y": new_data}, coords=wrong_length_coords, update_coords=False
        )

    new_data = np.random.normal(0, 1, size=(8, 20))
    mask = np.ones((8, 20), dtype=bool)
    mask[0, :] = False

    with pytest.raises(PyMCWrapperError, match="Mask shape"):
        wrong_mask = np.ones((9, 20), dtype=bool)
        wrapper.set_data({"Y": new_data}, coords=coords, mask={"Y": wrong_mask})

    try:
        wrapper.set_data({"Y": new_data}, coords=coords, mask={"Y": mask})
        masked_data = wrapper.observed_data["Y"]
        if hasattr(masked_data, "mask"):
            assert np.all(masked_data.mask[0, :])
    except Exception as e:
        if "mask" not in str(e):
            raise


def test_observed_data_access_errors():
    """Test error handling when accessing observed data incorrectly."""

    class MockWrapper:
        def __init__(self):
            self.observed_data = {}

    mock_wrapper = MockWrapper()
    mock_wrapper._validate_observed_var = PyMCWrapper._validate_observed_var.__get__(
        mock_wrapper, MockWrapper
    )

    with pytest.raises(PyMCWrapperError, match="not found in observed data"):
        mock_wrapper._validate_observed_var("nonexistent_var")

    mock_wrapper.observed_data = {"y": np.random.normal(0, 1, size=10)}
    mock_wrapper._validate_observed_var("y")


def test_wrapper_initialization_with_var_names(simple_model):
    """Test wrapper initialization with var_names parameter."""
    model, idata = simple_model

    var_names = ["alpha", "beta"]
    wrapper = PyMCWrapper(model, idata, var_names=var_names)
    assert wrapper.var_names == var_names

    original_data = idata.observed_data["y"].values.copy()
    wrapper.set_data({"y": original_data})
    assert len(wrapper.observed_data) > 0

    log_like_i = wrapper.log_likelihood_i(0, idata)
    assert isinstance(log_like_i, xr.DataArray)
    assert set(log_like_i.dims) == {"chain", "draw"}


def test_log_likelihood_i_invalid_idata(simple_model):
    """Test log_likelihood_i with invalid InferenceData object."""
    model, idata = simple_model
    wrapper = PyMCWrapper(model, idata)

    idata_invalid = InferenceData()

    with pytest.raises(PyMCWrapperError, match="must contain posterior samples"):
        wrapper.log_likelihood_i(0, idata_invalid)

    idata_copy = copy.deepcopy(idata)

    original_compute_ll = pm.compute_log_likelihood
    try:

        def mock_compute_ll(*args, **kwargs):
            return {}

        pm.compute_log_likelihood = mock_compute_ll
        with pytest.raises(PyMCWrapperError, match="Failed to compute log likelihood"):
            wrapper.log_likelihood_i(0, idata_copy)
    finally:
        pm.compute_log_likelihood = original_compute_ll


def test_get_shape_behavior(simple_model, hierarchical_model):
    """Test the get_shape method's behavior with different variables."""
    model, idata = simple_model
    wrapper = PyMCWrapper(model, idata)

    shape = wrapper.get_shape("y")
    assert shape is not None
    assert shape == (100,)

    shape_alpha = wrapper.get_shape("alpha")
    assert shape_alpha is not None
    assert shape_alpha == () or shape_alpha == (1,)

    shape_nonexistent = wrapper.get_shape("nonexistent_var")
    assert shape_nonexistent is None

    model, idata = hierarchical_model
    wrapper = PyMCWrapper(model, idata)

    shape_Y = wrapper.get_shape("Y")
    assert shape_Y is not None
    assert len(shape_Y) == 2

    shape_group_effects = wrapper.get_shape("group_effects_raw")
    assert shape_group_effects is not None
    assert len(shape_group_effects) > 0


def test_initialization_error_handling(caplog):
    """Test error handling during initialization."""

    class MockVar:
        def __init__(self, name, shape=()):
            self.name = name
            self.shape = shape
            self.tag = None

    class MockModel:
        def __init__(self):
            self.var1 = MockVar("var1", ())
            self.var2 = MockVar("var2", ())
            self.named_vars = {"var1": self.var1, "var2": self.var2}
            self.free_RVs = []
            self.rvs_to_transforms = {}
            self.observed_RVs = []
            self.deterministics = []

    mock_idata = InferenceData()

    with pytest.raises(PyMCWrapperError, match="must contain posterior samples"):
        PyMCWrapper(MockModel(), mock_idata)


def test_error_handling(simple_model):
    """Test comprehensive error handling across various methods."""
    model, idata = simple_model
    wrapper = PyMCWrapper(model, idata)

    with pytest.raises(PyMCWrapperError, match="Variable .* not found in model"):
        wrapper.set_data({"nonexistent_var": np.zeros(10)})

    wrapper_no_observed = copy.deepcopy(wrapper)
    wrapper_no_observed.observed_data = {}
    with pytest.raises(PyMCWrapperError, match="No observed variables found"):
        wrapper_no_observed.select_observations(slice(0, 10))

    with pytest.raises(PyMCWrapperError, match="No observed variables found"):
        wrapper_no_observed.get_observed_name()

    with pytest.raises((IndexError, PyMCWrapperError)):
        wrapper.select_observations(object())

    with pytest.raises(PyMCWrapperError, match="not found in model"):
        wrapper.log_likelihood_i(0, idata, "invalid_var")

    with pytest.raises(IndexError):
        wrapper.log_likelihood_i(1000, idata)

    bad_idata = copy.deepcopy(idata)
    if hasattr(bad_idata, "log_likelihood"):
        delattr(bad_idata, "log_likelihood")

    try:
        wrapper.log_likelihood_i(0, bad_idata)
    except PyMCWrapperError:
        pass
