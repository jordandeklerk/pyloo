"""Tests for the base functionality of LOO approximations."""

import numpy as np
import pytest
import xarray as xr

from ...approximations.base import LooApproximation, thin_draws


class MockApproximation(LooApproximation):

    def compute_approximation(self, log_likelihood, n_draws=None):
        """Mock implementation that returns the mean of log_likelihood."""
        if n_draws is not None:
            log_likelihood = thin_draws(log_likelihood, n_draws)
        return np.mean(log_likelihood, axis=-1).values


def test_loo_approximation_abstract():
    with pytest.raises(TypeError):
        LooApproximation()


def test_mock_approximation():
    approx = MockApproximation()
    log_likelihood = xr.DataArray(
        np.random.randn(10, 100),
        dims=["obs_id", "__sample__"],
        coords={"obs_id": range(10), "__sample__": range(100)},
    )
    result = approx.compute_approximation(log_likelihood)
    assert isinstance(result, np.ndarray)
    assert result.shape == (10,)


def test_thin_draws_none():
    data = xr.DataArray(
        np.random.randn(10, 100),
        dims=["obs_id", "__sample__"],
        coords={"obs_id": range(10), "__sample__": range(100)},
    )
    result = thin_draws(data, None)
    assert result is data


def test_thin_draws_subset():
    data = xr.DataArray(
        np.random.randn(10, 100),
        dims=["obs_id", "__sample__"],
        coords={"obs_id": range(10), "__sample__": range(100)},
    )
    n_draws = 50
    result = thin_draws(data, n_draws)
    assert result.sizes["__sample__"] == n_draws
    assert result.dims == data.dims
    assert set(result.coords["obs_id"].values) == set(data.coords["obs_id"].values)


def test_thin_draws_all():
    data = xr.DataArray(
        np.random.randn(10, 100),
        dims=["obs_id", "__sample__"],
        coords={"obs_id": range(10), "__sample__": range(100)},
    )
    n_draws = 100
    result = thin_draws(data, n_draws)
    assert result.sizes["__sample__"] == n_draws
    assert result.dims == data.dims
    assert set(result.coords["obs_id"].values) == set(data.coords["obs_id"].values)


def test_thin_draws_too_many():
    data = xr.DataArray(
        np.random.randn(10, 100),
        dims=["obs_id", "__sample__"],
        coords={"obs_id": range(10), "__sample__": range(100)},
    )
    n_draws = 200
    with pytest.raises(
        ValueError,
        match=(
            f"Requested {n_draws} draws but only {data.sizes['__sample__']} are"
            " available"
        ),
    ):
        thin_draws(data, n_draws)


def test_thin_draws_dataset():
    data = xr.Dataset(
        {
            "var1": (["obs_id", "__sample__"], np.random.randn(10, 100)),
            "var2": (["obs_id", "__sample__"], np.random.randn(10, 100)),
        },
        coords={"obs_id": range(10), "__sample__": range(100)},
    )
    n_draws = 50
    result = thin_draws(data, n_draws)
    assert result.sizes["__sample__"] == n_draws
    assert set(result.data_vars) == set(data.data_vars)
    assert set(result.coords["obs_id"].values) == set(data.coords["obs_id"].values)


def test_thin_draws_chain_draw():
    data = xr.Dataset(
        {
            "var1": (["chain", "draw", "obs_id"], np.random.randn(4, 25, 10)),
        },
        coords={"chain": range(4), "draw": range(25), "obs_id": range(10)},
    )
    n_draws = 50
    result = thin_draws(data, n_draws)
    assert "__sample__" in result.dims
    assert result.sizes["__sample__"] == n_draws
    assert "chain" not in result.dims
    assert "draw" not in result.dims
