"""Tests for ELPD base class."""
import numpy as np
import pytest
import xarray as xr
from arviz import InferenceData

from ...elpd import ELPD
from ...utils import compute_log_mean_exp
from ..helpers import (
    assert_arrays_allclose,
    assert_arrays_equal,
    assert_dtype,
    assert_shape_equal,
)


class SimpleELPD(ELPD):
    """Simple ELPD implementation for testing."""

    def pointwise_estimates(self):
        """Compute simple pointwise estimates."""
        elpd = compute_log_mean_exp(self.log_likelihood, axis=0)
        return {"elpd": elpd, "ic": -2 * elpd}


def create_multidim_data():
    """Create test data with multiple dimensions."""
    # Create data with dimensions (chain, draw, geo, time)
    n_chains = 4
    n_draws = 100
    n_geo = 3
    n_time = 5

    log_lik = np.random.normal(size=(n_chains, n_draws, n_geo, n_time))

    coords = {
        "chain": np.arange(n_chains),
        "draw": np.arange(n_draws),
        "geo": [f"region_{i}" for i in range(n_geo)],
        "time": np.arange(n_time),
    }

    ds = xr.Dataset(
        {
            "obs": (["chain", "draw", "geo", "time"], log_lik),
        },
        coords=coords,
    )

    return InferenceData(log_likelihood=ds)


def test_elpd_init_with_inference_data(centered_eight):
    """Test ELPD initialization with InferenceData."""
    elpd = SimpleELPD(centered_eight)
    log_lik = centered_eight.log_likelihood.obs.stack(__sample__=("chain", "draw"))

    assert_arrays_equal(elpd.log_likelihood, log_lik.values.T)
    assert_dtype(elpd.log_likelihood, np.float64)

    assert elpd.n_samples == log_lik.sizes["__sample__"]
    assert elpd.n_observations == log_lik.sizes["school"]

    assert elpd.chain_ids is not None
    assert len(elpd.chain_ids) == elpd.n_samples


def test_elpd_init_with_array(numpy_arrays):
    """Test ELPD initialization with numpy array."""
    log_likelihood = numpy_arrays["random_weights"]
    elpd = SimpleELPD(log_likelihood)

    assert_arrays_equal(elpd.log_likelihood, log_likelihood)
    assert elpd.chain_ids is None
    assert elpd.n_samples == log_likelihood.shape[0]
    assert elpd.n_observations == log_likelihood.shape[1]


def test_elpd_with_multidim_data():
    """Test ELPD with high-dimensional data."""
    idata = create_multidim_data()
    n_chains = len(idata.log_likelihood.chain)
    n_draws = len(idata.log_likelihood.draw)
    n_geo = len(idata.log_likelihood.geo)
    n_time = len(idata.log_likelihood.time)

    elpd = SimpleELPD(idata)
    assert elpd.n_samples == n_chains * n_draws
    assert elpd.n_observations == n_geo * n_time
    assert set(elpd.dims.keys()) == {"geo", "time"}
    assert elpd.dims["geo"] == n_geo
    assert elpd.dims["time"] == n_time

    str_output = str(elpd)
    assert "Original dimensions" in str_output
    assert "geo: 3" in str_output
    assert "time: 5" in str_output
    assert "combined into a single observation dimension" in str_output


def test_elpd_with_numpy_multidim():
    """Test ELPD with high-dimensional numpy array."""
    data = np.random.normal(size=(4, 100, 3, 5))
    elpd = SimpleELPD(data)

    assert elpd.n_samples == 4 * 100  # chains * draws
    assert elpd.n_observations == 3 * 5  # dim1 * dim2
    assert elpd.dims is None  # No dimension info for numpy arrays


def test_elpd_estimates(centered_eight):
    """Test ELPD estimate calculations."""
    elpd = SimpleELPD(centered_eight)

    pointwise = elpd.pointwise_estimates()
    assert isinstance(pointwise, dict)
    assert set(pointwise.keys()) == {"elpd", "ic"}

    assert_shape_equal(pointwise["elpd"], np.zeros(elpd.n_observations))
    assert_shape_equal(pointwise["ic"], np.zeros(elpd.n_observations))
    assert_arrays_allclose(pointwise["ic"], -2 * pointwise["elpd"])

    agg = elpd.aggregate_estimates()
    assert isinstance(agg, dict)
    assert set(agg.keys()) == {"elpd", "ic"}

    for metric in agg:
        assert set(agg[metric].keys()) == {"estimate", "se"}
        assert isinstance(agg[metric]["estimate"], np.ndarray)
        assert isinstance(agg[metric]["se"], np.ndarray)
        assert agg[metric]["estimate"].shape == (1,)
        assert agg[metric]["se"].shape == (1,)


def test_elpd_with_extreme_data(extreme_data):
    """Test ELPD with data containing extreme values."""
    elpd = SimpleELPD(extreme_data)
    pointwise = elpd.pointwise_estimates()

    assert np.all(np.isfinite(pointwise["elpd"]))
    assert np.all(np.isfinite(pointwise["ic"]))


def test_elpd_str_repr(centered_eight):
    """Test string and repr representations."""
    elpd = SimpleELPD(centered_eight)

    str_output = str(elpd)
    assert "ELPD estimates" in str_output
    assert str(elpd.n_samples) in str_output
    assert str(elpd.n_observations) in str_output
    assert "Estimate:" in str_output
    assert "SE:" in str_output

    repr_output = repr(elpd)
    assert "SimpleELPD" in repr_output
    assert f"n_samples={elpd.n_samples}" in repr_output
    assert f"n_observations={elpd.n_observations}" in repr_output


def test_elpd_invalid_input():
    """Test ELPD initialization with invalid input."""
    with pytest.raises(TypeError):
        SimpleELPD([1, 2, 3])

    with pytest.raises(ValueError):
        SimpleELPD(np.array([]))

    with pytest.raises(ValueError):
        SimpleELPD(np.zeros((1, 10)))
