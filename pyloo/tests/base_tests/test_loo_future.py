"""Tests for the loo_future function."""

import logging

import numpy as np
import pytest
import xarray as xr

from ...loo_future import loo_future
from ...wrapper.pymc import PyMCWrapper

logger = logging.getLogger(__name__)


def test_basic_functionality(time_series_model):
    """Test basic functionality of loo_future with default parameters."""
    model, idata = time_series_model
    wrapper = PyMCWrapper(model, idata)

    L = 130
    M = 1

    loo_future_result = loo_future(wrapper, M=M, L=L)

    assert loo_future_result is not None
    assert "elpd_lfo" in loo_future_result
    assert "se" in loo_future_result
    assert "refit_indices" in loo_future_result
    assert len(loo_future_result["refit_indices"]) >= 1

    logger.info(loo_future_result)


def test_m_step_ahead_prediction(time_series_model):
    """Test M-step-ahead prediction with M > 1."""
    model, idata = time_series_model
    wrapper = PyMCWrapper(model, idata)

    L = 130
    M = 4

    result = loo_future(wrapper, M=M, L=L)

    assert result is not None
    assert "elpd_lfo" in result
    assert "M" in result
    assert result["M"] == M

    logger.info(result)


def test_pointwise_output(time_series_model):
    """Test pointwise output of loo_future."""
    model, idata = time_series_model
    wrapper = PyMCWrapper(model, idata)

    L = 100
    M = 1

    result = loo_future(wrapper, M=M, L=L, pointwise=True)

    assert "lfo_i" in result
    assert isinstance(result["lfo_i"], xr.DataArray)
    assert "pareto_k" in result
    assert isinstance(result["pareto_k"], xr.DataArray)

    N = len(wrapper.observed_data[wrapper.get_observed_name()])
    expected_size = N - M - L
    assert result["lfo_i"].shape[0] == expected_size


def test_different_methods(time_series_model):
    """Test loo_future with different importance sampling methods."""
    model, idata = time_series_model
    wrapper = PyMCWrapper(model, idata)

    L = 100
    M = 1

    result_psis = loo_future(wrapper, M=M, L=L, method="psis")
    assert result_psis is not None
    assert not np.isnan(result_psis["elpd_lfo"])

    result_sis = loo_future(wrapper, M=M, L=L, method="sis")
    assert result_sis is not None
    assert not np.isnan(result_sis["elpd_lfo"])


def test_different_k_threshold(time_series_model):
    """Test loo_future with different k_threshold values."""
    model, idata = time_series_model
    wrapper = PyMCWrapper(model, idata)

    L = 100
    M = 1

    result_default = loo_future(wrapper, M=M, L=L, k_threshold=0.7)
    result_strict = loo_future(wrapper, M=M, L=L, k_threshold=0.5)

    assert result_default is not None
    assert result_strict is not None

    assert not np.isnan(result_default["elpd_lfo"])
    assert not np.isnan(result_strict["elpd_lfo"])


def test_different_scales(time_series_model):
    """Test loo_future with different output scales."""
    model, idata = time_series_model
    wrapper = PyMCWrapper(model, idata)

    L = 100
    M = 1

    result_log = loo_future(wrapper, M=M, L=L, scale="log")
    result_neglog = loo_future(wrapper, M=M, L=L, scale="negative_log")
    result_deviance = loo_future(wrapper, M=M, L=L, scale="deviance")

    assert result_log["scale"] == "log"
    assert result_neglog["scale"] == "negative_log"
    assert result_deviance["scale"] == "deviance"

    assert np.isclose(result_log["elpd_lfo"], -result_neglog["elpd_lfo"], rtol=1e-4)
    assert np.isclose(
        result_log["elpd_lfo"], -0.5 * result_deviance["elpd_lfo"], rtol=1e-4
    )


def test_error_handling(time_series_model):
    """Test error handling in loo_future."""
    model, idata = time_series_model
    wrapper = PyMCWrapper(model, idata)

    L = 100
    M = 1

    with pytest.raises(TypeError, match="must be a PyMCWrapper instance"):
        loo_future("not a wrapper", M=M, L=L)

    with pytest.raises(ValueError, match="M .* must be >= 1"):
        loo_future(wrapper, M=0, L=L)

    with pytest.raises(ValueError, match="L .* must be >= 0"):
        loo_future(wrapper, M=M, L=-1)

    with pytest.raises(ValueError):
        loo_future(wrapper, M=M, L=L, method="invalid_method")

    with pytest.raises(TypeError, match="Valid scale values"):
        loo_future(wrapper, M=M, L=L, scale="invalid_scale")

    with pytest.raises(ValueError, match="Not enough data points"):
        loo_future(wrapper, M=100, L=100)


def test_with_custom_sampling_args(time_series_model):
    """Test loo_future with custom sampling arguments."""
    model, idata = time_series_model
    wrapper = PyMCWrapper(model, idata)

    L = 100
    M = 1

    custom_args = {"draws": 100, "tune": 50, "chains": 1, "random_seed": 42}

    result = loo_future(wrapper, M=M, L=L, **custom_args)

    assert result is not None
    assert "elpd_lfo" in result
    assert "n_samples" in result
    assert result["n_samples"] == custom_args["draws"] * custom_args["chains"]


def test_with_fast_minimal_run(time_series_model):
    """Run a very fast minimal test with small numbers for quick testing."""
    model, idata = time_series_model
    wrapper = PyMCWrapper(model, idata)

    L = 100
    M = 1

    minimal_args = {"draws": 50, "tune": 25, "chains": 1, "random_seed": 42}

    result = loo_future(wrapper, M=M, L=L, **minimal_args)

    assert result is not None
    assert "elpd_lfo" in result


def test_result_attributes(time_series_model):
    """Test that the warning attribute is set correctly."""
    model, idata = time_series_model
    wrapper = PyMCWrapper(model, idata)

    L = 100
    M = 1

    result = loo_future(wrapper, M=M, L=L)

    assert hasattr(result, "warning")
    assert isinstance(result.warning, bool)
