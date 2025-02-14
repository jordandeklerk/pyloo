"""Tests for effective sample size calculations."""

import numpy as np
import pytest

from ...ess import _mat_to_chains, mcmc_eff_size, psis_eff_size, rel_eff
from ..helpers import (
    assert_arrays_allclose,
    assert_bounded,
    assert_positive,
    generate_psis_data,
)


def test_rel_eff_1d(centered_eight):
    """Test rel_eff with 1D input from real data."""
    x = centered_eight.posterior.mu.values.flatten()
    chain_id = np.repeat(
        np.arange(1, centered_eight.posterior.chain.size + 1),
        centered_eight.posterior.draw.size,
    )
    r_eff = rel_eff(x, chain_id)
    assert isinstance(r_eff, np.ndarray)
    assert r_eff.shape == (1,)
    assert_bounded(r_eff, lower=0, upper=1)


def test_rel_eff_2d(centered_eight):
    """Test rel_eff with 2D input from real data."""
    x = centered_eight.posterior.theta.values.reshape(-1, 8)
    chain_id = np.repeat(
        np.arange(1, centered_eight.posterior.chain.size + 1),
        centered_eight.posterior.draw.size,
    )
    r_eff = rel_eff(x, chain_id)
    assert isinstance(r_eff, np.ndarray)
    assert r_eff.shape == (8,)
    assert_bounded(r_eff, lower=0, upper=1)


def test_rel_eff_3d(multidim_data):
    """Test rel_eff with 3D input from multidimensional data."""
    x = multidim_data["llm"]  # shape: (4, 23, 15, 2)
    x_reshaped = x.reshape(4, 23, -1)  # combine last two dimensions
    r_eff = rel_eff(x_reshaped)
    assert isinstance(r_eff, np.ndarray)
    assert r_eff.shape == (30,)  # 15 * 2 = 30
    assert_bounded(r_eff, lower=0, upper=1)


def test_rel_eff_function(centered_eight, rng):
    """Test rel_eff with function input using real data."""
    draws = centered_eight.posterior.theta.values.reshape(-1, 8)
    chain_id = np.repeat(
        np.arange(1, centered_eight.posterior.chain.size + 1),
        centered_eight.posterior.draw.size,
    )

    def log_lik_fun(data_i, draws):
        # data_i is a slice with one element, we need its integer index
        idx = data_i.item()
        loc = draws[:, int(idx)].reshape(-1, 1)
        return rng.normal(loc=loc, scale=1)

    r_eff = rel_eff(log_lik_fun, chain_id, data=np.arange(8), draws=draws)
    assert isinstance(r_eff, np.ndarray)
    assert r_eff.shape == (8,)
    assert_bounded(r_eff, lower=0, upper=1)


def test_rel_eff_parallel(multidim_data):
    """Test rel_eff with parallel processing using real data."""
    x = multidim_data["ll1"]  # shape: (4, 23, 30)
    r_eff_single = rel_eff(x, cores=1)
    r_eff_parallel = rel_eff(x, cores=2)
    assert_arrays_allclose(r_eff_single, r_eff_parallel, rtol=1e-10)


def test_psis_eff_size_1d(rng):
    """Test psis_eff_size with 1D input using generated PSIS data."""
    psis_data = generate_psis_data(rng, n_samples=1000, n_obs=1)
    w = np.exp(psis_data["log_ratios"] - psis_data["log_ratios"].max(axis=0))
    w = w / w.sum(axis=0)

    n_eff = psis_eff_size(w.flatten())
    assert isinstance(n_eff, (float, np.floating))
    assert_positive(n_eff)
    assert n_eff <= 1000

    r_eff = 0.5
    n_eff_adj = psis_eff_size(w.flatten(), r_eff)
    assert isinstance(n_eff_adj, (float, np.floating))
    assert_arrays_allclose(n_eff_adj, n_eff * r_eff)


def test_psis_eff_size_2d(rng):
    """Test psis_eff_size with 2D input using generated PSIS data."""
    psis_data = generate_psis_data(rng, n_samples=1000, n_obs=5)
    w = np.exp(psis_data["log_ratios"] - psis_data["log_ratios"].max(axis=0))
    w = w / w.sum(axis=0)

    n_eff = psis_eff_size(w)
    assert isinstance(n_eff, np.ndarray)
    assert n_eff.shape == (5,)
    assert_positive(n_eff)
    assert_bounded(n_eff, upper=1000)

    r_eff = np.full(5, 0.5)
    n_eff_adj = psis_eff_size(w, r_eff)
    assert_arrays_allclose(n_eff_adj, n_eff * r_eff)


def test_mcmc_eff_size(centered_eight, non_centered_eight):
    """Test mcmc_eff_size with real MCMC samples."""
    x_centered = centered_eight.posterior.theta.values.reshape(-1, 8)
    ess_centered = mcmc_eff_size(x_centered)
    assert isinstance(ess_centered, (float, np.floating))
    assert_positive(ess_centered)

    x_non_centered = non_centered_eight.posterior.theta.values.reshape(-1, 8)
    ess_non_centered = mcmc_eff_size(x_non_centered)
    assert isinstance(ess_non_centered, (float, np.floating))
    assert_positive(ess_non_centered)


def test_mcmc_eff_size_methods(centered_eight):
    """Test different ESS calculation methods."""
    x = centered_eight.posterior.theta.values.reshape(-1, 8)

    ess_bulk = mcmc_eff_size(x, method="bulk")
    assert isinstance(ess_bulk, (float, np.floating))
    assert_positive(ess_bulk)

    ess_tail_default = mcmc_eff_size(x, method="tail")  # default prob=0.1
    ess_tail_custom = mcmc_eff_size(x, method="tail", prob=0.05)
    assert isinstance(ess_tail_default, (float, np.floating))
    assert isinstance(ess_tail_custom, (float, np.floating))
    assert_positive(ess_tail_default)
    assert_positive(ess_tail_custom)

    ess_mean = mcmc_eff_size(x, method="mean")
    assert isinstance(ess_mean, (float, np.floating))
    assert_positive(ess_mean)

    ess_sd = mcmc_eff_size(x, method="sd")
    assert isinstance(ess_sd, (float, np.floating))
    assert_positive(ess_sd)

    ess_median = mcmc_eff_size(x, method="median")
    assert isinstance(ess_median, (float, np.floating))
    assert_positive(ess_median)

    ess_mad = mcmc_eff_size(x, method="mad")
    assert isinstance(ess_mad, (float, np.floating))
    assert_positive(ess_mad)

    ess_local = mcmc_eff_size(x, method="local", prob=(0.25, 0.75))
    assert isinstance(ess_local, (float, np.floating))
    assert_positive(ess_local)


def test_mcmc_eff_size_single_chain():
    """Test ESS calculation with single chain input."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=1000)

    methods = ["bulk", "tail", "mean", "sd", "median", "mad"]
    for method in methods:
        ess = mcmc_eff_size(x, method=method)
        assert isinstance(
            ess, (int, float, np.floating, np.integer)
        ), f"ESS return type for method {method} is {type(ess)}"
        assert_positive(ess)
        assert ess <= len(x)

    ess_local = mcmc_eff_size(x, method="local", prob=(0.25, 0.75))
    assert isinstance(ess_local, (int, float, np.floating, np.integer))
    assert_positive(ess_local)
    assert ess_local <= len(x)


def test_mcmc_eff_size_validation():
    """Test input validation and error handling for ESS calculation."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(100, 4))  # 100 iterations, 4 chains

    with pytest.raises(ValueError, match="Unknown method"):
        mcmc_eff_size(x, method="invalid")

    with pytest.raises(ValueError, match="local method requires prob"):
        mcmc_eff_size(x, method="local")

    with pytest.raises(ValueError, match="local method requires prob"):
        mcmc_eff_size(x, method="local", prob=0.5)  # should be tuple

    with pytest.raises(ValueError):
        mcmc_eff_size(x.reshape(4, 25, 4))  # 3D input not supported

    with pytest.warns(RuntimeWarning, match="Input contains NaN values"):
        x_nan = np.full((100, 4), np.nan)
        ess = mcmc_eff_size(x_nan)
        assert np.isnan(ess)

    with pytest.warns(RuntimeWarning, match="Input contains infinite values"):
        x_inf = np.full((100, 4), np.inf)
        ess = mcmc_eff_size(x_inf)
        assert np.isnan(ess)


def test_mat_to_chains(centered_eight):
    """Test conversion from matrix to 3D array organized by chains using real data."""
    mat = centered_eight.posterior.theta.values.reshape(-1, 8)
    chain_id = np.repeat(
        np.arange(1, centered_eight.posterior.chain.size + 1),
        centered_eight.posterior.draw.size,
    )
    arr = _mat_to_chains(mat, chain_id)

    expected_shape = (
        centered_eight.posterior.draw.size,
        centered_eight.posterior.chain.size,
        8,
    )
    assert arr.shape == expected_shape

    for i in range(centered_eight.posterior.chain.size):
        start_idx = i * centered_eight.posterior.draw.size
        end_idx = (i + 1) * centered_eight.posterior.draw.size
        assert_arrays_allclose(arr[:, i, :], mat[start_idx:end_idx])

    bad_chain_id = np.array([1] * 600 + [2] * 400)
    with pytest.raises(ValueError):
        _mat_to_chains(mat, bad_chain_id)

    with pytest.raises(ValueError):
        _mat_to_chains(mat, chain_id[:-1])


def test_input_validation(multidim_data, rng):
    """Test input validation and error messages."""
    x_4d = multidim_data["llm"]  # shape: (4, 23, 15, 2)
    with pytest.raises(ValueError):
        rel_eff(x_4d)

    x_2d = multidim_data["ll1"].reshape(-1, 30)  # shape: (92, 30)
    with pytest.raises(ValueError):
        rel_eff(x_2d)

    def dummy_fun(data_i, draws):
        return rng.normal(size=(1000, 1))

    with pytest.raises(ValueError):
        rel_eff(dummy_fun, chain_id=np.repeat([1, 2], 500))

    psis_data = generate_psis_data(rng, n_samples=1000, n_obs=5)
    w = np.exp(psis_data["log_ratios"] - psis_data["log_ratios"].max(axis=0))
    w = w / w.sum(axis=0)

    with pytest.raises(ValueError):
        psis_eff_size(w, r_eff=np.ones(3))


def test_rel_eff_methods(centered_eight):
    """Test rel_eff with different ESS calculation methods."""
    x = centered_eight.posterior.theta.values.reshape(-1, 8)
    chain_id = np.repeat(
        np.arange(1, centered_eight.posterior.chain.size + 1),
        centered_eight.posterior.draw.size,
    )

    methods = ["bulk", "tail", "mean", "sd", "median", "mad"]
    for method in methods:
        r_eff = rel_eff(x, chain_id, method=method)
        assert isinstance(r_eff, np.ndarray)
        assert r_eff.shape == (8,)
        assert_bounded(r_eff, lower=0, upper=1)

    r_eff_local = rel_eff(x, chain_id, method="local", prob=(0.25, 0.75))
    assert isinstance(r_eff_local, np.ndarray)
    assert r_eff_local.shape == (8,)
    assert_bounded(r_eff_local, lower=0, upper=1)

    r_eff_tail = rel_eff(x, chain_id, method="tail", prob=0.05)
    assert isinstance(r_eff_tail, np.ndarray)
    assert r_eff_tail.shape == (8,)
    assert_bounded(r_eff_tail, lower=0, upper=1)
