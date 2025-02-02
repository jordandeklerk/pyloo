"""Tests for effective sample size calculations."""

import numpy as np
import pytest

from ...effective_sample_sizes import (
    _autocovariance,
    _convert_matrix_to_chains,
    _fft_next_good_size,
    compute_mcmc_effective_size,
    compute_psis_effective_size,
    compute_relative_efficiency,
)


def test_compute_relative_efficiency_1d():
    """Test compute_relative_efficiency with 1D input."""
    x = np.random.normal(size=1000)
    chain_id = np.repeat([1, 2], 500)
    r_eff = compute_relative_efficiency(x, chain_id)
    assert isinstance(r_eff, np.ndarray)
    assert r_eff.shape == (1,)
    assert 0 < r_eff[0] <= 1


def test_compute_relative_efficiency_2d():
    """Test compute_relative_efficiency with 2D input."""
    x = np.random.normal(size=(1000, 5))
    chain_id = np.repeat([1, 2], 500)
    r_eff = compute_relative_efficiency(x, chain_id)
    assert isinstance(r_eff, np.ndarray)
    assert r_eff.shape == (5,)
    assert np.all((0 < r_eff) & (r_eff <= 1))


def test_compute_relative_efficiency_3d():
    """Test compute_relative_efficiency with 3D input."""
    x = np.random.normal(size=(500, 2, 5))
    r_eff = compute_relative_efficiency(x)
    assert isinstance(r_eff, np.ndarray)
    assert r_eff.shape == (5,)
    assert np.all((0 < r_eff) & (r_eff <= 1))


def test_compute_relative_efficiency_function():
    """Test compute_relative_efficiency with function input."""
    data = np.random.normal(size=(10, 2))
    draws = np.random.normal(size=(1000, 3))
    chain_id = np.repeat([1, 2], 500)

    def log_lik_fun(data_i, draws):
        return np.exp(np.random.normal(size=(len(draws), 1)))

    r_eff = compute_relative_efficiency(log_lik_fun, chain_id, data=data, draws=draws)
    assert isinstance(r_eff, np.ndarray)
    assert r_eff.shape == (10,)
    assert np.all((0 < r_eff) & (r_eff <= 1))


def test_compute_relative_efficiency_parallel():
    """Test compute_relative_efficiency with parallel processing."""
    x = np.random.normal(size=(500, 2, 5))
    r_eff_single = compute_relative_efficiency(x, cores=1)
    r_eff_parallel = compute_relative_efficiency(x, cores=2)
    np.testing.assert_allclose(r_eff_single, r_eff_parallel, rtol=1e-10)


def test_compute_psis_effective_size_1d():
    """Test compute_psis_effective_size with 1D input."""
    w = np.random.dirichlet(np.ones(1000))
    n_eff = compute_psis_effective_size(w)
    assert isinstance(n_eff, float)
    assert 0 < n_eff <= 1000

    r_eff = 0.5
    n_eff_adj = compute_psis_effective_size(w, r_eff)
    assert isinstance(n_eff_adj, float)
    assert n_eff_adj == n_eff * r_eff


def test_compute_psis_effective_size_2d():
    """Test compute_psis_effective_size with 2D input."""
    w = np.random.dirichlet(np.ones(1000), size=5).T
    n_eff = compute_psis_effective_size(w)
    assert isinstance(n_eff, np.ndarray)
    assert n_eff.shape == (5,)
    assert np.all((0 < n_eff) & (n_eff <= 1000))

    r_eff = np.full(5, 0.5)
    n_eff_adj = compute_psis_effective_size(w, r_eff)
    np.testing.assert_allclose(n_eff_adj, n_eff * r_eff)


def test_compute_mcmc_effective_size():
    """Test compute_mcmc_effective_size with various inputs."""
    x = np.random.normal(size=(1000, 2))
    ess = compute_mcmc_effective_size(x)
    assert isinstance(ess, float)
    assert 0 < ess <= 2000

    rho = 0.9
    n = 1000
    x_corr = np.empty((n, 2))
    x_corr[0] = np.random.normal(size=2)
    for i in range(1, n):
        x_corr[i] = rho * x_corr[i - 1] + np.sqrt(1 - rho**2) * np.random.normal(size=2)

    ess_corr = compute_mcmc_effective_size(x_corr)
    assert ess_corr < ess


def test_autocovariance():
    """Test autocovariance calculation."""
    x = np.random.normal(size=1000)
    acf = _autocovariance(x)
    assert isinstance(acf, np.ndarray)
    assert len(acf) == len(x)
    assert np.abs(acf[0] - 1.0) < 1e-10
    assert np.all(np.abs(acf[1:]) < 0.2)

    rho = 0.9
    x_ar = np.empty(1000)
    x_ar[0] = np.random.normal()
    for i in range(1, 1000):
        x_ar[i] = rho * x_ar[i - 1] + np.sqrt(1 - rho**2) * np.random.normal()

    acf_ar = _autocovariance(x_ar)
    assert np.abs(acf_ar[0] - 1.0) < 1e-10
    assert acf_ar[1] > 0.5


def test_fft_next_good_size():
    """Test FFT size optimization."""
    assert _fft_next_good_size(1) == 2
    assert _fft_next_good_size(2) == 2
    assert _fft_next_good_size(3) == 3
    assert _fft_next_good_size(4) == 4
    assert _fft_next_good_size(5) == 5
    assert _fft_next_good_size(7) == 8
    assert _fft_next_good_size(9) == 9
    assert _fft_next_good_size(11) == 12


def test_convert_matrix_to_chains():
    """Test conversion from matrix to 3D array organized by chains."""
    mat = np.random.normal(size=(1000, 5))
    chain_id = np.repeat([1, 2], 500)
    arr = _convert_matrix_to_chains(mat, chain_id)

    assert arr.shape == (500, 2, 5)
    np.testing.assert_array_equal(arr[:, 0, :], mat[:500])
    np.testing.assert_array_equal(arr[:, 1, :], mat[500:])

    bad_chain_id = np.array([1] * 600 + [2] * 400)
    with pytest.raises(ValueError):
        _convert_matrix_to_chains(mat, bad_chain_id)

    with pytest.raises(ValueError):
        _convert_matrix_to_chains(mat, chain_id[:-1])


def test_input_validation():
    """Test input validation and error messages."""
    x_4d = np.random.normal(size=(10, 10, 10, 10))
    with pytest.raises(ValueError):
        compute_relative_efficiency(x_4d)

    x_2d = np.random.normal(size=(1000, 5))
    with pytest.raises(ValueError):
        compute_relative_efficiency(x_2d)

    def dummy_fun(data_i, draws):
        return np.random.normal(size=(1000, 1))

    with pytest.raises(ValueError):
        compute_relative_efficiency(dummy_fun, chain_id=np.repeat([1, 2], 500))

    w = np.random.dirichlet(np.ones(1000), size=5).T
    with pytest.raises(ValueError):
        compute_psis_effective_size(w, r_eff=np.ones(3))

    with pytest.raises(ValueError):
        _autocovariance(np.array([1.0]))
