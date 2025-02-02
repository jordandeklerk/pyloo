"""Tests for effective sample size calculations."""

import numpy as np
import pytest

from ...effective_sample_sizes import (
    _autocovariance,
    _fft_next_good_size,
    _llmatrix_to_array,
    ess_rfun,
    psis_n_eff,
    relative_eff,
)


def test_relative_eff_1d():
    """Test relative_eff with 1D input."""
    x = np.random.normal(size=1000)
    chain_id = np.repeat([1, 2], 500)
    r_eff = relative_eff(x, chain_id)
    assert isinstance(r_eff, np.ndarray)
    assert r_eff.shape == (1,)
    assert 0 < r_eff[0] <= 1


def test_relative_eff_2d():
    """Test relative_eff with 2D input."""
    x = np.random.normal(size=(1000, 5))
    chain_id = np.repeat([1, 2], 500)
    r_eff = relative_eff(x, chain_id)
    assert isinstance(r_eff, np.ndarray)
    assert r_eff.shape == (5,)
    assert np.all((0 < r_eff) & (r_eff <= 1))


def test_relative_eff_3d():
    """Test relative_eff with 3D input."""
    x = np.random.normal(size=(500, 2, 5))
    r_eff = relative_eff(x)
    assert isinstance(r_eff, np.ndarray)
    assert r_eff.shape == (5,)
    assert np.all((0 < r_eff) & (r_eff <= 1))


def test_relative_eff_function():
    """Test relative_eff with function input."""
    data = np.random.normal(size=(10, 2))
    draws = np.random.normal(size=(1000, 3))
    chain_id = np.repeat([1, 2], 500)

    def log_lik_fun(data_i, draws):
        return np.exp(np.random.normal(size=(len(draws), 1)))

    r_eff = relative_eff(log_lik_fun, chain_id, data=data, draws=draws)
    assert isinstance(r_eff, np.ndarray)
    assert r_eff.shape == (10,)
    assert np.all((0 < r_eff) & (r_eff <= 1))


def test_relative_eff_parallel():
    """Test relative_eff with parallel processing."""
    x = np.random.normal(size=(500, 2, 5))
    r_eff_single = relative_eff(x, cores=1)
    r_eff_parallel = relative_eff(x, cores=2)
    np.testing.assert_allclose(r_eff_single, r_eff_parallel, rtol=1e-10)


def test_psis_n_eff_1d():
    """Test psis_n_eff with 1D input."""
    w = np.random.dirichlet(np.ones(1000))
    n_eff = psis_n_eff(w)
    assert isinstance(n_eff, float)
    assert 0 < n_eff <= 1000

    r_eff = 0.5
    n_eff_adj = psis_n_eff(w, r_eff)
    assert isinstance(n_eff_adj, float)
    assert n_eff_adj == n_eff * r_eff


def test_psis_n_eff_2d():
    """Test psis_n_eff with 2D input."""
    w = np.random.dirichlet(np.ones(1000), size=5).T
    n_eff = psis_n_eff(w)
    assert isinstance(n_eff, np.ndarray)
    assert n_eff.shape == (5,)
    assert np.all((0 < n_eff) & (n_eff <= 1000))

    r_eff = np.full(5, 0.5)
    n_eff_adj = psis_n_eff(w, r_eff)
    np.testing.assert_allclose(n_eff_adj, n_eff * r_eff)


def test_ess_rfun():
    """Test ess_rfun with various inputs."""
    x = np.random.normal(size=(1000, 2))
    ess = ess_rfun(x)
    assert isinstance(ess, float)
    assert 0 < ess <= 2000

    rho = 0.9
    n = 1000
    x_corr = np.empty((n, 2))
    x_corr[0] = np.random.normal(size=2)
    for i in range(1, n):
        x_corr[i] = rho * x_corr[i - 1] + np.sqrt(1 - rho**2) * np.random.normal(size=2)

    ess_corr = ess_rfun(x_corr)
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


def test_llmatrix_to_array():
    """Test conversion from matrix to 3D array."""
    mat = np.random.normal(size=(1000, 5))
    chain_id = np.repeat([1, 2], 500)
    arr = _llmatrix_to_array(mat, chain_id)

    assert arr.shape == (500, 2, 5)
    np.testing.assert_array_equal(arr[:, 0, :], mat[:500])
    np.testing.assert_array_equal(arr[:, 1, :], mat[500:])

    bad_chain_id = np.array([1] * 600 + [2] * 400)
    with pytest.raises(ValueError):
        _llmatrix_to_array(mat, bad_chain_id)

    with pytest.raises(ValueError):
        _llmatrix_to_array(mat, chain_id[:-1])


def test_input_validation():
    """Test input validation and error messages."""
    x_4d = np.random.normal(size=(10, 10, 10, 10))
    with pytest.raises(ValueError):
        relative_eff(x_4d)

    x_2d = np.random.normal(size=(1000, 5))
    with pytest.raises(ValueError):
        relative_eff(x_2d)

    def dummy_fun(data_i, draws):
        return np.random.normal(size=(1000, 1))

    with pytest.raises(ValueError):
        relative_eff(dummy_fun, chain_id=np.repeat([1, 2], 500))

    w = np.random.dirichlet(np.ones(1000), size=5).T
    with pytest.raises(ValueError):
        psis_n_eff(w, r_eff=np.ones(3))

    with pytest.raises(ValueError):
        _autocovariance(np.array([1.0]))
