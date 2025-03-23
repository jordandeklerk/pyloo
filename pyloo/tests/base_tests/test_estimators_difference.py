"""Tests for the difference estimator implementation."""

import numpy as np
import pytest

from ...estimators.difference import (
    DifferenceEstimator,
    DiffEstimate,
    diff_srs_estimate,
)
from ..helpers import assert_allclose


def test_difference_estimator_basic():
    """Test basic functionality of the difference estimator."""
    N = 100
    m = 10
    y_approx = np.random.randn(N)
    y_idx = np.sort(np.random.choice(N, size=m, replace=False))
    y = y_approx[y_idx] + np.random.randn(m) * 0.1

    estimator = DifferenceEstimator()
    result = estimator.estimate(y_approx=y_approx, y=y, y_idx=y_idx)

    assert isinstance(result, DiffEstimate)
    assert np.isfinite(result.y_hat)
    assert np.isfinite(result.v_y_hat)
    assert np.isfinite(result.hat_v_y)
    assert result.m == m
    assert result.N == N
    assert np.isfinite(result.subsampling_SE)

    t_pi_tilde = np.sum(y_approx)
    e_i = y - y_approx[y_idx]
    t_e = N * np.mean(e_i)
    expected_y_hat = t_pi_tilde + t_e

    assert_allclose(result.y_hat, expected_y_hat)


def test_difference_estimator_validation():
    """Test input validation in the difference estimator."""
    N = 100
    m = 10
    y_approx = np.random.randn(N)
    y_idx = np.sort(np.random.choice(N, size=m, replace=False))
    y = y_approx[y_idx] + np.random.randn(m) * 0.1

    estimator = DifferenceEstimator()

    with pytest.raises(ValueError, match="y and y_idx must have same length"):
        estimator.estimate(y_approx=y_approx, y=y[:-1], y_idx=y_idx)

    invalid_y_idx = np.array([0, N])
    invalid_y = np.random.randn(len(invalid_y_idx))

    with pytest.raises(ValueError, match="y_idx contains invalid indices"):
        estimator.estimate(y_approx=y_approx, y=invalid_y, y_idx=invalid_y_idx)


def test_difference_estimator_multidimensional():
    """Test the difference estimator with multidimensional data."""
    N = 5
    dim = 2

    y_approx = np.ones((N, dim))
    for i in range(N):
        for j in range(dim):
            y_approx[i, j] = i + j * 0.1

    y_idx = np.array([1, 3])

    y = y_approx[y_idx].copy()
    y[0, 0] += 0.5
    y[1, 1] -= 0.2

    estimator = DifferenceEstimator()
    result = estimator.estimate(y_approx=y_approx, y=y, y_idx=y_idx)

    assert isinstance(result, DiffEstimate)
    assert np.isfinite(result.y_hat)
    assert np.isfinite(result.v_y_hat)
    assert np.isfinite(result.hat_v_y)

    result2 = estimator.estimate(y_approx=y_approx, y=y, y_idx=y_idx)
    assert_allclose(result.y_hat, result2.y_hat, rtol=1e-10)

    y_modified = y.copy()
    y_modified[0, 0] += 1.0

    result_modified = estimator.estimate(y_approx=y_approx, y=y_modified, y_idx=y_idx)
    assert result_modified.y_hat != result.y_hat

    y_approx_1d = y_approx.mean(axis=1)
    y_1d = y.mean(axis=1)

    result_1d = estimator.estimate(y_approx=y_approx_1d, y=y_1d, y_idx=y_idx)

    assert_allclose(result.y_hat, result_1d.y_hat, rtol=1e-10)


def test_difference_estimator_single_sample():
    """Test the difference estimator with a single sample."""
    N = 100
    m = 1
    y_approx = np.random.randn(N)
    y_idx = np.array([0])
    y = np.array([y_approx[0] + 0.1])

    estimator = DifferenceEstimator()
    result = estimator.estimate(y_approx=y_approx, y=y, y_idx=y_idx)

    assert isinstance(result, DiffEstimate)
    assert np.isfinite(result.y_hat)
    assert result.v_y_hat == np.inf
    assert result.hat_v_y == np.inf
    assert result.m == m
    assert result.N == N
    assert result.subsampling_SE == np.inf


def test_diff_srs_estimate():
    """Test the diff_srs_estimate convenience function."""
    N = 100
    m = 10
    elpd_loo_approximation = np.random.randn(N)
    sample_indices = np.sort(np.random.choice(N, size=m, replace=False))
    elpd_loo_i = elpd_loo_approximation[sample_indices] + np.random.randn(m) * 0.1

    result = diff_srs_estimate(
        elpd_loo_i=elpd_loo_i,
        elpd_loo_approximation=elpd_loo_approximation,
        sample_indices=sample_indices,
    )

    assert isinstance(result, DiffEstimate)
    assert np.isfinite(result.y_hat)
    assert np.isfinite(result.v_y_hat)
    assert np.isfinite(result.hat_v_y)
    assert result.m == m
    assert result.N == N
    assert np.isfinite(result.subsampling_SE)

    estimator = DifferenceEstimator()
    direct_result = estimator.estimate(
        y_approx=elpd_loo_approximation, y=elpd_loo_i, y_idx=sample_indices
    )

    assert_allclose(result.y_hat, direct_result.y_hat)
    assert_allclose(result.v_y_hat, direct_result.v_y_hat)
    assert_allclose(result.hat_v_y, direct_result.hat_v_y)
