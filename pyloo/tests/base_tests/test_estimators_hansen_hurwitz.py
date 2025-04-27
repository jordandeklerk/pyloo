"""Tests for the Hansen-Hurwitz estimator implementation."""

import numpy as np
import pytest

from ...estimators.hansen_hurwitz import (
    HansenHurwitzEstimator,
    HHEstimate,
    compute_sampling_probabilities,
    estimate_elpd_loo,
    hansen_hurwitz_estimate,
)
from ..helpers import assert_allclose


def test_hansen_hurwitz_estimator_basic():
    N = 100
    m = 10
    z = np.random.rand(m)
    z = z / np.sum(z)
    m_i = np.ones(m, dtype=int)
    y = np.random.randn(m)

    estimator = HansenHurwitzEstimator()
    result = estimator.estimate(z=z, m_i=m_i, y=y, N=N)

    assert isinstance(result, HHEstimate)
    assert np.isfinite(result.y_hat)
    assert np.isfinite(result.v_y_hat)
    assert np.isfinite(result.hat_v_y)
    assert result.m == m
    assert result.N == N
    assert np.isfinite(result.subsampling_SE)

    expected_y_hat = np.sum(m_i * (y / z)) / np.sum(m_i)
    assert_allclose(result.y_hat, expected_y_hat)


def test_hansen_hurwitz_estimator_validation():
    N = 100
    m = 10
    z = np.random.rand(m)
    z = z / np.sum(z)
    m_i = np.ones(m, dtype=int)
    y = np.random.randn(m)

    estimator = HansenHurwitzEstimator()

    z_invalid = z.copy()
    z_invalid[0] = 0
    with pytest.raises(ValueError, match="All probabilities .* must be positive"):
        estimator.estimate(z=z_invalid, m_i=m_i, y=y, N=N)

    m_i_invalid = m_i.copy()
    m_i_invalid[0] = 0
    with pytest.raises(ValueError, match="All sample counts .* must be positive"):
        estimator.estimate(z=z, m_i=m_i_invalid, y=y, N=N)

    with pytest.raises(ValueError, match="All input arrays must have same length"):
        estimator.estimate(z=z[:-1], m_i=m_i, y=y, N=N)


def test_hansen_hurwitz_estimator_with_counts():
    N = 100
    m = 5
    z = np.random.rand(m)
    z = z / np.sum(z)
    m_i = np.array([3, 1, 2, 1, 3])
    y = np.random.randn(m)

    estimator = HansenHurwitzEstimator()
    result = estimator.estimate(z=z, m_i=m_i, y=y, N=N)

    assert isinstance(result, HHEstimate)
    assert np.isfinite(result.y_hat)
    assert np.isfinite(result.v_y_hat)
    assert np.isfinite(result.hat_v_y)
    assert result.m == np.sum(m_i)
    assert result.N == N
    assert np.isfinite(result.subsampling_SE)

    expected_y_hat = np.sum(m_i * (y / z)) / np.sum(m_i)
    assert_allclose(result.y_hat, expected_y_hat)


def test_compute_sampling_probabilities():
    elpd_loo_approximation = np.array([-1.0, -2.0, -3.0, -4.0])
    probs = compute_sampling_probabilities(elpd_loo_approximation)

    assert len(probs) == len(elpd_loo_approximation)
    assert np.all(probs > 0)
    assert_allclose(np.sum(probs), 1.0)

    expected = np.abs(elpd_loo_approximation)
    expected = expected / np.sum(expected)
    assert_allclose(probs, expected)

    elpd_loo_approximation = np.array([0.0, 0.0, 0.0, 0.0])
    probs = compute_sampling_probabilities(elpd_loo_approximation)

    assert len(probs) == len(elpd_loo_approximation)
    assert np.all(probs > 0)
    assert_allclose(np.sum(probs), 1.0)

    expected = np.ones_like(elpd_loo_approximation) / len(elpd_loo_approximation)
    assert_allclose(probs, expected)

    elpd_loo_approximation = np.array([-1.0, -2.0, -3.0, -4.0])
    probs = compute_sampling_probabilities(elpd_loo_approximation)

    assert len(probs) == len(elpd_loo_approximation)
    assert np.all(probs > 0)
    assert_allclose(np.sum(probs), 1.0)

    expected = np.abs(elpd_loo_approximation)
    expected = expected / np.sum(expected)
    assert_allclose(probs, expected)


def test_hansen_hurwitz_estimate_function():
    N = 100
    m = 10
    z = np.random.rand(m)
    z = z / np.sum(z)
    m_i = np.ones(m, dtype=int)
    y = np.random.randn(m)

    result = hansen_hurwitz_estimate(z=z, m_i=m_i, y=y, N=N)

    assert isinstance(result, HHEstimate)
    assert np.isfinite(result.y_hat)
    assert np.isfinite(result.v_y_hat)
    assert np.isfinite(result.hat_v_y)
    assert result.m == m
    assert result.N == N
    assert np.isfinite(result.subsampling_SE)

    estimator = HansenHurwitzEstimator()
    direct_result = estimator.estimate(z=z, m_i=m_i, y=y, N=N)

    assert_allclose(result.y_hat, direct_result.y_hat)
    assert_allclose(result.v_y_hat, direct_result.v_y_hat)
    assert_allclose(result.hat_v_y, direct_result.hat_v_y)


def test_estimate_elpd_loo():
    N = 100
    m = 10
    elpd_loo_approximation = np.random.randn(N)
    sample_indices = np.sort(np.random.choice(N, size=m, replace=False))
    elpd_loo_i = elpd_loo_approximation[sample_indices] + np.random.randn(m) * 0.1
    m_i = np.ones(m, dtype=int)

    result = estimate_elpd_loo(
        elpd_loo_i=elpd_loo_i,
        elpd_loo_approximation=elpd_loo_approximation,
        sample_indices=sample_indices,
        m_i=m_i,
        N=N,
    )

    assert isinstance(result, HHEstimate)
    assert np.isfinite(result.y_hat)
    assert np.isfinite(result.v_y_hat)
    assert np.isfinite(result.hat_v_y)
    assert result.m == m
    assert result.N == N
    assert np.isfinite(result.subsampling_SE)

    z = compute_sampling_probabilities(elpd_loo_approximation)
    z_sample = z[sample_indices]

    direct_result = hansen_hurwitz_estimate(z=z_sample, m_i=m_i, y=elpd_loo_i, N=N)

    assert_allclose(result.y_hat, direct_result.y_hat)
    assert_allclose(result.v_y_hat, direct_result.v_y_hat)
    assert_allclose(result.hat_v_y, direct_result.hat_v_y)


def test_estimate_elpd_loo_with_non_uniform_counts():
    N = 100
    elpd_loo_approximation = np.random.randn(N)

    raw_indices = np.random.choice(
        N,
        size=10,
        replace=True,
        p=np.abs(elpd_loo_approximation) / np.sum(np.abs(elpd_loo_approximation)),
    )
    sample_indices, m_i = np.unique(raw_indices, return_counts=True)

    elpd_loo_i = (
        elpd_loo_approximation[sample_indices]
        + np.random.randn(len(sample_indices)) * 0.1
    )

    result = estimate_elpd_loo(
        elpd_loo_i=elpd_loo_i,
        elpd_loo_approximation=elpd_loo_approximation,
        sample_indices=sample_indices,
        m_i=m_i,
        N=N,
    )

    assert isinstance(result, HHEstimate)
    assert np.isfinite(result.y_hat)
    assert np.isfinite(result.v_y_hat)
    assert np.isfinite(result.hat_v_y)
    assert result.m == np.sum(m_i)
    assert result.N == N
    assert np.isfinite(result.subsampling_SE)
