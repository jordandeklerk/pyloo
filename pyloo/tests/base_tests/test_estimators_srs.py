"""Tests for the Simple Random Sampling (SRS) estimator implementation."""

import numpy as np

from ...estimators.srs import (
    SimpleRandomSamplingEstimator,
    SRSEstimate,
    estimate_elpd_loo,
    srs_estimate,
)
from ..helpers import assert_allclose


def test_simple_random_sampling_estimator_basic():
    N = 100
    m = 10
    y = np.random.randn(m)

    estimator = SimpleRandomSamplingEstimator()
    result = estimator.estimate(y=y, N=N)

    assert isinstance(result, SRSEstimate)
    assert np.isfinite(result.y_hat)
    assert np.isfinite(result.v_y_hat)
    assert np.isfinite(result.hat_v_y)
    assert result.m == m
    assert result.N == N
    assert np.isfinite(result.subsampling_SE)

    expected_y_hat = N * np.mean(y)
    assert_allclose(result.y_hat, expected_y_hat)

    expected_v_y_hat = N**2 * (1 - m / N) * np.var(y, ddof=1) / m
    assert_allclose(result.v_y_hat, expected_v_y_hat)

    assert_allclose(result.subsampling_SE, np.sqrt(expected_v_y_hat))


def test_simple_random_sampling_estimator_single_sample():
    N = 100
    m = 1
    y = np.array([1.0])

    estimator = SimpleRandomSamplingEstimator()
    result = estimator.estimate(y=y, N=N)

    assert isinstance(result, SRSEstimate)
    assert np.isfinite(result.y_hat)
    assert np.isnan(result.v_y_hat)
    assert np.isnan(result.hat_v_y)
    assert result.m == m
    assert result.N == N
    assert np.isnan(result.subsampling_SE)

    expected_y_hat = N * np.mean(y)
    assert_allclose(result.y_hat, expected_y_hat)


def test_simple_random_sampling_estimator_full_sample():
    N = 10
    m = N
    y = np.random.randn(m)

    estimator = SimpleRandomSamplingEstimator()
    result = estimator.estimate(y=y, N=N)

    assert isinstance(result, SRSEstimate)
    assert np.isfinite(result.y_hat)
    assert_allclose(result.v_y_hat, 0.0)
    assert np.isfinite(result.hat_v_y)
    assert result.m == m
    assert result.N == N
    assert_allclose(result.subsampling_SE, 0.0)

    expected_y_hat = N * np.mean(y)
    assert_allclose(result.y_hat, expected_y_hat)


def test_srs_estimate_function():
    N = 100
    m = 10
    y = np.random.randn(m)

    result = srs_estimate(y=y, N=N)

    assert isinstance(result, SRSEstimate)
    assert np.isfinite(result.y_hat)
    assert np.isfinite(result.v_y_hat)
    assert np.isfinite(result.hat_v_y)
    assert result.m == m
    assert result.N == N
    assert np.isfinite(result.subsampling_SE)

    estimator = SimpleRandomSamplingEstimator()
    direct_result = estimator.estimate(y=y, N=N)

    assert_allclose(result.y_hat, direct_result.y_hat)
    assert_allclose(result.v_y_hat, direct_result.v_y_hat)
    assert_allclose(result.hat_v_y, direct_result.hat_v_y)


def test_estimate_elpd_loo():
    N = 100
    m = 10
    elpd_loo_i = np.random.randn(m)

    result = estimate_elpd_loo(elpd_loo_i=elpd_loo_i, N=N)

    assert isinstance(result, SRSEstimate)
    assert np.isfinite(result.y_hat)
    assert np.isfinite(result.v_y_hat)
    assert np.isfinite(result.hat_v_y)
    assert result.m == m
    assert result.N == N
    assert np.isfinite(result.subsampling_SE)

    direct_result = srs_estimate(y=elpd_loo_i, N=N)

    assert_allclose(result.y_hat, direct_result.y_hat)
    assert_allclose(result.v_y_hat, direct_result.v_y_hat)
    assert_allclose(result.hat_v_y, direct_result.hat_v_y)


def test_srs_with_extreme_values():
    N = 100
    m = 10

    y = np.random.randn(m)
    y[0] = 1e10
    y[1] = -1e10

    estimator = SimpleRandomSamplingEstimator()
    result = estimator.estimate(y=y, N=N)

    assert isinstance(result, SRSEstimate)
    assert np.isfinite(result.y_hat)
    assert np.isfinite(result.v_y_hat)
    assert np.isfinite(result.hat_v_y)

    expected_y_hat = N * np.mean(y)
    assert_allclose(result.y_hat, expected_y_hat)


def test_srs_with_constant_values():
    N = 100
    m = 10

    y = np.ones(m) * 5.0

    estimator = SimpleRandomSamplingEstimator()
    result = estimator.estimate(y=y, N=N)

    assert isinstance(result, SRSEstimate)
    assert np.isfinite(result.y_hat)
    assert_allclose(result.v_y_hat, 0.0)
    assert_allclose(result.hat_v_y, 0.0)
    assert result.m == m
    assert result.N == N
    assert_allclose(result.subsampling_SE, 0.0)

    expected_y_hat = N * 5.0
    assert_allclose(result.y_hat, expected_y_hat)


def test_srs_with_different_sample_sizes():
    N = 1000
    sample_sizes = [10, 50, 100, 500]

    population = np.random.randn(N)

    for m in sample_sizes:
        indices = np.random.choice(N, size=m, replace=False)
        y = population[indices]

        estimator = SimpleRandomSamplingEstimator()
        result = estimator.estimate(y=y, N=N)

        assert isinstance(result, SRSEstimate)
        assert np.isfinite(result.y_hat)
        assert np.isfinite(result.v_y_hat)
        assert np.isfinite(result.hat_v_y)
        assert result.m == m
        assert result.N == N
        assert np.isfinite(result.subsampling_SE)

        expected_y_hat = N * np.mean(y)
        assert_allclose(result.y_hat, expected_y_hat)

        if m > 1:
            expected_v_y_hat = N**2 * (1 - m / N) * np.var(y, ddof=1) / m
            assert_allclose(result.v_y_hat, expected_v_y_hat)
