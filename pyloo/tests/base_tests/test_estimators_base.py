"""Tests for the base functionality of LOO-CV subsampling estimators."""

import numpy as np
import pytest

from ...estimators.base import (
    BaseEstimate,
    SubsampleIndices,
    compare_indices,
    subsample_indices,
)


def test_base_estimate():
    """Test the BaseEstimate dataclass."""
    estimate = BaseEstimate(
        y_hat=1.0,
        v_y_hat=0.1,
        hat_v_y=0.2,
        m=10,
        subsampling_SE=0.3,
        N=100,
    )

    assert estimate.y_hat == 1.0
    assert estimate.v_y_hat == 0.1
    assert estimate.hat_v_y == 0.2
    assert estimate.m == 10
    assert estimate.subsampling_SE == 0.3
    assert estimate.N == 100


def test_subsample_indices():
    """Test the subsample_indices function."""
    n_obs = 100
    elpd_loo_approximation = np.random.randn(n_obs)
    observations = 10

    np.random.seed(42)

    indices_diff_srs = subsample_indices(
        "diff_srs", elpd_loo_approximation, observations
    )
    assert isinstance(indices_diff_srs, SubsampleIndices)
    assert len(indices_diff_srs.idx) == observations
    assert np.all(indices_diff_srs.m_i == 1)
    assert np.all(np.array(indices_diff_srs.idx) < n_obs)

    indices_srs = subsample_indices("srs", elpd_loo_approximation, observations)
    assert isinstance(indices_srs, SubsampleIndices)
    assert len(indices_srs.idx) == observations
    assert np.all(indices_srs.m_i == 1)
    assert np.all(np.array(indices_srs.idx) < n_obs)

    indices_hh = subsample_indices("hh_pps", elpd_loo_approximation, observations)
    assert isinstance(indices_hh, SubsampleIndices)
    assert len(indices_hh.idx) <= observations
    assert np.sum(indices_hh.m_i) == observations
    assert np.all(indices_hh.idx < n_obs)

    with pytest.raises(ValueError, match="Unknown estimator"):
        subsample_indices("invalid", elpd_loo_approximation, observations)

    with pytest.raises(
        ValueError, match="Number of observations cannot exceed total sample size"
    ):
        subsample_indices("srs", elpd_loo_approximation, n_obs + 1)


def test_subsample_indices_reproducibility():
    """Test that subsample_indices is reproducible with fixed seed."""
    n_obs = 100
    elpd_loo_approximation = np.random.randn(n_obs)
    observations = 10

    np.random.seed(42)
    indices1 = subsample_indices("diff_srs", elpd_loo_approximation, observations)

    np.random.seed(42)
    indices2 = subsample_indices("diff_srs", elpd_loo_approximation, observations)

    np.testing.assert_array_equal(indices1.idx, indices2.idx)
    np.testing.assert_array_equal(indices1.m_i, indices2.m_i)

    np.random.seed(43)
    indices3 = subsample_indices("diff_srs", elpd_loo_approximation, observations)
    assert not np.array_equal(indices1.idx, indices3.idx)


def test_compare_indices():
    """Test the compare_indices function."""
    current_indices = SubsampleIndices(
        idx=np.array([0, 1, 2, 3, 4]),
        m_i=np.array([1, 1, 1, 1, 1]),
    )

    new_indices = SubsampleIndices(
        idx=np.array([2, 3, 4, 5, 6]),
        m_i=np.array([1, 1, 1, 1, 1]),
    )

    result = compare_indices(new_indices, current_indices)

    assert "new" in result
    assert "add" in result
    assert "remove" in result

    np.testing.assert_array_equal(result["new"].idx, np.array([5, 6]))
    np.testing.assert_array_equal(result["add"].idx, np.array([2, 3, 4]))
    np.testing.assert_array_equal(result["remove"].idx, np.array([0, 1]))

    np.testing.assert_array_equal(result["new"].m_i, np.array([1, 1]))
    np.testing.assert_array_equal(result["add"].m_i, np.array([1, 1, 1]))
    np.testing.assert_array_equal(result["remove"].m_i, np.array([1, 1]))


def test_compare_indices_with_counts():
    """Test compare_indices with non-uniform counts."""
    current_indices = SubsampleIndices(
        idx=np.array([0, 1, 2, 3]),
        m_i=np.array([2, 1, 3, 1]),
    )

    new_indices = SubsampleIndices(
        idx=np.array([1, 2, 3, 4]),
        m_i=np.array([2, 1, 2, 3]),
    )

    result = compare_indices(new_indices, current_indices)

    np.testing.assert_array_equal(result["new"].idx, np.array([4]))
    np.testing.assert_array_equal(result["add"].idx, np.array([1, 2, 3]))
    np.testing.assert_array_equal(result["remove"].idx, np.array([0]))

    np.testing.assert_array_equal(result["new"].m_i, np.array([3]))
    np.testing.assert_array_equal(result["add"].m_i, np.array([2, 1, 2]))
    np.testing.assert_array_equal(result["remove"].m_i, np.array([2]))


def test_compare_indices_edge_cases():
    """Test compare_indices with edge cases."""
    current_indices = SubsampleIndices(
        idx=np.array([0, 1, 2]),
        m_i=np.array([1, 1, 1]),
    )

    new_indices = SubsampleIndices(
        idx=np.array([3, 4, 5]),
        m_i=np.array([1, 1, 1]),
    )

    result = compare_indices(new_indices, current_indices)

    assert "new" in result
    assert "remove" in result
    assert "add" not in result

    np.testing.assert_array_equal(result["new"].idx, np.array([3, 4, 5]))
    np.testing.assert_array_equal(result["remove"].idx, np.array([0, 1, 2]))

    current_indices = SubsampleIndices(
        idx=np.array([0, 1, 2]),
        m_i=np.array([1, 1, 1]),
    )

    new_indices = SubsampleIndices(
        idx=np.array([0, 1, 2]),
        m_i=np.array([1, 1, 1]),
    )

    result = compare_indices(new_indices, current_indices)

    assert "add" in result
    assert "new" not in result
    assert "remove" not in result

    np.testing.assert_array_equal(result["add"].idx, np.array([0, 1, 2]))

    current_indices = SubsampleIndices(
        idx=np.array([]),
        m_i=np.array([]),
    )

    new_indices = SubsampleIndices(
        idx=np.array([0, 1, 2]),
        m_i=np.array([1, 1, 1]),
    )

    result = compare_indices(new_indices, current_indices)

    assert "new" in result
    assert "add" not in result
    assert "remove" not in result

    np.testing.assert_array_equal(result["new"].idx, np.array([0, 1, 2]))

    current_indices = SubsampleIndices(
        idx=np.array([0, 1, 2]),
        m_i=np.array([1, 1, 1]),
    )

    new_indices = SubsampleIndices(
        idx=np.array([]),
        m_i=np.array([]),
    )

    result = compare_indices(new_indices, current_indices)

    assert "remove" in result
    assert "new" not in result
    assert "add" not in result

    np.testing.assert_array_equal(result["remove"].idx, np.array([0, 1, 2]))
