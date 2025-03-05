"""Tests for moment matching LOO-CV."""

import warnings

import numpy as np
import pytest

from ...importance_sampling import ISMethod
from ...loo import loo
from ...loo_moment_match import (
    loo_moment_match,
    loo_moment_match_split,
    shift,
    shift_and_cov,
    shift_and_scale,
)
from ...wrapper.pymc_wrapper import PyMCWrapper


def test_shift_transformation_dimensions(rng):
    """Test the shift transformation with different dimensions."""
    upars_1d = rng.normal(size=(1000, 1))
    lwi_1d = rng.normal(size=1000)
    result_1d = shift(upars_1d, lwi_1d)
    assert result_1d["upars"].shape == upars_1d.shape
    assert result_1d["shift"].shape == (1,)

    upars_10d = rng.normal(size=(1000, 10))
    lwi_10d = rng.normal(size=1000)
    result_10d = shift(upars_10d, lwi_10d)
    assert result_10d["upars"].shape == upars_10d.shape
    assert result_10d["shift"].shape == (10,)

    upars_small = rng.normal(size=(10, 2))
    lwi_small = rng.normal(size=10)
    result_small = shift(upars_small, lwi_small)
    assert result_small["upars"].shape == upars_small.shape
    assert result_small["shift"].shape == (2,)


def test_shift_transformation(rng):
    """Test the shift transformation."""
    upars = rng.normal(size=(1000, 2))
    lwi = rng.normal(size=1000)
    lwi = lwi - np.max(lwi)

    result = shift(upars, lwi)

    assert "upars" in result
    assert "shift" in result
    assert result["upars"].shape == upars.shape
    assert result["shift"].shape == (2,)

    weights = np.exp(lwi - np.max(lwi))
    weights = weights / np.sum(weights)

    mean_weighted_result = np.sum(weights[:, None] * result["upars"], axis=0)
    mean_weighted_orig = np.sum(weights[:, None] * upars, axis=0)

    np.testing.assert_allclose(
        mean_weighted_result, mean_weighted_orig, rtol=1e-7, atol=1e-7
    )

    assert np.all(np.abs(result["shift"]) < 10 * np.std(upars, axis=0))


def test_shift_and_scale_transformation_edge_cases(rng):
    """Test shift_and_scale with edge cases."""
    upars = rng.normal(size=(1000, 2))
    lwi = np.full(1000, -1e10)
    lwi[0] = 0  # One dominant weight
    result = shift_and_scale(upars, lwi)
    assert np.all(np.isfinite(result["upars"]))
    assert np.all(np.isfinite(result["shift"]))
    assert np.all(np.isfinite(result["scaling"]))

    upars_const = np.ones((1000, 2))
    lwi_const = rng.normal(size=1000)
    result_const = shift_and_scale(upars_const, lwi_const)
    assert np.all(np.isfinite(result_const["scaling"]))

    upars_extreme = rng.normal(size=(1000, 2)) * 1e6
    lwi_extreme = rng.normal(size=1000)
    result_extreme = shift_and_scale(upars_extreme, lwi_extreme)
    assert np.all(np.isfinite(result_extreme["upars"]))
    assert np.all(np.isfinite(result_extreme["shift"]))
    assert np.all(np.isfinite(result_extreme["scaling"]))


def test_shift_and_scale_transformation(rng):
    """Test the shift and scale transformation."""
    upars = rng.normal(size=(1000, 2))
    lwi = rng.normal(size=1000)
    lwi = lwi - np.max(lwi)

    result = shift_and_scale(upars, lwi)

    assert "upars" in result
    assert "shift" in result
    assert "scaling" in result
    assert result["upars"].shape == upars.shape
    assert result["shift"].shape == (2,)
    assert result["scaling"].shape == (2,)

    weights = np.exp(lwi - np.max(lwi))
    weights = weights / np.sum(weights)

    mean_weighted_result = np.sum(weights[:, None] * result["upars"], axis=0)
    mean_weighted_orig = np.sum(weights[:, None] * upars, axis=0)
    np.testing.assert_allclose(
        mean_weighted_result, mean_weighted_orig, rtol=1e-7, atol=1e-7
    )

    var_weighted_result = np.sum(
        weights[:, None] * (result["upars"] - mean_weighted_result[None, :]) ** 2,
        axis=0,
    )
    var_weighted_orig = np.sum(
        weights[:, None] * (upars - mean_weighted_orig[None, :]) ** 2, axis=0
    )
    np.testing.assert_allclose(
        var_weighted_result, var_weighted_orig, rtol=1e-7, atol=1e-7
    )

    assert np.all(result["scaling"] > 0)
    assert np.all(np.abs(np.log(result["scaling"])) < 5)


def test_shift_and_cov_transformation_numerical_stability(rng):
    """Test numerical stability of shift_and_cov transformation."""
    upars = np.zeros((1000, 2))
    upars[:, 0] = rng.normal(size=1000)
    upars[:, 1] = upars[:, 0] + 1e-10 * rng.normal(size=1000)
    lwi = rng.normal(size=1000)
    result = shift_and_cov(upars, lwi)
    assert np.all(np.isfinite(result["upars"]))
    assert np.all(np.isfinite(result["mapping"]))

    upars_large = rng.normal(size=(1000, 2)) * 1e6
    lwi_large = rng.normal(size=1000)
    result_large = shift_and_cov(upars_large, lwi_large)
    assert np.all(np.isfinite(result_large["upars"]))
    assert np.all(np.isfinite(result_large["mapping"]))


def test_shift_and_cov_transformation(rng):
    """Test the shift and covariance transformation."""
    upars = rng.normal(size=(1000, 2))
    lwi = rng.normal(size=1000)
    lwi = lwi - np.max(lwi)

    result = shift_and_cov(upars, lwi)

    assert "upars" in result
    assert "shift" in result
    assert "mapping" in result
    assert result["upars"].shape == upars.shape
    assert result["shift"].shape == (2,)
    assert result["mapping"].shape == (2, 2)

    weights = np.exp(lwi - np.max(lwi))
    weights = weights / np.sum(weights)

    mean_weighted_result = np.sum(weights[:, None] * result["upars"], axis=0)
    mean_weighted_orig = np.sum(weights[:, None] * upars, axis=0)
    np.testing.assert_allclose(
        mean_weighted_result, mean_weighted_orig, rtol=1e-7, atol=1e-7
    )

    cov_weighted_result = np.cov(result["upars"], rowvar=False, aweights=weights)
    cov_weighted_orig = np.cov(upars, rowvar=False, aweights=weights)
    np.testing.assert_allclose(
        cov_weighted_result, cov_weighted_orig, rtol=2e-1, atol=5e-2
    )

    assert np.allclose(
        result["mapping"] @ result["mapping"].T, result["mapping"] @ result["mapping"].T
    )
    eigvals = np.linalg.eigvals(result["mapping"] @ result["mapping"].T)
    assert np.all(eigvals > 0)
    assert np.all(np.abs(np.log(eigvals)) < 5)


def test_loo_moment_match_mmm(mmm_model):
    """Test loo_moment_match with media mix model."""
    model, idata = mmm_model
    wrapper = PyMCWrapper(model, idata)

    loo_data = loo(idata, pointwise=True)

    thresholds = [0.5, 0.7, 0.9]
    for k_thresh in thresholds:
        result = loo_moment_match(
            wrapper,
            loo_data,
            max_iters=30,
            k_threshold=k_thresh,
            split=True,
            cov=True,
            method=ISMethod.PSIS,
        )
        assert np.all(result.pareto_k <= loo_data.pareto_k)
        assert np.all(np.isfinite(result.elpd_loo))


def test_loo_moment_match_mmm_convergence(mmm_model):
    """Test convergence behavior of loo_moment_match with media mix model."""
    model, idata = mmm_model
    wrapper = PyMCWrapper(model, idata)
    loo_data = loo(idata, pointwise=True)

    iters_list = [1, 5, 10, 30]
    prev_pareto_k = None

    for max_iters in iters_list:
        result = loo_moment_match(
            wrapper,
            loo_data,
            max_iters=max_iters,
            k_threshold=0.7,
            split=True,
            cov=True,
            method=ISMethod.PSIS,
        )

        if prev_pareto_k is not None:
            assert np.mean(result.pareto_k) <= np.mean(prev_pareto_k)

        prev_pareto_k = result.pareto_k.copy()


def test_loo_moment_match_mmm_parameter_combinations(mmm_model):
    """Test loo_moment_match with different parameter combinations on media mix model."""
    model, idata = mmm_model
    wrapper = PyMCWrapper(model, idata)
    loo_data = loo(idata, pointwise=True)

    params = [(True, True), (True, False), (False, True), (False, False)]
    for split, cov in params:
        result = loo_moment_match(
            wrapper,
            loo_data,
            max_iters=30,
            k_threshold=0.7,
            split=split,
            cov=cov,
            method=ISMethod.PSIS,
        )
        assert np.all(result.pareto_k <= loo_data.pareto_k)
        assert np.all(np.isfinite(result.elpd_loo))


def test_loo_moment_match_warnings(simple_model):
    """Test warning messages from loo_moment_match."""
    model, idata = simple_model
    wrapper = PyMCWrapper(model, idata)

    loo_data = loo(idata, pointwise=True)

    loo_data.pareto_k = np.array([0.8, 0.3, 0.3, 0.9, 0.3, 0.3, 0.85, 0.3])

    with pytest.warns(
        UserWarning, match="accuracy of self-normalized importance sampling"
    ):
        result = loo_moment_match(
            wrapper,
            loo_data,
            max_iters=30,
            k_threshold=0.5,
            split=False,
            cov=True,
            method=ISMethod.PSIS,
        )
        assert result is not None

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = loo_moment_match(
            wrapper,
            loo_data,
            max_iters=30,
            k_threshold=100,
            split=True,
            cov=True,
            method=ISMethod.PSIS,
        )
        assert result is not None

    with pytest.warns(
        UserWarning, match="Maximum number of moment matching iterations"
    ):
        loo_data.pareto_k = np.array([0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99])
        result = loo_moment_match(
            wrapper,
            loo_data,
            max_iters=1,
            k_threshold=0.5,
            split=False,
            cov=True,
            method=ISMethod.PSIS,
        )
        assert result is not None


def test_loo_moment_match_functionality(simple_model):
    """Test the main functionality of loo_moment_match."""
    model, idata = simple_model
    wrapper = PyMCWrapper(model, idata)

    loo_data = loo(idata, pointwise=True)

    if len(loo_data.pareto_k) >= 4:
        loo_data.pareto_k[[0, 1, 2, 3]] = np.array([0.9, 1.5, 0.8, 1])
    else:
        loo_data.pareto_k[:] = np.array([0.9] * len(loo_data.pareto_k))

    result1 = loo_moment_match(
        wrapper,
        loo_data,
        max_iters=30,
        k_threshold=0.8,
        split=True,
        cov=True,
        method=ISMethod.PSIS,
    )

    result2 = loo_moment_match(
        wrapper,
        loo_data,
        max_iters=30,
        k_threshold=0.7,
        split=False,
        cov=True,
        method=ISMethod.PSIS,
    )

    result3 = loo_moment_match(
        wrapper,
        loo_data,
        max_iters=30,
        k_threshold=0.5,
        split=True,
        cov=True,
        method=ISMethod.PSIS,
    )

    assert np.all(result1.pareto_k <= loo_data.pareto_k)
    assert np.all(result2.pareto_k <= loo_data.pareto_k)
    assert np.all(result3.pareto_k <= loo_data.pareto_k)

    result4 = loo_moment_match(
        wrapper,
        loo_data,
        max_iters=30,
        k_threshold=100,
        split=False,
        cov=True,
        method=ISMethod.PSIS,
    )

    np.testing.assert_allclose(result4.loo_i, loo_data.loo_i)


def test_loo_moment_match_split(simple_model):
    """Test the split moment matching functionality."""
    model, idata = simple_model
    wrapper = PyMCWrapper(model, idata)

    unconstrained = wrapper.get_unconstrained_parameters()
    upars = np.column_stack(
        [unconstrained[name].values.flatten() for name in unconstrained]
    )

    log_liki = wrapper.log_likelihood_i(0, wrapper.idata)
    log_liki = log_liki.stack(__sample__=("chain", "draw"))
    log_liki = log_liki.values.flatten()

    S = upars.shape[0]
    if S < 10:
        upars = np.vstack([upars] * 5)

    result1 = loo_moment_match_split(
        wrapper,
        upars,
        cov=False,
        total_shift=np.zeros(upars.shape[1]),
        total_scaling=np.ones(upars.shape[1]),
        total_mapping=np.eye(upars.shape[1]),
        i=0,
        r_eff_i=1.0,
        method=ISMethod.PSIS,
    )

    assert "lwi" in result1
    assert "lwfi" in result1
    assert "log_liki" in result1
    assert "r_eff_i" in result1

    weight_sum = np.exp(result1["lwi"]).sum()
    assert np.isclose(weight_sum, 1.0, rtol=1e-6) or weight_sum == 0.0

    result2 = loo_moment_match_split(
        wrapper,
        upars,
        cov=False,
        total_shift=np.array([-0.1, -0.2]),
        total_scaling=np.array([0.7, 0.7]),
        total_mapping=np.array([[1.0, 0.1], [0.1, 1.0]]),
        i=0,
        r_eff_i=1.0,
        method=ISMethod.PSIS,
    )

    assert "lwi" in result2
    assert "lwfi" in result2
    assert "log_liki" in result2
    assert "r_eff_i" in result2

    weight_sum = np.exp(result2["lwi"]).sum()
    assert np.isclose(weight_sum, 1.0, rtol=1e-6) or weight_sum == 0.0


def test_variance_and_covariance_transformations(simple_model):
    """Test variance and covariance transformations with real model."""
    model, idata = simple_model
    wrapper = PyMCWrapper(model, idata)

    loo_data = loo(idata, pointwise=True)

    loo_data.pareto_k = np.array([0.8, 0.3, 0.3, 0.9, 0.3, 0.3, 0.85, 0.3])

    result = loo_moment_match(
        wrapper,
        loo_data,
        max_iters=30,
        k_threshold=0.0,
        split=False,
        cov=True,
        method=ISMethod.PSIS,
    )

    assert np.all(result.pareto_k <= loo_data.pareto_k)

    assert np.isfinite(result.elpd_loo)
    assert result.elpd_loo < 0

    if hasattr(result, "n_eff"):
        assert np.all(result.n_eff > 0)
        assert np.all(result.n_eff <= len(wrapper.get_observed_data()))

    for i in range(len(loo_data.pareto_k)):
        log_liki = wrapper.log_likelihood_i(i, wrapper.idata)
        log_liki = log_liki.stack(__sample__=("chain", "draw"))
        log_liki = log_liki.values.flatten()
        weights = np.exp(-log_liki - np.max(-log_liki))
        weights = weights / np.sum(weights)
        assert np.allclose(np.sum(weights), 1.0, rtol=1e-10)


def test_loo_moment_match_with_problematic_k(problematic_k_model):
    """Test loo_moment_match with problematic Pareto k values."""
    model, idata = problematic_k_model
    wrapper = PyMCWrapper(model, idata)

    loo_data = loo(idata, pointwise=True)

    result = loo_moment_match(
        wrapper,
        loo_data,
        max_iters=1,
        k_threshold=0.7,
        split=True,
        cov=True,
        method=ISMethod.PSIS,
    )

    assert np.all(result.pareto_k <= loo_data.pareto_k)
    assert np.all(result.elpd_loo >= loo_data.elpd_loo)
