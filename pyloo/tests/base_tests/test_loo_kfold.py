"""Tests for K-fold cross-validation functionality."""

import logging

import numpy as np
import pytest

from ...loo import loo
from ...loo_kfold import (
    _kfold_split_grouped,
    _kfold_split_random,
    _kfold_split_stratified,
    loo_kfold,
)
from ...wrapper.pymc.pymc import PyMCWrapper

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("scale", ["log", "negative_log", "deviance"])
def test_kfold_basic(large_regression_model, scale):
    model, idata = large_regression_model
    wrapper = PyMCWrapper(model, idata)

    result = loo_kfold(wrapper, K=10, scale=scale, pointwise=True, draws=2000, tune=500)

    assert result is not None
    assert "elpd_kfold" in result
    assert "p_kfold" in result
    assert "se" in result
    assert "kfold_i" in result
    assert result["scale"] == scale
    assert result.method == "kfold"
    assert result["n_data_points"] > 0
    assert result["warning"] is False


def test_kfold_compare_to_loo(large_regression_model):
    model, idata = large_regression_model
    wrapper = PyMCWrapper(model, idata)

    loo_result = loo(wrapper.idata)

    kfold_result = loo_kfold(
        wrapper, K=10, draws=1000, tune=500, chains=4, progressbar=False
    )

    loo_elpd = loo_result["elpd_loo"]
    kfold_elpd = kfold_result["elpd_kfold"]

    rel_diff = np.abs(loo_elpd - kfold_elpd) / np.abs(loo_elpd)

    assert np.sign(loo_elpd) == np.sign(kfold_elpd)
    assert (
        rel_diff < 0.5
    ), f"LOO and K-fold differ by {rel_diff:.2%}, which exceeds the 50% threshold"


def test_kfold_save_fits(large_regression_model):
    model, idata = large_regression_model
    wrapper = PyMCWrapper(model, idata)

    n_obs = len(wrapper.get_observed_data())
    k_folds = min(5, n_obs)

    result = loo_kfold(wrapper, K=k_folds, save_fits=True, draws=100, tune=100)

    assert "fits" in result
    assert len(result["fits"]) <= k_folds
    for fit, omitted in result["fits"]:
        assert fit is not None
        assert isinstance(omitted, np.ndarray)
        assert np.all(omitted < n_obs)


def test_kfold_custom_folds(large_regression_model):
    model, idata = large_regression_model
    wrapper = PyMCWrapper(model, idata)

    n_obs = len(wrapper.get_observed_data())

    custom_folds = np.ones(n_obs, dtype=int)
    split_point = n_obs // 2
    if split_point > 0:
        custom_folds[split_point:] = 2

    assert np.all(
        (custom_folds >= 1) & (custom_folds <= 2)
    ), "Custom folds outside valid range"
    assert len(custom_folds) == n_obs, "Custom folds length doesn't match data"

    result = loo_kfold(wrapper, folds=custom_folds, draws=100, tune=100)

    assert result is not None
    assert "elpd_kfold" in result
    assert result["n_data_points"] == n_obs


def test_kfold_split_random():
    N = 100
    K = 5

    folds = _kfold_split_random(K, N)
    assert len(folds) == N
    assert set(folds) == set(range(1, K + 1))

    folds1 = _kfold_split_random(K, N, seed=42)
    folds2 = _kfold_split_random(K, N, seed=42)
    assert np.array_equal(folds1, folds2)

    fold_counts = np.bincount(folds)[1:]
    assert len(fold_counts) == K
    assert np.all(fold_counts >= N // K)
    assert np.all(fold_counts <= N // K + 1)

    K_large = 150
    folds_large = _kfold_split_random(K_large, N)
    assert len(folds_large) == N
    assert max(folds_large) <= N
    assert min(folds_large) >= 1


def test_kfold_split_stratified():
    N = 100
    K = 5

    groups = np.zeros(N, dtype=int)
    groups[:30] = 1
    groups[30:70] = 2
    groups[70:] = 3

    folds = _kfold_split_stratified(K, groups, seed=42)
    assert len(folds) == N
    assert set(folds) == set(range(1, K + 1))

    for k in range(1, K + 1):
        fold_k_groups = groups[folds == k]
        assert np.sum(fold_k_groups == 1) / len(fold_k_groups) == pytest.approx(
            0.3, abs=0.15
        )
        assert np.sum(fold_k_groups == 2) / len(fold_k_groups) == pytest.approx(
            0.4, abs=0.15
        )
        assert np.sum(fold_k_groups == 3) / len(fold_k_groups) == pytest.approx(
            0.3, abs=0.15
        )

    K_large = 10
    folds_large = _kfold_split_stratified(K_large, groups, seed=42)
    assert len(folds_large) == N
    assert max(folds_large) <= K_large
    assert min(folds_large) >= 1


def test_kfold_split_grouped():
    N = 100
    K = 5
    n_groups = 10

    groups = np.repeat(np.arange(n_groups), N // n_groups)

    folds = _kfold_split_grouped(K, groups, seed=42)
    assert len(folds) == N
    assert set(folds) == set(range(1, K + 1))

    for group in range(n_groups):
        group_indices = np.where(groups == group)[0]
        group_folds = folds[group_indices]
        assert (
            len(np.unique(group_folds)) == 1
        ), f"Group {group} is split across multiple folds"

    folds1 = _kfold_split_grouped(K, groups, seed=42)
    folds2 = _kfold_split_grouped(K, groups, seed=42)
    assert np.array_equal(folds1, folds2)

    K_large = 20
    folds_large = _kfold_split_grouped(K_large, groups, seed=42)
    assert len(folds_large) == N
    assert max(folds_large) <= n_groups
    assert min(folds_large) >= 1


def test_kfold_split_stratified_continuous():
    N = 100
    K = 5

    x = np.linspace(0, 10, N)

    folds = _kfold_split_stratified(K, x, seed=42)
    assert len(folds) == N
    assert set(folds) == set(range(1, K + 1))

    for k in range(1, K + 1):
        fold_k_x = x[folds == k]
        assert np.mean(fold_k_x) == pytest.approx(np.mean(x), abs=1.2)


def test_kfold_hierarchical_model(large_regression_model):
    model, idata = large_regression_model
    wrapper = PyMCWrapper(model, idata)

    n_obs = len(wrapper.get_observed_data())
    k_folds = min(2, n_obs)

    result = loo_kfold(wrapper, K=k_folds, pointwise=True, draws=100, tune=100)

    assert result is not None
    assert "elpd_kfold" in result
    assert "p_kfold" in result
    assert "se" in result
    assert "kfold_i" in result


def test_kfold_poisson_model(large_regression_model):
    model, idata = large_regression_model
    wrapper = PyMCWrapper(model, idata)

    n_obs = len(wrapper.get_observed_data())
    k_folds = min(2, n_obs)

    result = loo_kfold(wrapper, K=k_folds, pointwise=True, draws=100, tune=100)

    assert result is not None
    assert "elpd_kfold" in result
    assert "p_kfold" in result
    assert "se" in result
    assert "kfold_i" in result


def test_kfold_multi_observed_model(large_regression_model):
    model, idata = large_regression_model
    wrapper = PyMCWrapper(model, idata)

    var_name = "y"

    var_data = wrapper.get_observed_data()
    n_obs = len(var_data)
    k_folds = min(2, n_obs)

    result = loo_kfold(wrapper, K=k_folds, var_name=var_name, draws=100, tune=100)

    assert result is not None
    assert "elpd_kfold" in result
    assert result["n_data_points"] <= n_obs


def test_kfold_stratified_example(large_regression_model):
    model, idata = large_regression_model
    wrapper = PyMCWrapper(model, idata)

    observed_data = wrapper.get_observed_data()
    n_obs = len(observed_data)

    groups = np.zeros(n_obs, dtype=int)
    groups[observed_data > np.median(observed_data)] = 1

    k_folds = min(3, n_obs - 1)

    folds = _kfold_split_stratified(K=k_folds, x=groups, seed=42)

    assert len(folds) == n_obs
    assert np.all(folds >= 1) and np.all(folds <= k_folds)

    result = loo_kfold(wrapper, folds=folds, pointwise=True, draws=100, tune=100)

    assert result is not None
    assert "elpd_kfold" in result
    assert "p_kfold" in result
    assert "se" in result
    assert "kfold_i" in result

    for k in range(1, k_folds + 1):
        fold_indices = np.where(folds == k)[0]
        if len(fold_indices) > 0:
            fold_k_groups = groups[fold_indices]
            tolerance = 0.3 if n_obs < 20 else 0.2
            assert np.mean(fold_k_groups) == pytest.approx(
                np.mean(groups), abs=tolerance
            )


def test_kfold_grouped_parameter(large_regression_model):
    model, idata = large_regression_model
    wrapper = PyMCWrapper(model, idata)

    observed_data = wrapper.get_observed_data()
    n_obs = len(observed_data)

    n_groups = 5
    group_ids = np.repeat(np.arange(n_groups), n_obs // n_groups + 1)[:n_obs]

    k_folds = min(3, n_groups)
    random_seed = 42

    result_grouped = loo_kfold(
        wrapper,
        K=k_folds,
        groups=group_ids,
        random_seed=random_seed,
        draws=100,
        tune=100,
        progressbar=False,
    )

    assert result_grouped is not None
    assert "elpd_kfold" in result_grouped
    assert result_grouped["n_data_points"] == n_obs
    assert np.isfinite(result_grouped["elpd_kfold"])
    assert np.isfinite(result_grouped["se"])

    with pytest.raises(ValueError, match="Length of groups .* must match observations"):
        loo_kfold(
            wrapper,
            K=k_folds,
            groups=np.array([0, 1]),
            draws=100,
            tune=100,
            progressbar=False,
        )

    strat_var = np.zeros(n_obs, dtype=int)
    strat_var[observed_data > np.median(observed_data)] = 1

    result_both = loo_kfold(
        wrapper,
        K=k_folds,
        groups=group_ids,
        stratify=strat_var,
        random_seed=random_seed,
        draws=100,
        tune=100,
        progressbar=False,
    )

    assert result_both is not None
    assert "elpd_kfold" in result_both
    assert result_both["stratified"] is False
    assert result_both.stratified is False


def test_kfold_large_model(large_regression_model):
    model, idata = large_regression_model
    wrapper = PyMCWrapper(model, idata)

    n_obs = len(wrapper.get_observed_data())

    k_folds = 5

    result_random = loo_kfold(
        wrapper, K=k_folds, draws=200, tune=200, progressbar=False
    )

    assert result_random is not None
    assert "elpd_kfold" in result_random
    assert result_random["n_data_points"] == n_obs
    assert np.isfinite(result_random["elpd_kfold"])
    assert np.isfinite(result_random["se"])

    observed_data = wrapper.get_observed_data()
    strat_var = observed_data

    strat_folds = _kfold_split_stratified(K=k_folds, x=strat_var, seed=42)

    assert len(strat_folds) == n_obs
    assert np.all(strat_folds >= 1) and np.all(strat_folds <= k_folds)

    result_strat = loo_kfold(
        wrapper, folds=strat_folds, draws=200, tune=200, progressbar=False
    )

    assert result_strat is not None
    assert "elpd_kfold" in result_strat
    assert result_strat["n_data_points"] == n_obs
    assert np.isfinite(result_strat["elpd_kfold"])
    assert np.isfinite(result_strat["se"])

    result_with_fits = loo_kfold(
        wrapper, K=3, save_fits=True, draws=100, tune=100, progressbar=False
    )

    assert "fits" in result_with_fits
    assert len(result_with_fits["fits"]) <= 3

    for fit, omitted in result_with_fits["fits"]:
        assert fit is not None
        assert isinstance(omitted, np.ndarray)
        assert np.all(omitted < n_obs)


def test_kfold_stratify_parameter(large_regression_model):
    model, idata = large_regression_model
    wrapper = PyMCWrapper(model, idata)

    observed_data = wrapper.get_observed_data()
    n_obs = len(observed_data)

    strat_var = np.zeros(n_obs, dtype=int)
    strat_var[observed_data > np.median(observed_data)] = 1

    k_folds = 5
    random_seed = 42

    result_stratify = loo_kfold(
        wrapper,
        K=k_folds,
        stratify=strat_var,
        random_seed=random_seed,
        draws=200,
        tune=200,
        progressbar=False,
    )

    assert result_stratify is not None
    assert "elpd_kfold" in result_stratify
    assert result_stratify["n_data_points"] == n_obs
    assert np.isfinite(result_stratify["elpd_kfold"])
    assert np.isfinite(result_stratify["se"])

    custom_folds = np.ones(n_obs, dtype=int)
    custom_folds[n_obs // 2 :] = 2

    result_both = loo_kfold(
        wrapper,
        folds=custom_folds,
        stratify=strat_var,
        draws=200,
        tune=200,
        progressbar=False,
    )

    assert result_both is not None
    assert "elpd_kfold" in result_both
    assert result_both["n_data_points"] == n_obs
    assert np.isfinite(result_both["elpd_kfold"])
    assert np.isfinite(result_both["se"])

    with pytest.raises(
        ValueError, match="Length of stratify .* must match observations"
    ):
        loo_kfold(
            wrapper,
            K=k_folds,
            stratify=np.array([0, 1]),
            draws=200,
            tune=200,
            progressbar=False,
        )


def test_kfold_improved_validation(large_regression_model):
    model, idata = large_regression_model
    wrapper = PyMCWrapper(model, idata)
    n_obs = len(wrapper.get_observed_data())

    folds = _kfold_split_random(K=n_obs + 5, N=n_obs)
    assert len(folds) == n_obs
    assert np.max(folds) <= n_obs

    result = loo_kfold(
        wrapper,
        K=3,
        stratify=list(range(n_obs)),
        draws=100,
        tune=100,
        progressbar=False,
    )
    assert result is not None
    assert "elpd_kfold" in result

    with pytest.raises(ValueError, match="Failed to create stratified folds"):
        loo_kfold(
            wrapper,
            K=2,
            stratify=np.array([np.nan] * n_obs),
            draws=100,
            tune=100,
            progressbar=False,
        )
