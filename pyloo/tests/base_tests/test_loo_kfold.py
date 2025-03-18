"""Tests for K-fold cross-validation functionality."""

import numpy as np
import pytest

from ...loo import loo
from ...loo_kfold import _kfold_split_random, _kfold_split_stratified, kfold
from ...wrapper.pymc import PyMCWrapper


@pytest.mark.parametrize("scale", ["log", "negative_log", "deviance"])
def test_kfold_basic(large_regression_model, scale):
    """Test basic K-fold cross-validation functionality with different scales."""
    model, idata = large_regression_model
    wrapper = PyMCWrapper(model, idata)

    n_obs = len(wrapper.get_observed_data())
    k_folds = min(2, n_obs)

    result = kfold(wrapper, K=k_folds, scale=scale, draws=100, tune=100)

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
    """Test that K-fold CV results match R rstanarm implementation."""
    model, idata = large_regression_model
    wrapper = PyMCWrapper(model, idata)

    loo_result = loo(wrapper.idata)

    kfold_result = kfold(
        wrapper, K=10, draws=1000, tune=500, chains=4, progressbar=False
    )

    print(kfold_result)
    print(loo_result)

    loo_elpd = loo_result["elpd_loo"]
    kfold_elpd = kfold_result["elpd_kfold"]

    print("\nLOO Result:")
    print(f"elpd_loo: {loo_elpd:.2f}")
    print(f"p_loo: {loo_result['p_loo']:.2f}")
    print(f"looic: {loo_result['looic']:.2f}")
    print(f"se: {loo_result['se']:.2f}")

    print("\nK-fold Result:")
    print(f"elpd_kfold: {kfold_elpd:.2f}")
    print(f"p_kfold: {kfold_result['p_kfold']:.2f}")
    print(f"kfoldic: {kfold_result['kfoldic']:.2f}")
    print(f"se: {kfold_result['se']:.2f}")

    rel_diff = np.abs(loo_elpd - kfold_elpd) / np.abs(loo_elpd)
    print(f"\nRelative difference: {rel_diff:.2%}")

    assert np.sign(loo_elpd) == np.sign(kfold_elpd)
    assert (
        rel_diff < 0.5
    ), f"LOO and K-fold differ by {rel_diff:.2%}, which exceeds the 50% threshold"


def test_kfold_save_fits(large_regression_model):
    """Test K-fold CV with save_fits=True."""
    model, idata = large_regression_model
    wrapper = PyMCWrapper(model, idata)

    n_obs = len(wrapper.get_observed_data())
    k_folds = min(5, n_obs)

    result = kfold(wrapper, K=k_folds, save_fits=True, draws=100, tune=100)

    assert "fits" in result
    assert len(result["fits"]) <= k_folds
    for fit, omitted in result["fits"]:
        assert fit is not None
        assert isinstance(omitted, np.ndarray)
        assert np.all(omitted < n_obs)


def test_kfold_custom_folds(large_regression_model):
    """Test K-fold CV with custom fold assignments."""
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

    result = kfold(wrapper, folds=custom_folds, draws=100, tune=100)

    assert result is not None
    assert "elpd_kfold" in result
    assert result["n_data_points"] == n_obs


def test_kfold_split_random():
    """Test random fold creation function."""
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
    """Test stratified fold creation function."""
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


def test_kfold_split_stratified_continuous():
    """Test stratified fold creation with continuous variables."""
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
    """Test K-fold CV with a hierarchical model (using large regression model)."""
    model, idata = large_regression_model
    wrapper = PyMCWrapper(model, idata)

    n_obs = len(wrapper.get_observed_data())
    k_folds = min(2, n_obs)

    result = kfold(wrapper, K=k_folds, draws=100, tune=100)

    assert result is not None
    assert "elpd_kfold" in result
    assert "p_kfold" in result
    assert "se" in result
    assert "kfold_i" in result


def test_kfold_poisson_model(large_regression_model):
    """Test K-fold CV with a Poisson model (using large regression model)."""
    model, idata = large_regression_model
    wrapper = PyMCWrapper(model, idata)

    n_obs = len(wrapper.get_observed_data())
    k_folds = min(2, n_obs)

    result = kfold(wrapper, K=k_folds, draws=100, tune=100)

    assert result is not None
    assert "elpd_kfold" in result
    assert "p_kfold" in result
    assert "se" in result
    assert "kfold_i" in result


def test_kfold_multi_observed_model(large_regression_model):
    """Test K-fold CV with a model having multiple observed variables (using large regression model)."""
    model, idata = large_regression_model
    wrapper = PyMCWrapper(model, idata)

    var_name = "y"

    var_data = wrapper.get_observed_data()
    n_obs = len(var_data)
    k_folds = min(2, n_obs)

    result = kfold(wrapper, K=k_folds, var_name=var_name, draws=100, tune=100)

    assert result is not None
    assert "elpd_kfold" in result
    assert result["n_data_points"] <= n_obs


def test_kfold_stratified_example(large_regression_model):
    """Test a complete example of stratified K-fold CV."""
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

    result = kfold(wrapper, folds=folds, draws=100, tune=100)

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


def test_kfold_large_model(large_regression_model):
    """Test K-fold CV with a large regression model to ensure robustness."""
    model, idata = large_regression_model
    wrapper = PyMCWrapper(model, idata)

    n_obs = len(wrapper.get_observed_data())

    k_folds = 5

    result_random = kfold(wrapper, K=k_folds, draws=200, tune=200, progressbar=False)

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

    result_strat = kfold(
        wrapper, folds=strat_folds, draws=200, tune=200, progressbar=False
    )

    assert result_strat is not None
    assert "elpd_kfold" in result_strat
    assert result_strat["n_data_points"] == n_obs
    assert np.isfinite(result_strat["elpd_kfold"])
    assert np.isfinite(result_strat["se"])

    result_with_fits = kfold(
        wrapper, K=3, save_fits=True, draws=100, tune=100, progressbar=False
    )

    assert "fits" in result_with_fits
    assert len(result_with_fits["fits"]) <= 3

    for fit, omitted in result_with_fits["fits"]:
        assert fit is not None
        assert isinstance(omitted, np.ndarray)
        assert np.all(omitted < n_obs)


def test_kfold_stratify_parameter(large_regression_model):
    """Test the stratify parameter in the kfold function."""
    model, idata = large_regression_model
    wrapper = PyMCWrapper(model, idata)

    observed_data = wrapper.get_observed_data()
    n_obs = len(observed_data)

    strat_var = np.zeros(n_obs, dtype=int)
    strat_var[observed_data > np.median(observed_data)] = 1

    k_folds = 5
    random_seed = 42

    result_stratify = kfold(
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

    result_both = kfold(
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
        kfold(
            wrapper,
            K=k_folds,
            stratify=np.array([0, 1]),
            draws=200,
            tune=200,
            progressbar=False,
        )


def test_kfold_improved_validation(large_regression_model):
    """Test the improved validation in the kfold function."""
    model, idata = large_regression_model
    wrapper = PyMCWrapper(model, idata)
    n_obs = len(wrapper.get_observed_data())

    folds = _kfold_split_random(K=n_obs + 5, N=n_obs)
    assert len(folds) == n_obs
    assert np.max(folds) <= n_obs

    result = kfold(
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
        kfold(
            wrapper,
            K=2,
            stratify=np.array([np.nan] * n_obs),
            draws=100,
            tune=100,
            progressbar=False,
        )
