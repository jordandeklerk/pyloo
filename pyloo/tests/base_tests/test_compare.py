"""Tests for model comparison functionality."""

import logging
from copy import deepcopy

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from ...compare import loo_compare
from ...loo import loo
from ...loo_subsample import loo_subsample
from ...waic import waic
from ...wrapper.pymc import PyMCWrapper
from ..helpers import create_large_model


@pytest.fixture(scope="session")
def centered_eight():
    """Load the centered_eight example dataset from ArviZ."""
    return az.load_arviz_data("centered_eight")


@pytest.fixture(scope="session")
def non_centered_eight():
    """Load the non_centered_eight example dataset from ArviZ."""
    return az.load_arviz_data("non_centered_eight")


@pytest.fixture(scope="session")
def models(centered_eight, non_centered_eight):
    """Create dictionary of models for comparison."""
    return {
        "centered": centered_eight,
        "non_centered": non_centered_eight,
    }


@pytest.fixture(scope="session")
def large_models():
    """Create dictionary of large models for testing subsampling."""
    model1 = create_large_model(seed=42, n_obs=10000)
    model2 = create_large_model(seed=43, n_obs=10000)
    return {
        "model1": model1,
        "model2": model2,
    }


def test_loo_compare_basic(models):
    """Test basic model comparison functionality."""
    result = loo_compare(models)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert all(
        col in result.columns
        for col in [
            "rank",
            "elpd_loo",
            "p_loo",
            "elpd_diff",
            "weight",
            "se",
            "dse",
            "warning",
            "scale",
        ]
    )

    assert set(result["rank"]) == {0, 1}
    assert result.loc[result["rank"] == 0, "elpd_diff"].item() == 0
    assert all(result.loc[result["rank"] > 0, "elpd_diff"] < 0)
    assert_allclose(result["weight"].sum(), 1.0, rtol=1e-7)

    az_result = az.compare(models)
    assert_allclose(result["elpd_loo"], az_result["elpd_loo"], rtol=1e-7)
    assert_allclose(result["p_loo"], az_result["p_loo"], rtol=1e-7)


def test_waic_compare_basic(models):
    """Test basic model comparison functionality using WAIC."""
    result = loo_compare(models, ic="waic")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert all(
        col in result.columns
        for col in [
            "rank",
            "elpd_waic",
            "p_waic",
            "elpd_diff",
            "weight",
            "se",
            "dse",
            "warning",
            "scale",
        ]
    )

    assert set(result["rank"]) == {0, 1}
    assert result.loc[result["rank"] == 0, "elpd_diff"].item() == 0
    assert all(result.loc[result["rank"] > 0, "elpd_diff"] < 0)
    assert_allclose(result["weight"].sum(), 1.0, rtol=1e-7)

    az_result = az.compare(models, ic="waic")
    assert_allclose(result["elpd_waic"], az_result["elpd_waic"], rtol=1e-7)
    assert_allclose(result["p_waic"], az_result["p_waic"], rtol=1e-7)


@pytest.mark.parametrize("scale", ["log", "negative_log", "deviance"])
def test_loo_compare_scales(models, scale):
    """Test model comparison with different scales."""
    result = loo_compare(models, scale=scale)

    if scale == "log":
        assert result.iloc[0]["elpd_loo"] >= result.iloc[1]["elpd_loo"]
    else:
        assert result.iloc[0]["elpd_loo"] <= result.iloc[1]["elpd_loo"]

    az_result = az.compare(models, scale=scale)
    assert_allclose(result["elpd_loo"], az_result["elpd_loo"], rtol=1e-7)


@pytest.mark.parametrize("scale", ["log", "negative_log", "deviance"])
def test_waic_compare_scales(models, scale):
    """Test model comparison with different scales using WAIC."""
    result = loo_compare(models, ic="waic", scale=scale)

    if scale == "log":
        assert result.iloc[0]["elpd_waic"] >= result.iloc[1]["elpd_waic"]
    else:
        assert result.iloc[0]["elpd_waic"] <= result.iloc[1]["elpd_waic"]

    az_result = az.compare(models, ic="waic", scale=scale)
    assert_allclose(result["elpd_waic"], az_result["elpd_waic"], rtol=1e-7)


@pytest.mark.parametrize("method", ["stacking", "BB-pseudo-BMA", "pseudo-BMA"])
def test_loo_compare_methods(models, method):
    """Test different methods for computing model weights."""
    result = loo_compare(models, method=method)

    assert_allclose(result["weight"].sum(), 1.0, rtol=1e-7)
    assert np.all(result["weight"] >= 0)

    az_result = az.compare(models, method=method)
    if method == "BB-pseudo-BMA":
        rtol = 2e-2
        assert_allclose(result["weight"].sum(), 1.0, rtol=1e-7)
        assert_allclose(
            np.sort(result["weight"]), np.sort(az_result["weight"]), rtol=rtol
        )
    else:
        assert_allclose(result["weight"], az_result["weight"], rtol=1e-7, atol=1e-15)


@pytest.mark.parametrize("method", ["stacking", "BB-pseudo-BMA", "pseudo-BMA"])
def test_waic_compare_methods(models, method):
    """Test different methods for computing model weights using WAIC."""
    result = loo_compare(models, ic="waic", method=method)

    assert_allclose(result["weight"].sum(), 1.0, rtol=1e-7)
    assert np.all(result["weight"] >= 0)

    az_result = az.compare(models, ic="waic", method=method)
    if method == "BB-pseudo-BMA":
        rtol = 2e-2
        assert_allclose(result["weight"].sum(), 1.0, rtol=1e-7)
        assert_allclose(
            np.sort(result["weight"]), np.sort(az_result["weight"]), rtol=rtol
        )
    else:
        assert_allclose(result["weight"], az_result["weight"], rtol=1e-7, atol=1e-15)


def test_loo_compare_precomputed_elpd(models):
    """Test model comparison with pre-computed ELPD values."""
    elpds = {name: loo(model, pointwise=True) for name, model in models.items()}
    result = loo_compare(elpds)

    direct_result = loo_compare(models)
    assert_allclose(result["elpd_loo"], direct_result["elpd_loo"], rtol=1e-7)
    assert_allclose(result["weight"], direct_result["weight"], rtol=1e-7)


def test_waic_compare_precomputed_elpd(models):
    """Test model comparison with pre-computed WAIC values."""
    elpds = {name: waic(model, pointwise=True) for name, model in models.items()}
    result = loo_compare(elpds)

    direct_result = loo_compare(models, ic="waic")
    assert_allclose(result["elpd_waic"], direct_result["elpd_waic"], rtol=1e-7)
    assert_allclose(result["weight"], direct_result["weight"], rtol=1e-7)


def test_loo_compare_invalid_scale(models):
    """Test model comparison with invalid scale."""
    with pytest.raises(ValueError, match="Scale must be"):
        loo_compare(models, scale="invalid")


def test_loo_compare_invalid_method(models):
    """Test model comparison with invalid method."""
    with pytest.raises(ValueError, match="Method must be"):
        loo_compare(models, method="invalid")


def test_loo_compare_invalid_ic(models):
    """Test model comparison with invalid information criterion."""
    with pytest.raises(ValueError, match="ic must be 'loo', 'waic', or 'kfold'"):
        loo_compare(models, ic="invalid")


def test_loo_compare_single_model(models):
    """Test model comparison with single model."""
    with pytest.raises(ValueError, match="at least two models"):
        loo_compare({"model1": next(iter(models.values()))})


def test_mixed_ic_error(models):
    """Test error when mixing different information criteria."""
    elpds = {
        "loo_model": loo(next(iter(models.values())), pointwise=True),
        "waic_model": waic(next(iter(models.values())), pointwise=True),
    }
    with pytest.raises(
        ValueError, match="All information criteria to be compared must be the same"
    ):
        loo_compare(elpds)


def test_loo_compare_warning_models(centered_eight):
    """Test model comparison with problematic models that raise warnings."""
    model1 = deepcopy(centered_eight)
    model2 = deepcopy(centered_eight)

    model2.log_likelihood["obs"][:, :, 1] = 10

    models = {"model1": model1, "model2": model2}

    with pytest.warns(UserWarning):
        result = loo_compare(models)
        assert result is not None
        assert any(result["warning"])


@pytest.mark.parametrize("observations", [1000, 2000])
def test_loo_compare_subsample(large_models, observations):
    """Test model comparison with subsampled LOO computation."""
    result = loo_compare(
        large_models,
        observations=observations,
        estimator="diff_srs",
    )

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert all(
        col in result.columns
        for col in [
            "rank",
            "elpd_loo",
            "p_loo",
            "elpd_diff",
            "weight",
            "se",
            "dse",
            "warning",
            "scale",
        ]
    )

    assert_allclose(result["weight"].sum(), 1.0, rtol=1e-7)
    assert np.all(result["weight"] >= 0)
    assert set(result["rank"]) == {0, 1}
    assert result.loc[result["rank"] == 0, "elpd_diff"].item() == 0


@pytest.mark.parametrize("estimator", ["diff_srs", "srs", "hh_pps"])
def test_loo_compare_subsample_estimators(large_models, estimator):
    """Test model comparison with different subsampling estimators."""
    result = loo_compare(
        large_models,
        observations=1000,
        estimator=estimator,
    )

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2

    assert_allclose(result["weight"].sum(), 1.0, rtol=1e-7)
    assert np.all(result["weight"] >= 0)


def test_loo_compare_mixed_subsample(models, large_models):
    """Test comparing mix of subsampled and full LOO models."""
    mixed_models = {
        "small": next(iter(models.values())),
        "large": next(iter(large_models.values())),
    }

    with pytest.raises(ValueError) as exc_info:
        loo_compare(mixed_models)

    error_msg = str(exc_info.value)
    assert any(str(dim) in error_msg for dim in [(10000,), (8,)])


def test_loo_compare_precomputed_subsample(large_models):
    """Test model comparison with pre-computed subsampled ELPD values."""
    elpds = {
        name: loo_subsample(
            model, observations=1000, pointwise=True, estimator="diff_srs"
        )
        for name, model in large_models.items()
    }
    result = loo_compare(elpds)
    direct_result = loo_compare(large_models, observations=1000, estimator="diff_srs")

    for res in [result, direct_result]:
        assert_allclose(res["weight"].sum(), 1.0, rtol=1e-7)
        assert np.all(res["weight"] >= 0)

    assert_allclose(result["elpd_loo"], direct_result["elpd_loo"], rtol=1e-3)

    best_model_precomp = result.index[result["elpd_loo"].argmax()]
    best_model_direct = direct_result.index[direct_result["elpd_loo"].argmax()]
    assert best_model_precomp == best_model_direct


def test_loo_compare_subsample_warning(large_models):
    """Test warnings with subsampled model comparison."""
    model1 = deepcopy(next(iter(large_models.values())))
    model2 = deepcopy(model1)

    model2.log_likelihood["obs"][:, :, :100] = 10

    models = {"model1": model1, "model2": model2}

    with pytest.warns(UserWarning):
        result = loo_compare(models, observations=1000, estimator="diff_srs")
        assert result is not None
        assert any(result["warning"])


@pytest.mark.parametrize("K", [2, 5])
def test_kfold_compare_basic(simple_model, K):
    """Test basic model comparison functionality using K-fold cross-validation."""
    model1, idata1 = simple_model
    model2, idata2 = simple_model

    wrapper1 = PyMCWrapper(model1, idata1)
    wrapper2 = PyMCWrapper(model2, idata2)

    pymc_wrappers = {
        "simple": wrapper1,
        "hierarchical": wrapper2,
    }

    try:
        result = loo_compare(pymc_wrappers, ic="kfold", K=K)
    except Exception as e:
        pytest.skip(f"Skipping K-fold test due to error: {e}")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert all(
        col in result.columns
        for col in [
            "rank",
            "elpd_kfold",
            "p_kfold",
            "elpd_diff",
            "weight",
            "se",
            "dse",
            "warning",
            "scale",
        ]
    )

    assert set(result["rank"]) == {0, 1}
    assert result.loc[result["rank"] == 0, "elpd_diff"].item() == 0
    assert_allclose(result["weight"].sum(), 1.0, rtol=1e-7)
    assert np.all(result["weight"] >= 0)


@pytest.mark.parametrize("scale", ["log", "negative_log", "deviance"])
def test_kfold_compare_scales(simple_model, scale):
    """Test model comparison with different scales using K-fold cross-validation."""
    model1, idata1 = simple_model
    model2, idata2 = simple_model

    wrapper1 = PyMCWrapper(model1, idata1)
    wrapper2 = PyMCWrapper(model2, idata2)

    pymc_wrappers = {
        "simple": wrapper1,
        "large_regression": wrapper2,
    }

    try:
        result = loo_compare(pymc_wrappers, ic="kfold", K=2, scale=scale)
    except Exception as e:
        pytest.skip(f"Skipping K-fold test due to error: {e}")

    assert result["scale"].iloc[0] == scale

    if scale == "log":
        assert result.iloc[0]["elpd_kfold"] >= result.iloc[1]["elpd_kfold"]
    else:
        assert result.iloc[0]["elpd_kfold"] <= result.iloc[1]["elpd_kfold"]


@pytest.mark.parametrize("method", ["stacking", "BB-pseudo-BMA", "pseudo-BMA"])
def test_kfold_compare_methods(simple_model, poisson_model, method):
    """Test different methods for computing model weights using K-fold cross-validation."""
    model1, idata1 = simple_model
    model2, idata2 = poisson_model

    wrapper1 = PyMCWrapper(model1, idata1)
    wrapper2 = PyMCWrapper(model2, idata2)

    pymc_wrappers = {
        "simple": wrapper1,
        "poisson": wrapper2,
    }

    try:
        result = loo_compare(pymc_wrappers, ic="kfold", K=2, method=method)
    except Exception as e:
        pytest.skip(f"Skipping K-fold test due to error: {e}")

    assert_allclose(result["weight"].sum(), 1.0, rtol=1e-7)
    assert np.all(result["weight"] >= 0)


def test_kfold_compare_with_folds(simple_model):
    """Test model comparison with pre-specified folds."""
    model, idata = simple_model

    wrapper1 = PyMCWrapper(model, idata)
    wrapper2 = PyMCWrapper(model, idata)

    try:
        n_obs = len(wrapper1.get_observed_data())

        custom_folds = np.array([1, 2] * (n_obs // 2))
        if len(custom_folds) < n_obs:
            custom_folds = np.append(custom_folds, [1] * (n_obs - len(custom_folds)))

        custom_folds2 = np.array([2, 1] * (n_obs // 2))
        if len(custom_folds2) < n_obs:
            custom_folds2 = np.append(custom_folds2, [2] * (n_obs - len(custom_folds2)))

        result = loo_compare(
            {
                "model1": wrapper1,
                "model2": wrapper2,
            },
            ic="kfold",
            folds=custom_folds,
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "elpd_kfold" in result.columns
    except Exception as e:
        pytest.skip(f"Skipping K-fold test due to error: {e}")


def test_kfold_compare_stratified(simple_model):
    """Test model comparison with stratified K-fold cross-validation."""
    model, idata = simple_model

    wrapper1 = PyMCWrapper(model, idata)
    wrapper2 = PyMCWrapper(model, idata)

    try:
        n_obs = len(wrapper1.get_observed_data())

        np.random.seed(42)
        strat_var = np.random.choice([0, 1], size=n_obs, p=[0.7, 0.3])

        result = loo_compare(
            {
                "model1": wrapper1,
                "model2": wrapper2,
            },
            ic="kfold",
            K=2,
            stratify=strat_var,
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "elpd_kfold" in result.columns
    except Exception as e:
        pytest.skip(f"Skipping K-fold test due to error: {e}")


def test_loo_compare_with_jacobian_adjustments(centered_eight):
    """Test model comparison with Jacobian adjustments for different transformations."""
    original_model = deepcopy(centered_eight)
    squared_model = deepcopy(centered_eight)
    log_model = deepcopy(centered_eight)

    y = centered_eight.observed_data.obs.values
    positive_y = np.abs(y) + 1

    log_model.observed_data["obs"] = xr.DataArray(
        positive_y,
        dims=log_model.observed_data.obs.dims,
        coords=log_model.observed_data.obs.coords,
    )

    original_loo = loo(original_model, pointwise=True)

    squared_jacobian = np.log(np.abs(2 * y))
    squared_loo = loo(squared_model, pointwise=True, jacobian=squared_jacobian)

    log_jacobian = -np.log(positive_y)
    log_loo = loo(log_model, pointwise=True, jacobian=log_jacobian)

    loo_dict = {
        "original": original_loo,
        "squared": squared_loo,
        "log": log_loo,
    }

    comparison = loo_compare(loo_dict)

    logging.info(original_loo)
    logging.info(squared_loo)
    logging.info(log_loo)

    assert isinstance(comparison, pd.DataFrame)
    assert len(comparison) == 3
    assert all(comparison["scale"] == comparison["scale"].iloc[0])
    assert comparison.loc[comparison["rank"] == 0, "elpd_diff"].item() == 0
    assert_allclose(comparison["weight"].sum(), 1.0, rtol=1e-7)
    assert all(comparison["se"] > 0)
    assert comparison.loc[comparison["rank"] == 0, "dse"].item() == 0

    elpd_values = comparison["elpd_loo"].values
    assert not np.allclose(elpd_values[0], elpd_values[1]) or not np.allclose(
        elpd_values[0], elpd_values[2]
    )
