"""Tests for model comparison functionality."""
from copy import deepcopy

import arviz as az
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal

from ...compare import loo_compare
from ...loo import loo
from ...loo_subsample import loo_subsample
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
        for col in ["rank", "elpd_loo", "p_loo", "elpd_diff", "weight", "se", "dse", "warning", "scale"]
    )

    assert set(result["rank"]) == {0, 1}

    assert result.loc[result["rank"] == 0, "elpd_diff"].item() == 0
    assert all(result.loc[result["rank"] > 0, "elpd_diff"] < 0)

    assert_allclose(result["weight"].sum(), 1.0, rtol=1e-7)

    az_result = az.compare(models)
    assert_allclose(result["elpd_loo"], az_result["elpd_loo"], rtol=1e-7)
    assert_allclose(result["p_loo"], az_result["p_loo"], rtol=1e-7)


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


@pytest.mark.parametrize("method", ["stacking", "BB-pseudo-BMA", "pseudo-BMA"])
def test_loo_compare_methods(models, method):
    """Test different methods for computing model weights."""
    result = loo_compare(models, method=method)

    assert_allclose(result["weight"].sum(), 1.0, rtol=1e-7)

    assert np.all(result["weight"] >= 0)

    az_result = az.compare(models, method=method)
    if method == "BB-pseudo-BMA":
        rtol = 2e-3
        assert_allclose(result["weight"].sum(), 1.0, rtol=1e-7)
        assert_allclose(np.sort(result["weight"]), np.sort(az_result["weight"]), rtol=rtol)
    else:
        # For stacking method
        assert_allclose(result["weight"], az_result["weight"], rtol=1e-7, atol=1e-15)


def test_loo_compare_precomputed_elpd(models):
    """Test model comparison with pre-computed ELPD values."""
    elpds = {name: loo(model, pointwise=True) for name, model in models.items()}
    result = loo_compare(elpds)

    direct_result = loo_compare(models)
    assert_allclose(result["elpd_loo"], direct_result["elpd_loo"], rtol=1e-7)
    assert_allclose(result["weight"], direct_result["weight"], rtol=1e-7)


def test_loo_compare_invalid_scale(models):
    """Test model comparison with invalid scale."""
    with pytest.raises(ValueError, match="Scale must be"):
        loo_compare(models, scale="invalid")


def test_loo_compare_invalid_method(models):
    """Test model comparison with invalid method."""
    with pytest.raises(ValueError, match="Method must be"):
        loo_compare(models, method="invalid")


def test_loo_compare_single_model(models):
    """Test model comparison with single model."""
    with pytest.raises(ValueError, match="at least two models"):
        loo_compare({"model1": next(iter(models.values()))})


def test_loo_compare_bb_pseudo_bma_seed(models):
    """Test reproducibility with BB-pseudo-BMA method."""
    result1 = loo_compare(models, method="BB-pseudo-BMA", seed=42)
    result2 = loo_compare(models, method="BB-pseudo-BMA", seed=42)
    result3 = loo_compare(models, method="BB-pseudo-BMA", seed=43)

    assert_array_almost_equal(result1["weight"], result2["weight"])

    with pytest.raises(AssertionError):
        assert_array_almost_equal(result1["weight"], result3["weight"])


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


def test_loo_compare_stacking_optimization(models):
    """Test that stacking optimization produces valid weights."""
    result = loo_compare(models, method="stacking")

    assert_allclose(result["weight"].sum(), 1.0, rtol=1e-7)

    assert np.all(result["weight"] >= 0)

    az_result = az.compare(models, method="stacking")
    # For stacking method
    assert_allclose(result["weight"], az_result["weight"], rtol=1e-7, atol=1e-15)


def test_loo_compare_missing_var_name(models):
    """Test model comparison with multiple log_likelihood variables."""
    models_copy = deepcopy(models)
    for model in models_copy.values():
        model.log_likelihood["obs2"] = model.log_likelihood["obs"]

    with pytest.raises(TypeError):
        loo_compare(models_copy)

    result = loo_compare(models_copy, var_name="obs")
    assert result is not None


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
        for col in ["rank", "elpd_loo", "p_loo", "elpd_diff", "weight", "se", "dse", "warning", "scale"]
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
        name: loo_subsample(model, observations=1000, pointwise=True, estimator="diff_srs")
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
