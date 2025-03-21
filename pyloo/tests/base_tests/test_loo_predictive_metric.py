"""Tests for the loo_predictive_metric module."""

import logging

import numpy as np
import pytest

from ...loo_predictive_metric import (
    _accuracy,
    _balanced_accuracy,
    _mae,
    _mse,
    _rmse,
    loo_predictive_metric,
)


def test_loo_predictive_metric_basic(centered_eight):
    """Test basic functionality of loo_predictive_metric."""
    idata = centered_eight
    y_obs = idata.observed_data.obs.values

    result = loo_predictive_metric(
        data=idata, y=y_obs, var_name="obs", log_lik_var_name="obs", metric="mae"
    )

    logging.info(result)

    assert isinstance(result, dict)
    assert "estimate" in result
    assert "se" in result
    assert result["estimate"] > 0
    assert result["se"] > 0


def test_loo_predictive_metric_mse(centered_eight):
    """Test loo_predictive_metric with MSE metric."""
    idata = centered_eight
    y_obs = idata.observed_data.obs.values

    result = loo_predictive_metric(
        data=idata, y=y_obs, var_name="obs", log_lik_var_name="obs", metric="mse"
    )

    assert isinstance(result, dict)
    assert "estimate" in result
    assert "se" in result
    assert result["estimate"] > 0
    assert result["se"] > 0


def test_loo_predictive_metric_rmse(centered_eight):
    """Test RMSE metric calculation."""
    idata = centered_eight
    y_obs = idata.observed_data.obs.values

    result_rmse = loo_predictive_metric(
        data=idata, y=y_obs, var_name="obs", log_lik_var_name="obs", metric="rmse"
    )

    result_mse = loo_predictive_metric(
        data=idata, y=y_obs, var_name="obs", log_lik_var_name="obs", metric="mse"
    )

    assert np.isclose(
        result_rmse["estimate"], np.sqrt(result_mse["estimate"]), rtol=1e-10
    )


def test_loo_predictive_metric_with_r_eff(centered_eight):
    """Test loo_predictive_metric with r_eff parameter."""
    idata = centered_eight
    y_obs = idata.observed_data.obs.values

    r_eff = 0.8

    result1 = loo_predictive_metric(
        data=idata,
        y=y_obs,
        var_name="obs",
        log_lik_var_name="obs",
        metric="mae",
        r_eff=r_eff,
    )

    result2 = loo_predictive_metric(
        data=idata,
        y=y_obs,
        var_name="obs",
        log_lik_var_name="obs",
        metric="mae",
        r_eff=1.0,
    )

    # Results should be different with different r_eff values
    assert result1["estimate"] != result2["estimate"] or result1["se"] != result2["se"]


def test_loo_predictive_metric_invalid_metric(centered_eight):
    """Test loo_predictive_metric with invalid metric."""
    idata = centered_eight
    y_obs = idata.observed_data.obs.values

    with pytest.raises(ValueError, match="Invalid metric"):
        loo_predictive_metric(
            data=idata,
            y=y_obs,
            var_name="obs",
            log_lik_var_name="obs",
            metric="invalid_metric",
        )


def test_loo_predictive_metric_dimension_mismatch(centered_eight):
    """Test loo_predictive_metric with mismatched dimensions."""
    idata = centered_eight
    y_obs = idata.observed_data.obs.values

    with pytest.raises(ValueError, match="Length of y"):
        loo_predictive_metric(
            data=idata,
            y=y_obs[:-1],  # Remove one observation to create mismatch
            var_name="obs",
            log_lik_var_name="obs",
            metric="mae",
        )


def test_loo_predictive_metric_missing_group(centered_eight):
    """Test loo_predictive_metric with missing group."""
    idata = centered_eight
    y_obs = idata.observed_data.obs.values

    with pytest.raises(ValueError, match="does not have a"):
        loo_predictive_metric(
            data=idata,
            y=y_obs,
            var_name="obs",
            group="nonexistent_group",  # This group doesn't exist
            log_lik_var_name="obs",
            metric="mae",
        )


def test_mae_function():
    """Test the _mae helper function."""
    y = np.array([1.0, 2.0, 3.0])
    yhat = np.array([1.1, 2.2, 2.7])

    result = _mae(y, yhat)

    assert isinstance(result, dict)
    assert "estimate" in result
    assert "se" in result

    errors = np.abs(y - yhat)
    expected_estimate = np.mean(errors)
    expected_se = np.std(errors, ddof=1) / np.sqrt(len(y))

    assert np.isclose(result["estimate"], expected_estimate)
    assert np.isclose(result["se"], expected_se)

    with pytest.raises(ValueError, match="y and yhat must have the same length"):
        _mae(y, yhat[:-1])


def test_mse_function():
    """Test the _mse helper function."""
    y = np.array([1.0, 2.0, 3.0])
    yhat = np.array([1.1, 2.2, 2.7])

    result = _mse(y, yhat)

    assert isinstance(result, dict)
    assert "estimate" in result
    assert "se" in result

    errors = (y - yhat) ** 2
    expected_estimate = np.mean(errors)
    expected_se = np.std(errors, ddof=1) / np.sqrt(len(y))

    assert np.isclose(result["estimate"], expected_estimate)
    assert np.isclose(result["se"], expected_se)

    with pytest.raises(ValueError, match="y and yhat must have the same length"):
        _mse(y, yhat[:-1])


def test_rmse_function():
    """Test the _rmse helper function."""
    y = np.array([1.0, 2.0, 3.0])
    yhat = np.array([1.1, 2.2, 2.7])

    result = _rmse(y, yhat)

    assert isinstance(result, dict)
    assert "estimate" in result
    assert "se" in result

    mse_result = _mse(y, yhat)
    expected_estimate = np.sqrt(mse_result["estimate"])

    assert np.isclose(result["estimate"], expected_estimate)

    with pytest.raises(ValueError, match="y and yhat must have the same length"):
        _rmse(y, yhat[:-1])


def test_accuracy_function():
    """Test the _accuracy helper function."""
    y = np.array([0, 1, 0, 1, 1])
    yhat = np.array([0.1, 0.9, 0.4, 0.6, 0.3])

    result = _accuracy(y, yhat)

    assert isinstance(result, dict)
    assert "estimate" in result
    assert "se" in result

    yhat_binary = (yhat > 0.5).astype(int)
    acc = (yhat_binary == y).astype(int)
    expected_estimate = np.mean(acc)
    expected_se = np.sqrt(expected_estimate * (1 - expected_estimate) / len(y))

    assert np.isclose(result["estimate"], expected_estimate)
    assert np.isclose(result["se"], expected_se)

    with pytest.raises(ValueError, match="y and yhat must have the same length"):
        _accuracy(y, yhat[:-1])

    with pytest.raises(ValueError, match="y must contain values between 0 and 1"):
        _accuracy(np.array([0, 2, 0]), yhat[:3])

    with pytest.raises(ValueError, match="yhat must contain values between 0 and 1"):
        _accuracy(y, np.array([0.1, 1.1, 0.4, 0.6, 0.3]))


def test_balanced_accuracy_function():
    """Test the _balanced_accuracy helper function."""
    y = np.array([0, 0, 0, 1, 1])
    yhat = np.array([0.1, 0.3, 0.6, 0.7, 0.4])

    result = _balanced_accuracy(y, yhat)

    assert isinstance(result, dict)
    assert "estimate" in result
    assert "se" in result

    yhat_binary = (yhat > 0.5).astype(int)
    mask = y == 0
    tn = np.mean(yhat_binary[mask] == y[mask])
    tp = np.mean(yhat_binary[~mask] == y[~mask])
    expected_estimate = (tp + tn) / 2

    assert np.isclose(result["estimate"], expected_estimate)

    with pytest.raises(ValueError, match="y and yhat must have the same length"):
        _balanced_accuracy(y, yhat[:-1])

    with pytest.raises(ValueError, match="y must contain values between 0 and 1"):
        _balanced_accuracy(np.array([0, 2, 0]), yhat[:3])

    with pytest.raises(ValueError, match="yhat must contain values between 0 and 1"):
        _balanced_accuracy(y, np.array([0.1, 1.1, 0.3, 0.7, 0.4]))
