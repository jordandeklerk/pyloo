"""Leave-one-out predictive metrics for model evaluation."""

from typing import Any, Literal, TypedDict

import numpy as np
from arviz import InferenceData

from .e_loo import e_loo
from .psis import psislw
from .utils import to_inference_data

__all__ = ["loo_predictive_metric"]


class MetricResult(TypedDict):
    """Result of a predictive metric calculation."""

    estimate: float
    se: float


def loo_predictive_metric(
    data: InferenceData | Any,
    y: np.ndarray,
    var_name: str | None = None,
    group: str = "posterior_predictive",
    log_lik_group: str = "log_likelihood",
    log_lik_var_name: str | None = None,
    metric: Literal["mae", "mse", "rmse", "acc", "balanced_acc"] = "mae",
    r_eff: float | np.ndarray = 1.0,
    **kwargs,
) -> MetricResult:
    """Estimate leave-one-out predictive performance metrics.

    This function computes estimates of leave-one-out predictive metrics given a set of
    predictions and observations. Currently supported metrics are mean absolute error,
    mean squared error and root mean squared error for continuous predictions, and
    accuracy and balanced accuracy for binary classification. Predictions are passed
    to the e_loo function, so this function assumes that the PSIS approximation is
    working well.

    Parameters
    ----------
    data : InferenceData or convertible object
        An ArviZ InferenceData object or any object that can be converted to
        InferenceData containing posterior predictive samples and log likelihood values.
    y : array-like
        A numeric vector of observations with shape (n_observations,).
    var_name : str, optional
        Name of the variable in the posterior predictive group to compute metrics for.
        If None and there is only one variable, that variable will be used.
    group : str, default "posterior_predictive"
        Name of the InferenceData group containing the predictions.
    log_lik_group : str, default "log_likelihood"
        Name of the InferenceData group containing the log likelihood values.
    log_lik_var_name : str, optional
        Name of the variable in the log likelihood group to use.
        If None and there is only one variable, that variable will be used.
    metric : str, default "mae"
        The type of predictive metric to be used. Currently supported options are:
        - "mae": Mean absolute error
        - "mse": Mean squared error
        - "rmse": Root mean squared error (square root of MSE)
        - "acc": Accuracy (proportion of predictions indicating the correct outcome)
        - "balanced_acc": Balanced accuracy (average of true positive and true negative rates)
    r_eff : float or array-like, default 1.0
        Relative effective sample size estimates. If a scalar, the same value is used
        for all observations. If an array, should contain one element per observation.
    **kwargs
        Additional arguments passed to e_loo.

    Returns
    -------
    dict
        A dictionary with the following components:
        - "estimate": Estimate of the given metric
        - "se": Standard error of the estimate

    Notes
    -----
    For binary classification metrics ("acc" and "balanced_acc"), the predictions and
    observations should be in the range [0, 1], where predictions are probabilities
    and observations are binary (0 or 1).

    Examples
    --------
    Compute leave-one-out predictive metrics using the centered_eight dataset from ArviZ:

    .. code-block:: python

        import arviz as az
        import pyloo as pl

        idata = az.load_arviz_data("centered_eight")
        y_obs = idata.observed_data.obs.values

        result_mae = pl.loo_predictive_metric(
            data=idata,
            y=y_obs,
            var_name="obs",  # Variable name in posterior_predictive group
            log_lik_var_name="obs",  # Variable name in log_likelihood group
            metric="mae"
        )

        result_mse = pl.loo_predictive_metric(
            data=idata,
            y=y_obs,
            var_name="obs",
            log_lik_var_name="obs",
            metric="mse"
        )

        result_rmse = pl.loo_predictive_metric(
            data=idata,
            y=y_obs,
            var_name="obs",
            log_lik_var_name="obs",
            metric="rmse"
        )

    For MCMC samples with autocorrelation, we can provide the relative effective sample size:

    .. code-block:: python

        import arviz as az
        import pyloo as pl

        idata = az.load_arviz_data("centered_eight")
        y_obs = idata.observed_data.obs.values

        log_likelihood = idata.log_likelihood.stack(__sample__=("chain", "draw"))
        shape = log_likelihood.__sample__.shape
        n_samples = shape[-1]

        posterior = idata.posterior
        n_chains = len(posterior.chain)
        n_draws = len(posterior.draw)

        ess_p = az.ess(posterior, method="mean")
        reff = (
            np.hstack([ess_p[v].values.flatten() for v in ess_p.data_vars]).mean()
            / n_samples
        )

        result_with_reff = loo_predictive_metric(
            data=idata,
            y=y_obs,
            var_name="obs",
            log_lik_var_name="obs",
            metric="mae",
            r_eff=reff
        )
    """
    y = np.asarray(y).flatten()

    idata = to_inference_data(data)

    if not hasattr(idata, group):
        raise ValueError(f"InferenceData object does not have a {group} group")
    if not hasattr(idata, log_lik_group):
        raise ValueError(f"InferenceData object does not have a {log_lik_group} group")

    pp_group = getattr(idata, group)

    if log_lik_var_name is None:
        ll_var_names = list(getattr(idata, log_lik_group).data_vars)
        if len(ll_var_names) == 1:
            log_lik_var_name = ll_var_names[0]
        else:
            raise ValueError(
                f"Multiple variables found in {log_lik_group} group. Please specify"
                f" log_lik_var_name from: {ll_var_names}"
            )
    elif log_lik_var_name not in getattr(idata, log_lik_group).data_vars:
        raise ValueError(
            f"Variable '{log_lik_var_name}' not found in {log_lik_group} group."
            f" Available variables: {list(getattr(idata, log_lik_group).data_vars)}"
        )

    x = pp_group[var_name]
    log_lik = getattr(idata, log_lik_group)[log_lik_var_name]

    if "__sample__" not in x.dims and "chain" in x.dims and "draw" in x.dims:
        x = x.stack(__sample__=("chain", "draw"))

    if (
        "__sample__" not in log_lik.dims
        and "chain" in log_lik.dims
        and "draw" in log_lik.dims
    ):
        log_lik = log_lik.stack(__sample__=("chain", "draw"))

    obs_dims = [dim for dim in x.dims if dim != "__sample__"]
    n_obs = x.sizes[obs_dims[0]] if obs_dims else 1

    if len(y) != n_obs:
        raise ValueError(
            f"Length of y ({len(y)}) must match the number of observations in x"
            f" ({n_obs})"
        )

    if metric not in ["mae", "mse", "rmse", "acc", "balanced_acc"]:
        raise ValueError(
            f"Invalid metric: {metric}. Must be one of: 'mae', 'mse', 'rmse', 'acc',"
            " 'balanced_acc'"
        )

    log_weights, _ = psislw(-log_lik, reff=r_eff)

    loo_result = e_loo(
        idata,
        var_name=var_name,
        group=group,
        log_weights=log_weights,
        log_ratios=-log_lik,
        type="mean",
        **kwargs,
    )

    pred_loo = loo_result.value.values

    if metric == "mae":
        return _mae(y, pred_loo)
    elif metric == "mse":
        return _mse(y, pred_loo)
    elif metric == "rmse":
        return _rmse(y, pred_loo)
    elif metric == "acc":
        return _accuracy(y, pred_loo)
    else:  # metric == "balanced_acc"
        return _balanced_accuracy(y, pred_loo)


def _mae(y: np.ndarray, yhat: np.ndarray) -> MetricResult:
    """Mean absolute error.

    Parameters
    ----------
    y : np.ndarray
        A vector of observed values
    yhat : np.ndarray
        A vector of predictions

    Returns
    -------
    dict
        Dictionary with estimate and standard error
    """
    n = _validate_lengths(y, yhat)
    e = np.abs(y - yhat)

    return {"estimate": np.mean(e), "se": np.std(e, ddof=1) / np.sqrt(n)}


def _mse(y: np.ndarray, yhat: np.ndarray) -> MetricResult:
    """Mean squared error.

    Parameters
    ----------
    y : np.ndarray
        A vector of observed values
    yhat : np.ndarray
        A vector of predictions

    Returns
    -------
    dict
        Dictionary with estimate and standard error
    """
    n = _validate_lengths(y, yhat)
    e = (y - yhat) ** 2

    return {"estimate": np.mean(e), "se": np.std(e, ddof=1) / np.sqrt(n)}


def _rmse(y: np.ndarray, yhat: np.ndarray) -> MetricResult:
    """Root mean squared error.

    Parameters
    ----------
    y : np.ndarray
        A vector of observed values
    yhat : np.ndarray
        A vector of predictions

    Returns
    -------
    dict
        Dictionary with estimate and standard error
    """
    mse_result = _mse(y, yhat)
    mean_mse = mse_result["estimate"]
    var_mse = mse_result["se"] ** 2

    # Variance of RMSE using first-order Taylor approximation
    var_rmse = var_mse / mean_mse / 4

    return {"estimate": np.sqrt(mean_mse), "se": np.sqrt(var_rmse)}


def _accuracy(y: np.ndarray, yhat: np.ndarray) -> MetricResult:
    """Classification accuracy.

    Parameters
    ----------
    y : np.ndarray
        A vector of observed values (binary: 0 or 1)
    yhat : np.ndarray
        A vector of predictions (probabilities between 0 and 1)

    Returns
    -------
    dict
        Dictionary with estimate and standard error
    """
    n = _validate_lengths(y, yhat)
    _validate_binary_inputs(y, yhat)

    yhat_binary = (yhat > 0.5).astype(int)
    acc = (yhat_binary == y).astype(int)
    est = np.mean(acc)

    # Standard error for proportion
    se = np.sqrt(est * (1 - est) / n)

    return {"estimate": est, "se": se}


def _balanced_accuracy(y: np.ndarray, yhat: np.ndarray) -> MetricResult:
    """Balanced classification accuracy.

    Parameters
    ----------
    y : np.ndarray
        A vector of observed values (binary: 0 or 1)
    yhat : np.ndarray
        A vector of predictions (probabilities between 0 and 1)

    Returns
    -------
    dict
        Dictionary with estimate and standard error
    """
    n = _validate_lengths(y, yhat)
    _validate_binary_inputs(y, yhat)

    yhat_binary = (yhat > 0.5).astype(int)
    mask = y == 0

    tn = np.mean(yhat_binary[mask] == y[mask])
    tp = np.mean(yhat_binary[~mask] == y[~mask])

    bls_acc = (tp + tn) / 2
    bls_acc_var = (tp * (1 - tp) + tn * (1 - tn)) / 4

    return {"estimate": bls_acc, "se": np.sqrt(bls_acc_var / n)}


def _validate_lengths(y: np.ndarray, yhat: np.ndarray) -> int:
    """Validate that y and yhat have the same length."""
    if len(y) != len(yhat):
        raise ValueError("y and yhat must have the same length")
    return len(y)


def _validate_binary_inputs(y: np.ndarray, yhat: np.ndarray) -> None:
    """Validate inputs for binary classification metrics."""
    if not np.all((y <= 1) & (y >= 0)):
        raise ValueError("y must contain values between 0 and 1")

    if not np.all((yhat <= 1) & (yhat >= 0)):
        raise ValueError("yhat must contain values between 0 and 1")
