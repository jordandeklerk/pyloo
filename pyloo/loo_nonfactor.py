# -*- coding: utf-8 -*-
"""Leave-one-out cross-validation (LOO-CV) for non-factorized multivariate normal models."""

import warnings
from typing import Any, Literal

import numpy as np
import xarray as xr
from arviz.data import InferenceData
from arviz.stats.diagnostics import ess
from numpy.linalg import inv

from .base import ISMethod, compute_importance_weights
from .elpd import ELPDData
from .rcparams import rcParams
from .utils import _logsumexp, to_inference_data, wrap_xarray_ufunc

__all__ = ["loo_nonfactor"]


def loo_nonfactor(
    data: InferenceData | Any,
    pointwise: bool | None = None,
    var_name: str | None = None,
    reff: float | None = None,
    scale: str | None = None,
    method: Literal["psis", "sis", "tis"] | ISMethod = "psis",
    mu_var_name: str = "mu",
    cov_var_name: str | None = None,
    prec_var_name: str | None = None,
) -> ELPDData:
    r"""Compute LOO-CV for multivariate normal models using importance sampling.

    Estimates the expected log pointwise predictive density (elpd) for non-factorized
    multivariate normal models using importance sampling leave-one-out cross-validation.

    In non-factorized models, the joint likelihood of the response values :math:`p(y | \theta)`
    is not factorized into observation-specific components, but rather given directly as one
    joint expression. For some models, an analytic factorized formulation is simply not available,
    while for others, a non-factorized form may be preferred for efficiency and numerical stability.

    For multivariate normal models with covariance matrix :math:`C`, the LOO predictive mean
    and standard deviation can be computed efficiently as

    .. math::
        \begin{align}
        \mu_{\tilde{y},-i} &= y_i-\bar{c}_{ii}^{-1} g_i \\
        \sigma_{\tilde{y},-i} &= \sqrt{\bar{c}_{ii}^{-1}}
        \end{align}

    where :math:`g_i` and :math:`\bar{c}_{ii}` are:

    .. math::
        \begin{align}
        g_i &= \left[C^{-1} y\right]_i \\
        \bar{c}_{ii} &= \left[C^{-1}\right]_{ii}
        \end{align}

    Using these results, the log predictive density of the :math:`i`-th observation is computed as

    .. math::
        \log p(y_i | y_{-i},\theta)
        = - \frac{1}{2}\log(2\pi)
        + \frac{1}{2}\log \bar{c}_{ii}
        - \frac{1}{2}\frac{g_i^2}{\bar{c}_{ii}}

    To obtain the leave-one-out predictive density :math:`p(y_i | y_{-i})`, we integrate over
    parameters :math:`\theta` using importance sampling

    .. math::
        p(y_i|y_{-i}) \approx
        \frac{ \sum_{s=1}^S p(y_i|y_{-i},\,\theta^{(s)}) \,w_i^{(s)}}{ \sum_{s=1}^S w_i^{(s)}}

    where :math:`w_i^{(s)}` are importance weights computed using Pareto smoothed importance
    sampling (PSIS) or other importance sampling methods.

    Parameters
    ----------
    data: InferenceData | Any
        Any object that can be converted to an :class:`arviz.InferenceData` object.
        Must contain posterior samples for the mean vector and covariance matrix
        (or precision matrix) of the multivariate normal likelihood, as well as
        the observed data `y`.
    pointwise: bool | None
        If True the pointwise predictive accuracy will be returned. Defaults to
        ``stats.ic_pointwise`` rcParam.
    var_name : str | None
        The name of the variable in observed_data group storing the response data `y`.
        If not provided, the function will try to infer it.
    reff: float | None
        Relative MCMC efficiency, ``ess / n`` i.e. number of effective samples divided by the number
        of actual samples. Computed from trace by default.
    scale: str | None
        Output scale for loo. Available options are:
        - "log": (default) log-score
        - "negative_log": -1 * log-score
        - "deviance": -2 * log-score
    method: Literal['psis', 'sis', 'tis'] | ISMethod
        The importance sampling method to use:
        - 'psis': Pareto Smoothed Importance Sampling (recommended)
        - 'sis': Standard Importance Sampling
        - 'tis': Truncated Importance Sampling
    mu_var_name: str
        Name of the variable in posterior group storing the mean vector.
        Defaults to "mu".
    cov_var_name: str | None
        Name of the variable in posterior group storing the covariance matrix.
        If not provided, the function will look for a variable named "cov".
    prec_var_name: str | None
        Name of the variable in posterior group storing the precision matrix.
        If not provided, the function will look for a variable named "prec".
        Only used if covariance matrix is not found.

    Returns
    -------
    ELPDData object (inherits from :class:`pandas.Series`) with the following row/attributes:
    elpd_loo: approximated expected log pointwise predictive density (elpd)
    se: standard error of the elpd
    p_loo: effective number of parameters
    p_loo_se: standard error of p_loo
    n_samples: number of samples
    n_data_points: number of data points
    warning: bool
        True if using PSIS and the estimated shape parameter of Pareto distribution
        is greater than ``good_k``.
    loo_i: :class:`~xarray.DataArray` with the pointwise predictive accuracy,
            only if pointwise=True
    diagnostic: array of diagnostic values, only if pointwise=True
        - For PSIS: Pareto shape parameter (pareto_k)
        - For SIS/TIS: Effective sample size (ess)
    scale: scale of the elpd
    looic: LOO information criterion (computed as -2 * elpd_loo)
    looic_se: standard error of looic
    good_k: For PSIS method and sample size :math:`S`, threshold computed as :math:`\\min(1 - 1/\\log_{10}(S), 0.7)`

    References
    ----------
    Bürkner, P. C., Gabry, J., & Vehtari, A. (2020). Efficient leave-one-out
    cross-validation for Bayesian non-factorized normal and Student-t models.
    Computational Statistics, 35(4), 1717-1750.
    """
    inference_data = to_inference_data(data)

    if not hasattr(inference_data, "observed_data"):
        raise TypeError("Must be able to extract an observed_data group from data.")
    if not hasattr(inference_data, "posterior"):
        raise TypeError("Must be able to extract a posterior group from data.")

    pointwise = rcParams["stats.ic_pointwise"] if pointwise is None else pointwise
    scale = rcParams["stats.ic_scale"] if scale is None else scale.lower()

    if scale == "deviance":
        scale_value = -2
    elif scale == "log":
        scale_value = 1
    elif scale == "negative_log":
        scale_value = -1
    else:
        raise TypeError('Valid scale values are "deviance", "log", "negative_log"')

    obs_group = inference_data.observed_data
    if var_name is None:
        obs_vars = list(obs_group.data_vars)
        if len(obs_vars) == 1:
            var_name = obs_vars[0]
        elif not obs_vars:
            raise ValueError("No variables found in observed_data group.")
        else:
            raise ValueError(
                f"Multiple variables found in observed_data: {obs_vars}. "
                "Please specify the response variable using `var_name`."
            )
    try:
        y = obs_group[var_name]
    except KeyError:
        raise ValueError(f"Variable '{var_name}' not found in observed_data group.")

    y_name = var_name
    if y.ndim != 1:
        # TODO: Extend to handle multivariate y (N x D)
        raise ValueError(
            f"Observed data '{y_name}' must be 1-dimensional (N,). Found shape"
            f" {y.shape}."
        )
    n_data_points = y.shape[0]

    post_group = inference_data.posterior
    try:
        mu = post_group[mu_var_name]
    except KeyError:
        raise ValueError(f"Posterior variable '{mu_var_name}' not found.")

    cov_matrix = None
    prec_matrix = None

    if cov_var_name:
        try:
            cov_matrix = post_group[cov_var_name]
        except KeyError:
            raise ValueError(f"Posterior variable '{cov_var_name}' not found.")
    elif prec_var_name:
        try:
            prec_matrix = post_group[prec_var_name]
        except KeyError:
            raise ValueError(f"Posterior variable '{prec_var_name}' not found.")
    else:
        try:
            cov_matrix = post_group["cov"]
            cov_var_name = "cov"
        except KeyError:
            try:
                prec_matrix = post_group["prec"]
                prec_var_name = "prec"
            except KeyError:
                pass

    if cov_matrix is None and prec_matrix is None:
        raise ValueError(
            "Could not find posterior samples for covariance ('cov') or precision"
            " ('prec') matrix. Specify the variable name using `cov_var_name` or"
            " `prec_var_name`."
        )
    if cov_matrix is not None and prec_matrix is not None:
        warnings.warn(
            f"Found both covariance ('{cov_var_name}') and precision"
            f" ('{prec_var_name}') matrices. Using covariance matrix '{cov_var_name}'.",
            UserWarning,
            stacklevel=2,
        )
        prec_matrix = None

    mu = mu.stack(__sample__=("chain", "draw"))
    if cov_matrix is not None:
        cov_matrix = cov_matrix.stack(__sample__=("chain", "draw"))
        if cov_matrix.shape[-3:] != (n_data_points, n_data_points, mu.shape[-1]):
            raise ValueError(
                f"Covariance matrix '{cov_var_name}' shape {cov_matrix.shape[:-1]} "
                f"is incompatible with observed data size {n_data_points} "
                f"and number of samples {mu.shape[-1]}."
            )
    if prec_matrix is not None:
        prec_matrix = prec_matrix.stack(__sample__=("chain", "draw"))
        if prec_matrix.shape[-3:] != (n_data_points, n_data_points, mu.shape[-1]):
            raise ValueError(
                f"Precision matrix '{prec_var_name}' shape {prec_matrix.shape[:-1]} "
                f"is incompatible with observed data size {n_data_points} "
                f"and number of samples {mu.shape[-1]}."
            )

    if mu.shape[-2] != n_data_points:
        raise ValueError(
            f"Mean vector '{mu_var_name}' shape {mu.shape[:-1]} is incompatible with "
            f"observed data size {n_data_points}."
        )

    n_samples = mu.shape[-1]

    if reff is None:
        if not hasattr(inference_data, "posterior"):
            raise TypeError("Must be able to extract a posterior group from data.")
        posterior = inference_data.posterior
        n_chains = len(posterior.chain)
        if n_chains == 1:
            reff = 1.0
        else:
            ess_p = ess(posterior, method="mean")
            reff = (
                np.hstack([ess_p[v].values.flatten() for v in ess_p.data_vars]).mean()
                / n_samples
            )

    try:
        method = method if isinstance(method, ISMethod) else ISMethod(method.lower())
    except ValueError:
        valid_methods = ", ".join(m.value for m in ISMethod)
        raise ValueError(f"Invalid method '{method}'. Must be one of: {valid_methods}")

    if method != ISMethod.PSIS:
        method_name = (
            method.value.upper() if isinstance(method, ISMethod) else method.upper()
        )
        warnings.warn(
            f"Using {method_name} for LOO computation. Note that PSIS is the"
            " recommended method as it is typically more efficient and reliable.",
            UserWarning,
            stacklevel=2,
        )

    dims = mu.dims
    coords = mu.coords

    if y_name in coords:
        obs_dim = y_name
    else:
        matching_dims = [d for d, size in coords.items() if size.size == n_data_points]
        if len(matching_dims) == 1:
            obs_dim = matching_dims[0]
        else:
            obs_dim = dims[-2] if len(dims) > 1 else dims[0]
            warnings.warn(
                "Could not reliably determine the observation dimension name. Assuming"
                f" '{obs_dim}'.",
                UserWarning,
                stacklevel=2,
            )

    log_lik_i = np.empty((n_samples, n_data_points))
    log_const = -0.5 * np.log(2 * np.pi)
    y_vals = y.values

    for s in range(n_samples):
        mu_s = mu.values[..., s]
        if cov_matrix is not None:
            cov_s = cov_matrix.values[..., s]
            try:
                cinv_s = inv(cov_s)
            except np.linalg.LinAlgError:
                warnings.warn(
                    f"Sample {s}: Covariance matrix is singular. Skipping.",
                    UserWarning,
                    stacklevel=2,
                )
                log_lik_i[s, :] = -np.inf
                continue
        else:
            cinv_s = prec_matrix.values[..., s]
            try:
                cinv_s = inv(cinv_s)
            except np.linalg.LinAlgError:
                warnings.warn(
                    f"Sample {s}: Precision matrix is singular. Skipping.",
                    UserWarning,
                    stacklevel=2,
                )
                log_lik_i[s, :] = -np.inf
                continue

        g_s = cinv_s @ (y_vals - mu_s)
        cbar_diag_s = np.diag(cinv_s)

        if np.any(cbar_diag_s <= 0):
            warnings.warn(
                f"Sample {s}: Non-positive values found in diagonal of inverse"
                " covariance (c̄_ii). This might indicate numerical instability."
                " Results may be unreliable. Setting log-likelihood to -inf for"
                " affected points.",
                UserWarning,
                stacklevel=2,
            )

            bad_idx = cbar_diag_s <= 0
            cbar_diag_s[bad_idx] = np.finfo(float).eps  # Avoid log(0) or log(-)

            log_lik_i[s, :] = (
                log_const + 0.5 * np.log(cbar_diag_s) - 0.5 * (g_s**2 / cbar_diag_s)
            )
            log_lik_i[s, bad_idx] = -np.inf
        else:
            log_lik_i[s, :] = (
                log_const + 0.5 * np.log(cbar_diag_s) - 0.5 * (g_s**2 / cbar_diag_s)
            )

    log_lik_i_xarray = xr.DataArray(
        log_lik_i.T,  # Transpose to (N, S)
        coords={obs_dim: coords[obs_dim], "__sample__": mu["__sample__"]},
        dims=[obs_dim, "__sample__"],
        name="conditional_log_likelihood",
    )

    if np.any(np.isnan(log_lik_i_xarray.values)):
        warnings.warn(
            "NaN values detected in conditional log-likelihood calculation, likely due"
            " to singular matrices or numerical issues. Replacing with -inf.",
            UserWarning,
            stacklevel=2,
        )
        log_lik_i_xarray = log_lik_i_xarray.fillna(-np.inf)

    if np.any(np.isneginf(log_lik_i_xarray.values)):
        warnings.warn(
            "Negative infinity values detected in conditional log-likelihood. "
            "These points will have zero weight in the final calculation.",
            UserWarning,
            stacklevel=2,
        )

    log_weights, diagnostic = compute_importance_weights(
        -log_lik_i_xarray, method=method, reff=reff
    )
    log_weights += log_lik_i_xarray  # log(w * p) = log(w) + log(p)

    warn_mg = False
    good_k = min(1 - 1 / np.log10(n_samples), 0.7) if n_samples > 1 else 0.7

    if method == ISMethod.PSIS:
        if np.any(diagnostic > good_k):
            n_high_k = np.sum(diagnostic > good_k)
            warnings.warn(
                "Estimated shape parameter of Pareto distribution is greater than"
                f" {good_k:.2f} for {n_high_k} observations. This indicates that"
                " importance sampling may be unreliable. Consider running moment"
                " matching or exact LOO-CV.",
                UserWarning,
                stacklevel=2,
            )
            warn_mg = True
    else:
        min_ess = np.min(diagnostic)
        if min_ess < n_samples * 0.1:
            warnings.warn(
                f"Low effective sample size detected (minimum ESS: {min_ess:.1f})."
                " Importance sampling approximation may be unreliable. Consider using"
                " PSIS.",
                UserWarning,
                stacklevel=2,
            )
            warn_mg = True

    ufunc_kwargs = {"n_dims": 1, "ravel": False}
    xarray_kwargs = {"input_core_dims": [["__sample__"]]}

    loo_lppd_i = scale_value * wrap_xarray_ufunc(
        _logsumexp,
        log_weights,
        ufunc_kwargs=ufunc_kwargs,
        **xarray_kwargs,
    )

    loo_lppd = loo_lppd_i.sum().item()
    loo_lppd_se = (n_data_points * loo_lppd_i.var().item()) ** 0.5

    lppd_i = wrap_xarray_ufunc(
        _logsumexp,
        log_lik_i_xarray,
        func_kwargs={"b_inv": n_samples},
        ufunc_kwargs=ufunc_kwargs,
        **xarray_kwargs,
    )

    lppd = lppd_i.sum().item()
    p_loo = lppd - loo_lppd / scale_value
    p_loo_se = np.sqrt(np.sum(np.var(loo_lppd_i.values)))

    looic = -2 * loo_lppd
    looic_se = 2 * loo_lppd_se

    loo_i_named = loo_lppd_i.rename("loo_i")
    diagnostic_name = "pareto_k" if method == ISMethod.PSIS else "ess"
    diagnostic_named = diagnostic.rename(diagnostic_name)

    result_data = [
        loo_lppd,
        loo_lppd_se,
        p_loo,
        p_loo_se,
        n_samples,
        n_data_points,
        warn_mg,
        scale,
        looic,
        looic_se,
    ]
    result_index = [
        "elpd_loo",
        "se",
        "p_loo",
        "p_loo_se",
        "n_samples",
        "n_data_points",
        "warning",
        "scale",
        "looic",
        "looic_se",
    ]

    if pointwise:
        result_data.insert(result_index.index("scale"), loo_i_named)
        result_index.insert(result_index.index("scale"), "loo_i")

        result_data.append(diagnostic_named)
        result_index.append(diagnostic_name)

        if method == ISMethod.PSIS:
            result_data.append(good_k)
            result_index.append("good_k")

    return ELPDData(data=result_data, index=result_index)
