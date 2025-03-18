"""Efficient approximate leave-one-out cross-validation (LOO) for posterior approximations."""

import warnings
from typing import Any, Literal

import numpy as np
import xarray as xr
from arviz.data import InferenceData
from arviz.stats.diagnostics import ess

from .base import ISMethod, compute_importance_weights
from .elpd import ELPDData
from .psis import psislw
from .rcparams import rcParams
from .utils import _logsumexp, get_log_likelihood, to_inference_data, wrap_xarray_ufunc

__all__ = ["loo_approximate_posterior"]


def loo_approximate_posterior(
    data: InferenceData | Any,
    log_p: np.ndarray,
    log_q: np.ndarray,
    pointwise: bool | None = None,
    var_name: str | None = None,
    reff: float | None = None,
    scale: str | None = None,
    method: Literal["psis", "sis", "tis"] | ISMethod = "psis",
    resample_method: str = "psis",
    seed: int | None = None,
) -> ELPDData:
    """Efficient approximate leave-one-out cross-validation (LOO) for posterior approximations.

    This function computes LOO-CV for posterior approximations where it is possible to compute
    the log density for the posterior approximation. Performs importance resampling
    for better numerical stability.

    Parameters
    ----------
    data : InferenceData or Any
        Any object that can be converted to an :class:`arviz.InferenceData` object.
        Refer to documentation of :func:`arviz.convert_to_dataset` for details.
    log_p : array-like
        The log-posterior (target) evaluated at S samples from the proposal distribution (q).
        A vector of length S where S is the number of samples.
    log_q : array-like
        The log-density (proposal) evaluated at S samples from the proposal distribution (q).
        A vector of length S.
    pointwise : bool, optional
        If True, returns pointwise values. Defaults to rcParams["stats.ic_pointwise"].
    var_name : str, optional
        The name of the variable in log_likelihood groups storing the pointwise log
        likelihood data to use for loo computation.
    reff : float, optional
        Relative MCMC efficiency, ``ess / n`` i.e. number of effective samples divided by the number
        of actual samples. Computed from trace by default.
    scale : str, optional
        Output scale for LOO. Available options are:
        - "log": (default) log-score
        - "negative_log": -1 * log-score
        - "deviance": -2 * log-score
    method : {'psis', 'sis', 'tis'}, default 'psis'
        The importance sampling method to use for LOO computation:
        - 'psis': Pareto Smoothed Importance Sampling (recommended)
        - 'sis': Standard Importance Sampling
        - 'tis': Truncated Importance Sampling
    resample_method : str, default "psis"
        Method to use for importance resampling:
        - "psis": Pareto Smoothed Importance Sampling (without replacement)
        - "psir": Pareto Smoothed Importance Resampling (with replacement)
        - "sis": Standard Importance Sampling (no smoothing)
    seed : int, optional
        Random seed for reproducible resampling.

    Returns
    -------
    ELPDData object (inherits from :class:`pandas.Series`) with the following row/attributes:
        elpd_loo: approximated expected log pointwise predictive density (elpd)
        se: standard error of the elpd
        p_loo: effective number of parameters
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
        good_k: For PSIS method and sample size S, threshold computed as min(1 - 1/log10(S), 0.7)
        approximate_posterior: dictionary with log_p and log_g values
        ess_ratio: effective sample size ratio from importance resampling
        resample_pareto_k: Pareto k diagnostic values from resampling (if using PSIS)
        method: The method used for computation ("loo_approximate_posterior")

        The returned object has a custom print method that overrides pd.Series method.

    See Also
    --------
    loo_subsample : Subsampled LOO-CV computation
    reloo : Exact LOO-CV computation for PyMC models
    loo_moment_match : LOO-CV computation using moment matching
    loo_kfold : K-fold cross-validation
    waic : Compute WAIC

    Examples
    --------
    Calculate LOO-CV for a variational inference using a Laplace approximation

    .. code-block:: python

        import pyloo as pl
        from pyloo.wrapper import Laplace

        wrapper = Laplace(model)
        result = wrapper.fit()

        # Get target and proposal log densities
        log_p = wrapper.compute_logp()
        log_g = wrapper.compute_logq()

        loo_result = pl.loo_approximate_posterior(
            result.idata,
            log_p,
            log_g,
            pointwise=True
        )
    """
    inference_data = to_inference_data(data)
    log_likelihood = get_log_likelihood(inference_data, var_name=var_name)
    pointwise = rcParams["stats.ic_pointwise"] if pointwise is None else pointwise

    log_likelihood = log_likelihood.stack(__sample__=("chain", "draw"))
    shape = log_likelihood.shape
    n_samples = shape[-1]
    n_data_points = np.prod(shape[:-1])
    scale = rcParams["stats.ic_scale"] if scale is None else scale.lower()

    if scale == "deviance":
        scale_value = -2
    elif scale == "log":
        scale_value = 1
    elif scale == "negative_log":
        scale_value = -1
    else:
        raise TypeError('Valid scale values are "deviance", "log", "negative_log"')

    if len(log_p) != len(log_q):
        raise ValueError(
            f"log_p and log_q must have the same length, got {len(log_p)} and"
            f" {len(log_q)}"
        )

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

    has_nan = np.any(np.isnan(log_likelihood.values))

    if has_nan:
        warnings.warn(
            "NaN values detected in log-likelihood. These will be ignored in the LOO"
            " calculation.",
            UserWarning,
            stacklevel=2,
        )
        log_likelihood = log_likelihood.where(~np.isnan(log_likelihood), -1e10)

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

    log_likelihood_np = log_likelihood.values
    n_obs = np.prod(log_likelihood_np.shape[:-1])
    log_likelihood_matrix = log_likelihood_np.reshape(n_obs, n_samples).T

    try:
        indices = importance_resample(
            log_p=log_p,
            log_q=log_q,
            method=resample_method,
            seed=seed,
        )

        log_likelihood_matrix = log_likelihood_matrix[indices, :]
        log_ratios_matrix = -log_likelihood_matrix.copy()

        for i in range(log_ratios_matrix.shape[1]):
            max_ratio = np.max(log_ratios_matrix[:, i])
            log_ratios_matrix[:, i] -= max_ratio

    except Exception as e:
        warnings.warn(
            f"Importance resampling failed: {str(e)}. Falling back to original"
            " samples.",
            UserWarning,
            stacklevel=2,
        )

    log_ratios_xr = xr.DataArray(
        log_ratios_matrix.T.reshape(log_likelihood_np.shape),
        dims=log_likelihood.dims,
        coords=log_likelihood.coords,
    )

    log_weights, diagnostic = compute_importance_weights(
        log_ratios_xr, method=method, reff=reff
    )

    resampled_log_likelihood = xr.DataArray(
        log_likelihood_matrix.T.reshape(log_likelihood_np.shape),
        dims=log_likelihood.dims,
        coords=log_likelihood.coords,
    )
    log_weights += resampled_log_likelihood

    warn_mg = False
    good_k = min(1 - 1 / np.log10(n_samples), 0.7)

    if method == ISMethod.PSIS:
        if np.any(diagnostic > good_k):
            n_high_k = np.sum(diagnostic > good_k)
            warnings.warn(
                "Estimated shape parameter of Pareto distribution is greater than"
                f" {good_k:.2f} for {n_high_k} observations. This indicates that"
                " importance sampling may be unreliable because the marginal posterior"
                " and LOO posterior are very different.",
                UserWarning,
                stacklevel=2,
            )
            warn_mg = True
    else:
        min_ess = np.min(diagnostic)
        if min_ess < n_samples * 0.1:
            warnings.warn(
                f"Low effective sample size detected (minimum ESS: {min_ess:.1f}). This"
                " indicates that the importance sampling approximation may be"
                " unreliable. Consider using PSIS which is more robust to such cases.",
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

    loo_lppd = loo_lppd_i.values.sum()
    loo_lppd_se = (n_data_points * np.var(loo_lppd_i.values)) ** 0.5

    lppd = np.sum(
        wrap_xarray_ufunc(
            _logsumexp,
            resampled_log_likelihood,
            func_kwargs={"b_inv": n_samples},
            ufunc_kwargs=ufunc_kwargs,
            **xarray_kwargs,
        ).values
    )

    p_loo = lppd - loo_lppd / scale_value
    looic = -2 * loo_lppd
    looic_se = 2 * loo_lppd_se

    result_data: list[Any] = []
    result_index: list[str] = []

    if not pointwise:
        result_data = [
            loo_lppd,
            loo_lppd_se,
            p_loo,
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
            "n_samples",
            "n_data_points",
            "warning",
            "scale",
            "looic",
            "looic_se",
        ]

        if method == ISMethod.PSIS:
            result_data.append(good_k)
            result_index.append("good_k")

        result = ELPDData(data=result_data, index=result_index)
        result.approximate_posterior = {"log_p": log_p, "log_q": log_q}

        return result

    if np.allclose(loo_lppd_i, loo_lppd_i[0]):
        warnings.warn(
            "The point-wise LOO is the same with the sum LOO, please double check "
            "the Observed RV in your model to make sure it returns element-wise logp.",
            stacklevel=2,
        )

    result_data = [
        loo_lppd,
        loo_lppd_se,
        p_loo,
        n_samples,
        n_data_points,
        warn_mg,
        loo_lppd_i.rename("loo_i"),
        scale,
        looic,
        looic_se,
    ]
    result_index = [
        "elpd_loo",
        "se",
        "p_loo",
        "n_samples",
        "n_data_points",
        "warning",
        "loo_i",
        "scale",
        "looic",
        "looic_se",
    ]

    if method == ISMethod.PSIS:
        result_data.append(diagnostic)
        result_index.append("pareto_k")
        result_data.append(good_k)
        result_index.append("good_k")
    else:
        result_data.append(diagnostic)
        result_index.append("ess")

    result = ELPDData(data=result_data, index=result_index)
    result.approximate_posterior = {"log_p": log_p, "log_q": log_q}

    return result


def importance_resample(
    log_p: np.ndarray,
    log_q: np.ndarray,
    method: str = "psis",
    seed: int | None = None,
) -> np.ndarray:
    """
    Importance resampling from approximate posterior samples.

    Parameters
    ----------
    log_p : np.ndarray
        Log density under target distribution
    log_q : np.ndarray
        Log density under proposal distribution
    method : str, default "psis"
        Method to use for importance sampling:
        - "psis": Pareto Smoothed Importance Sampling
        - "psir": Pareto Smoothed Importance Resampling (with replacement)
        - "sis": Standard Importance Sampling (no smoothing)
    seed : int | None, optional
        Random seed for reproducibility

    Returns
    -------
    np.ndarray
        Indices of resampled points
    """
    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
    draws = len(log_p)
    logiw = log_p - log_q

    # Handle non-finite importance weights
    valid_idx = np.isfinite(logiw)
    if not np.all(valid_idx):
        warnings.warn(
            f"Found {np.sum(~valid_idx)} non-finite importance weights. These will be"
            " excluded.",
            UserWarning,
            stacklevel=2,
        )
        if np.sum(valid_idx) == 0:
            raise ValueError("No valid importance weights found.")

        logiw = logiw[valid_idx]
    else:
        valid_idx = slice(None)

    replace = method == "psir"

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning, message="overflow encountered in exp"
        )
        if method in ["psis", "psir"]:
            try:
                logiw_smoothed, _ = psislw(logiw)
                logiw = logiw_smoothed
            except Exception as e:
                warnings.warn(
                    f"PSIS smoothing failed: {str(e)}.",
                    UserWarning,
                    stacklevel=2,
                )
        else:
            logiw = logiw - _logsumexp(logiw)

    p = np.exp(logiw)
    p = p / np.sum(p)

    # Resampling
    try:
        indices_subset = rng.choice(draws, size=draws, replace=replace, p=p)
    except ValueError as e:
        # Handle insufficient non-zero weights
        if "Fewer non-zero entries in p than size" in str(e) and not replace:
            warnings.warn(
                "Not enough non-zero weights for sampling without replacement. "
                "Switching to sampling with replacement.",
                UserWarning,
                stacklevel=2,
            )
            indices_subset = rng.choice(draws, size=draws, replace=True, p=p)
        else:
            warnings.warn(
                f"Resampling failed: {str(e)}. Using random indices.",
                UserWarning,
                stacklevel=2,
            )
            indices_subset = rng.choice(draws, size=draws)

    # Map back to original indices
    if isinstance(valid_idx, np.ndarray):
        orig_indices = np.where(valid_idx)[0]
        indices = orig_indices[indices_subset]
    else:
        indices = indices_subset

    return indices
