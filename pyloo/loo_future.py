"""Leave-Future-Out cross-validation (LFO-CV) using importance sampling."""

import copy
import logging
import warnings
from typing import Literal

import numpy as np
import xarray as xr
from arviz.stats.diagnostics import ess

from .base import ISMethod, compute_importance_weights
from .elpd import ELPDData
from .rcparams import rcParams
from .utils import _logsumexp
from .wrapper.pymc import PyMCWrapper

__all__ = ["loo_future"]

_log = logging.getLogger(__name__)


def loo_future(
    wrapper: PyMCWrapper,
    M: int,
    L: int,
    k_threshold: float = 0.7,
    pointwise: bool | None = None,
    reff: float | None = None,
    scale: str | None = None,
    method: Literal["psis", "sis", "tis"] | ISMethod = "psis",
    **kwargs,
) -> ELPDData:
    r"""Compute Leave-Future-Out cross-validation (LFO-CV).

    Estimates the expected log pointwise predictive density (elpd) for M-step-ahead
    predictions using LFO-CV. By default, uses Pareto-smoothed importance sampling (PSIS)
    approximation.

    Parameters
    ----------
    wrapper : PyMCWrapper
        A PyMCWrapper instance containing the fitted model and inference data.
        The model should be suitable for time series data.
    M : int
        The number of future steps to predict ahead (M-step-ahead prediction). Must be >= 1.
    L : int
        The minimum number of observations required before making predictions. Must be >= 0.
    k_threshold : float, optional
        Threshold value for Pareto k values above which model refitting occurs.
        Defaults to 0.7.
    pointwise: bool | None
        If True the pointwise predictive accuracy will be returned. Defaults to
        ``stats.ic_pointwise`` rcParam. Each "point" corresponds to the M-step-ahead
        prediction starting from a given time point `i`.
    reff: float | None
        Relative MCMC efficiency, ``ess / n`` i.e. number of effective samples divided by the number
        of actual samples. Computed from trace by default if available in the wrapper's idata.
        Used for PSIS calculation.
    scale: str | None
        Output scale for loo. Available options are:
        - "log": (default) log-score
        - "negative_log": -1 * log-score
        - "deviance": -2 * log-score
    method: Literal['psis', 'sis', 'tis'] | ISMethod
        The importance sampling method to use for approximating posteriors between refits:
        - 'psis': Pareto Smoothed Importance Sampling (recommended)
        - 'sis': Standard Importance Sampling
        - 'tis': Truncated Importance Sampling
    **kwargs:
        Additional keyword arguments passed to the underlying PyMC sampling function
        during refitting (e.g., `draws`, `tune`, `chains`).

    Returns
    -------
    ELPDData object
        An object containing the ELPD estimate, standard error, Pareto k diagnostics,
        and potentially pointwise values. The indices for pointwise values correspond
        to the starting time point `i` of the M-step-ahead prediction.

    Notes
    -----
    This function implements the PSIS-LFO-CV algorithm described in Bürkner, Gabry, & Vehtari (2020).
    It iteratively computes or approximates M-step-ahead predictions, refitting the model
    only when the PSIS approximation becomes unstable (indicated by high Pareto k values).

    The quality of the approximation depends on the similarity between the posterior distributions
    at different time points. If the time series exhibits strong non-stationarity, more frequent
    refits might be necessary.

    The `PyMCWrapper` must be properly configured for the time series model, ensuring that
    `log_likelihood_i` or similar methods can compute the required conditional likelihoods.
    This might require custom wrapper implementations or modifications for specific model types.

    References
    ----------
    Bürkner P. C., Gabry J., & Vehtari A. (2020). Approximate leave-future-out cross-validation
    for time series models. *Journal of Statistical Computation and Simulation*, 90(14):2499-2523.
    """
    if not isinstance(wrapper, PyMCWrapper):
        raise TypeError("`wrapper` must be a PyMCWrapper instance.")
    if M < 1:
        raise ValueError("M (steps ahead) must be >= 1.")
    if L < 0:
        raise ValueError("L (minimum observations) must be >= 0.")

    var_name = wrapper.get_observed_name()
    obs_shape = wrapper.get_shape(var_name)
    if obs_shape is None or len(obs_shape) == 0:
        raise ValueError("Could not determine shape of observed data.")
    N = obs_shape[0]

    if N <= L + M:
        raise ValueError(
            f"Not enough data points ({N}) for L={L} and M={M}. Need N > L + M."
        )

    required_methods = ["compute_conditional_loglik", "compute_m_step_loglik"]
    missing = wrapper.check_implemented_methods(required_methods)
    if missing:
        raise NotImplementedError(
            "The provided PyMCWrapper is missing required methods for LFO-CV:"
            f" {', '.join(missing)}. These methods need to compute conditional"
            " log-likelihoods suitable for time series."
        )

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

    n_pred_points = N - M - L
    elpds_i = np.full(n_pred_points, np.nan)
    pareto_ks = np.full(n_pred_points, np.nan)
    refit_indices = []
    idata_refit = None
    log_cumulative_ratios = None
    n_samples = 0

    _log.info(f"Initial refit using first {L} observations.")
    indices_past = slice(0, L)
    data_past, _ = wrapper.select_observations(indices_past, var_name=var_name)

    temp_wrapper = copy.deepcopy(wrapper)
    temp_wrapper.set_data({var_name: data_past})

    try:
        idata_refit = temp_wrapper.sample_posterior(**kwargs)
        n_samples = (
            idata_refit.posterior.dims["draw"] * idata_refit.posterior.dims["chain"]
        )
        log_cumulative_ratios = np.zeros(n_samples)
    except Exception as e:
        raise RuntimeError(f"Initial model refit failed: {e}") from e

    i_refit = L
    refit_indices.append(L)

    for i in range(L, N - M):
        _log.info(f"Processing prediction step starting at i = {i}")
        pred_idx = i - L
        k = 0.0

        if i > i_refit:
            _log.debug(
                f"  Calculating IS weights for step i={i} based on refit at {i_refit}"
            )
            idx_is = i
            try:
                log_lik_step_i = wrapper.compute_conditional_loglik(idata_refit, idx_is)
                log_lik_step_i = np.asarray(log_lik_step_i).flatten()
                if log_lik_step_i.shape[0] != n_samples:
                    raise ValueError(
                        f"Expected {n_samples} loglik values, got"
                        f" {log_lik_step_i.shape[0]}"
                    )

                log_cumulative_ratios += log_lik_step_i

                current_reff = reff
                if current_reff is None:
                    ess_ratios = ess(log_cumulative_ratios, method="mean")
                    current_reff = ess_ratios / n_samples if n_samples > 0 else 1.0

                psis_obj = compute_importance_weights(
                    log_cumulative_ratios, method=method, reff=current_reff
                )
                lw, k = psis_obj
                pareto_ks[pred_idx] = k
                _log.debug(f"  Pareto k = {k:.4f}")

            except Exception as e:
                warnings.warn(
                    f"Could not compute IS weights for step i={i}: {e}. Skipping"
                    " approximation.",
                    UserWarning,
                    stacklevel=2,
                )
                k = k_threshold + 1

        needs_refit = (i == L) or (i > i_refit and k > k_threshold)

        if needs_refit:
            _log.info(f"  Refitting model at i = {i} (k={k:.4f} > {k_threshold})")
            if i > L:
                indices_past = slice(0, i + 1)
                data_past, _ = wrapper.select_observations(
                    indices_past, var_name=var_name
                )
                temp_wrapper = copy.deepcopy(wrapper)
                temp_wrapper.set_data({var_name: data_past})
                try:
                    idata_refit = temp_wrapper.sample_posterior(**kwargs)
                    n_samples = (
                        idata_refit.posterior.dims["draw"]
                        * idata_refit.posterior.dims["chain"]
                    )
                    log_cumulative_ratios = np.zeros(n_samples)
                    i_refit = i + 1
                    refit_indices.append(i + 1)
                except Exception as e:
                    raise RuntimeError(f"Model refit failed at i={i}: {e}") from e

            _log.debug("  Computing exact M-step-ahead prediction.")
            indices_future = slice(i + 1, i + M + 1)
            try:
                loglik_m_step_matrix = wrapper.compute_m_step_loglik(
                    idata_refit, indices_future
                )
                loglik_m_step_matrix = np.asarray(loglik_m_step_matrix)
                if (
                    loglik_m_step_matrix.shape[0] != n_samples
                    or loglik_m_step_matrix.shape[1] != M
                ):
                    raise ValueError(
                        f"Expected ({n_samples}, {M}) loglik matrix, got"
                        f" {loglik_m_step_matrix.shape}"
                    )

                loglik_m_step = np.sum(loglik_m_step_matrix, axis=1)
                elpd_i = _logsumexp(loglik_m_step, b_inv=n_samples)
            except Exception as e:
                warnings.warn(
                    f"Could not compute exact M-step prediction at i={i}: {e}. Storing"
                    " NaN.",
                    UserWarning,
                    stacklevel=2,
                )
                elpd_i = np.nan

        else:
            _log.debug(f"  Computing approximate M-step-ahead prediction (k={k:.4f}).")
            indices_future = slice(i + 1, i + M + 1)
            try:
                loglik_m_step_matrix = wrapper.compute_m_step_loglik(
                    idata_refit, indices_future
                )
                loglik_m_step_matrix = np.asarray(loglik_m_step_matrix)
                if (
                    loglik_m_step_matrix.shape[0] != n_samples
                    or loglik_m_step_matrix.shape[1] != M
                ):
                    raise ValueError(
                        f"Expected ({n_samples}, {M}) loglik matrix, got"
                        f" {loglik_m_step_matrix.shape}"
                    )

                loglik_m_step = np.sum(loglik_m_step_matrix, axis=1)
                lw = np.asarray(lw).flatten()
                if lw.shape[0] != n_samples:
                    raise ValueError(
                        f"Shape mismatch: lw {lw.shape}, loglik_m_step"
                        f" {loglik_m_step.shape}"
                    )

                elpd_i = _logsumexp(lw + loglik_m_step)
            except Exception as e:
                warnings.warn(
                    f"Could not compute approximate M-step prediction at i={i}: {e}."
                    " Storing NaN.",
                    UserWarning,
                    stacklevel=2,
                )
                elpd_i = np.nan

        elpds_i[pred_idx] = elpd_i * scale_value
        _log.debug(f"  Stored ELPD contribution for i={i}: {elpds_i[pred_idx]:.4f}")

    valid_elpds = elpds_i[~np.isnan(elpds_i)]
    n_points_eff = len(valid_elpds)

    if n_points_eff == 0:
        warnings.warn(
            "Could not compute ELPD for any prediction step.",
            UserWarning,
            stacklevel=2,
        )
        elpd_lfo = np.nan
        elpd_se = np.nan
    else:
        elpd_lfo = np.sum(valid_elpds)
        elpd_se = (
            (n_points_eff * np.var(valid_elpds)) ** 0.5 if n_points_eff > 1 else 0.0
        )

    p_lfo = np.nan
    p_lfo_se = np.nan

    looic = elpd_lfo * (-2 / scale_value) if scale_value != 0 else np.nan
    looic_se = elpd_se * abs(-2 / scale_value) if scale_value != 0 else np.nan

    result_data = [
        elpd_lfo,
        elpd_se,
        p_lfo,
        p_lfo_se,
        n_samples,
        n_pred_points,
        n_points_eff,
        scale,
        looic,
        looic_se,
        M,
        L,
        refit_indices,
    ]
    result_index = [
        "elpd_lfo",
        "se",
        "p_lfo",
        "p_lfo_se",
        "n_samples",
        "n_pred_points",
        "n_points_eff",
        "scale",
        "looic",
        "looic_se",
        "M",
        "L",
        "refit_indices",
        "warning",
    ]

    if pointwise:
        pointwise_coords = np.arange(L, N - M)
        lfo_i = xr.DataArray(
            elpds_i,
            coords={f"{var_name}_dim_0": pointwise_coords},
            dims=[f"{var_name}_dim_0"],
            name="lfo_i",
        )
        result_data.append(lfo_i)
        result_index.append("lfo_i")

        if method == ISMethod.PSIS:
            pareto_k_da = xr.DataArray(
                pareto_ks,
                coords={f"{var_name}_dim_0": pointwise_coords},
                dims=[f"{var_name}_dim_0"],
                name="pareto_k",
            )
            result_data.append(pareto_k_da)
            result_index.append("pareto_k")

            result_data.append(k_threshold)
            result_index.append("k_threshold")

            good_k = min(1 - 1 / np.log10(n_samples), 0.7) if n_samples > 0 else 0.7
            result_data.append(good_k)
            result_index.append("good_k")

    result_data.append(False)
    result = ELPDData(data=result_data, index=result_index)

    if method == ISMethod.PSIS and np.any(
        pareto_ks[~np.isnan(pareto_ks)] > k_threshold
    ):
        result["warning"] = True
        warnings.warn(
            f"Some Pareto k values remained above the threshold {k_threshold:.2f} after"
            " refitting. This might indicate instability or model misspecification.",
            UserWarning,
            stacklevel=2,
        )

    _log.info(f"LFO-CV finished. Refitted {len(refit_indices)} times.")
    return result
