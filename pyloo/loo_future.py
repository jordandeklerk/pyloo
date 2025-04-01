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

    For a time series of observations :math:`y = (y_1, y_2, \ldots, y_N)`, we often want to
    predict a sequence of :math:`M` future observations - referred to as :math:`M`-step-ahead
    prediction (:math:`M`-SAP). With Leave-Future-Out cross-validation (LFO-CV), we assess
    the predictive performance by computing the predictive densities

    .. math::
        p(y_{i+1:i+M} \,|\, y_{1:i}) = p(y_{i+1}, \ldots, y_{i+M} \,|\, y_{1},...,y_{i})

    for each :math:`i \in \{L, \ldots, N-M\}`, where :math:`L` is the minimum number of
    observations required before making predictions.

    These predictive densities can be computed using the posterior distribution
    :math:`p(\theta \,|\, y_{1:i})`

    .. math::
        p(y_{i+1:i+M} \,|\, y_{1:i}) = \int p(y_{i+1:i+M} \,|\, y_{1:i}, \theta) \,
        p(\theta\,|\,y_{1:i}) \,d\theta.

    With :math:`S` posterior draws :math:`(\theta_{1:i}^{(1)}, \ldots, \theta_{1:i}^{(S)})`
    from :math:`p(\theta\,|\,y_{1:i})`, we can estimate this as

    .. math::
        p(y_{i+1:i+M} \,|\, y_{1:i}) \approx \frac{1}{S}\sum_{s=1}^S p(y_{i+1:i+M} \,|\,
        y_{1:i}, \theta_{1:i}^{(s)}).

    Exact LFO-CV requires refitting the model for each time point :math:`i`, which is
    computationally expensive. The PSIS-LFO-CV algorithm reduces this computational burden.

    The algorithm works as follows:

    1. Refit the model using the first :math:`L` observations, compute exact M-step-ahead
       prediction for :math:`p(y_{L+1:L+M} \,|\, y_{1:L})`, and set :math:`i^* = L` as current refit point.

    2. For each :math:`i > i^*`, approximate :math:`p(y_{i+1:i+M} \,|\, y_{1:i})` via importance sampling

       .. math::
           p(y_{i+1:i+M} \,|\, y_{1:i}) \approx \frac{\sum_{s=1}^S w_i^{(s)} p(y_{i+1:i+M} \,|\,
           y_{1:i}, \theta^{(s)})}{\sum_{s=1}^S w_i^{(s)}},

       where :math:`\theta^{(s)} = \theta^{(s)}_{1:i^*}` are draws from the posterior based on
       the first :math:`i^*` observations, and :math:`w_i^{(s)}` are importance weights.

    3. Compute raw importance ratios:

       .. math::
           r_i^{(s)} \propto \prod_{j \in (i^* + 1):i} p(y_j \,|\, y_{1:(j-1)}, \theta^{(s)})

       and stabilize them using PSIS.

    4. Check if Pareto k diagnostic exceeds threshold :math:`\tau` (default 0.7). If so:
       - Refit the model using observations up to :math:`i`
       - Update :math:`i^* = i`
       - Restart the process from step 2

    5. Continue until reaching observation :math:`i = N - M`.

    Parameters
    ----------
    wrapper : PyMCWrapper
        A PyMCWrapper instance containing the fitted model and inference data.
        The model should be suitable for time series data.
    M : int
        The number of future steps to predict ahead (:math:`M`-step-ahead prediction). Must be >= 1.
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
    ELPDData object (inherits from :class:`pandas.Series`) with the following row/attributes:
    elpd_lfo: approximated expected log pointwise predictive density for LFO-CV
    se: standard error of the elpd_lfo
    p_lfo: effective number of parameters for LFO-CV
    p_lfo_se: standard error of p_lfo
    n_samples: number of posterior samples
    n_pred_points: number of prediction points (N - M - L)
    n_points_eff: number of prediction points with valid ELPD values
    scale: scale of the elpd
    looic: LFO information criterion (-2 * elpd_lfo)
    looic_se: standard error of looic
    M: number of future steps predicted ahead
    L: minimum number of observations required before making predictions
    refit_indices: list of indices where model was refitted
    warning: bool
        True if using PSIS and any Pareto k values remained above threshold after refitting
    lfo_i: :class:`~xarray.DataArray` with the pointwise predictive accuracy,
            only if pointwise=True
    pareto_k: :class:`~xarray.DataArray` with the Pareto shape parameters,
            only if pointwise=True and method='psis'
    k_threshold: threshold for Pareto k values that trigger refitting,
            only if pointwise=True and method='psis'
    good_k: For PSIS method and sample size :math:`S`, threshold computed as :math:`\\min(1 - 1/\\log_{10}(S), 0.7)`,
            only if pointwise=True and method='psis'

    Notes
    -----
    The quality of the approximation depends on the similarity between the posterior distributions
    at different time points. If the time series exhibits strong non-stationarity, more frequent
    refits might be necessary.

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

    valid_methods = {"psis", "sis", "tis"}
    if isinstance(method, str) and method.lower() not in valid_methods:
        raise ValueError(
            f"Invalid method '{method}'. Must be one of: {', '.join(valid_methods)}."
        )
    elif not isinstance(method, (str, ISMethod)):
        raise TypeError(
            f"Method must be a string {valid_methods} or an ISMethod instance."
        )

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

    result_data.append(False)

    is_psis_method = False
    if isinstance(method, ISMethod):
        is_psis_method = method == ISMethod.PSIS
    elif isinstance(method, str):
        try:
            is_psis_method = ISMethod(method.lower()) == ISMethod.PSIS
        except ValueError:
            warnings.warn(
                f"Unknown method '{method}', defaulting to PSIS behavior for"
                " reporting.",
                UserWarning,
                stacklevel=2,
            )
            is_psis_method = method.lower() == "psis"

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

        if is_psis_method:
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

    result = ELPDData(data=result_data, index=result_index)

    if is_psis_method and np.any(pareto_ks[~np.isnan(pareto_ks)] > k_threshold):
        result["warning"] = True
        warnings.warn(
            f"Some Pareto k values remained above the threshold {k_threshold:.2f} after"
            " refitting. This might indicate instability or model misspecification.",
            UserWarning,
            stacklevel=2,
        )

    _log.info(f"LFO-CV finished. Refitted {len(refit_indices)} times.")
    return result
