"""Efficient approximate leave-one-out cross-validation (LOO-CV) using subsampling."""

import warnings
from typing import Any, cast

import numpy as np
import xarray as xr
from arviz.data import InferenceData
from arviz.stats.diagnostics import ess

from .approximations import (
    LPDApproximation,
    PLPDApproximation,
    SISApproximation,
    TISApproximation,
)
from .base import ISMethod, compute_importance_weights
from .constants import EstimatorMethod, LooApproximationMethod
from .elpd import ELPDData
from .estimators import SubsampleIndices, get_estimator, subsample_indices
from .estimators.hansen_hurwitz import compute_sampling_probabilities
from .loo import loo
from .loo_approximate_posterior import importance_resample
from .rcparams import rcParams
from .utils import _logsumexp, get_log_likelihood, to_inference_data, wrap_xarray_ufunc

APPROXIMATION_METHODS: dict[LooApproximationMethod, Any] = {
    LooApproximationMethod.LPD: lambda: LPDApproximation(),
    LooApproximationMethod.TIS: lambda: TISApproximation(),
    LooApproximationMethod.SIS: lambda: SISApproximation(),
}

__all__ = ["loo_subsample", "update_subsample"]


def loo_subsample(
    data: InferenceData | dict[str, Any],
    observations: int | np.ndarray | None = 100,
    loo_approximation: str = "plpd",
    estimator: str = "diff_srs",
    loo_approximation_draws: int | None = None,
    pointwise: bool | None = None,
    var_name: str | None = None,
    reff: float | None = None,
    scale: str | None = None,
    log_p: np.ndarray | None = None,
    log_q: np.ndarray | None = None,
    resample_method: str = "psis",
    seed: int | None = None,
) -> ELPDData:
    """Compute approximate LOO-CV using subsampling.

    Parameters
    ----------
    data : Union[InferenceData, Dict[str, Any]]
        Any object that can be converted to an InferenceData object
        containing the log likelihood data.
    observations : Optional[Union[int, np.ndarray]], default 400
        The subsample observations to use:
        - An integer specifying the number of observations to subsample
        - An array of integers providing specific indices to use
        - None to use all observations (equivalent to standard LOO)
    loo_approximation : str, default "plpd"
        The type of approximation to use for the loo_i values:
        - "plpd": Point estimate based approximation (default)
        - "lpd": Log predictive density
        - "tis": Truncated importance sampling
        - "sis": Standard importance sampling
    estimator : str, default "diff_srs"
        The estimation method to use:
        - "diff_srs": Difference estimator with simple random sampling (default)
        - "hh_pps": Hansen-Hurwitz estimator
        - "srs": Simple random sampling
    loo_approximation_draws : Optional[int], default None
        The number of posterior draws to use for approximation methods that
        require integration over the posterior.
    pointwise : Optional[bool], default None
        If True, returns pointwise values. Defaults to rcParams["stats.ic_pointwise"].
    var_name : Optional[str], default None
        Name of the variable in log_likelihood groups storing the pointwise log
        likelihood data.
    reff : Optional[float], default None
        Relative MCMC efficiency, ess / n. If not provided, computed from trace.
    scale : Optional[str], default None
        Output scale for LOO. Available options are:
        - "log": (default) log-score
        - "negative_log": -1 * log-score
        - "deviance": -2 * log-score
    log_p : Optional[np.ndarray], default None
        The log-posterior (target) evaluated at S samples from the proposal distribution (q).
        A vector of length S where S is the number of samples. If provided along with log_q,
        importance resampling will be performed for better numerical stability.
    log_q : Optional[np.ndarray], default None
        The log-density (proposal) evaluated at S samples from the proposal distribution (q).
        A vector of length S. Must be provided if log_p is provided.
    resample_method : str, default "psis"
        Method to use for importance resampling when log_p and log_q are provided:
        - "psis": Pareto Smoothed Importance Sampling (without replacement)
        - "psir": Pareto Smoothed Importance Resampling (with replacement)
        - "sis": Standard Importance Sampling (no smoothing)
    seed : Optional[int], default None
        Random seed for reproducible resampling. Only used when log_p and log_q are provided.

    Returns
    -------
    ELPDData object (inherits from :class:`pandas.Series`) with the following row/attributes:
    elpd_loo: approximated expected log pointwise predictive density (elpd)
    se: standard error of the elpd (includes both approximation and sampling uncertainty)
    subsampling_SE: standard error from subsampling uncertainty only
    p_loo: effective number of parameters
    n_samples: number of samples
    n_data_points: number of data points
    warning: bool
        True if using PSIS and the estimated shape parameter of Pareto distribution
        is greater than good_k
    loo_i: :class:`~xarray.DataArray` with the pointwise predictive accuracy,
        only if pointwise=True
    diagnostic: array of diagnostic values, only if pointwise=True
    scale: scale of the elpd
    good_k: threshold computed as min(1 - 1/log10(S), 0.7)

    The returned object has a custom print method that overrides pd.Series method.

    Notes
    -----
    This implementation follows the methodology described in:
    Magnusson et al. (2019) https://arxiv.org/abs/1902.06504

    Examples
    --------
    First, let's compute approximate LOO-CV using subsampling. We'll load a sample dataset,
    specify the number of observations to subsample, and use the point estimate based approximation

    .. code-block:: python

        import arviz as az
        from pyloo import loo_subsample

        data = az.load_arviz_data("centered_eight")

        # Subsample 4 out of 8 observations using point estimate based approximation
        result = loo_subsample(
            data,
            observations=4,
            loo_approximation="plpd",
        )

    Once we have initial results, we can update them with more observations

    .. code-block:: python

        from pyloo import update_subsample

        updated = update_subsample(result, observations=6)

    We can also update with specific observation indices

    .. code-block:: python

        import numpy as np

        indices = np.array([0, 2, 4, 6])
        updated_specific = update_subsample(result, observations=indices)

    See Also
    --------
    loo : Standard LOO-CV computation
    loo_moment_match : Leave-one-out cross-validation with moment matching
    loo_kfold : K-fold cross-validation
    loo_approximate_posterior : LOO-CV for posterior approximations
    waic : Compute WAIC
    """
    inference_data = to_inference_data(data)
    log_likelihood = get_log_likelihood(inference_data, var_name=var_name)
    pointwise = rcParams["stats.ic_pointwise"] if pointwise is None else pointwise

    try:
        loo_approx_method = LooApproximationMethod(loo_approximation.lower())
    except ValueError:
        raise ValueError(
            f"Invalid loo_approximation '{loo_approximation}'. "
            f"Must be one of: {', '.join(m.value for m in LooApproximationMethod)}"
        )

    try:
        est_method = EstimatorMethod(estimator.lower())
    except ValueError:
        raise ValueError(
            f"Invalid estimator '{estimator}'. "
            f"Must be one of: {', '.join(m.value for m in EstimatorMethod)}"
        )

    log_likelihood = log_likelihood.stack(__sample__=("chain", "draw"))
    shape = log_likelihood.shape
    n_samples = shape[-1]

    obs_dims = [dim for dim in log_likelihood.dims if dim != "__sample__"]
    n_data_points = np.prod([log_likelihood.sizes[dim] for dim in obs_dims])

    scale = rcParams["stats.ic_scale"] if scale is None else scale.lower()
    if scale == "deviance":
        scale_value = -2
    elif scale == "log":
        scale_value = 1
    elif scale == "negative_log":
        scale_value = -1
    else:
        raise TypeError('Valid scale values are "deviance", "log", "negative_log"')

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

    if observations is None:
        return loo(
            data=data,
            pointwise=pointwise,
            var_name=var_name,
            reff=reff,
            scale=scale,
        )

    if isinstance(observations, int):
        if observations <= 0 or observations > n_data_points:
            raise ValueError(
                f"Number of observations must be between 1 and {n_data_points}, "
                f"got {observations}"
            )
    elif isinstance(observations, np.ndarray):
        if observations.dtype != np.integer:
            raise TypeError("observations array must contain integers")
        if observations.min() < 0 or observations.max() >= n_data_points:
            raise ValueError(
                f"Observation indices must be between 0 and {n_data_points - 1}, "
                f"got range [{observations.min()}, {observations.max()}]"
            )
    else:
        raise TypeError(
            "observations must be None, an integer, or an array of integers"
        )

    if loo_approx_method == LooApproximationMethod.PLPD:
        if hasattr(inference_data, "posterior"):
            approximator = PLPDApproximation(posterior=inference_data.posterior)
        else:
            # Fall back to LPD approximation if posterior not available
            warnings.warn(
                "PLPD approximation requested but posterior draws not available. "
                "Falling back to LPD approximation.",
                UserWarning,
                stacklevel=2,
            )
            approximator = PLPDApproximation(LPDApproximation())
    else:
        approximator = cast(
            PLPDApproximation, APPROXIMATION_METHODS[loo_approx_method]()
        )

    elpd_loo_approx = approximator.compute_approximation(
        log_likelihood=log_likelihood,
        n_draws=loo_approximation_draws,
    )

    if isinstance(observations, np.ndarray):
        indices = SubsampleIndices(idx=observations, m_i=np.ones_like(observations))
    else:
        indices = subsample_indices(
            estimator=est_method.value,
            elpd_loo_approximation=elpd_loo_approx,
            observations=observations,
        )

    # For multidimensional data, we need to handle the indices differently
    if len(obs_dims) > 1:
        flat_idx = indices.idx
        idx_arrays = np.unravel_index(
            flat_idx, [log_likelihood.sizes[dim] for dim in obs_dims]
        )
        idx_dict = dict(zip(obs_dims, idx_arrays))
    else:
        idx_dict = {obs_dims[0]: indices.idx}

    if "__sample__" in log_likelihood.dims:
        idx_dict["__sample__"] = slice(None)

    log_likelihood_sample = log_likelihood.isel(idx_dict)

    # Apply posterior correction
    if log_p is not None and log_q is not None:
        if len(log_p) != len(log_q):
            raise ValueError(
                f"log_p and log_q must have the same length, got {len(log_p)} and"
                f" {len(log_q)}"
            )

        try:
            resample_indices = importance_resample(
                log_p=log_p,
                log_q=log_q,
                method=resample_method,
                seed=seed,
            )

            log_likelihood_np = log_likelihood_sample.values

            orig_shape = log_likelihood_np.shape
            n_obs = np.prod(orig_shape[:-1])
            log_likelihood_matrix = log_likelihood_np.reshape(n_obs, n_samples).T

            resampled_log_likelihood = log_likelihood_matrix[resample_indices, :]
            resampled_log_likelihood = resampled_log_likelihood.T.reshape(orig_shape)

            log_likelihood_sample = xr.DataArray(
                resampled_log_likelihood,
                dims=log_likelihood_sample.dims,
                coords=log_likelihood_sample.coords,
            )

        except Exception as e:
            warnings.warn(
                f"Importance resampling failed: {str(e)}. Falling back to original"
                " samples.",
                UserWarning,
                stacklevel=2,
            )

    log_weights, diagnostic = compute_importance_weights(
        -log_likelihood_sample,
        method=ISMethod.PSIS,
        reff=reff,
    )
    log_weights += log_likelihood_sample

    ufunc_kwargs = {"n_dims": 1, "ravel": False}
    kwargs = {"input_core_dims": [["__sample__"]]}

    loo_lppd_i = scale_value * wrap_xarray_ufunc(
        _logsumexp, log_weights, ufunc_kwargs=ufunc_kwargs, **kwargs
    )

    estimator_impl = get_estimator(est_method.value)
    p_loo_values = log_likelihood_sample.var(dim="__sample__").values

    # Calculate elpd_loo and p_loo based on the current estimator method
    if est_method == EstimatorMethod.HH_PPS:
        z = compute_sampling_probabilities(elpd_loo_approx)
        z_sample = z[indices.idx]

        estimates = estimator_impl.estimate(
            z=z_sample,
            m_i=indices.m_i,
            y=loo_lppd_i.values,
            N=n_data_points,
        )

        p_loo_estimates = estimator_impl.estimate(
            z=z_sample,
            m_i=indices.m_i,
            y=p_loo_values,
            N=n_data_points,
        )
    elif est_method == EstimatorMethod.SRS:
        estimates = estimator_impl.estimate(
            y=loo_lppd_i.values,
            N=n_data_points,
        )

        p_loo_estimates = estimator_impl.estimate(
            y=p_loo_values,
            N=n_data_points,
        )
    else:  # diff_srs
        estimates = estimator_impl.estimate(
            y_approx=elpd_loo_approx,
            y=loo_lppd_i.values,
            y_idx=indices.idx,
        )

        p_loo_approx = np.zeros_like(elpd_loo_approx)
        p_loo_estimates = estimator_impl.estimate(
            y_approx=p_loo_approx,
            y=p_loo_values,
            y_idx=indices.idx,
        )

    p_loo = p_loo_estimates.y_hat

    # Calculate standard errors and information criteria
    # Total variance (hat_v_y) for regular SE and subsampling variance
    # (v_y_hat) for subsampling SE
    se = np.sqrt(estimates.hat_v_y) if hasattr(estimates, "hat_v_y") else np.nan
    subsampling_se = (
        np.sqrt(estimates.v_y_hat) if hasattr(estimates, "v_y_hat") else np.nan
    )

    looic = -2 * estimates.y_hat
    looic_se = 2 * se
    looic_subsamp_se = 2 * subsampling_se

    good_k = min(1 - 1 / np.log10(n_samples), 0.7)
    max_k = np.nanmax(diagnostic) if not np.all(np.isnan(diagnostic)) else 0
    warn_mg = False

    if est_method == EstimatorMethod.SRS:
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
    else:
        if max_k > good_k:
            n_high_k = np.sum(diagnostic > good_k)

            warnings.warn(
                "Estimated shape parameter of Pareto distribution is greater than"
                f" {good_k:.2f} for {n_high_k} observations. This indicates that"
                " importance sampling may be unreliable because the marginal posterior"
                " and LOO posterior are very different. If you're using a PyMC model,"
                " consider using reloo() to compute exact LOO for these problematic"
                " observations, or moment matching to improve the estimates.",
                UserWarning,
                stacklevel=2,
            )

            warn_mg = True

    if len(obs_dims) > 1:
        full_shape = [log_likelihood.sizes[dim] for dim in obs_dims]
        loo_lppd_i_full = np.full(full_shape, np.nan)

        values = loo_lppd_i.values.reshape(-1)

        for i, idx in enumerate(indices.idx):
            idx_tuple = np.unravel_index([idx], full_shape)
            idx_tuple = tuple(x[0] for x in idx_tuple)
            loo_lppd_i_full[idx_tuple] = values[i]
    else:
        loo_lppd_i_full = np.full(n_data_points, np.nan)
        loo_lppd_i_full[indices.idx] = loo_lppd_i.values

    non_nan_values = loo_lppd_i_full[~np.isnan(loo_lppd_i_full)]
    if len(non_nan_values) > 0 and np.allclose(non_nan_values, non_nan_values[0]):
        warnings.warn(
            "The point-wise LOO is the same with the sum LOO, please double check "
            "the Observed RV in your model to make sure it returns element-wise logp.",
            UserWarning,
            stacklevel=2,
        )

    result_data: list[Any] = []
    result_index: list[str] = []

    if not pointwise:
        result_data = [
            estimates.y_hat,  # elpd_loo
            se,  # Total uncertainty (approximation + sampling)
            p_loo,
            n_samples,
            n_data_points,
            warn_mg,
            scale,
            good_k,
            subsampling_se,  # Only subsampling uncertainty
            len(indices.idx),  # subsample_size
            looic,
            looic_se,
            looic_subsamp_se,
            "loo_subsample",
        ]

        result_index = [
            "elpd_loo",
            "se",
            "p_loo",
            "n_samples",
            "n_data_points",
            "warning",
            "scale",
            "good_k",
            "subsampling_SE",
            "subsample_size",
            "looic",
            "looic_se",
            "looic_subsamp_se",
            "method",
        ]

        result = ELPDData(data=result_data, index=result_index)
    else:
        loo_i = xr.DataArray(loo_lppd_i_full, name="loo_i")

        result_data = [
            estimates.y_hat,
            se,
            p_loo,
            n_samples,
            n_data_points,
            warn_mg,
            loo_i,
            scale,
            good_k,
            subsampling_se,
            len(indices.idx),
            looic,
            looic_se,
            looic_subsamp_se,
            diagnostic,
            "loo_subsample",
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
            "good_k",
            "subsampling_SE",
            "subsample_size",
            "looic",
            "looic_se",
            "looic_subsamp_se",
            "pareto_k",
            "method",
        ]

        result = ELPDData(data=result_data, index=result_index)

    result.estimates = estimates
    result.estimates.data = inference_data
    result.estimates.loo_approximation = loo_approximation
    result.estimates.estimator = estimator
    result.estimates.loo_approximation_draws = loo_approximation_draws
    result.estimates.var_name = var_name
    result.method = "loo_subsample"

    if log_p is not None and log_q is not None:
        result.log_p = log_p
        result.log_q = log_q
        result.resample_method = resample_method
        result.seed = seed

    return result


def update_subsample(
    loo_data: ELPDData,
    observations: int | np.ndarray | None = None,
    **kwargs,
) -> ELPDData:
    """Update subsampling results with new observations or parameters.

    This function allows updating the subsampling results by recomputing with a new
    number of observations or other parameters while maintaining the original data
    and configuration.

    Parameters
    ----------
    loo_data : ELPDData
        The original LOO-CV results from loo_subsample()
    observations : int | np.ndarray | None, default None
        The new subsample observations to use:
        - An integer specifying the number of observations to subsample
        - An array of integers providing specific indices to use
        - None to use all observations (equivalent to standard LOO)
    **kwargs
        Additional keyword arguments to pass to loo_subsample()

    Returns
    -------
    ELPDData object (inherits from :class:`pandas.Series`):
    A new ELPDData object containing the updated results with the same attributes
    as described in :func:`loo_subsample`

    The returned object has a custom print method that overrides pd.Series method.

    Examples
    --------
    Let's see how to update our subsampling results with more observations. First, we'll
    compute the initial LOO-CV with a small subsample, then update it to use more observations:

    .. code-block:: python

        import arviz as az
        from pyloo import loo_subsample, update_subsample

        data = az.load_arviz_data("centered_eight")
        result = loo_subsample(data, observations=100)
        updated = update_subsample(result, observations=200)
    """
    if not isinstance(loo_data, ELPDData):
        raise TypeError("loo_data must be an ELPDData object from loo_subsample()")

    if not hasattr(loo_data.estimates, "data"):
        raise ValueError("Cannot update: original data not available")

    data = loo_data.estimates.data
    params = {
        "data": data,
        "observations": (
            observations if observations is not None else loo_data["subsample_size"]
        ),
        "loo_approximation": getattr(loo_data.estimates, "loo_approximation", "plpd"),
        "estimator": getattr(loo_data.estimates, "estimator", "diff_srs"),
        "loo_approximation_draws": getattr(
            loo_data.estimates, "loo_approximation_draws", None
        ),
        "pointwise": "loo_i" in loo_data,
        "var_name": getattr(loo_data.estimates, "var_name", None),
        "reff": loo_data.get("r_eff", None),
        "scale": loo_data["scale"],
        "log_p": getattr(loo_data, "log_p", None),
        "log_q": getattr(loo_data, "log_q", None),
        "resample_method": getattr(loo_data, "resample_method", "psis"),
        "seed": getattr(loo_data, "seed", None),
    }
    params.update(kwargs)

    return loo_subsample(**params)
