"""Moment matching for efficient approximate leave-one-out cross-validation (LOO)."""

import logging
import warnings
from copy import deepcopy
from typing import Literal

import numpy as np
import xarray as xr
from arviz.stats.diagnostics import ess

from .base import ISMethod, compute_importance_weights
from .elpd import ELPDData
from .helpers import (
    ParameterConverter,
    ShiftAndCovResult,
    ShiftAndScaleResult,
    ShiftResult,
    UpdateQuantitiesResult,
    extract_log_likelihood_for_observation,
    log_lik_i_upars,
    log_prob_upars,
)
from .split_moment_matching import loo_moment_match_split
from .utils import _logsumexp, wrap_xarray_ufunc
from .wrapper.pymc import PyMCWrapper

__all__ = ["loo_moment_match"]

_log = logging.getLogger(__name__)


def loo_moment_match(
    wrapper: PyMCWrapper,
    loo_data: ELPDData,
    max_iters: int = 30,
    k_threshold: float | None = None,
    split: bool = True,
    cov: bool = True,
    method: Literal["psis", "sis", "tis"] | ISMethod = "psis",
) -> ELPDData:
    r"""Moment matching algorithm for updating a loo object when Pareto k estimates are large.

    Parameters
    ----------
    wrapper : PyMCWrapper
        PyMC model wrapper instance
    loo_data : ELPDData
        A loo object to be modified
    max_iters : int
        Maximum number of moment matching iterations
    k_threshold : float | None
        Threshold value for Pareto k values above which moment matching is used
    split : bool
        Whether to do the split transformation at the end of moment matching
    cov : bool
        Whether to match the covariance matrix of the samples
    method : Literal['psis', 'sis', 'tis'] | ISMethod
        Importance sampling method to use

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
    pareto_k: :class:`~xarray.DataArray` with the Pareto shape parameter k diagnostic values,
            only if pointwise=True and using PSIS method
    scale: scale of the elpd
    looic: leave-one-out information criterion (looic = -2 * elpd_loo)
    looic_se: standard error of the looic

    Notes
    -----
    Moment matching can fail to improve LOO estimates for several reasons such as very high-dimensional
    parameter spaces, multi-modality, weight instability, and insufficient sample size.

    Split moment matching can be used to improve the estimates by transforming only half of the draws
    and using multiple importance sampling to combine them with untransformed draws. This strategy
    provides more stability than transforming all draws, particularly in cases where the transformation
    itself might be imperfect. However, split moment matching is not guaranteed to improve the estimates
    either.

    Examples
    --------
    When we have many Pareto k estimates above the threshold, we can use moment matching to improve the estimates
    and avoid the computational cost of refitting the model :math:`k` times.

    .. code-block:: python

        import pyloo as pl
        import arviz as az

        data = az.load_arviz_data("centered_eight")

        with pm.Model() as model:
            mu = pm.Normal('mu', mu=0, sigma=10)
            sigma = pm.HalfNormal('sigma', sigma=10)
            likelihood = pm.Normal('y', mu=mu, sigma=sigma, observed=data.y)
            idata = pm.sample(1000, tune=1000, idata_kwargs={"log_likelihood": True})

        wrapper = pl.PyMCWrapper(model, idata)

    We can first check the Pareto k estimates.

    .. code-block:: python

        loo_orig = pl.loo(wrapper, pointwise=True)
        print(loo_orig.pareto_k)

    If there are many Pareto k estimates above the threshold, we can use moment matching to improve the estimates.
    Moment matching allows us to match the mean and marginal variances of the posterior draws as well as the
    covariance matrix.

    .. code-block:: python

        loo_new = pl.loo_moment_match(
            wrapper,
            loo_orig,
            max_iters=30,
            k_threshold=0.7,
            split=False,
            cov=True,
        )

    If we want to use split moment matching, we can do the following. Split moment matching transforms only half
    of the draws and computes a single elpd using multiple importance sampling.

    .. code-block:: python

        loo_new = pl.loo_moment_match(
            wrapper,
            loo_orig,
            split=True,
            max_iters=30,
            k_threshold=0.7,
            cov=True,
            )

    See Also
    --------
    loo_subsample : Leave-one-out cross-validation with subsampling
    loo_kfold : K-fold cross-validation
    loo_approximate_posterior : Leave-one-out cross-validation for posterior approximations
    loo_score : Compute LOO score for continuous ranked probability score
    loo_group : Leave-one-group-out cross-validation
    waic : Compute WAIC
    """
    converter = ParameterConverter(wrapper)
    loo_data = deepcopy(loo_data)

    unconstrained = wrapper.get_unconstrained_parameters()
    upars = converter.dict_to_matrix(unconstrained)
    S = upars.shape[0]

    if k_threshold is None:
        k_threshold = min(1 - 1 / np.log10(S), 0.7)

    orig_log_prob = log_prob_upars(wrapper, unconstrained)

    # Check if we have pointwise results
    if hasattr(loo_data, "pareto_k"):
        ks = loo_data.pareto_k.values
    else:
        raise ValueError(
            "Moment matching requires pointwise LOO results with Pareto k values. "
            "Please recompute LOO with pointwise=True before using moment_match=True."
        )

    bad_obs = np.where(ks > k_threshold)[0]
    _log.info(f"Found {len(bad_obs)} observations with Pareto k > {k_threshold}")
    kfs = np.zeros_like(ks)

    for i in bad_obs:
        uparsi = upars.copy()
        ki = ks[i]
        kfi = 0

        try:
            log_lik_result = log_lik_i_upars(wrapper, unconstrained, pointwise=True)
            log_liki = extract_log_likelihood_for_observation(log_lik_result, i)
        except Exception as e:
            raise ValueError(
                f"Error computing log likelihood for observation {i}: {e}"
            ) from e

        posterior = wrapper.idata.posterior
        n_chains = len(posterior.chain)
        if n_chains == 1:
            r_eff_i = 1.0
        else:
            ess_i = ess(log_liki, method="mean")
            if isinstance(ess_i, xr.DataArray):
                ess_i = ess_i.values
            r_eff_i = float(ess_i / len(log_liki))

        is_obj = compute_importance_weights(-log_liki, method=method, reff=r_eff_i)
        lwi, initial_k = is_obj
        _log.info(f"Observation {i}: Initial Pareto k = {initial_k:.4f}")

        total_shift = np.zeros(upars.shape[1])
        total_scaling = np.ones(upars.shape[1])
        total_mapping = np.eye(upars.shape[1])

        iterind = 1
        _log.info(f"Processing observation {i} with Pareto k = {ks[i]:.4f}")

        while iterind <= max_iters and ki > k_threshold:
            if iterind == max_iters:
                warnings.warn(
                    "Maximum number of moment matching iterations reached. "
                    "Increasing max_iters may improve accuracy.",
                    stacklevel=2,
                )

            improved = False

            # Match means
            trans = shift(uparsi, lwi)
            try:
                quantities_i = update_quantities_i(
                    wrapper,
                    trans["upars"],
                    i,
                    orig_log_prob,
                    r_eff_i,
                    converter,
                    method,
                )
            except Exception as e:
                _log.warning(f"Error computing quantities for observation {i}: {e}")
                break

            if quantities_i["ki"] < ki:
                _log.info(
                    f"Observation {i}: Mean shift improved Pareto k from {ki:.4f} to"
                    f" {quantities_i['ki']:.4f}"
                )
                uparsi = trans["upars"]
                total_shift += trans["shift"]
                lwi = quantities_i["lwi"]
                ki = quantities_i["ki"]
                kfi = quantities_i["kfi"]
                log_liki = quantities_i["log_liki"]
                iterind += 1
                improved = True

            # Match means and marginal variances
            trans = shift_and_scale(uparsi, lwi)
            try:
                quantities_i_scale = update_quantities_i(
                    wrapper,
                    trans["upars"],
                    i,
                    orig_log_prob,
                    r_eff_i,
                    converter,
                    method,
                )
            except Exception as e:
                _log.warning(
                    f"Error computing scale quantities for observation {i}: {e}"
                )
                if improved:
                    continue
                else:
                    break

            if quantities_i_scale["ki"] < ki:
                _log.info(
                    f"Observation {i}: Mean and scale shift improved Pareto k from"
                    f" {ki:.4f} to {quantities_i_scale['ki']:.4f}"
                )
                uparsi = trans["upars"]
                total_shift += trans["shift"]
                total_scaling *= trans["scaling"]
                lwi = quantities_i_scale["lwi"]
                ki = quantities_i_scale["ki"]
                kfi = quantities_i_scale["kfi"]
                log_liki = quantities_i_scale["log_liki"]
                iterind += 1
                improved = True

            # Match means and covariances
            if cov:
                trans = shift_and_cov(uparsi, lwi)
                try:
                    quantities_i_cov = update_quantities_i(
                        wrapper,
                        trans["upars"],
                        i,
                        orig_log_prob,
                        r_eff_i,
                        converter,
                        method,
                    )
                except Exception as e:
                    _log.warning(
                        "Error computing covariance quantities for observation"
                        f" {i}: {e}"
                    )
                    if improved:
                        continue
                    else:
                        break

                if quantities_i_cov["ki"] < ki:
                    _log.info(
                        f"Observation {i}: Covariance shift improved Pareto k from"
                        f" {ki:.4f} to {quantities_i_cov['ki']:.4f}"
                    )
                    uparsi = trans["upars"]
                    total_shift += trans["shift"]
                    total_mapping = trans["mapping"] @ total_mapping
                    lwi = quantities_i_cov["lwi"]
                    ki = quantities_i_cov["ki"]
                    kfi = quantities_i_cov["kfi"]
                    log_liki = quantities_i_cov["log_liki"]
                    iterind += 1
                    improved = True

            # Only break if no transformation improved k
            if not improved:
                _log.info(
                    f"Observation {i}: No further improvement after"
                    f" {iterind} iterations"
                )
                break

        if max_iters == 1:
            warnings.warn(
                "Maximum number of moment matching iterations reached. "
                "Increasing max_iters may improve accuracy.",
                stacklevel=2,
            )

        # Split transformation if requested and transformations were successful
        if split and iterind > 1:
            _log.info(f"Performing split transformation for observation {i}")
            split_result = loo_moment_match_split(
                wrapper,
                upars,
                cov,
                total_shift,
                total_scaling,
                total_mapping,
                i,
                r_eff_i,
                method,
            )
            log_liki = split_result["log_liki"]
            lwi = split_result["lwi"]
            r_eff_i = split_result["r_eff_i"]

        new_elpd_i = _logsumexp(log_liki + lwi)

        # Update loo_i data with new estimates
        update_loo_data_i(loo_data, wrapper, i, new_elpd_i, ki, kfi, kfs)

    summary(loo_data, ks, k_threshold)

    if np.any(ks > k_threshold):
        warnings.warn(
            "Some Pareto k estimates are still above the threshold. "
            "The model may be misspecified or the data may be highly influential.",
            stacklevel=2,
        )

    if not split and np.any(kfs > k_threshold):
        warnings.warn(
            "The accuracy of self-normalized importance sampling may be bad. "
            "Setting split=True will likely improve accuracy.",
            stacklevel=2,
        )

    return loo_data


def update_quantities_i(
    wrapper: PyMCWrapper,
    upars: np.ndarray,
    i: int,
    orig_log_prob: np.ndarray,
    r_eff_i: float,
    converter: ParameterConverter,
    method: Literal["psis", "sis", "tis"] | ISMethod = "psis",
) -> UpdateQuantitiesResult:
    """Update the importance weights, Pareto diagnostic and log-likelihood for observation i.

    Parameters
    ----------
    wrapper : PyMCWrapper
        PyMC model wrapper instance
    upars : np.ndarray
        A matrix representing a sample of vector-valued parameters in the unconstrained space
    i : int
        Observation number
    orig_log_prob : np.ndarray
        Log probability densities of the original draws
    r_eff_i : float
        MCMC effective sample size divided by total sample size
    converter : ParameterConverter
        Parameter converter instance for efficient format conversions.
    method : Literal['psis', 'sis', 'tis'] | ISMethod
        Importance sampling method to use

    Returns
    -------
    dict
        dictionary containing updated quantities:
        - lwi: New log importance weights
        - lwfi: New log importance weights for full distribution
        - ki: New Pareto k value
        - kfi: New Pareto k value for full distribution
        - log_liki: New log likelihood values
    """
    upars_dict = converter.matrix_to_dict(upars)
    log_prob_new = log_prob_upars(wrapper, upars_dict)

    try:
        log_lik_result = log_lik_i_upars(wrapper, upars_dict, pointwise=True)
        log_liki_new = extract_log_likelihood_for_observation(log_lik_result, i)
    except Exception as e:
        raise ValueError(
            f"Error computing log likelihood for observation {i}: {e}"
        ) from e

    log_liki_new = np.array(log_liki_new, dtype=np.float64)
    log_prob_new = np.array(log_prob_new, dtype=np.float64)
    orig_log_prob = np.array(orig_log_prob, dtype=np.float64)

    lr = -log_liki_new + log_prob_new - orig_log_prob
    lr[np.isnan(lr)] = -np.inf

    lwi_new, ki_new = compute_importance_weights(lr, method=method, reff=r_eff_i)

    full_lr = log_prob_new - orig_log_prob
    full_lr[np.isnan(full_lr)] = -np.inf
    lwfi_new, kfi_new = compute_importance_weights(full_lr, method=method, reff=r_eff_i)

    return {
        "lwi": lwi_new,
        "lwfi": lwfi_new,
        "ki": ki_new,
        "kfi": kfi_new,
        "log_liki": log_liki_new,
    }


def shift(upars: np.ndarray, lwi: np.ndarray) -> ShiftResult:
    """Shift a matrix of parameters to their weighted mean.

    Parameters
    ----------
    upars : np.ndarray
        A matrix representing a sample of vector-valued parameters in the unconstrained space
    lwi : np.ndarray
        A vector representing the log-weight of each parameter

    Returns
    -------
    dict
        dictionary containing:
        - upars: The transformed parameter matrix
        - shift: The shift that was performed
    """
    mean_original = np.mean(upars, axis=0)
    mean_weighted = np.sum(np.exp(lwi)[:, None] * upars, axis=0)
    shift = mean_weighted - mean_original
    upars_new = upars + shift[None, :]

    return {"upars": upars_new, "shift": shift}


def shift_and_scale(upars: np.ndarray, lwi: np.ndarray) -> ShiftAndScaleResult:
    """Shift a matrix of parameters to their weighted mean and scale the marginal variances.

    Parameters
    ----------
    upars : np.ndarray
        A matrix representing a sample of vector-valued parameters in the unconstrained space
    lwi : np.ndarray
        A vector representing the log-weight of each parameter

    Returns
    -------
    dict
        dictionary containing:
        - upars: The transformed parameter matrix
        - shift: The shift that was performed
        - scaling: The scaling that was performed
    """
    S = upars.shape[0]
    mean_original = np.mean(upars, axis=0)
    mean_weighted = np.sum(np.exp(lwi)[:, None] * upars, axis=0)
    shift = mean_weighted - mean_original
    mii = np.exp(lwi)[:, None] * upars**2
    mii = np.sum(mii, axis=0) - mean_weighted**2
    mii = mii * S / (S - 1)
    scaling = np.sqrt(mii / np.var(upars, axis=0))

    upars_new = upars - mean_original[None, :]
    upars_new = upars_new * scaling[None, :]
    upars_new = upars_new + mean_weighted[None, :]

    return {"upars": upars_new, "shift": shift, "scaling": scaling}


def shift_and_cov(upars: np.ndarray, lwi: np.ndarray) -> ShiftAndCovResult:
    """Shift a matrix of parameters and scale the covariance to match the weighted covariance.

    Parameters
    ----------
    upars : np.ndarray
        A matrix representing a sample of vector-valued parameters in the unconstrained space
    lwi : np.ndarray
        A vector representing the log-weight of each parameter

    Returns
    -------
    dict
        dictionary containing:
        - upars: The transformed parameter matrix
        - shift: The shift that was performed
        - mapping: The mapping matrix that was used
    """
    mean_original = np.mean(upars, axis=0)
    mean_weighted = np.sum(np.exp(lwi)[:, None] * upars, axis=0)
    shift = mean_weighted - mean_original

    covv = np.cov(upars, rowvar=False)
    wcovv = np.cov(upars, rowvar=False, aweights=np.exp(lwi))

    try:
        chol1 = np.linalg.cholesky(wcovv)
        chol2 = np.linalg.cholesky(covv)
        mapping = chol1.T @ np.linalg.inv(chol2.T)
    except np.linalg.LinAlgError:
        mapping = np.eye(len(mean_original))

    upars_new = upars - mean_original[None, :]
    upars_new = upars_new @ mapping.T
    upars_new = upars_new + mean_weighted[None, :]

    return {"upars": upars_new, "shift": shift, "mapping": mapping}


def update_loo_data_i(
    loo_data: ELPDData,
    wrapper: PyMCWrapper,
    i: int,
    new_elpd_i: float,
    ki: float,
    kfi: float,
    kfs: np.ndarray,
) -> None:
    """Update LOO data for a specific observation with new ELPD and k values.

    Parameters
    ----------
    loo_data : ELPDData
        The LOO data object to update
    wrapper : PyMCWrapper
        PyMC model wrapper instance
    i : int
        Observation index
    new_elpd_i : float
        New ELPD value for the observation
    ki : float
        New Pareto k value
    kfi : float
        New Pareto k value for full distribution
    kfs : np.ndarray
        Array to store kfi values
    """
    if hasattr(loo_data, "loo_i"):
        # Multi-observation case
        old_elpd_i = loo_data.loo_i.values[i]
        loo_data.loo_i.values[i] = new_elpd_i

        old_elpd_total = loo_data["elpd_loo"]
        loo_data["elpd_loo"] = loo_data.loo_i.values.sum()
        new_elpd_total = loo_data["elpd_loo"]

        # Update SE
        n_data_points = loo_data.n_data_points
        loo_data["se"] = (n_data_points * np.var(loo_data.loo_i.values)) ** 0.5

        update_stats(loo_data, wrapper)

        _log.info(
            f"Observation {i}: ELPD changed from {old_elpd_i:.4f} to"
            f" {new_elpd_i:.4f} (diff: {new_elpd_i - old_elpd_i:.4f})"
        )
        _log.info(
            f"Total ELPD changed from {old_elpd_total:.4f} to"
            f" {new_elpd_total:.4f} (diff: {new_elpd_total - old_elpd_total:.4f})"
        )
    else:
        # Single observation case
        old_elpd_total = loo_data["elpd_loo"]
        loo_data["elpd_loo"] = new_elpd_i

        update_stats(loo_data, wrapper)

        _log.info(
            f"Total ELPD changed from {old_elpd_total:.4f} to {new_elpd_i:.4f} (diff:"
            f" {new_elpd_i - old_elpd_total:.4f})"
        )

    # Update Pareto k
    if hasattr(loo_data, "pareto_k"):
        old_k = loo_data.pareto_k.values[i]
        loo_data.pareto_k.values[i] = ki
        _log.info(
            f"Observation {i}: Pareto k changed from {old_k:.4f} to"
            f" {ki:.4f} (improvement: {old_k - ki:.4f})"
        )

    kfs[i] = kfi


def update_stats(
    loo_data: ELPDData, wrapper: PyMCWrapper, scale: str | None = None
) -> None:
    """Update derived statistics like p_loo and looic in the LOO data.

    Parameters
    ----------
    loo_data : ELPDData
        The LOO data object to update
    wrapper : PyMCWrapper
        PyMC model wrapper instance
    scale : str, optional
        Output scale for LOO. If None, uses the scale from loo_data.
    """
    if hasattr(wrapper.idata, "log_likelihood"):
        log_likelihood = wrapper.idata.log_likelihood
        var_name = list(log_likelihood.data_vars)[0]
        log_likelihood = log_likelihood[var_name].stack(__sample__=("chain", "draw"))
        n_samples = log_likelihood.shape[-1]

        scale = loo_data.get("scale", "log") if scale is None else scale
        if scale == "deviance":
            scale_value = -2
        elif scale == "log":
            scale_value = 1
        elif scale == "negative_log":
            scale_value = -1
        else:
            scale_value = 1  # Default to log scale if unknown

        ufunc_kwargs = {"n_dims": 1, "ravel": False}
        kwargs = {"input_core_dims": [["__sample__"]]}

        lppd = np.sum(
            wrap_xarray_ufunc(
                _logsumexp,
                log_likelihood,
                func_kwargs={"b_inv": n_samples},
                ufunc_kwargs=ufunc_kwargs,
                **kwargs,
            ).values
        )

        loo_data["p_loo"] = lppd - loo_data["elpd_loo"] / scale_value

        if "looic" in loo_data:
            loo_data["looic"] = -2 * loo_data["elpd_loo"]
            if "se" in loo_data:
                loo_data["looic_se"] = 2 * loo_data["se"]


def summary(loo_data: ELPDData, original_ks: np.ndarray, k_threshold: float) -> None:
    """Log a summary of improvements in Pareto k values.

    Parameters
    ----------
    loo_data : ELPDData
        The LOO data object
    original_ks : np.ndarray
        Original Pareto k values
    k_threshold : float
        Threshold for Pareto k values
    """
    if hasattr(loo_data, "pareto_k"):
        improved_indices = np.where(loo_data.pareto_k.values < original_ks)[0]
        n_improved = len(improved_indices)

        if n_improved > 0:
            avg_improvement = np.mean(
                original_ks[improved_indices]
                - loo_data.pareto_k.values[improved_indices]
            )
            _log.info(
                f"Improved Pareto k for {n_improved} observations. Average improvement:"
                f" {avg_improvement:.4f}"
            )
        else:
            _log.info("No improvements in Pareto k values")

        # Observations that still have high Pareto k values
        high_k_indices = np.where(loo_data.pareto_k.values > k_threshold)[0]
        if len(high_k_indices) > 0:
            _log.info(
                f"{len(high_k_indices)} observations still have Pareto k >"
                f" {k_threshold}"
            )
