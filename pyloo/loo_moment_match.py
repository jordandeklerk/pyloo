"""Moment matching for efficient approximate leave-one-out cross-validation (LOO)."""

import logging
import warnings
from typing import Literal

import numpy as np
import xarray as xr
from arviz.stats.diagnostics import ess

from .elpd import ELPDData
from .importance_sampling import ISMethod, compute_importance_weights
from .split_moment_matching import loo_moment_match_split
from .utils import _logsumexp
from .utils_moment_matching import (
    ShiftAndCovResult,
    ShiftAndScaleResult,
    ShiftResult,
    UpdateQuantitiesResult,
    log_prob_upars,
)
from .wrapper.pymc_wrapper import PyMCWrapper

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
    k_threshold : float, optional
        Threshold value for Pareto k values above which moment matching is used
    split : bool
        Whether to do the split transformation at the end of moment matching
    cov : bool
        Whether to match the covariance matrix of the samples
    method : ISMethod
        Importance sampling method to use

    Returns
    -------
    ELPDData
        Updated loo object with improved estimates

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
    Moment matching allows us to match the mean and marginal variances of the posterior draws as well as the covariance matrix.

    .. code-block:: python

        loo_new = pl.loo_moment_match(
            wrapper,
            loo_orig,
            max_iters=30,
            k_threshold=0.7,
            split=False,
            cov=True,
        )

    If we want to use split moment matching, we can do the following. Split moment matching transforms only half of the draws
    and computes a single elpd using multiple importance sampling.

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
    loo : Leave-one-out cross-validation
    loo_subsample : Subsampled LOO-CV computation
    reloo : Exact LOO-CV computation for PyMC models
    loo_kfold : K-fold cross-validation
    """
    unconstrained = wrapper.get_unconstrained_parameters()
    param_names = list(unconstrained.keys())

    param_arrays = []
    for name in param_names:
        param = unconstrained[name].values.flatten()
        param_arrays.append(param)

    min_size = min(len(arr) for arr in param_arrays)
    upars = np.column_stack([arr[:min_size] for arr in param_arrays])
    S = upars.shape[0]

    if k_threshold is None:
        k_threshold = min(1 - 1 / np.log10(S), 0.7)

    # Compute original log probabilities
    orig_log_prob = np.zeros(S)
    for name, param in unconstrained.items():
        var = wrapper.get_variable(name)
        if var is not None and hasattr(var, "logp"):
            if isinstance(param, xr.DataArray):
                param = param.values
            log_prob_part = var.logp(param).eval()
            orig_log_prob += log_prob_part

    if hasattr(loo_data, "pareto_k"):
        ks = loo_data.pareto_k
    else:
        ks = loo_data.diagnostics["pareto_k"]

    bad_obs = np.where(ks > k_threshold)[0]
    _log.info(f"Found {len(bad_obs)} observations with Pareto k > {k_threshold}")
    kfs = np.zeros_like(ks)

    for i in bad_obs:
        uparsi = upars.copy()
        ki = ks[i]
        kfi = 0

        try:
            # TODO: Placeholder for now, will be replaced with actual log likelihood computation
            log_liki = wrapper.log_likelihood_i(wrapper, i)
        except Exception as e:
            raise ValueError(
                f"Error computing log likelihood for observation {i}"
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
        lwi, _ = is_obj

        total_shift = np.zeros(upars.shape[1])
        total_scaling = np.ones(upars.shape[1])
        total_mapping = np.eye(upars.shape[1])

        iterind = 1
        _log.info(f"Processing observation {i} with Pareto k = {ks[i]}")

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
            quantities_i = update_quantities_i(
                wrapper, trans["upars"], i, orig_log_prob, r_eff_i, method
            )

            if quantities_i["ki"] < ki:
                uparsi = trans["upars"]
                total_shift += trans["shift"]
                lwi = quantities_i["lwi"]
                ki = quantities_i["ki"]
                kfi = quantities_i["kfi"]
                log_liki = quantities_i["log_liki"]
                iterind += 1
                improved = True
                continue

            # Match means and marginal variances
            trans = shift_and_scale(uparsi, lwi)
            quantities_i = update_quantities_i(
                wrapper, trans["upars"], i, orig_log_prob, r_eff_i, method
            )

            if quantities_i["ki"] < ki:
                uparsi = trans["upars"]
                total_shift += trans["shift"]
                total_scaling *= trans["scaling"]
                lwi = quantities_i["lwi"]
                ki = quantities_i["ki"]
                kfi = quantities_i["kfi"]
                log_liki = quantities_i["log_liki"]
                iterind += 1
                improved = True
                continue

            # Match means and covariances
            if cov and not improved:
                trans = shift_and_cov(uparsi, lwi)
                quantities_i = update_quantities_i(
                    wrapper, trans["upars"], i, orig_log_prob, r_eff_i, method
                )

                if quantities_i["ki"] < ki:
                    uparsi = trans["upars"]
                    total_shift += trans["shift"]
                    total_mapping = trans["mapping"] @ total_mapping
                    lwi = quantities_i["lwi"]
                    ki = quantities_i["ki"]
                    kfi = quantities_i["kfi"]
                    log_liki = quantities_i["log_liki"]
                    iterind += 1
                    improved = True
                    continue

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

        loo_data.elpd_loo = _logsumexp(log_liki + lwi)
        if hasattr(loo_data, "pareto_k"):
            loo_data.pareto_k[i] = ki
        kfs[i] = kfi

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
    method : ISMethod
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
    log_prob_new = log_prob_upars(wrapper, upars)
    try:
        # TODO: Placeholder for now, will be replaced with actual log likelihood computation
        log_liki_new = wrapper.log_likelihood_i(wrapper, i)
    except Exception as e:
        raise ValueError(f"Error computing log likelihood for observation {i}") from e

    lr = log_prob_new - orig_log_prob
    stable_mask = log_prob_new > orig_log_prob

    lwi = -log_liki_new + log_prob_new

    lwi[stable_mask] = lwi[stable_mask] - (
        log_prob_new[stable_mask]
        + np.log1p(np.exp(orig_log_prob[stable_mask] - log_prob_new[stable_mask]))
    )
    lwi[~stable_mask] = lwi[~stable_mask] - (
        orig_log_prob[~stable_mask]
        + np.log1p(np.exp(log_prob_new[~stable_mask] - orig_log_prob[~stable_mask]))
    )

    lwi[np.isnan(lwi)] = -np.inf
    lwi_new, ki_new = compute_importance_weights(lwi, method=method, reff=r_eff_i)
    lwfi_new, kfi_new = compute_importance_weights(lr, method=method, reff=r_eff_i)

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
