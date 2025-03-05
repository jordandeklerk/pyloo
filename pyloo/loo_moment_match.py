"""Moment matching for efficient approximate leave-one-out cross-validation (LOO)."""

import warnings
from typing import Any

import numpy as np
import xarray as xr
from arviz.stats.diagnostics import ess

from .importance_sampling import ISMethod, compute_importance_weights
from .utils import _logsumexp
from .wrapper.pymc_wrapper import PyMCWrapper


def loo_moment_match(
    wrapper: PyMCWrapper,
    loo_data: Any,
    max_iters: int = 30,
    k_threshold: float | None = None,
    split: bool = True,
    cov: bool = True,
    method: ISMethod = ISMethod.PSIS,
) -> Any:
    r"""Moment matching algorithm for updating a loo object when Pareto k estimates are large.

    Parameters
    ----------
    wrapper : PyMCWrapper
        PyMC model wrapper instance
    loo_data : Any
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
    Any
        Updated loo object with improved estimates

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

        loo_new = pl.loo_moment_match(wrapper, loo_orig)

    If we want to use split moment matching, we can do the following. Split moment matching transforms only half of the draws
    and computes a single elpd using multiple importance sampling.

    .. code-block:: python

        loo_new = pl.loo_moment_match(wrapper, loo_orig, split=True)

    """
    unconstrained = wrapper.get_unconstrained_parameters()
    param_names = list(unconstrained.keys())
    upars = np.column_stack(
        [unconstrained[name].values.flatten() for name in param_names]
    )
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
    kfs = np.zeros_like(ks)

    for i in bad_obs:
        uparsi = upars.copy()
        ki = ks[i]
        kfi = 0

        try:
            log_liki = _compute_log_likelihood(wrapper, i)
        except Exception as e:
            raise ValueError(
                f"Error computing log likelihood for observation {i}"
            ) from e

        if hasattr(wrapper.idata, "posterior"):
            posterior = wrapper.idata.posterior
            n_chains = len(posterior.chain)
            if n_chains == 1:
                r_eff_i = 1.0
            else:
                try:
                    log_liki_chains = _compute_log_likelihood(wrapper, i)
                except Exception as e:
                    raise ValueError(
                        f"Error computing log likelihood for observation {i}"
                    ) from e

                ess_i = ess(log_liki_chains, method="mean")
                if isinstance(ess_i, xr.DataArray):
                    ess_i = ess_i.values
                r_eff_i = float(ess_i / len(log_liki))
        else:
            r_eff_i = 1.0

        is_obj = compute_importance_weights(-log_liki, method=method, reff=r_eff_i)
        lwi, _ = is_obj

        total_shift = np.zeros(upars.shape[1])
        total_scaling = np.ones(upars.shape[1])
        total_mapping = np.eye(upars.shape[1])

        iterind = 1

        while iterind <= max_iters and ki > k_threshold:
            if iterind == max_iters:
                warnings.warn(
                    "Maximum number of moment matching iterations reached. "
                    "Increasing max_iters may improve accuracy.",
                    stacklevel=2,
                )

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
                continue

            # Match means and covariances
            if cov:
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
                    continue

            # None of the transformations improved khat
            break

        if max_iters == 1:
            warnings.warn(
                "Maximum number of moment matching iterations reached. "
                "Increasing max_iters may improve accuracy.",
                stacklevel=2,
            )

        # Split transformation if requested and transformations were successful
        if split and iterind > 1:
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

        # Update loo
        loo_data.loo_i[i] = _logsumexp(log_liki + lwi)
        if hasattr(loo_data, "pareto_k"):
            loo_data.pareto_k[i] = ki
        kfs[i] = kfi

    if np.any(ks > k_threshold):
        warnings.warn(
            "Some Pareto k estimates are still above the threshold. "
            "The model may be misspecified or the data may be highly influential.",
            stacklevel=2,
        )

    # TODO: This still gives a warning even if split=True
    if not split and np.any(kfs > k_threshold):
        warnings.warn(
            "The accuracy of self-normalized importance sampling may be bad. "
            "Setting split=True will likely improve accuracy.",
            stacklevel=2,
        )

    return loo_data


def loo_moment_match_split(
    wrapper: PyMCWrapper,
    upars: np.ndarray,
    cov: bool,
    total_shift: np.ndarray,
    total_scaling: np.ndarray,
    total_mapping: np.ndarray,
    i: int,
    r_eff_i: float,
    method: ISMethod = ISMethod.PSIS,
) -> dict[str, Any]:
    """Split moment matching for efficient approximate leave-one-out cross-validation.

    This function computes the split moment matching importance sampling loo.
    It transforms only half of the draws and computes a single elpd using
    multiple importance sampling.

    Parameters
    ----------
    wrapper : PyMCWrapper
        PyMC model wrapper instance
    upars : np.ndarray
        Matrix containing the model parameters in unconstrained space
    cov : bool
        Whether to match the covariance matrix of the samples
    total_shift : np.ndarray
        Total shift made by the moment matching algorithm
    total_scaling : np.ndarray
        Total scaling of marginal variance made by the moment matching algorithm
    total_mapping : np.ndarray
        Total covariance transformation made by the moment matching algorithm
    i : int
        Observation index
    r_eff_i : float
        MCMC relative effective sample size
    method : ISMethod
        Importance sampling method to use

    Returns
    -------
    dict
        dictionary containing:
        - lwi: Updated log importance weights
        - lwfi: Updated log importance weights for full distribution
        - log_liki: Updated log likelihood values
        - r_eff_i: Updated relative effective sample size
    """
    S = upars.shape[0]
    S_half = S // 2
    mean_original = np.mean(upars, axis=0)

    if total_shift.shape[0] != upars.shape[1]:
        total_shift = np.zeros(upars.shape[1])
    if total_scaling.shape[0] != upars.shape[1]:
        total_scaling = np.ones(upars.shape[1])
    if total_mapping.shape[0] != upars.shape[1]:
        total_mapping = np.eye(upars.shape[1])

    # Affine transformation
    upars_trans = upars - mean_original
    upars_trans = upars_trans * total_scaling[None, :]
    if cov:
        upars_trans = upars_trans @ total_mapping.T
    upars_trans = upars_trans + total_shift[None, :] + mean_original

    # Inverse affine transformation
    upars_trans_inv = upars - mean_original
    if cov:
        upars_trans_inv = upars_trans_inv @ np.linalg.inv(total_mapping).T
    upars_trans_inv = upars_trans_inv / total_scaling[None, :]
    upars_trans_inv = upars_trans_inv + mean_original - total_shift[None, :]

    # First half of upars_trans_half are T(theta)
    # Second half are theta
    upars_trans_half = upars.copy()
    upars_trans_half[:S_half] = upars_trans[:S_half]

    # First half of upars_half_inv are theta
    # Second half are T^-1(theta)
    upars_trans_half_inv = upars.copy()
    upars_trans_half_inv[S_half:] = upars_trans_inv[S_half:]

    constrained_params = {}
    param_names = list(wrapper.get_unconstrained_parameters().keys())

    # Transformed parameters
    for j, name in enumerate(param_names):
        constrained_params[name] = xr.DataArray(
            upars_trans_half[:, j],
            dims=["sample"],
            coords={"sample": np.arange(len(upars))},
        )
    constrained_trans = wrapper.constrain_parameters(constrained_params)

    # Inverse transformed parameters
    for j, name in enumerate(param_names):
        constrained_params[name] = xr.DataArray(
            upars_trans_half_inv[:, j],
            dims=["sample"],
            coords={"sample": np.arange(len(upars))},
        )
    constrained_trans_inv = wrapper.constrain_parameters(constrained_params)

    log_prob_half_trans = np.zeros(S)
    log_prob_half_trans_inv = np.zeros(S)

    for name in param_names:
        var = wrapper.get_variable(name)
        if var is not None and hasattr(var, "logp"):

            # Transformed parameters
            param_trans = constrained_trans[name]
            if isinstance(param_trans, xr.DataArray):
                param_trans = param_trans.values

            log_prob_trans = var.logp(param_trans).eval()
            log_prob_half_trans += log_prob_trans

            # Inverse transformed parameters
            param_inv = constrained_trans_inv[name]
            if isinstance(param_inv, xr.DataArray):
                param_inv = param_inv.values

            log_prob_inv = var.logp(param_inv).eval()
            log_prob_half_trans_inv += log_prob_inv

    # Adjust for transformation Jacobian
    log_prob_half_trans_inv = (
        log_prob_half_trans_inv
        - np.sum(np.log(total_scaling))
        - np.log(np.abs(np.linalg.det(total_mapping)))
    )

    try:
        log_liki_half = _compute_log_likelihood(wrapper, i)
    except Exception as e:
        raise ValueError(f"Error computing log likelihood for observation {i}") from e

    stable_S = log_prob_half_trans > log_prob_half_trans_inv
    lwi_half = -log_liki_half + log_prob_half_trans

    # For numerically stable regions
    lwi_half[stable_S] = lwi_half[stable_S] - (
        log_prob_half_trans[stable_S]
        + np.log1p(
            np.exp(log_prob_half_trans_inv[stable_S] - log_prob_half_trans[stable_S])
        )
    )

    # For numerically unstable regions
    lwi_half[~stable_S] = lwi_half[~stable_S] - (
        log_prob_half_trans_inv[~stable_S]
        + np.log1p(
            np.exp(log_prob_half_trans[~stable_S] - log_prob_half_trans_inv[~stable_S])
        )
    )

    lr = lwi_half.copy()
    lr[np.isnan(lr)] = -np.inf

    is_obj_half = compute_importance_weights(lr, method=method, reff=r_eff_i)
    lwi_half, _ = is_obj_half

    # Compute weights for the integrand
    lr = lwi_half + log_liki_half
    lr[np.isnan(lr)] = -np.inf
    is_obj_f_half = compute_importance_weights(lr, method=method, reff=r_eff_i)
    lwfi_half, _ = is_obj_f_half

    # Compute relative effective sample size
    # Currently ignores chain information since we have two proposal distributions
    log_liki_half_1 = log_liki_half[S_half:]
    log_liki_half_2 = log_liki_half[:S_half]
    r_eff_i1 = r_eff_i2 = r_eff_i

    if hasattr(wrapper.idata, "posterior"):
        posterior = wrapper.idata.posterior
        n_chains = len(posterior.chain)
        if n_chains == 1:
            r_eff_i1 = r_eff_i2 = 1.0
        else:
            # Calculate ESS for each half's log likelihood
            try:
                log_liki_chains = _compute_log_likelihood(wrapper, i)
            except Exception as e:
                raise ValueError(
                    f"Error computing log likelihood for observation {i}"
                ) from e

            n_chains = wrapper.idata.posterior.chain.size
            n_draws = wrapper.idata.posterior.draw.size
            log_liki_chains = log_liki_chains.reshape(n_chains, n_draws)

            # Split into two halves
            if log_liki_chains[:, S_half:].size > 0 and n_chains > 0:
                try:
                    ess_i1 = ess(log_liki_chains[:, S_half:], method="mean")
                    if isinstance(ess_i1, xr.DataArray):
                        ess_i1 = ess_i1.values
                    if ess_i1.size > 0:
                        r_eff_i1 = float(ess_i1 / max(1, len(log_liki_half_1)))
                except (ValueError, np.linalg.LinAlgError) as e:
                    warnings.warn(
                        f"Error computing effective sample size for half 1: {e}",
                        stacklevel=2,
                    )
                    r_eff_i1 = r_eff_i

            if log_liki_chains[:, :S_half].size > 0 and n_chains > 0:
                try:
                    ess_i2 = ess(log_liki_chains[:, :S_half], method="mean")
                    if isinstance(ess_i2, xr.DataArray):
                        ess_i2 = ess_i2.values
                    if ess_i2.size > 0:
                        r_eff_i2 = float(ess_i2 / max(1, len(log_liki_half_2)))
                except (ValueError, np.linalg.LinAlgError) as e:
                    warnings.warn(
                        f"Error computing effective sample size for half 2: {e}",
                        stacklevel=2,
                    )
                    r_eff_i2 = r_eff_i
    else:
        r_eff_i1 = r_eff_i2 = 1.0

    r_eff_i = min(r_eff_i1, r_eff_i2)

    return {
        "lwi": lwi_half,
        "lwfi": lwfi_half,
        "log_liki": log_liki_half,
        "r_eff_i": r_eff_i,
    }


def update_quantities_i(
    wrapper: PyMCWrapper,
    upars: np.ndarray,
    i: int,
    orig_log_prob: np.ndarray,
    r_eff_i: float,
    method: ISMethod = ISMethod.PSIS,
) -> dict[str, Any]:
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
    log_prob_new = _compute_log_prob(wrapper, upars)
    try:
        log_liki_new = _compute_log_likelihood(wrapper, i)
    except Exception as e:
        raise ValueError(f"Error computing log likelihood for observation {i}") from e

    lr = -log_liki_new + log_prob_new - orig_log_prob
    lr[np.isnan(lr)] = -np.inf

    lwi_new, ki_new = compute_importance_weights(lr, method=method, reff=r_eff_i)
    lwfi_new, kfi_new = compute_importance_weights(
        log_prob_new - orig_log_prob, method=method, reff=r_eff_i
    )

    return {
        "lwi": lwi_new,
        "lwfi": lwfi_new,
        "ki": ki_new,
        "kfi": kfi_new,
        "log_liki": log_liki_new,
    }


def shift(upars: np.ndarray, lwi: np.ndarray) -> dict[str, Any]:
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
    weights = _compute_weights(lwi)
    mean_weighted, mean_original = _compute_means(upars, weights)
    shift = mean_weighted - mean_original

    upars_new = upars + shift
    upars_new, correction = _apply_correction(upars_new, weights, mean_weighted)

    return {"upars": upars_new, "shift": shift + correction}


def shift_and_scale(upars: np.ndarray, lwi: np.ndarray) -> dict[str, Any]:
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
    weights = _compute_weights(lwi)
    mean_weighted, mean_original = _compute_means(upars, weights)
    shift = mean_weighted - mean_original

    # Compute variance scaling
    centered_upars = upars - mean_weighted[None, :]
    var_weighted_orig = np.sum(weights[:, None] * centered_upars**2, axis=0)
    var_original = np.var(upars, axis=0)
    scaling = np.sqrt(var_weighted_orig / var_original)

    # Apply transformation
    upars_new = (upars - mean_original[None, :]) * scaling[None, :] + mean_weighted[
        None, :
    ]
    upars_new, correction = _apply_correction(upars_new, weights, mean_weighted)

    # Correct scaling
    centered_upars_new = upars_new - mean_weighted[None, :]
    var_weighted_new = np.sum(weights[:, None] * centered_upars_new**2, axis=0)
    scaling_correction = np.sqrt(var_weighted_orig / var_weighted_new)
    upars_new = (
        mean_weighted[None, :] + centered_upars_new * scaling_correction[None, :]
    )

    return {"upars": upars_new, "shift": shift + correction, "scaling": scaling}


def shift_and_cov(upars: np.ndarray, lwi: np.ndarray) -> dict[str, Any]:
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
    weights = _compute_weights(lwi)
    mean_weighted, mean_original = _compute_means(upars, weights)
    shift = mean_weighted - mean_original

    # Compute covariance mapping
    centered_upars = upars - mean_weighted[None, :]
    cov_weighted_orig = np.cov(centered_upars, rowvar=False, aweights=weights)
    cov_original = np.cov(upars - mean_original[None, :], rowvar=False)

    try:
        chol1 = np.linalg.cholesky(cov_weighted_orig)
        chol2 = np.linalg.cholesky(cov_original)
        mapping = chol1.T @ np.linalg.inv(chol2.T)
    except np.linalg.LinAlgError:
        mapping = np.eye(len(mean_original))

    # Apply transformation
    upars_new = (upars - mean_original[None, :]) @ mapping.T + mean_weighted[None, :]
    upars_new, correction = _apply_correction(upars_new, weights, mean_weighted)

    return {"upars": upars_new, "shift": shift + correction, "mapping": mapping}


def _compute_log_prob(wrapper: PyMCWrapper, upars: np.ndarray) -> np.ndarray:
    """Compute log probability for parameters."""
    constrained_params = {}
    param_names = list(wrapper.get_unconstrained_parameters().keys())
    for j, name in enumerate(param_names):
        constrained_params[name] = xr.DataArray(
            upars[:, j],
            dims=["sample"],
            coords={"sample": np.arange(len(upars))},
        )
    constrained = wrapper.constrain_parameters(constrained_params)

    log_prob = np.zeros(len(upars))
    for name, param in constrained.items():
        var = wrapper.get_variable(name)
        if var is not None and hasattr(var, "logp"):
            if isinstance(param, xr.DataArray):
                param = param.values
            log_prob_part = var.logp(param).eval()
            log_prob += log_prob_part
    return log_prob


def _compute_log_likelihood(wrapper: PyMCWrapper, i: int) -> np.ndarray:
    """Compute log likelihood for observation i."""
    log_liki = wrapper.log_likelihood_i(i, wrapper.idata)
    log_liki = log_liki.stack(__sample__=("chain", "draw"))
    return log_liki.values.flatten()


def _compute_weights(lwi: np.ndarray) -> np.ndarray:
    """Compute normalized weights from log weights with numerical stability."""
    weights = np.exp(lwi - np.max(lwi))  # Numerical stability
    return weights / np.sum(weights)


def _compute_means(
    upars: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute weighted and original means of parameters."""
    mean_weighted = np.sum(weights[:, None] * upars, axis=0)
    mean_original = np.mean(upars, axis=0)
    return mean_weighted, mean_original


def _apply_correction(
    upars_new: np.ndarray, weights: np.ndarray, mean_weighted: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Apply mean correction to transformed parameters."""
    mean_weighted_new = np.sum(weights[:, None] * upars_new, axis=0)
    correction = mean_weighted - mean_weighted_new
    return upars_new + correction[None, :], correction
