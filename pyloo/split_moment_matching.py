"""Split moment matching for efficient approximate leave-one-out cross-validation."""

import warnings
from typing import Literal

import numpy as np
import xarray as xr
from arviz.stats.diagnostics import ess

from .importance_sampling import ISMethod, compute_importance_weights
from .utils_moment_matching import (
    SplitMomentMatchResult,
    _initialize_array,
    compute_log_likelihood,
)
from .wrapper.pymc_wrapper import PyMCWrapper

__all__ = ["loo_moment_match_split"]


def loo_moment_match_split(
    wrapper: PyMCWrapper,
    upars: np.ndarray,
    cov: bool,
    total_shift: np.ndarray,
    total_scaling: np.ndarray,
    total_mapping: np.ndarray,
    i: int,
    r_eff_i: float,
    method: Literal["psis", "sis", "tis"] | ISMethod = "psis",
) -> SplitMomentMatchResult:
    r"""Split moment matching for efficient approximate leave-one-out cross-validation.

    Instead of transforming all posterior draws, we apply transformations to only half of
    them while leaving the other half unchanged. This creates two different but complementary
    approximations of the leave-one-out posterior. When we combine these halves using multiple
    importance sampling, we get more reliable estimates while maintaining computational efficiency.

    The split moment matching approach works as follows:

    Let :math:`S` be the total number of posterior draws and :math:`S_{half} = S/2`. The approach applies
    transformations :math:`T` to only the first half of draws while leaving the other half unchanged

    .. math::
        \\begin{align}
        \\text{For } s = 1 \\text{ to } S_{half}: & \\quad \\theta^*_s = T(\\theta_s) \\\\
        \\text{For } s = S_{half}+1 \\text{ to } S: & \\quad \\theta^*_s = \\theta_s
        \\end{align}

    The transformation :math:`T` is a composition of simpler transformations that match moments

    .. math::
        \\begin{align}
        T_1(\\theta) &= \\theta - \\bar{\\theta} + \\bar{\\theta}_w \\quad \\text{(match means)}, \\\\
        T_2(\\theta) &= v^{1/2}_w \\circ v^{-1/2} \\circ (\\theta - \\bar{\\theta}) + \\bar{\\theta}_w
        \\quad \\text{(match marginal variances)}, \\\\
        T_3(\\theta) &= L_w L^{-1} (\\theta - \\bar{\\theta}) + \\bar{\\theta}_w
        \\quad \\text{(match covariance)},
        \\end{align}

    where :math:`\\bar{\\theta}` is the sample mean, :math:`\\bar{\\theta}_w` is the weighted mean,
    :math:`v` and :math:`v_w` are the sample and weighted variances, and :math:`L` comes from the
    Cholesky decomposition of the covariance matrix (:math:`LL^T = \\Sigma`) and :math:`L_w` from
    the weighted covariance (:math:`L_w L^T_w = \\Sigma_w`).

    For the inverse transformation, we apply :math:`T^{-1}` to the second half

    .. math::
        \\begin{align}
        \\text{For } s = 1 \\text{ to } S_{half}: & \\quad \\theta^*_{inv,s} = \\theta_s, \\\\
        \\text{For } s = S_{half}+1 \\text{ to } S: & \\quad \\theta^*_{inv,s} = T^{-1}(\\theta_s).
        \\end{align}

    The importance weights are then computed using a deterministic mixture distribution

    .. math::
        w^{(s)}_{mix} = \\frac{p(\\theta^*_s | y_{-i})}{g_{mix}(\\theta^*_s)}

    where :math:`g_{mix}(\\theta)` is the implicit mixture

    .. math::
        g_{mix}(\\theta) \\propto p(\\theta | y) + |J_T|^{-1} p(T^{-1}(\\theta) | y)

    and :math:`|J_T|` is the determinant of the Jacobian of transformation :math:`T`.

    This approach effectively creates a split proposal that better approximates the target
    leave-one-out posterior distribution while maintaining stability.

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

    References
    ----------
    Paananen, T., Piironen, J., Buerkner, P.-C., Vehtari, A. (2020). Implicitly Adaptive Importance
    Sampling. arXiv preprint arXiv:1906.08850.
    """
    S = upars.shape[0]
    S_half = S // 2
    mean_original = np.mean(upars, axis=0)
    dim = upars.shape[1]

    total_shift = _initialize_array(total_shift, np.zeros, dim)
    total_scaling = _initialize_array(total_scaling, np.ones, dim)
    total_mapping = _initialize_array(total_mapping, np.eye, dim)

    # Forward transformation
    upars_trans = upars - mean_original[None, :]
    upars_trans = upars_trans * total_scaling[None, :]
    if cov:
        upars_trans = upars_trans @ total_mapping.T
    upars_trans = upars_trans + (total_shift + mean_original)[None, :]

    # Inverse transformation
    upars_trans_inv = upars - mean_original[None, :]
    if cov:
        upars_trans_inv = upars_trans_inv @ np.linalg.inv(total_mapping).T
    upars_trans_inv = upars_trans_inv / total_scaling[None, :]
    upars_trans_inv = upars_trans_inv + (mean_original - total_shift)[None, :]

    # Split transformations - first half gets forward transform
    upars_trans_half = upars.copy()
    upars_trans_half[:S_half] = upars_trans[:S_half]

    # Second half gets inverse transform
    upars_trans_half_inv = upars.copy()
    upars_trans_half_inv[S_half:] = upars_trans_inv[S_half:]

    try:
        param_names = list(wrapper.get_unconstrained_parameters().keys())

        # Constrained parameters for transformed set
        constrained_params = {}
        for j, name in enumerate(param_names):
            constrained_params[name] = xr.DataArray(
                upars_trans_half[:, j],
                dims=["sample"],
                coords={"sample": np.arange(S)},
            )
        constrained_trans = wrapper.constrain_parameters(constrained_params)

        # Constrained parameters for inverse transformed set
        constrained_params = {}
        for j, name in enumerate(param_names):
            constrained_params[name] = xr.DataArray(
                upars_trans_half_inv[:, j],
                dims=["sample"],
                coords={"sample": np.arange(S)},
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

        # Adjust for Jacobian
        log_prob_half_trans_inv = (
            log_prob_half_trans_inv
            - np.sum(np.log(total_scaling))
            - np.log(np.abs(np.linalg.det(total_mapping)))
        )
    except Exception as e:
        raise ValueError(
            f"Error computing log probabilities for transformed parameters {e}"
        ) from e

    try:
        log_liki_half = compute_log_likelihood(wrapper, i)
    except Exception as e:
        raise ValueError(f"Error computing log likelihood for observation {i}") from e

    log_prob_half_trans_inv = (
        log_prob_half_trans_inv
        - np.sum(np.log(np.abs(total_scaling)))
        - np.log(np.abs(np.linalg.det(total_mapping)))
    )

    # Determine stable regions for computation
    stable_S = log_prob_half_trans > log_prob_half_trans_inv
    lwi_half = -log_liki_half + log_prob_half_trans

    lwi_half[stable_S] = lwi_half[stable_S] - (
        log_prob_half_trans[stable_S]
        + np.log1p(
            np.exp(log_prob_half_trans_inv[stable_S] - log_prob_half_trans[stable_S])
        )
    )

    lwi_half[~stable_S] = lwi_half[~stable_S] - (
        log_prob_half_trans_inv[~stable_S]
        + np.log1p(
            np.exp(log_prob_half_trans[~stable_S] - log_prob_half_trans_inv[~stable_S])
        )
    )

    lwi_half[np.isnan(lwi_half)] = -np.inf
    lwi_half[np.isinf(lwi_half) & (lwi_half > 0)] = -np.inf

    is_obj_half = compute_importance_weights(lwi_half, method=method, reff=r_eff_i)
    lwi_half, _ = is_obj_half

    lr = lwi_half + log_liki_half
    lr[np.isnan(lr) | (np.isinf(lr) & (lr > 0))] = -np.inf
    is_obj_f_half = compute_importance_weights(lr, method=method, reff=r_eff_i)
    lwfi_half, _ = is_obj_f_half

    r_eff_i = _compute_updated_r_eff(wrapper, i, log_liki_half, S_half, r_eff_i)

    return {
        "lwi": lwi_half,
        "lwfi": lwfi_half,
        "log_liki": log_liki_half,
        "r_eff_i": r_eff_i,
    }


def _compute_updated_r_eff(
    wrapper: PyMCWrapper,
    i: int,
    log_liki_half: np.ndarray,
    S_half: int,
    r_eff_i: float,
) -> float:
    """Compute updated relative effective sample size.

    Parameters
    ----------
    wrapper : PyMCWrapper
        PyMC model wrapper instance
    i : int
        Index of the observation
    log_liki_half : np.ndarray
        Log likelihood values for observation i, shape (n_samples,)
    S_half : int
        Half the number of samples
    r_eff_i : float
        Current relative effective sample size for observation i

    Returns
    -------
    float
        Updated relative effective sample size (min of the two halves)
    """
    log_liki_half_1 = log_liki_half[S_half:]
    log_liki_half_2 = log_liki_half[:S_half]

    r_eff_i1 = r_eff_i2 = r_eff_i

    posterior = wrapper.idata.posterior
    n_chains = len(posterior.chain)

    if n_chains <= 1:
        r_eff_i1 = r_eff_i2 = 1.0
    else:
        try:
            log_liki_chains = compute_log_likelihood(wrapper, i)
            n_draws = posterior.draw.size
            log_liki_chains = log_liki_chains.reshape(n_chains, n_draws)

            # Calculate ESS for first half
            if log_liki_chains[:, S_half:].size > 0:
                ess_i1 = ess(log_liki_chains[:, S_half:], method="mean")
                if isinstance(ess_i1, xr.DataArray):
                    ess_i1 = ess_i1.values
                if ess_i1.size > 0:
                    r_eff_i1 = float(ess_i1 / max(1, len(log_liki_half_1)))

            # Calculate ESS for second half
            if log_liki_chains[:, :S_half].size > 0:
                ess_i2 = ess(log_liki_chains[:, :S_half], method="mean")
                if isinstance(ess_i2, xr.DataArray):
                    ess_i2 = ess_i2.values
                if ess_i2.size > 0:
                    r_eff_i2 = float(ess_i2 / max(1, len(log_liki_half_2)))
        except Exception as e:
            warnings.warn(
                f"Error calculating ESS for observation {i}, using original"
                f" r_eff_i: {e}",
                stacklevel=2,
            )
            return r_eff_i

    return min(r_eff_i1, r_eff_i2)
