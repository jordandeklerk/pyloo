"""Split moment matching for efficient approximate leave-one-out cross-validation."""

from typing import Literal

import numpy as np

from .base import ISMethod, compute_importance_weights
from .helpers import (
    ParameterConverter,
    SplitMomentMatchResult,
    _initialize_array,
    compute_updated_r_eff,
    extract_log_likelihood_for_observation,
    log_lik_i_upars,
    log_prob_upars,
)
from .wrapper.pymc import PyMCWrapper

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

    converter = ParameterConverter(wrapper)
    upars_trans_half_dict = converter.matrix_to_dict(upars_trans_half)
    upars_trans_half_inv_dict = converter.matrix_to_dict(upars_trans_half_inv)

    try:
        log_prob_half_trans = log_prob_upars(wrapper, upars_trans_half_dict)
        log_prob_half_trans_inv = log_prob_upars(wrapper, upars_trans_half_inv_dict)

        # Jacobian adjustment
        log_prob_half_trans_inv = (
            log_prob_half_trans_inv
            - np.sum(np.log(total_scaling))
            - np.log(np.abs(np.linalg.det(total_mapping)))
        )
    except Exception as e:
        raise ValueError(
            f"Error computing log probabilities for transformed parameters: {e}"
        ) from e

    try:
        log_lik_result = log_lik_i_upars(wrapper, upars_trans_half_dict, pointwise=True)
        log_liki_half = extract_log_likelihood_for_observation(log_lik_result, i)
    except Exception as e:
        raise ValueError(
            f"Error computing log likelihood for observation {i}: {e}"
        ) from e

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

    r_eff_i = compute_updated_r_eff(wrapper, i, log_liki_half, S_half, r_eff_i)

    return {
        "lwi": lwi_half,
        "lwfi": lwfi_half,
        "log_liki": log_liki_half,
        "r_eff_i": r_eff_i,
    }
