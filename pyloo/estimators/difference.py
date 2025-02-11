"""Difference estimator implementation for LOO-CV subsampling."""

import numpy as np

from .base import BaseEstimate


class DiffEstimate(BaseEstimate):
    """Container for difference estimation results."""

    pass


def difference_estimate(
    y_approx: np.ndarray,
    y: np.ndarray,
    y_idx: np.ndarray,
) -> DiffEstimate:
    """Compute the difference estimator for SRS-WOR sampling.

    Parameters
    ----------
    y_approx : np.ndarray
        Approximated values for all observations
    y : np.ndarray
        The observed values for sampled observations
    y_idx : np.ndarray
        Indices of sampled observations in y_approx

    Returns
    -------
    DiffEstimate
        Named tuple containing:
        * y_hat: Point estimate
        * v_y_hat: Variance of point estimate
        * hat_v_y: Estimated variance of y
        * m: Sample size
        * N: Population size

    Notes
    -----
    The difference estimator is computed as:
        y_hat = t_pi_tilde + t_e
    where:
        t_pi_tilde = sum(y_approx)  # Population total of approximations
        t_e = N * mean(e_i)  # Estimated total of differences
        e_i = y - y_approx_m  # Differences for sampled observations

    References
    ----------
    Magnusson et al. (2020) https://arxiv.org/abs/2001.09660
    """
    if len(y) != len(y_idx):
        raise ValueError("y and y_idx must have same length")
    if np.max(y_idx) >= len(y_approx):
        raise ValueError("y_idx contains invalid indices")

    N = len(y_approx)  # Population size
    m = len(y)  # Sample size

    y_approx_m = y_approx[y_idx]

    if y.ndim > 1 or y_approx_m.ndim > 1:
        if y.ndim > 1:
            y = y.mean(axis=tuple(range(1, y.ndim)))
        if y_approx_m.ndim > 1:
            y_approx_m = y_approx_m.mean(axis=tuple(range(1, y_approx_m.ndim)))

    e_i = y - y_approx_m

    t_pi_tilde = np.sum(y_approx)
    t_pi2_tilde = np.sum(y_approx**2)

    t_e = N * np.mean(e_i)
    t_hat_epsilon = N * np.mean(y**2 - y_approx_m**2)

    y_hat = t_pi_tilde + t_e

    if m > 1:
        e_mean = np.mean(e_i)
        reg_term = 1e-12 * np.abs(e_mean)
        v_y_hat = (N**2) * (1 - m / N) * (np.sum((e_i - e_mean) ** 2) + reg_term) / (m * (m - 1))
    else:
        v_y_hat = np.inf

    # Estimated variance of y (equation 9)
    # Note: The paper has a typo in equation 9, first row second + should be -
    # See supplementary material equation (6) for correct version
    hat_v_y = (t_pi2_tilde + t_hat_epsilon) - (1 / N) * (t_e**2 - v_y_hat + 2 * t_pi_tilde * y_hat - t_pi_tilde**2)

    return DiffEstimate(y_hat=y_hat, v_y_hat=v_y_hat, hat_v_y=hat_v_y, m=m, N=N, subsampling_SE=np.sqrt(v_y_hat))


def diff_srs_estimate(
    elpd_loo_i: np.ndarray,
    elpd_loo_approximation: np.ndarray,
    sample_indices: np.ndarray,
) -> DiffEstimate:
    """Estimate expected log pointwise predictive density using difference estimator.

    Parameters
    ----------
    elpd_loo_i : np.ndarray
        LOO values for sampled observations
    elpd_loo_approximation : np.ndarray
        Approximations for all observations
    sample_indices : np.ndarray
        Indices of sampled observations

    Returns
    -------
    DiffEstimate
        The estimated ELPD and its variance
    """
    return difference_estimate(y_approx=elpd_loo_approximation, y=elpd_loo_i, y_idx=sample_indices)
