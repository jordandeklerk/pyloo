"""Hansen-Hurwitz estimator implementation for LOO-CV subsampling."""

import numpy as np

from .base import BaseEstimate


class HHEstimate(BaseEstimate):
    """Container for Hansen-Hurwitz estimation results."""


def hansen_hurwitz_estimate(
    z: np.ndarray,
    m_i: np.ndarray,
    y: np.ndarray,
    N: int,
) -> HHEstimate:
    """Compute the weighted Hansen-Hurwitz estimator.

    Parameters
    ----------
    z : np.ndarray
        Normalized probabilities for each observation (should sum to 1)
    m_i : np.ndarray
        Number of times each observation was selected
    y : np.ndarray
        The observed values
    N : int
        Total population size

    Returns
    -------
    HHEstimate
        Named tuple containing:
        * y_hat: Point estimate
        * v_y_hat: Variance of point estimate
        * hat_v_y: Estimated variance of y
        * m: Total number of samples

    Notes
    -----
    The estimator is computed as:
        y_hat = (1/m) * sum(m_i * (y_i/z_i))
    where:
        m = sum(m_i)

    The variance estimator accounts for the PPS sampling design and includes
    a finite population correction factor.

    References
    ----------
    Magnusson et al. (2019) https://arxiv.org/abs/1902.06504
    """
    if not np.allclose(z.sum(), 1.0):
        raise ValueError("Probabilities (z) must sum to 1")
    if not np.all(z > 0):
        raise ValueError("All probabilities (z) must be positive")
    if not np.all(m_i > 0):
        raise ValueError("All sample counts (m_i) must be positive")
    if not len(z) == len(m_i) == len(y):
        raise ValueError("All input arrays must have same length")

    m = np.sum(m_i)

    y_hat = np.sum(m_i * (y / z)) / m

    v_y_hat = (np.sum(m_i * ((y / z - y_hat) ** 2)) / m) / (m - 1)

    # Compute estimated variance of y
    # This includes the finite population correction factor (1/N)
    # See supplementary material of the paper for derivation
    hat_v_y = (np.sum(m_i * (y**2 / z)) / m) + v_y_hat / N - y_hat**2 / N

    return HHEstimate(y_hat=y_hat, v_y_hat=v_y_hat, hat_v_y=hat_v_y, m=m, subsampling_SE=np.sqrt(v_y_hat))


def compute_sampling_probabilities(
    elpd_loo_approximation: np.ndarray,
) -> np.ndarray:
    """Compute PPS sampling probabilities from LOO approximations.

    Parameters
    ----------
    elpd_loo_approximation : np.ndarray
        Vector of LOO approximations for all observations

    Returns
    -------
    np.ndarray
        Normalized sampling probabilities that sum to 1

    Notes
    -----
    This follows the R implementation in loo_subsample.R which uses:
        pi_values = abs(elpd_loo_approximation)
        pi_values = pi_values/sum(pi_values)
    """
    pi_values = np.abs(elpd_loo_approximation)

    if np.all(pi_values <= 0):
        # If all values are zero or negative, use uniform probabilities
        pi_values = np.ones_like(pi_values)

    pi_values = np.maximum(pi_values, np.finfo(float).tiny)
    pi_values = pi_values / pi_values.sum()

    return pi_values


def estimate_elpd_loo(
    elpd_loo_i: np.ndarray,  # LOO values for sampled observations
    elpd_loo_approximation: np.ndarray,  # Approximations for all observations
    sample_indices: np.ndarray,  # Indices of sampled observations
    m_i: np.ndarray,  # Number of times each observation was sampled
    N: int,  # Total number of observations
) -> HHEstimate:
    """Estimate expected log pointwise predictive density using Hansen-Hurwitz.

    Parameters
    ----------
    elpd_loo_i : np.ndarray
        LOO values for sampled observations
    elpd_loo_approximation : np.ndarray
        Approximations for all observations
    sample_indices : np.ndarray
        Indices of sampled observations
    m_i : np.ndarray
        Number of times each observation was sampled
    N : int
        Total number of observations

    Returns
    -------
    HHEstimate
        The estimated ELPD and its variance
    """
    z = compute_sampling_probabilities(elpd_loo_approximation)
    z_sample = z[sample_indices]

    return hansen_hurwitz_estimate(z=z_sample, m_i=m_i, y=elpd_loo_i, N=N)
