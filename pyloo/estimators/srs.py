"""Simple random sampling (SRS) estimator implementation for LOO-CV subsampling."""

import numpy as np

from .base import BaseEstimate


class SRSEstimate(BaseEstimate):
    """Container for simple random sampling estimation results."""


def srs_estimate(
    y: np.ndarray,
    N: int,
) -> SRSEstimate:
    """Compute the simple random sampling estimator.

    This implements the standard SRS-WOR estimator for a population total.
    While simpler than the difference estimator, it doesn't leverage auxiliary
    information and thus may be less efficient.

    Parameters
    ----------
    y : np.ndarray
        The observed values
    N : int
        Total population size

    Returns
    -------
    SRSEstimate
        Named tuple containing:
        * y_hat: Point estimate
        * v_y_hat: Variance of point estimate
        * hat_v_y: Estimated variance of y
        * m: Sample size
        * N: Population size

    Notes
    -----
    The SRS estimator for the population total is:
        y_hat = N * mean(y)

    The variance estimators include finite population correction factors
    to account for sampling without replacement from a finite population.
    """
    m = len(y)
    y_hat = N * np.mean(y)
    sample_var = np.var(y, ddof=1)

    # Variance of point estimate (includes finite population correction)
    v_y_hat = N**2 * (1 - m / N) * sample_var / m
    hat_v_y = N * sample_var

    return SRSEstimate(y_hat=y_hat, v_y_hat=v_y_hat, hat_v_y=hat_v_y, m=m, N=N, subsampling_SE=np.sqrt(v_y_hat))


def estimate_elpd_loo(
    elpd_loo_i: np.ndarray,
    N: int,
) -> SRSEstimate:
    """Estimate expected log pointwise predictive density using SRS.

    Parameters
    ----------
    elpd_loo_i : np.ndarray
        LOO values for sampled observations
    N : int
        Total number of observations

    Returns
    -------
    SRSEstimate
        The estimated ELPD and its variance
    """
    return srs_estimate(y=elpd_loo_i, N=N)
