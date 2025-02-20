"""Hansen-Hurwitz estimator implementation for LOO-CV subsampling."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from .base import BaseEstimate, EstimatorProtocol


@dataclass
class HHEstimate(BaseEstimate):
    """Container for Hansen-Hurwitz estimation results."""

    pass


class HansenHurwitzEstimator(EstimatorProtocol[HHEstimate]):
    """Implementation of the weighted Hansen-Hurwitz estimator."""

    def estimate(self, **kwargs: Any) -> HHEstimate:
        """Compute the weighted Hansen-Hurwitz estimator.

        Parameters
        ----------
        **kwargs : Any
            Must contain:
            - z: Normalized probabilities for each observation
            - m_i: Number of times each observation was selected
            - y: The observed values
            - N: Total population size

        Returns
        -------
        HHEstimate
            The computed estimate including point estimate and variance

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
        z = np.asarray(kwargs["z"])
        m_i = np.asarray(kwargs["m_i"])
        y = np.asarray(kwargs["y"])
        N = int(kwargs["N"])

        if not np.all(z > 0):
            raise ValueError("All probabilities (z) must be positive")
        if not np.all(m_i > 0):
            raise ValueError("All sample counts (m_i) must be positive")
        if not len(z) == len(m_i) == len(y):
            raise ValueError("All input arrays must have same length")

        z = z / np.sum(z)
        m = np.sum(m_i)
        y_hat = np.sum(m_i * (y / z)) / m
        v_y_hat = (np.sum(m_i * ((y / z - y_hat) ** 2)) / m) / (m - 1)
        hat_v_y = (np.sum(m_i * (y**2 / z)) / m) + v_y_hat / N - y_hat**2 / N
        return HHEstimate(
            y_hat=y_hat,
            v_y_hat=v_y_hat,
            hat_v_y=hat_v_y,
            m=m,
            N=N,
            subsampling_SE=np.sqrt(v_y_hat),
        )


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
    pi_values = pi_values / np.sum(pi_values)

    return pi_values


def hansen_hurwitz_estimate(
    z: np.ndarray,
    m_i: np.ndarray,
    y: np.ndarray,
    N: int,
) -> HHEstimate:
    """Compute the weighted Hansen-Hurwitz estimate.

    Parameters
    ----------
    z : np.ndarray
        Normalized probabilities for each observation
    m_i : np.ndarray
        Number of times each observation was selected
    y : np.ndarray
        The observed values
    N : int
        Total population size

    Returns
    -------
    HHEstimate
        The estimated values and their variance
    """
    estimator = HansenHurwitzEstimator()
    return estimator.estimate(z=z, m_i=m_i, y=y, N=N)


def estimate_elpd_loo(
    elpd_loo_i: np.ndarray,
    elpd_loo_approximation: np.ndarray,
    sample_indices: np.ndarray,
    m_i: np.ndarray,
    N: int,
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
