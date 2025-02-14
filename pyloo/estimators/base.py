"""Base functionality for LOO-CV subsampling estimators."""

from dataclasses import dataclass
from typing import Any, Dict, Protocol, TypeVar, runtime_checkable

import numpy as np

T_co = TypeVar("T_co", bound="BaseEstimate", covariant=True)


@dataclass
class BaseEstimate:
    """Base class for all estimation results.

    Parameters
    ----------
    y_hat : float
        Point estimate
    v_y_hat : float
        Variance of point estimate
    hat_v_y : float
        Estimated variance of y
    m : int
        Sample size
    subsampling_SE : float
        Standard error of the subsampling estimate
    N : int, optional
        Population size, defaults to 0 for estimators that don't use it
    """

    y_hat: float
    v_y_hat: float
    hat_v_y: float
    m: int
    subsampling_SE: float
    N: int = 0


@dataclass
class SubsampleIndices:
    """Container for subsampling indices and counts.

    Parameters
    ----------
    idx : np.ndarray
        Array of sampled observation indices
    m_i : np.ndarray
        Array of counts for each observation
    """

    idx: np.ndarray
    m_i: np.ndarray


@runtime_checkable
class EstimatorProtocol(Protocol[T_co]):
    """Protocol defining the interface for estimator implementations."""

    def estimate(self, **kwargs: Any) -> T_co:
        """Compute the estimate based on provided data.

        Parameters
        ----------
        **kwargs : Any
            Keyword arguments specific to each estimator implementation

        Returns
        -------
        T_co
            The computed estimate including point estimate and variance
        """
        pass


def subsample_indices(
    estimator: str,
    elpd_loo_approximation: np.ndarray,
    observations: int,
) -> SubsampleIndices:
    """Draw a subsample of observations based on the specified estimator.

    Parameters
    ----------
    estimator : str
        The estimation method to use:
        * "diff_srs": Difference estimator with simple random sampling
        * "hh_pps": Hansen-Hurwitz estimator with probability proportional to size
        * "srs": Simple random sampling
    elpd_loo_approximation : np.ndarray
        Vector of LOO approximations for all observations
    observations : int
        Number of observations to sample

    Returns
    -------
    SubsampleIndices
        Named tuple containing:
        * idx: Array of sampled observation indices
        * m_i: Array of counts for each observation
    """
    if estimator == "hh_pps":
        pi_values = np.abs(elpd_loo_approximation)
        pi_values = pi_values / pi_values.sum()
        idx = np.random.choice(
            len(elpd_loo_approximation), size=observations, replace=True, p=pi_values
        )
        unique_idx, counts = np.unique(idx, return_counts=True)
        return SubsampleIndices(idx=unique_idx, m_i=counts)

    elif estimator in ("diff_srs", "srs"):
        if observations > len(elpd_loo_approximation):
            raise ValueError(
                "Number of observations cannot exceed total sample size "
                "when using SRS without replacement"
            )
        idx = sorted(
            np.random.choice(
                len(elpd_loo_approximation), size=observations, replace=False
            )
        )
        return SubsampleIndices(idx=idx, m_i=np.ones_like(idx))

    else:
        raise ValueError(f"Unknown estimator: {estimator}")


def compare_indices(
    new_indices: SubsampleIndices,
    current_indices: SubsampleIndices,
) -> Dict[str, SubsampleIndices]:
    """Compare new and current indices to determine updates needed.

    Parameters
    ----------
    new_indices : SubsampleIndices
        The newly sampled indices
    current_indices : SubsampleIndices
        The currently used indices

    Returns
    -------
    Dict[str, SubsampleIndices]
        Dictionary containing:
        * 'new': Indices not in current set
        * 'add': Indices in both sets
        * 'remove': Indices only in current set
    """
    result = {}

    new_mask = ~np.isin(new_indices.idx, current_indices.idx)
    if new_mask.any():
        result["new"] = SubsampleIndices(
            idx=new_indices.idx[new_mask], m_i=new_indices.m_i[new_mask]
        )

    add_mask = np.isin(new_indices.idx, current_indices.idx)
    if add_mask.any():
        result["add"] = SubsampleIndices(
            idx=new_indices.idx[add_mask], m_i=new_indices.m_i[add_mask]
        )

    remove_mask = ~np.isin(current_indices.idx, new_indices.idx)
    if remove_mask.any():
        result["remove"] = SubsampleIndices(
            idx=current_indices.idx[remove_mask], m_i=current_indices.m_i[remove_mask]
        )

    return result
