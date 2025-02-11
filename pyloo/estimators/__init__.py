"""Estimators for LOO-CV subsampling."""

from typing import Dict, Type

from .base import (
    BaseEstimate,
    EstimatorProtocol,
    SubsampleIndices,
    compare_indices,
    subsample_indices,
)
from .difference import DifferenceEstimator, DiffEstimate, diff_srs_estimate
from .hansen_hurwitz import (
    HansenHurwitzEstimator,
    HHEstimate,
    compute_sampling_probabilities,
)
from .hansen_hurwitz import estimate_elpd_loo as hh_estimate_elpd_loo
from .hansen_hurwitz import hansen_hurwitz_estimate
from .srs import SimpleRandomSamplingEstimator, SRSEstimate
from .srs import estimate_elpd_loo as srs_estimate_elpd_loo
from .srs import srs_estimate

ESTIMATOR_REGISTRY: Dict[str, Type[EstimatorProtocol]] = {
    "diff_srs": DifferenceEstimator,
    "hh_pps": HansenHurwitzEstimator,
    "srs": SimpleRandomSamplingEstimator,
}


def get_estimator(method: str) -> EstimatorProtocol:
    """Get an estimator implementation by method name.

    Parameters
    ----------
    method : str
        The estimation method to use:
        * "diff_srs": Difference estimator with simple random sampling
        * "hh_pps": Hansen-Hurwitz estimator with probability proportional to size
        * "srs": Simple random sampling

    Returns
    -------
    EstimatorProtocol
        An instance of the requested estimator

    Raises
    ------
    ValueError
        If the requested method is not found in the registry
    """
    try:
        estimator_cls = ESTIMATOR_REGISTRY[method]
        return estimator_cls()
    except KeyError:
        raise ValueError(
            f"Unknown estimator method: {method}. " f"Must be one of: {', '.join(ESTIMATOR_REGISTRY.keys())}"
        )


__all__ = [
    # Base types and functions
    "BaseEstimate",
    "EstimatorProtocol",
    "SubsampleIndices",
    "subsample_indices",
    "compare_indices",
    "get_estimator",
    # Difference estimator
    "DiffEstimate",
    "DifferenceEstimator",
    "diff_srs_estimate",
    # Hansen-Hurwitz estimator
    "HHEstimate",
    "HansenHurwitzEstimator",
    "hansen_hurwitz_estimate",
    "hh_estimate_elpd_loo",
    "compute_sampling_probabilities",
    # Simple random sampling estimator
    "SRSEstimate",
    "SimpleRandomSamplingEstimator",
    "srs_estimate",
    "srs_estimate_elpd_loo",
]
