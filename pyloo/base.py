"""Base class for importance sampling methods."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import numpy as np


class ImportanceSampling(ABC):
    """Abstract base class for importance sampling methods.

    This class defines the interface that all importance sampling methods must implement.
    Currently supported methods are:
    - PSIS: Pareto Smoothed Importance Sampling
    - TIS: Truncated Importance Sampling
    - SIS: Standard Importance Sampling
    """

    @abstractmethod
    def compute_weights(
        self,
        log_ratios: np.ndarray,
        r_eff: Optional[Union[float, np.ndarray]] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Compute importance sampling weights.

        Parameters
        ----------
        log_ratios : np.ndarray
            Array of shape (n_samples, n_observations) containing log importance
            ratios (for example, log-likelihood values).
        r_eff : Optional[Union[float, np.ndarray]], optional
            Relative MCMC efficiency (effective sample size / total samples) used in
            tail length calculation. Can be a scalar or array of length
            n_observations. Default is None.
        **kwargs
            Additional keyword arguments specific to each method.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]
            - Array of smoothed log weights
            - Array of diagnostic values (e.g., Pareto k values for PSIS)
            - Optional array of effective sample sizes
        """
        pass
