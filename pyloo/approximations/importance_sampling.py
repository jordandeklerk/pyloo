"""Importance sampling based approximations for LOO-CV."""

from typing import Optional

import numpy as np
import xarray as xr

from ..importance_sampling import ISMethod, compute_importance_weights
from ..utils import _logsumexp, wrap_xarray_ufunc
from .base import LooApproximation, thin_draws


class ImportanceSamplingApproximation(LooApproximation):
    """Base class for importance sampling based approximations."""

    def __init__(self, method: ISMethod):
        """Initialize with specific importance sampling method.

        Parameters
        ----------
        method : ISMethod
            The importance sampling method to use:
            - PSIS: Pareto Smoothed Importance Sampling (default)
            - SIS: Standard Importance Sampling
            - TIS: Truncated Importance Sampling
        """
        self.method = method

    def compute_approximation(
        self,
        log_likelihood: xr.DataArray,
        n_draws: Optional[int] = None,
    ) -> np.ndarray:
        """Compute importance sampling based approximation of LOO values.

        Parameters
        ----------
        log_likelihood : xr.DataArray
            Log likelihood values with shape (n_obs, n_samples)
        n_draws : Optional[int]
            Number of posterior draws to use for approximation.
            If None, uses all available draws.

        Returns
        -------
        np.ndarray
            Array of approximated LOO values with shape (n_obs,)

        Notes
        -----
        The importance sampling approximation is computed using the specified
        method (PSIS, SIS, or TIS) to obtain importance weights, which are then
        used to compute the LOO values.

        Raises
        ------
        ValueError
            If n_draws exceeds available draws
        """
        if n_draws is not None:
            log_likelihood = thin_draws(log_likelihood, n_draws)

        log_weights, _ = compute_importance_weights(-log_likelihood, method=self.method)
        log_weights += log_likelihood

        ufunc_kwargs = {"n_dims": 1, "ravel": False}
        kwargs = {"input_core_dims": [["__sample__"]]}

        loo_lppd = wrap_xarray_ufunc(_logsumexp, log_weights, ufunc_kwargs=ufunc_kwargs, **kwargs)

        return loo_lppd.values


class TISApproximation(ImportanceSamplingApproximation):
    """Truncated Importance Sampling approximation."""

    def __init__(self):
        """Initialize with TIS method."""
        super().__init__(method=ISMethod.TIS)


class SISApproximation(ImportanceSamplingApproximation):
    """Standard Importance Sampling approximation."""

    def __init__(self):
        """Initialize with SIS method."""
        super().__init__(method=ISMethod.SIS)
