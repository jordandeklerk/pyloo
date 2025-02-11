"""Log Predictive Density (LPD) approximation for LOO-CV."""

from typing import Optional

import numpy as np
import xarray as xr

from ..utils import _logsumexp, wrap_xarray_ufunc
from .base import LooApproximation, thin_draws


class LPDApproximation(LooApproximation):
    """Log Predictive Density approximation."""

    def compute_approximation(
        self,
        log_likelihood: xr.DataArray,
        n_draws: Optional[int] = None,
    ) -> np.ndarray:
        """Compute LPD approximation of LOO values.

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
        The LPD approximation is computed as:
            log p(y_i | y)
        which is the log predictive density for each observation.
        This method uses all posterior draws to compute the predictive density,
        making it more accurate but computationally more expensive than PLPD.

        Raises
        ------
        ValueError
            If n_draws exceeds available draws
        """
        if n_draws is not None:
            log_likelihood = thin_draws(log_likelihood, n_draws)

        ufunc_kwargs = {"n_dims": 1, "ravel": False}
        kwargs = {"input_core_dims": [["__sample__"]]}

        lpds = wrap_xarray_ufunc(
            _logsumexp,
            log_likelihood,
            func_kwargs={"b_inv": len(log_likelihood.__sample__)},
            ufunc_kwargs=ufunc_kwargs,
            **kwargs,
        )

        return lpds.values
