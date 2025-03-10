"""Point Log Predictive Density (PLPD) approximation for LOO-CV."""

from typing import Optional, Union

import numpy as np
import xarray as xr

from .base import LooApproximation, thin_draws


class PLPDApproximation(LooApproximation):
    """Point Log Predictive Density approximation.

    This approximation uses posterior expectations (point estimates) to compute
    an approximation of the LOO values.
    """

    def __init__(self, posterior: Optional[Union[np.ndarray, xr.DataArray]] = None):
        """Initialize with posterior draws.

        Parameters
        ----------
        posterior : Optional[Union[np.ndarray, xr.DataArray]]
            Posterior draws used to compute point estimates. If None, will use
            log likelihood values directly.
        """
        self.posterior = posterior

    def compute_approximation(
        self,
        log_likelihood: xr.DataArray,
        n_draws: Optional[int] = None,
    ) -> np.ndarray:
        """Compute PLPD approximation of LOO values.

        Parameters
        ----------
        log_likelihood : xr.DataArray
            Log likelihood values with shape (n_obs, n_samples)
        n_draws : Optional[int]
            Number of draws to use for point estimates. If None, uses all draws.

        Returns
        -------
        np.ndarray
            Array of approximated LOO values with shape (n_obs,)

        Notes
        -----
        The PLPD approximation is computed as:
            log p(y_i | E[θ])
        where E[θ] is the posterior mean of the parameters.

        Raises
        ------
        ValueError
            If no posterior samples are provided.
        """
        if self.posterior is None:
            raise ValueError(
                "No posterior samples provided. PLPD approximation requires posterior"
                " samples to compute point estimates. Please sample your model and"
                " provide the posterior draws."
            )

        if n_draws is not None:
            log_likelihood = thin_draws(log_likelihood, n_draws)
            self.posterior = thin_draws(self.posterior, n_draws)

        if isinstance(self.posterior, xr.Dataset):
            posterior_vars = list(self.posterior.data_vars)
            if not posterior_vars:
                raise ValueError("Empty posterior Dataset")
            posterior_data = self.posterior[posterior_vars[0]]
        else:
            posterior_data = self.posterior

        if isinstance(posterior_data, xr.DataArray):
            if "chain" in posterior_data.dims and "draw" in posterior_data.dims:
                posterior_data = posterior_data.stack(__sample__=("chain", "draw"))
            point_est = posterior_data.mean(dim="__sample__")
        else:
            posterior_da = xr.DataArray(posterior_data, dims=["__sample__"])
            point_est = posterior_da.mean(dim="__sample__")

        # Use point estimates to compute log likelihood
        point_log_likelihood = log_likelihood.isel(__sample__=0).copy()

        for var in point_est.coords:
            if var != "__sample__":
                existing_dims = set(point_log_likelihood.dims)
                if isinstance(point_est, xr.Dataset):
                    coords_to_assign = {
                        var: point_est[var]
                        for var in point_est.data_vars
                        if set(point_est[var].dims).issubset(existing_dims)
                    }
                else:
                    if set(point_est.dims).issubset(existing_dims):
                        coords_to_assign = {point_est.name: point_est}
                    else:
                        coords_to_assign = {}

                if coords_to_assign:
                    point_log_likelihood = point_log_likelihood.assign_coords(
                        coords_to_assign
                    )

        return point_log_likelihood.values
