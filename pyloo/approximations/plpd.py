"""Point Log Predictive Density (PLPD) approximation for LOO-CV."""

import warnings

import numpy as np
import xarray as xr

from .base import LooApproximation, thin_draws


class PLPDApproximation(LooApproximation):
    """Point Log Predictive Density approximation.

    This approximation uses posterior expectations (point estimates) to compute
    an approximation of the LOO values.
    """

    def __init__(self, posterior=None, log_likelihood_fn=None, data=None):
        """Initialize with posterior draws and likelihood function."""
        self.posterior = posterior
        self.log_likelihood_fn = log_likelihood_fn
        self.data = data

    def compute_approximation(self, log_likelihood, n_draws=None):
        r"""Compute PLPD approximation of LOO values.

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

        .. math::
            \log p(y_i | \mathbb{E}[\theta])

        where :math:`\mathbb{E}[\theta]` is the posterior mean of the parameters.

        Raises
        ------
        ValueError
            If no posterior samples are provided.
        """
        if self.posterior is None:
            raise ValueError("No posterior samples provided for PLPD approximation")

        if n_draws is not None:
            posterior = thin_draws(self.posterior, n_draws)
        else:
            posterior = self.posterior

        if isinstance(posterior, xr.Dataset):
            point_est = {}
            for var in posterior.data_vars:
                param_data = posterior[var]
                if "chain" in param_data.dims and "draw" in param_data.dims:
                    param_data = param_data.stack(__sample__=("chain", "draw"))
                point_est[var] = param_data.mean(dim="__sample__").values
        else:
            if isinstance(posterior, xr.DataArray):
                if "chain" in posterior.dims and "draw" in posterior.dims:
                    posterior = posterior.stack(__sample__=("chain", "draw"))
                point_est = posterior.mean(dim="__sample__").values
            else:
                point_est = np.mean(posterior, axis=0)

        if self.log_likelihood_fn is not None and self.data is not None:
            n_obs = (
                len(self.data)
                if hasattr(self.data, "__len__")
                else log_likelihood.shape[0]
            )
            plpd_values = np.zeros(n_obs)

            for i in range(n_obs):
                obs_data = self.data[i : i + 1] if self.data is not None else i
                plpd_values[i] = self.log_likelihood_fn(obs_data, point_est)

            return plpd_values

        warnings.warn(
            "Using approximate PLPD calculation. For better accuracy, provide "
            "log likelihood and data to compute log likelihoods directly.",
            UserWarning,
            stacklevel=2,
        )

        if "__sample__" in log_likelihood.dims:
            return log_likelihood.mean(dim="__sample__").values
        elif "chain" in log_likelihood.dims and "draw" in log_likelihood.dims:
            return log_likelihood.mean(dim=["chain", "draw"]).values
        else:
            return np.mean(log_likelihood, axis=-1)
