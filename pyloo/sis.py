"""Standard Importance Sampling (SIS) implementation."""

from typing import Optional, Tuple, Union

import numpy as np

from .base import ImportanceSampling
from .ess import mcmc_eff_size
from .utils import _logsumexp


def sislw(
    log_ratios: np.ndarray,
    r_eff: Union[float, np.ndarray] = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Standard Importance Sampling (SIS) log weights.

    Parameters
    ----------
    log_ratios : np.ndarray
        Array of shape (n_samples, n_observations) containing log importance
        ratios (for example, log-likelihood values).
    r_eff : Union[float, np.ndarray], optional
        Relative MCMC efficiency (effective sample size / total samples).
        Can be a scalar or array of length n_observations. Default is 1.0.

    Returns
    -------
    log_weights : np.ndarray
        Array of same shape as log_ratios containing log weights
    pareto_k : np.ndarray
        Array of zeros with length n_observations (not used in SIS)
    ess : np.ndarray
        Array of effective sample sizes

    Notes
    -----
    Standard importance sampling simply uses the raw importance ratios as weights,
    without any smoothing or stabilization. This can be less stable than methods
    like PSIS when the importance ratios have high variance.

    Examples
    --------
    Calculate SIS weights for log-likelihood values:

    .. ipython::

        In [1]: import numpy as np
           ...: from pyloo import sislw
           ...: log_liks = np.random.normal(size=(1000, 100))
           ...: weights, k, ess = sislw(log_liks)
           ...: print(f"Mean ESS: {ess.mean():.1f}")

    See Also
    --------
    psis : Pareto Smoothed Importance Sampling
    """
    if not isinstance(log_ratios, np.ndarray):
        log_ratios = np.asarray(log_ratios)

    if log_ratios.ndim == 1:
        log_ratios = log_ratios.reshape(-1, 1)

    if log_ratios.ndim != 2:
        raise ValueError("log_ratios must be 1D or 2D array")

    _, n_obs = log_ratios.shape

    if isinstance(r_eff, (int, float)):
        r_eff = float(r_eff)
    elif len(r_eff) != n_obs:
        raise ValueError("r_eff must be a scalar or have length equal to n_observations")

    log_weights = log_ratios.copy()
    for i in range(n_obs):
        log_weights[:, i] = log_weights[:, i] - _logsumexp(log_weights[:, i])

    pareto_k = np.zeros(n_obs)

    ess = np.zeros(n_obs)
    for i in range(n_obs):
        weights = np.exp(log_weights[:, i])
        ess[i] = mcmc_eff_size(weights.reshape(-1, 1), method="bulk")

    if log_ratios.shape[1] == 1:
        log_weights = log_weights.ravel()
        pareto_k = pareto_k.reshape(())
        ess = ess.reshape(())

    return log_weights, pareto_k, ess


class StandardImportanceSampling(ImportanceSampling):
    """Standard Importance Sampling implementation."""

    def compute_weights(
        self,
        log_ratios: np.ndarray,
        r_eff: Optional[Union[float, np.ndarray]] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Compute standard importance sampling weights.

        Parameters
        ----------
        log_ratios : np.ndarray
            Array of shape (n_samples, n_observations) containing log importance
            ratios (for example, log-likelihood values).
        r_eff : Optional[Union[float, np.ndarray]], optional
            Relative MCMC efficiency (effective sample size / total samples).
            Can be a scalar or array of length n_observations. Default is None.
        **kwargs
            Additional keyword arguments (not used in SIS).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]
            - Array of log weights
            - Array of zeros (Pareto k values, not used in SIS)
            - Array of effective sample sizes
        """
        return sislw(log_ratios, r_eff if r_eff is not None else 1.0)
