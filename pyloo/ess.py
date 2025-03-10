"""ESS functions for pyloo with adaptations from Arviz."""

import multiprocessing as mp
from typing import Callable, Literal

import numpy as np
from scipy import stats

from .utils import autocov


def rel_eff(
    x: np.ndarray | Callable,
    chain_id: np.ndarray | None = None,
    cores: int = 1,
    data: np.ndarray | None = None,
    draws: np.ndarray | None = None,
    method: Literal["bulk", "tail", "mean", "sd", "median", "mad", "local"] = "bulk",
    prob: float | tuple[float, float] | None = None,
) -> np.ndarray:
    """Compute the MCMC effective sample size divided by the total sample size.

    Parameters
    ----------
    x : Union[np.ndarray, Callable]
        Input values in one of the following forms:
        * Vector (1-D array)
        * Matrix (2-D array)
        * 3-D array
        * Function that returns likelihood values
        For use with loo(), the values should be likelihood values (exp(log_lik)),
        not on the log scale. For generic use with psis(), the values should be
        the reciprocal of the importance ratios (exp(-log_ratios)).
    chain_id : Optional[np.ndarray]
        Vector of length NROW(x) containing MCMC chain indexes for each row
        of x (if matrix) or each value in x (if vector). Not needed if x
        is a 3-D array. If there are C chains then valid indexes are 1:C.
    cores : int
        Number of cores to use for parallelization. Default is 1.
    data : Optional[np.ndarray]
        Data array (only used if x is a function).
    draws : Optional[np.ndarray]
        Draws array (only used if x is a function).
    method : str
        Method to use for ESS calculation. Options are:
        - "bulk": ESS for bulk of the distribution
        - "tail": ESS focusing on distribution tails
        - "mean": ESS for mean estimation
        - "sd": ESS for standard deviation estimation
        - "median": ESS for median estimation
        - "mad": ESS for median absolute deviation
        - "local": ESS for specific probability region
    prob : Optional[Union[float, tuple[float, float]]]
        Probability value(s) for tail or local ESS calculation.
        For tail: single value p gives min(ESS(p), ESS(1-p))
        For local: tuple (p1, p2) gives ESS for region between p1 and p2

    Returns
    -------
    np.ndarray
        Vector of relative effective sample sizes, bounded between 0 and 1.
        Values closer to 1 indicate better mixing of the chains.

    Examples
    --------
    Calculate relative efficiency for MCMC samples:

    .. code-block:: python

        import numpy as np
        from pyloo import rel_eff

        # Generate fake MCMC samples
        samples = np.random.normal(size=(1000, 4, 10))  # 1000 iterations, 4 chains, 10 parameters
        r_eff = rel_eff(samples)

        print(f"Mean relative efficiency: {r_eff.mean():.3f}")

    See Also
    --------
    mcmc_eff_size : Calculate raw effective sample size
    psis_eff_size : Compute effective sample size for PSIS
    """
    if callable(x):
        return _relative_eff_function(
            x, chain_id, cores, data, draws, method=method, prob=prob
        )

    x = np.asarray(x)

    if x.ndim == 1:
        x = x.reshape(-1, 1)
        if chain_id is not None:
            x = _mat_to_chains(x, chain_id)
        else:
            x = x.reshape(x.shape[0], 1, 1)
    elif x.ndim == 2:
        if chain_id is None:
            raise ValueError("chain_id required for 2-D input")
        x = _mat_to_chains(x, chain_id)
    elif x.ndim != 3:
        raise ValueError("Input must be 1-D, 2-D, or 3-D array")

    S = x.shape[0] * x.shape[1]

    if cores == 1:
        n_eff = np.array([
            mcmc_eff_size(x[:, :, i], method=method, prob=prob)
            for i in range(x.shape[2])
        ])
    else:
        with mp.Pool(cores) as pool:
            n_eff = np.array(
                pool.starmap(
                    mcmc_eff_size,
                    [(x[:, :, i], method, prob) for i in range(x.shape[2])],
                )
            )

    return np.minimum(n_eff / S, 1.0)


def psis_eff_size(
    w: np.ndarray, r_eff: float | np.ndarray | None = None
) -> float | np.ndarray:
    """Compute effective sample size for Pareto Smoothed Importance Sampling (PSIS).

    Parameters
    ----------
    w : np.ndarray
        Array of importance weights. Can be 1-D (vector) or 2-D (matrix).
        For 2-D input, rows represent draws and columns represent observations.
    r_eff : Optional[Union[float, np.ndarray]]
        Relative MCMC efficiency (effective sample size / total samples).
        Can be a scalar or array of length ncol(w). Default is None.

    Returns
    -------
    Union[float, np.ndarray]
        Effective sample size(s). Returns a float for 1-D input w,
        or an array of length ncol(w) for 2-D input.

    Examples
    --------
    Calculate PSIS effective sample size:

    .. code-block:: python

        import numpy as np
        from pyloo import psis_eff_size
        # Generate fake importance weights
        weights = np.random.gamma(1, 1, size=(1000, 100))
        weights /= weights.sum(axis=0)  # normalize
        n_eff = psis_eff_size(weights)
        print(f"Mean effective sample size: {n_eff.mean():.1f}")

    See Also
    --------
    rel_eff : Compute MCMC relative efficiency
    mcmc_eff_size : Calculate raw effective sample size
    """
    w = np.asarray(w)
    if w.ndim == 1:
        ss = np.sum(w**2)
        if r_eff is None:
            return 1.0 / ss
        if not np.isscalar(r_eff):
            raise ValueError("r_eff must be scalar when w is 1-D")
        return (1.0 / ss) * r_eff

    ss = np.sum(w**2, axis=0)
    if r_eff is None:
        return 1.0 / ss

    r_eff = np.asarray(r_eff)
    if len(r_eff) != len(ss) and len(r_eff) != 1:
        raise ValueError("r_eff must have length 1 or ncol(w)")

    return (1.0 / ss) * r_eff


def mcmc_eff_size(
    sims: np.ndarray,
    method: str = "bulk",
    prob: float | tuple[float, float] | None = None,
) -> float:
    """Calculate MCMC effective sample size using various methods.

    Parameters
    ----------
    sims : np.ndarray
        Array of MCMC draws. Can be 1-D (vector) or 2-D (matrix).
        For 2-D input, rows represent iterations and columns represent chains.
    method : str
        Method to use for ESS calculation. Options are:
        - "bulk": ESS for bulk of the distribution (default)
        - "tail": ESS focusing on distribution tails
        - "mean": ESS for mean estimation
        - "sd": ESS for standard deviation estimation
        - "median": ESS for median estimation
        - "mad": ESS for median absolute deviation
        - "local": ESS for specific probability region
    prob : Optional[Union[float, tuple[float, float]]]
        Probability value(s) for tail or local ESS calculation.

    Returns
    -------
    float
        The effective sample size estimate.

    Notes
    -----
    This function implements multiple approaches to computing effective sample size,
    each suited for different aspects of the posterior distribution. The implementation
    uses FFT for efficient computation of autocovariances and includes adjustments
    to ensure reliable estimates even with relatively few samples.

    Examples
    --------
    Calculate effective sample size for MCMC draws:

    .. code-block:: python

        import numpy as np
        from pyloo import mcmc_eff_size
        # Generate fake MCMC samples
        samples = np.random.normal(size=(1000, 4))  # 1000 iterations, 4 chains
        ess = mcmc_eff_size(samples)
        print(f"Effective sample size: {ess:.1f}")

    See Also
    --------
    rel_eff : Compute relative efficiency
    psis_eff_size : Compute effective sample size for PSIS

    References
    ----------
    .. [1] Stan Development Team. (2024). Stan Reference Manual.
           https://mc-stan.org/docs/reference-manual/effective-sample-size.html
    .. [2] Vehtari et al. (2021). Rank-normalization, folding, and
           localization: An improved R-hat for assessing convergence of MCMC.
           Bayesian Analysis, 16(2):667-718.
    """
    sims = np.asarray(sims)
    if sims.ndim == 1:
        sims = sims.reshape(-1, 1)

    if not _is_valid_draws(sims):
        return np.nan

    if method == "bulk":
        return _ess_bulk(sims)
    elif method == "tail":
        return _ess_tail(sims, prob)
    elif method == "mean":
        return _ess_mean(sims)
    elif method == "sd":
        return _ess_sd(sims)
    elif method == "median":
        return _ess_median(sims)
    elif method == "mad":
        return _ess_mad(sims)
    elif method == "local":
        if not isinstance(prob, tuple) or len(prob) != 2:
            raise ValueError("local method requires prob=(lower, upper)")
        return _ess_local(sims, prob)
    else:
        raise ValueError(f"Unknown method: {method}")


def _is_valid_draws(x: np.ndarray, min_draws: int = 4, min_chains: int = 1) -> bool:
    """Check if the input array has valid dimensions for ESS calculation."""
    if x.size == 0:
        return False
    if np.any(~np.isfinite(x)):
        import warnings

        if np.any(np.isnan(x)):
            warnings.warn("Input contains NaN values", RuntimeWarning, stacklevel=2)
        if np.any(np.isinf(x)):
            warnings.warn(
                "Input contains infinite values", RuntimeWarning, stacklevel=2
            )
        return False
    if len(x.shape) > 2:
        raise ValueError("Input array must be 1-D or 2-D")
    if x.shape[0] < min_draws:
        return False
    if len(x.shape) > 1 and x.shape[1] < min_chains:
        return False
    return True


def _z_scale(x: np.ndarray) -> np.ndarray:
    """Apply rank normalization and z-score transformation."""
    ranks = stats.rankdata(x.ravel(), method="average")
    ranks = (ranks - 0.375) / (len(ranks) + 0.25)  # Blom transformation
    return stats.norm.ppf(ranks).reshape(x.shape)


def _split_chains(x: np.ndarray) -> np.ndarray:
    """Split chains into first and second half."""
    n_samples = x.shape[0]
    split_point = n_samples // 2
    if len(x.shape) == 1:
        return np.vstack((x[:split_point], x[split_point:]))
    return np.vstack((x[:split_point, :], x[split_point:, :]))


def _ess_bulk(x: np.ndarray) -> float:
    """Compute bulk ESS using rank normalization."""
    z = _z_scale(_split_chains(x))
    return min(_ess_raw(z), x.size)


def _ess_tail(x: np.ndarray, prob: float | tuple[float, float] | None = None) -> float:
    """Compute tail ESS."""
    if prob is None:
        prob_low, prob_high = 0.05, 0.95
    elif isinstance(prob, tuple):
        prob_low, prob_high = prob
    else:
        prob_low, prob_high = prob, 1 - prob

    q1 = np.quantile(x, prob_low)
    q2 = np.quantile(x, prob_high)
    return min(
        _ess_raw(_split_chains(x <= q1)), _ess_raw(_split_chains(x >= q2)), x.size
    )


def _ess_mean(x: np.ndarray) -> float:
    """Compute ESS for mean estimation."""
    return min(_ess_raw(_split_chains(x)), x.size)


def _ess_sd(x: np.ndarray) -> float:
    """Compute ESS for standard deviation estimation."""
    centered = x - np.mean(x)
    return min(_ess_raw(_split_chains(centered**2)), x.size)


def _ess_median(x: np.ndarray) -> float:
    """Compute ESS for median estimation."""
    median = np.median(x)
    return min(_ess_raw(_split_chains(x <= median)), x.size)


def _ess_mad(x: np.ndarray) -> float:
    """Compute ESS for median absolute deviation."""
    median = np.median(x)
    mad = np.median(np.abs(x - median))
    return min(_ess_raw(_split_chains(np.abs(x - median) <= mad)), x.size)


def _ess_local(x: np.ndarray, prob: tuple[float, float]) -> float:
    """Compute ESS for a specific probability region."""
    lower, upper = prob
    q1 = np.quantile(x, lower)
    q2 = np.quantile(x, upper)
    return min(_ess_raw(_split_chains((x >= q1) & (x <= q2))), x.size)


def _ess_raw(x: np.ndarray) -> float:
    """Core ESS calculation using Geyer's initial monotone sequence."""
    chains = x.shape[1] if len(x.shape) > 1 else 1
    n_samples = x.shape[0]

    if chains == 1:
        acov = autocov(x.ravel())
    else:
        acov_chains = [autocov(x[:, i]) for i in range(chains)]
        acov = np.column_stack(acov_chains)

    chain_mean = np.mean(x, axis=0)
    mean_var = np.mean(acov[0]) * n_samples / (n_samples - 1)
    var_plus = mean_var * (n_samples - 1) / n_samples

    if chains > 1:
        var_plus += np.var(chain_mean, ddof=1)

    rho_hat_t = np.zeros(n_samples)
    rho_hat_even = 1.0
    rho_hat_t[0] = rho_hat_even
    rho_hat_odd = 1.0 - (mean_var - np.mean(acov[1])) / var_plus
    rho_hat_t[1] = rho_hat_odd

    # Geyer's initial positive sequence
    t = 1
    while t < (n_samples - 3) and (rho_hat_even + rho_hat_odd) > 0:
        rho_hat_even = 1.0 - (mean_var - np.mean(acov[t + 1])) / var_plus
        rho_hat_odd = 1.0 - (mean_var - np.mean(acov[t + 2])) / var_plus
        if (rho_hat_even + rho_hat_odd) >= 0:
            rho_hat_t[t + 1] = rho_hat_even
            rho_hat_t[t + 2] = rho_hat_odd
        t += 2

    max_t = t - 2
    if rho_hat_even > 0:
        rho_hat_t[max_t + 1] = rho_hat_even

    # Geyer's initial monotone sequence
    t = 1
    while t <= max_t - 2:
        if (rho_hat_t[t + 1] + rho_hat_t[t + 2]) > (rho_hat_t[t - 1] + rho_hat_t[t]):
            rho_hat_t[t + 1] = (rho_hat_t[t - 1] + rho_hat_t[t]) / 2
            rho_hat_t[t + 2] = rho_hat_t[t + 1]
        t += 2

    ess = chains * n_samples
    tau_hat = -1.0 + 2.0 * np.sum(rho_hat_t[: max_t + 1]) + rho_hat_t[max_t + 1]
    tau_hat = max(tau_hat, 1.0 / np.log10(ess))

    return ess / tau_hat


def _relative_eff_function(
    func: Callable,
    chain_id: np.ndarray,
    cores: int,
    data: np.ndarray,
    draws: np.ndarray | None = None,
    method: Literal["bulk", "tail", "mean", "sd", "median", "mad", "local"] = "bulk",
    prob: float | tuple[float, float] | None = None,
) -> np.ndarray:
    """Compute relative efficiency for a function that returns likelihood values."""
    if data is None:
        raise ValueError("data required when x is a function")

    N = data.shape[0]

    def process_one(i: int) -> float:
        val_i = func(data_i=data[i : i + 1], draws=draws)
        return rel_eff(val_i, chain_id=chain_id, cores=1, method=method, prob=prob)[0]

    if cores == 1:
        n_eff = np.array([process_one(i) for i in range(N)])
    else:
        with mp.Pool(cores) as pool:
            n_eff = np.array(pool.map(process_one, range(N)))

    return n_eff


def _mat_to_chains(mat: np.ndarray, chain_id: np.ndarray) -> np.ndarray:
    """Convert matrix of MCMC draws to 3-D array organized by chain."""
    chain_id = np.asarray(chain_id)
    if len(chain_id) != mat.shape[0]:
        raise ValueError("chain_id must have length equal to nrow(mat)")

    chains = len(np.unique(chain_id))
    iterations = np.bincount(chain_id - 1)
    if not np.all(iterations == iterations[0]):
        raise ValueError("All chains must have same number of iterations")

    iterations = iterations[0]
    variables = mat.shape[1]

    arr = np.empty((iterations, chains, variables))
    for i in range(chains):
        arr[:, i, :] = mat[chain_id == i + 1]

    return arr
