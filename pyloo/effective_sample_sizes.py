"""Functions for computing effective sample sizes and relative efficiencies."""

import multiprocessing as mp
from typing import Callable, Optional, Union

import numpy as np


def compute_relative_efficiency(
    x: Union[np.ndarray, Callable],
    chain_id: Optional[np.ndarray] = None,
    cores: int = 1,
    data: Optional[np.ndarray] = None,
    draws: Optional[np.ndarray] = None,
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

    Returns
    -------
    np.ndarray
        Vector of relative effective sample sizes, bounded between 0 and 1.
        Values closer to 1 indicate better mixing of the chains.

    Examples
    --------
    Calculate relative efficiency for MCMC samples:

    .. ipython::

        In [1]: import numpy as np
           ...: from pyloo import compute_relative_efficiency
           ...: # Generate fake MCMC samples
           ...: samples = np.random.normal(size=(1000, 4, 10))  # 1000 iterations, 4 chains, 10 parameters
           ...: r_eff = compute_relative_efficiency(samples)
           ...: print(f"Mean relative efficiency: {r_eff.mean():.3f}")

    See Also
    --------
    compute_mcmc_effective_size : Calculate raw effective sample size
    compute_psis_effective_size : Compute effective sample size for PSIS
    """
    if callable(x):
        return _relative_eff_function(x, chain_id, cores, data, draws)

    x = np.asarray(x)

    if x.ndim == 1:
        x = x.reshape(-1, 1)
        if chain_id is not None:
            x = _convert_matrix_to_chains(x, chain_id)
        else:
            x = x.reshape(x.shape[0], 1, 1)
    elif x.ndim == 2:
        if chain_id is None:
            raise ValueError("chain_id required for 2-D input")
        x = _convert_matrix_to_chains(x, chain_id)
    elif x.ndim != 3:
        raise ValueError("Input must be 1-D, 2-D, or 3-D array")

    S = x.shape[0] * x.shape[1]

    if cores == 1:
        n_eff = np.array([compute_mcmc_effective_size(x[:, :, i]) for i in range(x.shape[2])])
    else:
        with mp.Pool(cores) as pool:
            n_eff = np.array(pool.map(compute_mcmc_effective_size, [x[:, :, i] for i in range(x.shape[2])]))

    return np.minimum(n_eff / S, 1.0)


def _relative_eff_function(
    func: Callable,
    chain_id: np.ndarray,
    cores: int,
    data: np.ndarray,
    draws: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute relative efficiency for a function that returns likelihood values."""
    if data is None:
        raise ValueError("data required when x is a function")

    N = data.shape[0]

    def process_one(i: int) -> float:
        val_i = func(data_i=data[i : i + 1], draws=draws)
        return compute_relative_efficiency(val_i, chain_id=chain_id, cores=1)[0]

    if cores == 1:
        n_eff = np.array([process_one(i) for i in range(N)])
    else:
        with mp.Pool(cores) as pool:
            n_eff = np.array(pool.map(process_one, range(N)))

    return n_eff


def compute_psis_effective_size(
    w: np.ndarray, r_eff: Optional[Union[float, np.ndarray]] = None
) -> Union[float, np.ndarray]:
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

    .. ipython::

        In [1]: import numpy as np
           ...: from pyloo import compute_psis_effective_size
           ...: # Generate fake importance weights
           ...: weights = np.random.gamma(1, 1, size=(1000, 100))
           ...: weights /= weights.sum(axis=0)  # normalize
           ...: n_eff = compute_psis_effective_size(weights)
           ...: print(f"Mean effective sample size: {n_eff.mean():.1f}")

    See Also
    --------
    compute_relative_efficiency : Compute MCMC relative efficiency
    compute_mcmc_effective_size : Calculate raw effective sample size
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


def compute_mcmc_effective_size(sims: np.ndarray) -> float:
    """Calculate MCMC effective sample size using Stan's approach.

    Parameters
    ----------
    sims : np.ndarray
        Array of MCMC draws. Can be 1-D (vector) or 2-D (matrix).
        For 2-D input, rows represent iterations and columns represent chains.

    Returns
    -------
    float
        The effective sample size estimate.

    Notes
    -----
    This function implements Stan's approach to computing effective sample size,
    which accounts for autocorrelation within chains and cross-chain correlation.
    The implementation uses FFT for efficient computation of autocovariances
    and includes adjustments to ensure reliable estimates even with relatively
    few samples.

    Examples
    --------
    Calculate effective sample size for MCMC draws:

    .. ipython::

        In [1]: import numpy as np
           ...: from pyloo import compute_mcmc_effective_size
           ...: # Generate fake MCMC samples
           ...: samples = np.random.normal(size=(1000, 4))  # 1000 iterations, 4 chains
           ...: ess = compute_mcmc_effective_size(samples)
           ...: print(f"Effective sample size: {ess:.1f}")

    See Also
    --------
    compute_relative_efficiency : Compute relative efficiency
    compute_psis_effective_size : Compute effective sample size for PSIS

    References
    ----------
    .. [1] Stan Development Team. (2024). Stan Reference Manual.
           https://mc-stan.org/docs/reference-manual/effective-sample-size.html
    """
    sims = np.asarray(sims)
    if sims.ndim == 1:
        sims = sims.reshape(-1, 1)

    chains = sims.shape[1]
    n_samples = sims.shape[0]

    acov_chains = []
    for i in range(chains):
        chain = sims[:, i]
        acov = _autocovariance(chain)
        acov_chains.append(acov)
    acov = np.column_stack(acov_chains)

    chain_mean = np.mean(sims, axis=0)
    mean_var = np.mean(acov[0]) * n_samples / (n_samples - 1)
    var_plus = mean_var * (n_samples - 1) / n_samples

    if chains > 1:
        var_plus += np.var(chain_mean, ddof=1)

    rho_hat_t = np.zeros(n_samples)
    t = 0
    rho_hat_even = 1.0
    rho_hat_t[t] = rho_hat_even
    rho_hat_odd = 1.0 - (mean_var - np.mean(acov[t + 1])) / var_plus
    rho_hat_t[t + 1] = rho_hat_odd

    while t < len(acov) - 5 and not np.isnan(rho_hat_even + rho_hat_odd) and (rho_hat_even + rho_hat_odd > 0):
        t += 2
        rho_hat_even = 1.0 - (mean_var - np.mean(acov[t])) / var_plus
        rho_hat_odd = 1.0 - (mean_var - np.mean(acov[t + 1])) / var_plus
        if (rho_hat_even + rho_hat_odd) >= 0:
            rho_hat_t[t] = rho_hat_even
            rho_hat_t[t + 1] = rho_hat_odd

    max_t = t
    if rho_hat_even > 0:
        rho_hat_t[max_t] = rho_hat_even

    t = 0
    while t <= max_t - 4:
        t += 2
        if rho_hat_t[t] + rho_hat_t[t + 1] > rho_hat_t[t - 2] + rho_hat_t[t - 1]:
            rho_hat_t[t] = (rho_hat_t[t - 2] + rho_hat_t[t - 1]) / 2
            rho_hat_t[t + 1] = rho_hat_t[t]

    ess = chains * n_samples
    tau_hat = -1.0 + 2.0 * np.sum(rho_hat_t[:max_t]) + rho_hat_t[max_t]
    tau_hat = max(tau_hat, 1.0 / np.log10(ess))
    ess = ess / tau_hat

    return float(ess)


def _autocovariance(x: np.ndarray) -> np.ndarray:
    """Compute autocovariance estimates for every lag."""
    n = len(x)
    if n < 2:
        raise ValueError("Array too short")

    n_fft = _fft_next_good_size(2 * n)

    x = x - np.mean(x)
    var = np.var(x, ddof=1)
    if var == 0:
        return np.zeros(n)

    y = np.zeros(n_fft)
    y[:n] = x

    f = np.fft.fft(y)
    acf = np.fft.ifft(f * np.conjugate(f))[:n]

    acf = acf.real / (n * var)
    acf[0] = 1.0

    return acf


def _fft_next_good_size(n: int) -> int:
    """Find optimal next size for FFT."""
    if n <= 2:
        return 2

    while True:
        m = n
        while (m % 2) == 0:
            m //= 2
        while (m % 3) == 0:
            m //= 3
        while (m % 5) == 0:
            m //= 5
        if m <= 1:
            return n
        n += 1


def _convert_matrix_to_chains(mat: np.ndarray, chain_id: np.ndarray) -> np.ndarray:
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
