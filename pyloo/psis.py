"""Functions for Pareto smoothed importance sampling (PSIS)."""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Union
from collections.abc import Sequence

@dataclass
class PSISObject:
    """Object containing smoothed log weights and diagnostics from PSIS calculations.
    
    Attributes
    ----------
    log_weights : np.ndarray
        Smoothed log weights
    pareto_k : np.ndarray
        Estimated shape parameters for the generalized Pareto distribution
    n_eff : Optional[np.ndarray]
        Effective sample size estimate (if computed)
    r_eff : Optional[np.ndarray]
        Relative efficiency estimate (if provided)
    tail_len : Optional[Union[int, np.ndarray]]
        Length of the tail for Pareto smoothing. Can be a scalar or array.
    """
    log_weights: np.ndarray
    pareto_k: np.ndarray
    n_eff: Optional[np.ndarray] = None
    r_eff: Optional[np.ndarray] = None
    tail_len: Optional[Union[int, np.ndarray]] = None

def psislw(
    log_ratios: np.ndarray,
    r_eff: Union[float, np.ndarray] = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Pareto smoothed importance sampling (PSIS) log weights.
    
    Parameters
    ----------
    log_ratios : np.ndarray
        Array of shape (n_samples, n_observations) containing log importance ratios
        (for example, log-likelihood values).
    r_eff : Union[float, np.ndarray], optional
        Relative MCMC efficiency (effective sample size / total samples) used in 
        tail length calculation. Can be a scalar or array of length n_observations.
        Default is 1.0 (for independent draws).
        
    Returns
    -------
    smoothed_log_weights : np.ndarray
        Array of same shape as log_ratios containing smoothed log weights
    pareto_k : np.ndarray
        Array of length n_observations containing estimated shape parameters for the
        generalized Pareto distribution fit to the tail of the log ratios
        
    Notes
    -----
    The Pareto k diagnostic values indicate the reliability of the importance sampling
    estimates:
    * k < 0.5   : excellent
    * 0.5 <= k <= 0.7 : good
    * k > 0.7   : unreliable
    
    Examples
    --------
    Calculate PSIS weights for log-likelihood values:

    .. ipython::

        In [1]: import numpy as np
           ...: from pyloo import psislw
           ...: log_liks = np.random.normal(size=(1000, 100))
           ...: weights, k = psislw(log_liks)
           ...: print(f"Mean Pareto k: {k.mean():.3f}")
    
    See Also
    --------
    PSISObject : Container for PSIS results including diagnostics
    
    References
    ----------
    .. [1] Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P. C. (2024).
           Pareto smoothed importance sampling. Journal of Machine Learning Research, 25(72):1-58.
    """
    if log_ratios.ndim == 1:
        log_ratios = log_ratios.reshape(-1, 1)
    
    n_samples, n_obs = log_ratios.shape

    if isinstance(r_eff, (int, float)):
        r_eff = float(r_eff)  
    elif len(r_eff) != n_obs:
        raise ValueError("r_eff must be a scalar or have length equal to n_observations")

    tail_len = np.ceil(np.minimum(0.2 * n_samples, 3 * np.sqrt(n_samples / r_eff))).astype(int)
    smoothed_log_weights = log_ratios.copy()
    pareto_k = np.zeros(n_obs)

    for i in range(n_obs):
        x = smoothed_log_weights[:, i]
        x = x - np.max(x)
        
        sorted_idx = np.argsort(x)
        cutoff_idx = -int(tail_len[i] if isinstance(tail_len, np.ndarray) else tail_len) - 1
        cutoff = np.maximum(x[sorted_idx[cutoff_idx]], np.log(np.finfo(float).tiny))
        
        tail_ids = np.where(x > cutoff)[0] 
        n_tail = len(tail_ids)
        
        if n_tail <= 4:
            # Not enough tail samples for GPD fit
            pareto_k[i] = np.inf
        else:
            tail_order = np.argsort(x[tail_ids])
            x_tail = x[tail_ids][tail_order]
            exp_cutoff = np.exp(cutoff)
            x_tail = np.exp(x_tail) - exp_cutoff
            
            k, sigma = _gpdfit(x_tail)
            pareto_k[i] = k
            
            if np.isfinite(k):
                sti = (np.arange(0.5, n_tail) / n_tail)
                smoothed_tail = _gpinv(sti, k, sigma)
                smoothed_tail = np.log(smoothed_tail + exp_cutoff)
                
                x[tail_ids[tail_order]] = smoothed_tail
                x[x > 0] = 0
            
        smoothed_log_weights[:, i] = x - _logsumexp(x)

    if log_ratios.ndim == 1:
        smoothed_log_weights = smoothed_log_weights.ravel()
        pareto_k = pareto_k.ravel()
        
    return smoothed_log_weights, pareto_k

def _gpdfit(x: np.ndarray) -> Tuple[float, float]:
    """Estimate generalized Pareto distribution parameters using method of moments."""
    prior_bs = 3
    prior_k = 10
    n = len(x)
    m_est = 30 + int(n ** 0.5)
    
    b_ary = 1 - np.sqrt(m_est / (np.arange(1, m_est + 1) - 0.5))
    b_ary /= prior_bs * x[int(n/4 + 0.5) - 1]
    b_ary += 1 / x[-1]
    
    k_ary = np.mean(np.log1p(-b_ary[:, None] * x), axis=1)
    len_scale = n * (np.log(-b_ary / k_ary) - k_ary - 1)
    weights = 1 / np.sum(np.exp(len_scale - len_scale[:, None]), axis=1)
    
    real_idxs = weights >= 10 * np.finfo(float).eps
    if not np.all(real_idxs):
        weights = weights[real_idxs]
        b_ary = b_ary[real_idxs]
    
    weights = weights / np.sum(weights)
    b_post = np.sum(b_ary * weights)
    k_post = np.mean(np.log1p(-b_post * x))
    
    sigma = -k_post / b_post
    k_post = (n * k_post + prior_k * 0.5) / (n + prior_k)
    
    return k_post, sigma

def _gpinv(probs: np.ndarray, kappa: float, sigma: float) -> np.ndarray:
    """Compute inverse generalized Pareto distribution function."""
    x = np.full_like(probs, np.nan)
    if sigma <= 0:
        return x
    
    ok = (probs > 0) & (probs < 1)
    if np.all(ok):
        if np.abs(kappa) < np.finfo(float).eps:
            x = -np.log1p(-probs)
        else:
            x = np.expm1(-kappa * np.log1p(-probs)) / kappa
        x *= sigma
    else:
        if np.abs(kappa) < np.finfo(float).eps:
            x[ok] = -np.log1p(-probs[ok])
        else:
            x[ok] = np.expm1(-kappa * np.log1p(-probs[ok])) / kappa
        x *= sigma
        x[probs == 0] = 0
        x[probs == 1] = np.inf if kappa >= 0 else -sigma / kappa
    
    return x

def _logsumexp(ary, *, b=None, b_inv=None, axis=None, keepdims=False, out=None, copy=True):
    """Stable logsumexp implementation."""
    ary = np.asarray(ary)
    if ary.dtype.kind == "i":
        ary = ary.astype(np.float64)
    dtype = ary.dtype.type
    shape = ary.shape
    shape_len = len(shape)

    if isinstance(axis, Sequence):
        axis = tuple(axis_i if axis_i >= 0 else shape_len + axis_i for axis_i in axis)
        agroup = axis
    else:
        axis = axis if (axis is None) or (axis >= 0) else shape_len + axis
        agroup = (axis,)

    shape_max = (
        tuple(1 for _ in shape)
        if axis is None
        else tuple(1 if i in agroup else d for i, d in enumerate(shape))
    )

    if out is None:
        if not keepdims:
            out_shape = (
                tuple()
                if axis is None
                else tuple(d for i, d in enumerate(shape) if i not in agroup)
            )
        else:
            out_shape = shape_max
        out = np.empty(out_shape, dtype=dtype)

    if b_inv == 0:
        return np.full_like(out, np.inf, dtype=dtype) if out.shape else np.inf
    if b_inv is None and b == 0:
        return np.full_like(out, -np.inf) if out.shape else -np.inf

    ary_max = np.empty(shape_max, dtype=dtype)
    ary.max(axis=axis, keepdims=True, out=ary_max)
    if copy:
        ary = ary.copy()
    ary -= ary_max
    np.exp(ary, out=ary)
    ary.sum(axis=axis, keepdims=keepdims, out=out)
    np.log(out, out=out)

    if b_inv is not None:
        ary_max -= np.log(b_inv)
    elif b:
        ary_max += np.log(b)
    out += ary_max if keepdims else ary_max.squeeze()

    return out if out.shape else dtype(out)
