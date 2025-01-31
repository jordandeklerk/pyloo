"""Pareto smoothed importance sampling (PSIS).

Reference: Vehtari et al. (2024). Pareto smoothed importance sampling. 
Journal of Machine Learning Research, 25(72):1-58.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Union

@dataclass
class PSISObject:
    """Object containing smoothed log weights and diagnostics."""
    log_weights: np.ndarray
    pareto_k: np.ndarray
    n_eff: Optional[np.ndarray] = None
    r_eff: Optional[np.ndarray] = None

def psislw(
    log_ratios: np.ndarray,
    r_eff: Union[float, np.ndarray] = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Main PSIS implementation."""
    if log_ratios.ndim == 1:
        log_ratios = log_ratios.reshape(-1, 1)
    
    n_samples, n_obs = log_ratios.shape

    if isinstance(r_eff, (int, float)):
        r_eff = float(r_eff)  
    elif len(r_eff) != n_obs:
        raise ValueError("r_eff must be a scalar or have length equal to n_observations")

    tail_len = int(np.ceil(min(0.2 * n_samples, 3 * np.sqrt(n_samples / r_eff))))
    smoothed_log_weights = np.empty_like(log_ratios)
    pareto_k = np.full(n_obs, np.inf)

    for i in range(n_obs):
        x = log_ratios[:, i]
        x = x - np.max(x)
        
        sorted_idx = np.argsort(x)
        cutoff_idx = -tail_len - 1
        cutoff = max(x[sorted_idx[cutoff_idx]], np.log(np.finfo(float).tiny))
        
        tail_ids = x > cutoff
        x_tail = x[tail_ids]
        n_tail = len(x_tail)
        
        if n_tail <= 4 or len(np.unique(x_tail)) == 1:
            k = np.inf
            smoothed_log_weights[:, i] = x
        else:
            tail_order = np.argsort(x_tail)
            exp_cutoff = np.exp(cutoff)
            x_tail = np.exp(x_tail) - exp_cutoff
            
            k, sigma = _gpdfit(x_tail[tail_order])
            
            if np.isfinite(k):
                sti = (np.arange(0.5, n_tail) / n_tail)
                smoothed_tail = _gpinv(sti, k, sigma)
                smoothed_tail = np.log(smoothed_tail + exp_cutoff)
                
                x_smooth = x.copy()
                x_smooth[tail_ids] = smoothed_tail[np.argsort(tail_order)]
                x_smooth[x_smooth > 0] = 0
                smoothed_log_weights[:, i] = x_smooth
            else:
                smoothed_log_weights[:, i] = x
            
            pareto_k[i] = k
            
        smoothed_log_weights[:, i] = smoothed_log_weights[:, i] - _logsumexp(smoothed_log_weights[:, i])

    if log_ratios.ndim == 1:
        smoothed_log_weights = smoothed_log_weights.ravel()
        pareto_k = pareto_k.ravel()
        
    return smoothed_log_weights, pareto_k

def _gpdfit(x: np.ndarray) -> Tuple[float, float]:
    """Estimate GPD parameters."""
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
    """Inverse GPD function."""
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

def _logsumexp(x: np.ndarray) -> float:
    """Stable implementation of log(sum(exp(x)))."""
    x_max = np.max(x)
    if x_max == -np.inf:
        return -np.inf
    
    sum_exp = np.sum(np.exp(x - x_max))
    return x_max + np.log(sum_exp)