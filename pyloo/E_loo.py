"""Functions for computing weighted expectations using PSIS weights."""

import numpy as np
from scipy.stats import pareto
from dataclasses import dataclass
from typing import Optional

from .psis import PSISObject


MIN_TAIL_LENGTH = 5
DEFAULT_TAIL_LENGTH = 25
PARETO_K_WARN = 0.7
PARETO_K_GOOD = 0.5


@dataclass
class ExpectationResult:
    """Container for expectation calculation results."""
    value: np.ndarray
    pareto_k: np.ndarray


def validate_psis_inputs(
    x: np.ndarray,
    psis_object: PSISObject,
    log_ratios: Optional[np.ndarray] = None,
) -> None:
    """Validate inputs for PSIS-based calculations."""
    if not isinstance(psis_object, PSISObject):
        raise ValueError("psis_object must be a PSISObject instance")
        
    if x.shape[0] != psis_object.log_weights.shape[0]:
        raise ValueError("x and psis_object.log_weights must have same first dimension")
        
    if log_ratios is not None:
        if log_ratios.shape != psis_object.log_weights.shape:
            raise ValueError("log_ratios must have same shape as psis_object.log_weights")


def validate_weights(weights: np.ndarray) -> None:
    """Validate importance sampling weights."""
    if not np.all(np.isfinite(weights)):
        raise ValueError("weights must be finite")
        
    if not np.all(weights >= 0):
        raise ValueError("weights must be non-negative")
        
    if not np.allclose(np.sum(weights, axis=0), 1.0):
        raise ValueError("weights must sum to 1 along first axis")


def normalize_weights(weights: np.ndarray, log: bool = True) -> np.ndarray:
    """Normalize importance sampling weights."""
    if log:
        weights = np.exp(weights - np.max(weights, axis=0))
    weights = weights / np.sum(weights, axis=0)
    return weights


def e_loo(
    x: np.ndarray,
    psis_object: PSISObject,
    log_ratios: Optional[np.ndarray] = None,
    type: str = "mean",
    probs: Optional[np.ndarray] = None,
) -> ExpectationResult:
    """Compute weighted expectations using PSIS weights.
    
    Parameters
    ----------
    x : np.ndarray
        Values to compute expectations for
    psis_object : PSISObject
        PSIS object containing smoothed log weights
    log_ratios : np.ndarray, optional
        Raw (not smoothed) log ratios. If working with log-likelihood values,
        these are the negative of those values. Used for more accurate Pareto k diagnostics.
    type : str, optional
        Type of expectation to compute: "mean", "variance", "sd", or "quantile"
    probs : np.ndarray, optional
        For type="quantile", the probabilities at which to compute quantiles
        
    Returns
    -------
    ExpectationResult
        Container with computed values and diagnostics
    """
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    
    validate_psis_inputs(x, psis_object, log_ratios)
    
    if type not in ["mean", "variance", "sd", "quantile"]:
        raise ValueError("type must be 'mean', 'variance', 'sd' or 'quantile'")
        
    if type == "quantile" and probs is None:
        raise ValueError("probs must be provided when type='quantile'")
        
    if probs is not None:
        if not np.all((probs > 0) & (probs < 1)):
            raise ValueError("probs must be between 0 and 1")
    
    weights = normalize_weights(psis_object.log_weights)
    validate_weights(weights)
    
    # Compute expectation based on type
    if type == "mean":
        value = _wmean(x, weights)
    elif type == "variance":
        value = _wvar(x, weights)
    elif type == "sd":
        value = _wsd(x, weights)
    else:  # type == "quantile"
        value = _wquantile(x, weights, probs)
        
    # Compute function-specific k diagnostic
    if log_ratios is None:
        # Use of smoothed ratios gives slightly optimistic Pareto k estimates
        log_ratios = psis_object.log_weights
        
    h = None
    if type == "mean":
        h = x
    elif type in ["variance", "sd"]:
        h = x**2
        
    pareto_k = _e_loo_khat(h, log_ratios, psis_object)
    
    return ExpectationResult(value=value, pareto_k=pareto_k)


def _wmean(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Compute weighted mean."""
    return np.sum(weights * x, axis=0)


def _wvar(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Compute weighted variance.
    
    The denominator (1 - sum(w^2)) is equal to (ESS-1)/ESS, where effective sample size
    ESS is estimated with the generic target quantity invariant estimate 1/sum(w^2).
    See "Monte Carlo theory, methods and examples" by Owen (2013).
    """
    x_mean = _wmean(x, weights)
    x2_mean = _wmean(x**2, weights)
    w_sum_sq = np.sum(weights**2, axis=0)
    return (x2_mean - x_mean**2) / (1 - w_sum_sq)


def _wsd(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Compute weighted standard deviation."""
    return np.sqrt(_wvar(x, weights))


def _wquantile(x: np.ndarray, weights: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Compute weighted quantiles."""
    if not np.all((probs > 0) & (probs < 1)):
        raise ValueError("probs must be between 0 and 1")
        
    if np.allclose(weights - weights[0, 0], 0):
        return np.quantile(x, probs, axis=0)
    
    n_probs = len(probs)
    n_obs = x.shape[1]
    result = np.empty((n_probs, n_obs))
    
    for i in range(n_obs):
        x_i = x[:, i]
        w_i = weights[:, i]
        
        sort_idx = np.argsort(x_i)
        x_sorted = x_i[sort_idx]
        w_sorted = w_i[sort_idx]
        
        w_cumsum = np.cumsum(w_sorted)
        w_cumsum /= w_cumsum[-1]
        
        for j, prob in enumerate(probs):
            # Find indices where cumsum crosses probability
            idx = np.searchsorted(w_cumsum, prob)
            
            if idx == 0:
                result[j, i] = x_sorted[0]
            else:
                # Linear interpolation
                w1 = w_cumsum[idx - 1]
                w2 = w_cumsum[idx]
                x1 = x_sorted[idx - 1]
                x2 = x_sorted[idx]
                t = (prob - w1) / (w2 - w1)
                result[j, i] = x1 + t * (x2 - x1)
                
    return result


def _e_loo_khat(
    h: Optional[np.ndarray],
    log_ratios: np.ndarray,
    psis_object: PSISObject,
) -> np.ndarray:
    """Compute function-specific k-hat diagnostics."""
    n_obs = log_ratios.shape[1]
    k = np.zeros(n_obs)
    
    for i in range(n_obs):
        r = np.exp(log_ratios[:, i] - np.max(log_ratios[:, i]))
        tail_len = getattr(psis_object, "tail_len", None)
        if tail_len is None:
            tail_len = min(DEFAULT_TAIL_LENGTH, len(r) // 4)
        tail_len = int(tail_len) if np.isscalar(tail_len) else int(tail_len[i])
        
        # Get k estimate for right tail of r
        sorted_r = np.sort(r)
        threshold = sorted_r[-tail_len]
        tail_r = r[r > threshold] - threshold
        
        if len(tail_r) < MIN_TAIL_LENGTH:
            k[i] = np.inf
            continue
            
        k_r = pareto.fit(tail_r, floc=0, scale=tail_r.mean())[0]
        
        if h is None or _is_degenerate(h[:, i]):
            k[i] = k_r
        else:
            # Get k estimate for both tails of h*r
            hr = h[:, i] * r
            sorted_hr = np.sort(hr)
            threshold_left = sorted_hr[tail_len-1]
            threshold_right = sorted_hr[-tail_len]
            
            tail_left = threshold_left - hr[hr < threshold_left]
            tail_right = hr[hr > threshold_right] - threshold_right
            
            if len(tail_left) < MIN_TAIL_LENGTH or len(tail_right) < MIN_TAIL_LENGTH:
                k[i] = k_r
                continue
                
            k_hr_left = pareto.fit(tail_left, floc=0, scale=tail_left.mean())[0]
            k_hr_right = pareto.fit(tail_right, floc=0, scale=tail_right.mean())[0]
            
            k[i] = max(k_r, k_hr_left, k_hr_right)
            
    return k


def _is_degenerate(x: np.ndarray) -> bool:
    """Check if array is constant, binary, or contains non-finite values."""
    if not np.all(np.isfinite(x)):
        return True
    x = np.unique(x)
    return len(x) <= 2