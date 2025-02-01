"""Functions for computing weighted expectations using importance sampling weights."""

import numpy as np
from typing import Optional, Union, Sequence
from dataclasses import dataclass
from .psis import PSISObject, _logsumexp, _gpdfit


@dataclass
class ExpectationResult:
    """Container for results from expectation calculations.

    Attributes
    ----------
    value : Union[float, np.ndarray]
        The computed expectation value. For matrix inputs with type="quantile" and
        multiple probabilities, this will be a matrix with shape (len(probs),
        n_cols). Otherwise it will be a scalar or vector.
    pareto_k : Union[float, np.ndarray]
        Function-specific Pareto k diagnostic value(s). For matrix inputs this will
        be a vector with length n_cols, for vector inputs it will be a scalar.
    """

    value: Union[float, np.ndarray]
    pareto_k: Union[float, np.ndarray]


# Core weighted statistics functions
def _wmean(x: np.ndarray, w: np.ndarray) -> float:
    """Compute weighted mean."""
    return np.sum(w * x)


def _wvar(x: np.ndarray, w: np.ndarray) -> float:
    """Compute weighted variance using bias-corrected estimator.

    The denominator (1 - sum(w^2)) equals (ESS-1)/ESS, where effective sample size
    ESS is estimated as 1/sum(w^2). See "Monte Carlo theory, methods and examples"
    by Owen (2013).
    """
    if np.allclose(x, x[0]):
        return 0.0

    w_sum_sq = np.sum(w**2)
    if np.isclose(w_sum_sq, 1.0):
        return 0.0

    mean = _wmean(x, w)
    mean_sq = _wmean(x**2, w)
    var = (mean_sq - mean**2) / (1 - w_sum_sq)
    return max(var, 0.0)


def _wsd(x: np.ndarray, w: np.ndarray) -> float:
    """Compute weighted standard deviation."""
    var = _wvar(x, w)
    return np.sqrt(var) if var > 0 else 0.0


def _wquant(x: np.ndarray, w: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Compute weighted quantiles."""
    if np.allclose(w, w[0]):
        return np.quantile(x, probs)

    idx = np.argsort(x)
    x, w = x[idx], w[idx]

    ww = np.cumsum(w) / np.sum(w)  # Normalized cumulative weights

    qq = np.zeros_like(probs)
    for j, prob in enumerate(probs):
        ids = np.where(ww >= prob)[0]
        wi = ids[0]
        if wi == 0:
            qq[j] = x[0]
        else:
            w1 = ww[wi - 1]
            x1 = x[wi - 1]
            qq[j] = x1 + (x[wi] - x1) * (prob - w1) / (ww[wi] - w1)

    return qq


def _e_loo_khat(
    x: Optional[np.ndarray],
    log_ratios: np.ndarray,
    tail_len: Union[int, np.ndarray],
) -> float:
    """Compute Pareto k diagnostic."""
    r_theta = np.exp(log_ratios - np.max(log_ratios))

    # Handle right tail of importance ratios
    x_tail = -np.sort(-r_theta)[:tail_len]
    if len(x_tail) < 5 or np.allclose(x_tail, x_tail[0]):
        khat_r = np.inf
    else:
        exp_cutoff = np.exp(np.log(x_tail[-1]))
        khat_r, _ = _gpdfit(x_tail - exp_cutoff)

    # Early return for special cases
    if (
        x is None
        or np.allclose(x, x[0])
        or len(np.unique(x)) == 2
        or np.any(np.isnan(x))
        or np.any(np.isinf(x))
    ):
        return khat_r

    # Handle both tails of h*r
    hr = x * r_theta
    x_tail_left = np.sort(hr)[:tail_len]
    x_tail_right = -np.sort(-hr)[:tail_len]
    x_tail = np.concatenate([x_tail_left, x_tail_right])

    if len(x_tail) < 5 or np.allclose(x_tail, x_tail[0]):
        khat_hr = np.inf
    else:
        exp_cutoff = np.exp(np.log(x_tail[-1]))
        khat_hr, _ = _gpdfit(x_tail - exp_cutoff)

    if np.isnan(khat_hr) and np.isnan(khat_r):
        return np.nan
    return max(khat_hr, khat_r)


def _validate_inputs(
    x: np.ndarray,
    psis_object: PSISObject,
    type: str,
    probs: Optional[Union[float, Sequence[float]]],
    log_ratios: Optional[np.ndarray],
) -> None:
    """Validate input parameters."""
    if not isinstance(psis_object, PSISObject):
        raise TypeError("psis_object must be a PSISObject")

    if x.ndim == 1:
        if len(x) != len(psis_object.log_weights):
            raise ValueError("x and psis_object must have same length")
        if log_ratios is not None and len(log_ratios) != len(x):
            raise ValueError("log_ratios must have same length as x")
    else:  # x.ndim == 2
        if x.shape != psis_object.log_weights.shape:
            raise ValueError("x and psis_object must have same shape")
        if log_ratios is not None and log_ratios.shape != x.shape:
            raise ValueError("log_ratios must have same shape as x")

    if not np.all(np.isfinite(psis_object.log_weights)):
        raise ValueError("log weights must be finite")

    if type not in ["mean", "variance", "sd", "quantile"]:
        raise ValueError("type must be 'mean', 'variance', 'sd' or 'quantile'")

    if type == "quantile":
        if probs is None:
            raise ValueError("probs must be provided for quantile calculation")
        probs = np.asarray(probs)
        if not np.all((probs > 0) & (probs < 1)):
            raise ValueError("probs must be between 0 and 1")


def e_loo(
    x: Union[np.ndarray, Sequence[float]],
    psis_object: PSISObject,
    *,
    type: str = "mean",
    probs: Optional[Union[float, Sequence[float]]] = None,
    log_ratios: Optional[np.ndarray] = None,
) -> ExpectationResult:
    """Compute weighted expectations using importance sampling weights.

    Parameters
    ----------
    x : array-like
        Values to compute expectations for. Can be a vector or 2D array.
    psis_object : PSISObject
        Object containing importance sampling weights from PSIS.
    type : str, optional
        Type of expectation to compute. Options are:
        - "mean": weighted mean (default)
        - "variance": weighted variance
        - "sd": weighted standard deviation
        - "quantile": weighted quantiles
    probs : array-like, optional
        Probabilities for computing quantiles. Required if type="quantile".
    log_ratios : array-like, optional
        Raw (not smoothed) log ratios with same shape as x. If provided, these
        are used to compute more accurate Pareto k diagnostics.

    Returns
    -------
    ExpectationResult
        Container with computed expectation value and diagnostics.
    """
    x = np.asarray(x)
    if x.ndim == 1:
        return _e_loo_vector(
            x, psis_object, type=type, probs=probs, log_ratios=log_ratios
        )
    elif x.ndim == 2:
        return _e_loo_matrix(
            x, psis_object, type=type, probs=probs, log_ratios=log_ratios
        )
    else:
        raise ValueError("x must be 1D or 2D")


def _e_loo_vector(
    x: np.ndarray,
    psis_object: PSISObject,
    *,
    type: str = "mean",
    probs: Optional[Union[float, Sequence[float]]] = None,
    log_ratios: Optional[np.ndarray] = None,
) -> ExpectationResult:
    """Compute expectations for vector inputs."""
    _validate_inputs(x, psis_object, type, probs, log_ratios)

    # Normalize weights
    log_weights = psis_object.log_weights - _logsumexp(psis_object.log_weights)
    w = np.exp(np.clip(log_weights, -100, 0))

    # Compute expectation based on type
    if type == "mean":
        value = _wmean(x, w)
    elif type == "variance":
        value = _wvar(x, w)
    elif type == "sd":
        value = _wsd(x, w)
    else:  # type == "quantile"
        value = _wquant(x, w, np.asarray(probs))

    # Compute diagnostics
    if log_ratios is None:
        log_ratios = psis_object.log_weights

    h = None if type == "quantile" else x**2 if type in ("variance", "sd") else x
    pareto_k = _e_loo_khat(h, log_ratios, psis_object.tail_len)

    return ExpectationResult(value=value, pareto_k=pareto_k)


def _e_loo_matrix(
    x: np.ndarray,
    psis_object: PSISObject,
    *,
    type: str = "mean",
    probs: Optional[Union[float, Sequence[float]]] = None,
    log_ratios: Optional[np.ndarray] = None,
) -> ExpectationResult:
    """Compute expectations for matrix inputs."""
    _validate_inputs(x, psis_object, type, probs, log_ratios)

    # Normalize weights
    w = np.exp(psis_object.log_weights - _logsumexp(psis_object.log_weights, axis=0))

    # Initialize output arrays
    n_cols = x.shape[1]
    if type == "quantile":
        probs = np.asarray(probs)
        value = np.zeros((len(probs), n_cols))
    else:
        value = np.zeros(n_cols)

    # Compute expectations column by column
    for i in range(n_cols):
        if type == "mean":
            value[i] = _wmean(x[:, i], w[:, i])
        elif type == "variance":
            value[i] = _wvar(x[:, i], w[:, i])
        elif type == "sd":
            value[i] = _wsd(x[:, i], w[:, i])
        else:  # type == "quantile"
            value[:, i] = _wquant(x[:, i], w[:, i], probs)

    # Compute diagnostics
    if log_ratios is None:
        log_ratios = psis_object.log_weights

    h = None if type == "quantile" else x**2 if type in ("variance", "sd") else x
    pareto_k = np.array(
        [
            _e_loo_khat(
                None if h is None else h[:, i],
                log_ratios[:, i],
                (
                    psis_object.tail_len[i]
                    if isinstance(psis_object.tail_len, np.ndarray)
                    else psis_object.tail_len
                ),
            )
            for i in range(n_cols)
        ]
    )

    return ExpectationResult(value=value, pareto_k=pareto_k)
