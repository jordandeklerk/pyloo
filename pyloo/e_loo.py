"""Functions for computing weighted expectations using importance sampling."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .psis import _gpdfit
from .utils import _logsumexp


@dataclass
class ExpectationResult:
    """Container for results from expectation calculations.

    Attributes
    ----------
    value : float | np.ndarray
        The computed expectation value. For matrix inputs with type="quantile"
        and multiple probabilities, this will be a matrix with shape
        (len(probs), n_cols). Otherwise it will be a scalar or vector.
    pareto_k : float | np.ndarray
        Function-specific Pareto k diagnostic value(s). For matrix inputs this
        will be a vector with length n_cols, for vector inputs it will be a
        scalar.
    """

    value: float | np.ndarray
    pareto_k: float | np.ndarray


def e_loo(
    x: np.ndarray | Sequence[float],
    log_weights: np.ndarray,
    *,
    type: str = "mean",
    probs: float | Sequence[float] | None = None,
    log_ratios: np.ndarray | None = None,
    pareto_k: float | np.ndarray | None = None,
) -> ExpectationResult:
    """Compute weighted expectations using importance sampling weights.

    Parameters
    ----------
    x : array-like
        Values to compute expectations for. Can be a vector or 2D array.
    log_weights : array-like
        Log weights from importance sampling. Can be a vector or 2D array.
    type : str, optional
        Type of expectation to compute. Options are:
        - "mean": weighted mean (default)
        - "variance": weighted variance
        - "sd": weighted standard deviation
        - "quantile": weighted quantiles
    probs : array-like, optional
        Probabilities for computing quantiles. Required if type="quantile".
    log_ratios : array-like, optional
        Raw (not smoothed) log ratios with same shape as x. If provided,
        these are used to compute more accurate Pareto k diagnostics.
    pareto_k : float or array-like, optional
        Pre-computed Pareto k values. If not provided, they will be estimated
        from the log weights.

    Returns
    -------
    ExpectationResult
        Container with computed expectation value and diagnostics.

    Examples
    --------
    Compute weighted mean using importance sampling:

    .. code-block:: python

        import numpy as np
        from pyloo import compute_importance_weights, e_loo

        x = np.random.normal(size=(1000, 100))

        log_ratios = np.random.normal(size=(1000, 100))
        weights, k = compute_importance_weights(log_ratios)
        result = e_loo(x, weights, pareto_k=k)

        print(f"Mean value: {result.value.mean():.3f}")
    """
    x = np.asarray(x)
    if x.ndim == 1:
        return _e_loo_vector(
            x,
            log_weights,
            type=type,
            probs=probs,
            log_ratios=log_ratios,
            pareto_k=pareto_k,
        )
    elif x.ndim == 2:
        return _e_loo_matrix(
            x,
            log_weights,
            type=type,
            probs=probs,
            log_ratios=log_ratios,
            pareto_k=pareto_k,
        )
    else:
        raise ValueError("x must be 1D or 2D")


def _validate_inputs(
    x: np.ndarray,
    log_weights: np.ndarray,
    type: str,
    probs: float | Sequence[float] | None,
    log_ratios: np.ndarray | None,
) -> None:
    """Validate input parameters."""
    if x.ndim == 1:
        if len(x) != len(log_weights):
            raise ValueError("x and log_weights must have same length")
        if log_ratios is not None and len(log_ratios) != len(x):
            raise ValueError("log_ratios must have same length as x")
    else:  # x.ndim == 2
        if x.shape != log_weights.shape:
            raise ValueError("x and log_weights must have same shape")
        if log_ratios is not None and log_ratios.shape != x.shape:
            raise ValueError("log_ratios must have same shape as x")

    if not np.all(np.isfinite(log_weights)):
        raise ValueError("log weights must be finite")

    if type not in ["mean", "variance", "sd", "quantile"]:
        raise ValueError("type must be 'mean', 'variance', 'sd' or 'quantile'")

    if type == "quantile":
        if probs is None:
            raise ValueError("probs must be provided for quantile calculation")
        probs_array = np.asarray(probs)
        if not np.all((probs_array > 0) & (probs_array < 1)):
            raise ValueError("probs must be between 0 and 1")


def _wmean(x: np.ndarray, w: np.ndarray) -> float:
    """Compute weighted mean."""
    return np.sum(w * x)


def _wvar(x: np.ndarray, w: np.ndarray) -> float:
    """Compute weighted variance using bias-corrected estimator.

    The denominator (1 - sum(w^2)) equals (ESS-1)/ESS, where effective sample
    size ESS is estimated as 1/sum(w^2). See "Monte Carlo theory, methods and
    examples" by Owen (2013).
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

    ww = np.cumsum(w) / np.sum(w)

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
    x: np.ndarray | None,
    log_ratios: np.ndarray,
    tail_len: int = 20,
) -> float:
    """Compute Pareto k diagnostic."""
    r_theta = np.exp(log_ratios - np.max(log_ratios))

    x_tail = -np.sort(-r_theta)[:tail_len]
    if len(x_tail) < 5 or np.allclose(x_tail, x_tail[0]):
        khat_r = np.inf
    else:
        exp_cutoff = np.exp(np.log(x_tail[-1]))
        khat_r, _ = _gpdfit(x_tail - exp_cutoff)

    if (
        x is None
        or np.allclose(x, x[0])
        or len(np.unique(x)) == 2
        or np.any(np.isnan(x))
        or np.any(np.isinf(x))
    ):
        return khat_r

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


def _e_loo_vector(
    x: np.ndarray,
    log_weights: np.ndarray,
    *,
    type: str = "mean",
    probs: float | Sequence[float] | None = None,
    log_ratios: np.ndarray | None = None,
    pareto_k: float | None = None,
) -> ExpectationResult:
    """Compute expectations for vector inputs."""
    _validate_inputs(x, log_weights, type, probs, log_ratios)

    log_weights = log_weights - _logsumexp(log_weights)
    w = np.exp(np.clip(log_weights, -100, 0))

    if type == "mean":
        value = _wmean(x, w)
    elif type == "variance":
        value = _wvar(x, w)
    elif type == "sd":
        value = _wsd(x, w)
    else:
        value = _wquant(x, w, np.asarray(probs))

    if pareto_k is None:
        if log_ratios is None:
            log_ratios = log_weights
        h = None if type == "quantile" else x**2 if type in ("variance", "sd") else x
        pareto_k = _e_loo_khat(h, log_ratios)

    return ExpectationResult(value=value, pareto_k=pareto_k)


def _e_loo_matrix(
    x: np.ndarray,
    log_weights: np.ndarray,
    *,
    type: str = "mean",
    probs: float | Sequence[float] | None = None,
    log_ratios: np.ndarray | None = None,
    pareto_k: np.ndarray | None = None,
) -> ExpectationResult:
    """Compute expectations for matrix inputs."""
    _validate_inputs(x, log_weights, type, probs, log_ratios)

    w = np.exp(log_weights - _logsumexp(log_weights, axis=0))

    n_cols = x.shape[1]
    if type == "quantile":
        if probs is None:
            raise ValueError("probs must be provided for quantile calculation")
        probs_array = np.asarray(probs)
        value = np.zeros((len(probs_array), n_cols))
    else:
        value = np.zeros(n_cols)

    for i in range(n_cols):
        if type == "mean":
            value[i] = _wmean(x[:, i], w[:, i])
        elif type == "variance":
            value[i] = _wvar(x[:, i], w[:, i])
        elif type == "sd":
            value[i] = _wsd(x[:, i], w[:, i])
        else:  # type == "quantile"
            value[:, i] = _wquant(x[:, i], w[:, i], probs_array)

    if pareto_k is None:
        if log_ratios is None:
            log_ratios = log_weights
        h = None if type == "quantile" else x**2 if type in ("variance", "sd") else x
        pareto_k = np.array([
            _e_loo_khat(None if h is None else h[:, i], log_ratios[:, i])
            for i in range(n_cols)
        ])

    return ExpectationResult(value=value, pareto_k=pareto_k)
