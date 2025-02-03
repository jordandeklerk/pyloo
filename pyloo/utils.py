"""Helper functions for pyloo package with some adaptations from ArviZ."""

import warnings
from collections.abc import Sequence
from typing import Any, Dict, Optional, Tuple, Union

import arviz as az
import numpy as np
import pandas as pd
from arviz import InferenceData
from scipy.fftpack import next_fast_len
from scipy.interpolate import CubicSpline

_FLOAT_EPS = np.finfo(float).eps


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

    shape_max = tuple(1 for _ in shape) if axis is None else tuple(1 if i in agroup else d for i, d in enumerate(shape))

    if out is None:
        if not keepdims:
            out_shape = () if axis is None else tuple(d for i, d in enumerate(shape) if i not in agroup)
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


def autocov(ary: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute autocovariance estimates for every lag for the input array.

    Parameters
    ----------
    ary : np.ndarray
        An array containing MCMC samples
    axis : int, optional
        The axis along which to compute the autocovariance.
        Default is -1 (last axis).

    Returns
    -------
    np.ndarray
        Array of autocovariance estimates for every lag
    """
    axis = axis if axis > 0 else len(ary.shape) + axis
    n = ary.shape[axis]
    m = next_fast_len(2 * n)

    ary = ary - ary.mean(axis, keepdims=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ifft_ary = np.fft.rfft(ary, n=m, axis=axis)
        ifft_ary *= np.conjugate(ifft_ary)

        shape = tuple(slice(None) if dim_len != axis else slice(0, n) for dim_len, _ in enumerate(ary.shape))
        cov = np.fft.irfft(ifft_ary, n=m, axis=axis)[shape]
        cov /= n

    return cov


def autocorr(ary: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute autocorrelation using FFT for every lag for the input array.

    Parameters
    ----------
    ary : np.ndarray
        An array containing MCMC samples
    axis : int, optional
        The axis along which to compute the autocorrelation.
        Default is -1 (last axis).

    Returns
    -------
    np.ndarray
        Array of autocorrelation estimates for every lag
    """
    corr = autocov(ary, axis=axis)
    axis = axis if axis > 0 else len(corr.shape) + axis
    norm = tuple(slice(None, None) if dim != axis else slice(None, 1) for dim, _ in enumerate(corr.shape))
    with np.errstate(invalid="ignore"):
        corr /= corr[norm]
    return corr


def to_inference_data(obj: Any) -> InferenceData:
    """Convert supported objects to ArviZ InferenceData.

    Parameters
    ----------
    obj : Any
        Object to convert. Supported types include:
        - ArviZ InferenceData
        - PyMC model/trace
        - CmdStanPy fit
        - NumPyro mcmc
        - Pyro MCMC
        - TensorFlow Probability
        - xarray Dataset
        - numpy array
        - dict with array-like values

    Returns
    -------
    InferenceData
        ArviZ InferenceData object

    Raises
    ------
    ValueError
        If conversion fails or object type is not supported
    """
    if isinstance(obj, InferenceData):
        return obj

    if isinstance(obj, (list, tuple)):
        raise ValueError("Lists and tuples cannot be converted to InferenceData directly")

    if isinstance(obj, dict) and not all(isinstance(v, (np.ndarray, list)) for v in obj.values()):
        raise ValueError("Dictionary values must be array-like")

    try:
        return az.convert_to_inference_data(obj)
    except Exception as e:
        allowable_types = (
            "xarray Dataset",
            "xarray DataArray",
            "dict with array-like values",
            "numpy array",
            "pystan fit",
            "emcee fit",
            "pyro mcmc fit",
            "numpyro mcmc fit",
            "cmdstan fit",
            "cmdstanpy fit",
        )
        raise ValueError(
            f'Can only convert {", ".join(allowable_types)} to InferenceData, ' f"not {obj.__class__.__name__}"
        ) from e


def extract_log_likelihood(idata: InferenceData, var_name: str = "obs") -> Tuple[np.ndarray, np.ndarray]:
    """Extract log likelihood values and chain IDs from InferenceData.

    Parameters
    ----------
    idata : InferenceData
        ArviZ InferenceData object
    var_name : str, optional
        Name of log likelihood variable in log_likelihood group
        Default is "obs" as it's commonly used in ArviZ

    Returns
    -------
    tuple
        (log_likelihood_array, chain_ids)
        - log_likelihood_array: Array with shape (chains*draws, observations)
        - chain_ids: Array of chain IDs for each draw

    Raises
    ------
    ValueError
        If log likelihood values cannot be extracted
    """
    if not hasattr(idata, "log_likelihood"):
        raise ValueError("InferenceData object must have log_likelihood group")

    if var_name not in idata.log_likelihood.data_vars:
        raise ValueError(f"Variable '{var_name}' not found in log_likelihood group")

    log_lik = idata.log_likelihood[var_name]

    if log_lik.ndim == 2:
        return log_lik.values, np.ones(len(log_lik))

    n_chains = log_lik.shape[0]
    n_draws = log_lik.shape[1]
    chain_ids = np.repeat(np.arange(1, n_chains + 1), n_draws)

    reshaped_values = log_lik.values.reshape(-1, log_lik.shape[-1])
    return reshaped_values, chain_ids


def compute_log_mean_exp(x: np.ndarray, axis: Optional[int] = None) -> Union[float, np.ndarray]:
    """Compute log(mean(exp(x))) in a numerically stable way."""
    if axis is None:
        log_S = np.log(x.size)
        return _logsumexp(x.ravel()) - log_S
    else:
        log_S = np.log(x.shape[axis])
        return _logsumexp(x, axis=axis) - log_S


def compute_estimates(x: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute point estimates and standard errors."""
    estimate = np.sum(x, axis=0)
    se = np.sqrt(x.shape[0] * np.var(x, axis=0, ddof=1))
    return {"estimate": estimate, "se": se}


def validate_data(
    x: Union[np.ndarray, InferenceData, Any],
    var_name: Optional[str] = None,
    check_shape: Optional[Tuple[int, ...]] = None,
    allow_inf: bool = False,
    min_chains: int = 2,
    min_draws: int = 4,
    check_nan: bool = True,
    nan_axis: Optional[int] = None,
    nan_policy: str = "any",
    raise_on_failure: bool = True,
) -> Union[np.ndarray, InferenceData]:
    """Validate input data for PSIS-LOO-CV computations.

    This function performs comprehensive validation of input data to ensure it meets
    the requirements for PSIS-LOO-CV computations. It handles both raw arrays and
    InferenceData objects.

    Parameters
    ----------
    x : Union[np.ndarray, InferenceData, Any]
        Data to validate. Can be:
        - numpy array of log-likelihood values
        - ArviZ InferenceData object
        - Any object that can be converted to InferenceData
    var_name : str, optional
        For InferenceData objects, the name of the log-likelihood variable
        Default is None
    check_shape : tuple of int, optional
        Expected shape of the array. If provided, validates the array shape.
        Default is None
    allow_inf : bool, optional
        Whether to allow infinite values (useful for log-likelihoods)
        Default is False
    min_chains : int, optional
        Minimum number of chains required. Default is 2.
    min_draws : int, optional
        Minimum number of draws required. Default is 4.
    check_nan : bool, optional
        Whether to check for NaN values. Default is True.
    nan_axis : int, optional
        Axis along which to check for NaN values. Default is None (check all).
    nan_policy : str, optional
        How to handle NaN values: "any" or "all". Default is "any".
    raise_on_failure : bool, optional
        Whether to raise exceptions on validation failures. If False, will
        only issue warnings. Default is True.

    Returns
    -------
    Union[np.ndarray, InferenceData]
        Validated data object

    Raises
    ------
    TypeError
        If input type is not supported or conversion fails
    ValueError
        If validation fails for any of these reasons:
        - Array contains NaN values
        - Array contains infinite values when not allowed
        - Array shape doesn't match expected shape
        - InferenceData missing required groups or variables
        - Numerical instability detected
        - Insufficient chains or draws
    """

    def _raise_or_warn(msg, error_type=ValueError):
        if raise_on_failure:
            raise error_type(msg)
        import warnings

        warnings.warn(msg, stacklevel=2)

    if isinstance(x, InferenceData) or not isinstance(x, np.ndarray):
        try:
            if not isinstance(x, InferenceData):
                x = to_inference_data(x)

            if not hasattr(x, "log_likelihood"):
                _raise_or_warn("InferenceData object must have a log_likelihood group for PSIS-LOO-CV")

            if var_name is not None and var_name not in x.log_likelihood.data_vars:
                available_vars = list(x.log_likelihood.data_vars.keys())
                _raise_or_warn(
                    f"Variable '{var_name}' not found in log_likelihood group. "
                    f"Available variables are: {available_vars}"
                )

            if hasattr(x.log_likelihood, "chain"):
                n_chains = x.log_likelihood.chain.size
                if n_chains < min_chains:
                    _raise_or_warn(f"Number of chains ({n_chains}) is less than min_chains ({min_chains})")

            if hasattr(x.log_likelihood, "draw"):
                n_draws = x.log_likelihood.draw.size
                if n_draws < min_draws:
                    _raise_or_warn(f"Number of draws ({n_draws}) is less than min_draws ({min_draws})")

            return x

        except Exception as e:
            raise TypeError(f"Failed to validate or convert input: {str(e)}")

    if not isinstance(x, np.ndarray):
        raise TypeError(f"Expected numpy array or InferenceData, got {type(x).__name__}")

    if check_nan:
        isnan = np.isnan(x)
        if nan_axis is not None:
            if nan_policy.lower() == "all":
                has_nan = isnan.all(axis=nan_axis)
            else:
                has_nan = isnan.any(axis=nan_axis)
        else:
            if nan_policy.lower() == "all":
                has_nan = isnan.all()
            else:
                has_nan = isnan.any()

        if has_nan:
            _raise_or_warn("Input contains NaN values")

    if not allow_inf and np.any(np.isinf(x)):
        _raise_or_warn("Input contains infinite values. If these are log-likelihoods, set allow_inf=True")

    if check_shape is not None and x.shape != check_shape:
        _raise_or_warn(f"Array has incorrect shape. Expected {check_shape}, got {x.shape}")

    shape = x.shape
    if len(shape) >= 2:
        n_chains = shape[0]
        n_draws = shape[1]
    else:
        n_chains = 1
        n_draws = shape[0]

    if n_chains < min_chains:
        _raise_or_warn(f"Number of chains ({n_chains}) is less than min_chains ({min_chains})")
    if n_draws < min_draws:
        _raise_or_warn(f"Number of draws ({n_draws}) is less than min_draws ({min_draws})")

    if not allow_inf:
        abs_max = np.max(np.abs(x[np.isfinite(x)]))
        if abs_max > 1e38:
            _raise_or_warn(
                f"Array contains very large values (max abs: {abs_max}). " "This may cause numerical instability."
            )

    return x


def reshape_draws(x: np.ndarray, chain_ids: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Reshape MCMC draws between matrix and array formats."""
    if x.ndim == 3:
        return x.reshape(-1, x.shape[2]), None
    elif x.ndim == 2 and chain_ids is not None:
        n_chains = len(np.unique(chain_ids))
        n_iter = len(x) // n_chains
        return x.reshape(n_iter, n_chains, -1), chain_ids
    else:
        return x, chain_ids


def is_constant(x: np.ndarray, tol: float = _FLOAT_EPS) -> bool:
    """Check if array is constant within tolerance."""
    return np.abs(np.max(x) - np.min(x)) < tol


def get_log_likelihood(idata: InferenceData, var_name: Optional[str] = None) -> Any:
    """Retrieve the log likelihood dataarray of a given variable.

    Parameters
    ----------
    idata : InferenceData
        ArviZ InferenceData object
    var_name : str, optional
        Name of the variable in log_likelihood group.
        If None and there is only one variable, return it.
        If None and there are multiple variables, raise an error.

    Returns
    -------
    Any
        The log likelihood values for the specified variable

    Raises
    ------
    TypeError
        If log likelihood not found or variable name not found
    """
    if not hasattr(idata, "log_likelihood"):
        raise TypeError("log likelihood not found in inference data object")

    if var_name is None:
        var_names = list(idata.log_likelihood.data_vars)
        if len(var_names) > 1:
            raise TypeError(f"Found several log likelihood arrays {var_names}, var_name cannot be None")
        return idata.log_likelihood[var_names[0]]
    else:
        try:
            log_likelihood = idata.log_likelihood[var_name]
        except KeyError as err:
            raise TypeError(f"No log likelihood data named {var_name} found") from err
        return log_likelihood


def smooth_data(obs_vals: np.ndarray, pp_vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Smooth data using a cubic spline.

    This function is particularly useful for discrete data in PSIS-LOO-CV
    visualizations, helping to create smoother plots.

    Parameters
    ----------
    obs_vals : np.ndarray
        Observed data with shape (N,)
    pp_vals : np.ndarray
        Posterior predictive samples with shape (S, N), where S is the
        number of samples and N is the number of observations

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - Smoothed observed data with shape (N,)
        - Smoothed posterior predictive samples with shape (S, N)
    """
    x = np.linspace(0, 1, len(obs_vals))
    csi = CubicSpline(x, obs_vals)
    obs_vals = csi(np.linspace(0.01, 0.99, len(obs_vals)))

    x = np.linspace(0, 1, pp_vals.shape[1])
    csi = CubicSpline(x, pp_vals, axis=1)
    pp_vals = csi(np.linspace(0.01, 0.99, pp_vals.shape[1]))

    return obs_vals, pp_vals


# Format constants following ArviZ
BASE_FMT = """Computed from {{n_samples}} posterior samples and \
{{n_points}} observations.

{{pad:{width}}} Estimate       SE
{{scale}}_{{kind}} {{estimate:8.2f}}  {{se:7.2f}}
p_{{kind:{width}}} {{p_value:8.2f}}        -"""

DIAGNOSTIC_FMT = """
Pareto k diagnostic values:
                         Count    Pct.
(-Inf, {{k_thres:.2f}}]  (good)    {{counts[0]:4d}}  {{percentages[0]:5.1f}}%
 ({{k_thres:.2f}}, 1.00]   (bad)    {{counts[1]:4d}}  {{percentages[1]:5.1f}}%
  (1.00, Inf)   (very bad)  {{counts[2]:4d}}  {{percentages[2]:5.1f}}%"""

SCALE_DICT = {"deviance": "deviance", "log": "elpd", "negative_log": "-elpd"}


class ELPDData(pd.Series):
    """Class to contain ELPD (Expected Log Pointwise Predictive Density) data.

    This class extends pandas.Series to provide a specialized container for ELPD
    information criterion data like PSIS-LOO-CV or WAIC. It includes methods for
    pretty printing, validation, and analysis of the results.

    Required Data
    ------------
    The Series must contain:
    - elpd_{kind}: Expected log pointwise predictive density
    - p_{kind}: Effective number of parameters
    - scale: One of ["deviance", "log", "negative_log"]
    - n_samples: Number of posterior samples used
    - n_data_points: Number of observations

    Optional Data
    ------------
    - warning: bool, indicates potential issues
    - pareto_k: array of Pareto k values (for LOO only)
    - good_k: float, threshold for good Pareto k values (default: 0.7)
    """

    @property
    def kind(self) -> str:
        kind = self.index[0].split("_")[1]
        if kind not in ("loo", "waic"):
            raise ValueError("Invalid ELPDData object")
        return kind

    @property
    def scale_str(self) -> str:
        return SCALE_DICT[self["scale"]]

    def has_warnings(self) -> bool:
        return hasattr(self, "warning") and self.warning

    def has_pareto_k(self) -> bool:
        return self.kind == "loo" and hasattr(self, "pareto_k")

    def get_pareto_k_summary(self) -> Dict[str, Any]:
        if not self.has_pareto_k():
            raise ValueError("Pareto k diagnostics not available")

        k_thres = getattr(self, "good_k", 0.7)
        bins = np.array([-np.inf, k_thres, 1, np.inf])
        counts, *_ = np.histogram(self.pareto_k.values, bins=bins)
        percentages = counts / np.sum(counts) * 100

        return {
            "k_threshold": k_thres,
            "counts": counts,
            "percentages": percentages,
            "categories": ["good", "bad", "very bad"],
        }

    def validate(self) -> None:
        required_fields = [f"elpd_{self.kind}", f"p_{self.kind}", "scale", "n_samples", "n_data_points"]
        missing = [f for f in required_fields if f not in self]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")

        if self["scale"] not in SCALE_DICT:
            raise ValueError(f"Invalid scale '{self['scale']}'. Must be one of {list(SCALE_DICT.keys())}")

    def __str__(self) -> str:
        self.validate()

        padding = len(self.scale_str) + len(self.kind) + 1
        elpd_key = f"elpd_{self.kind}"
        p_key = f"p_{self.kind}"

        # First format the width parameter
        base_template = BASE_FMT.format(width=padding)

        # Then format the rest of the parameters
        base = base_template.format(
            pad="",
            kind=self.kind,
            scale=self.scale_str,
            estimate=self[elpd_key],
            se=self.get(f"{elpd_key}_se", 0.0),
            p_value=self[p_key],
            n_samples=self.n_samples,
            n_points=self.n_data_points,
        )

        if self.has_warnings():
            base += "\n\nWarning: Some diagnostics indicate potential problems."

        if self.has_pareto_k():
            summary = self.get_pareto_k_summary()
            diag = DIAGNOSTIC_FMT.format(
                k_thres=summary["k_threshold"], counts=summary["counts"], percentages=summary["percentages"]
            )
            base = "\n".join([base, diag])

        return base

    def __repr__(self) -> str:
        return self.__str__()

    def copy(self, deep: bool = True) -> "ELPDData":
        copied = super().copy(deep=deep)
        result = ELPDData(copied)

        if hasattr(self, "pareto_k"):
            result.pareto_k = self.pareto_k.copy(deep=deep)
        if hasattr(self, "warning"):
            result.warning = self.warning
        if hasattr(self, "good_k"):
            result.good_k = self.good_k

        return result
