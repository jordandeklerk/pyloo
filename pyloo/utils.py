"""Helper functions for pyloo package with adaptations from ArviZ."""

import warnings
from collections.abc import Sequence
from typing import Any, Dict, Optional, Tuple, Union

import arviz as az
import numpy as np
from arviz import InferenceData
from scipy.fftpack import next_fast_len
from scipy.interpolate import CubicSpline
from xarray import apply_ufunc

_FLOAT_EPS = np.finfo(float).eps


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

        shape = tuple(
            slice(None) if dim_len != axis else slice(0, n)
            for dim_len, _ in enumerate(ary.shape)
        )
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
    norm = tuple(
        slice(None, None) if dim != axis else slice(None, 1)
        for dim, _ in enumerate(corr.shape)
    )
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
        raise ValueError(
            "Lists and tuples cannot be converted to InferenceData directly"
        )

    if isinstance(obj, dict) and not all(
        isinstance(v, (np.ndarray, list)) for v in obj.values()
    ):
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
            f'Can only convert {", ".join(allowable_types)} to InferenceData, '
            f"not {obj.__class__.__name__}"
        ) from e


def compute_log_mean_exp(
    x: np.ndarray, axis: Optional[int] = None
) -> Union[float, np.ndarray]:
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
                _raise_or_warn(
                    "InferenceData object must have a log_likelihood group for"
                    " PSIS-LOO-CV"
                )

            if var_name is not None and var_name not in x.log_likelihood.data_vars:
                available_vars = list(x.log_likelihood.data_vars.keys())
                _raise_or_warn(
                    f"Variable '{var_name}' not found in log_likelihood group. "
                    f"Available variables are: {available_vars}"
                )

            if hasattr(x.log_likelihood, "chain"):
                n_chains = x.log_likelihood.chain.size
                if n_chains < min_chains:
                    _raise_or_warn(
                        f"Number of chains ({n_chains}) is less than min_chains"
                        f" ({min_chains})"
                    )

            if hasattr(x.log_likelihood, "draw"):
                n_draws = x.log_likelihood.draw.size
                if n_draws < min_draws:
                    _raise_or_warn(
                        f"Number of draws ({n_draws}) is less than min_draws"
                        f" ({min_draws})"
                    )

            return x

        except Exception as e:
            raise TypeError(f"Failed to validate or convert input: {str(e)}")

    if not isinstance(x, np.ndarray):
        raise TypeError(
            f"Expected numpy array or InferenceData, got {type(x).__name__}"
        )

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
        _raise_or_warn(
            "Input contains infinite values. If these are log-likelihoods, set"
            " allow_inf=True"
        )

    if check_shape is not None and x.shape != check_shape:
        _raise_or_warn(
            f"Array has incorrect shape. Expected {check_shape}, got {x.shape}"
        )

    shape = x.shape
    if len(shape) >= 2:
        n_chains = shape[0]
        n_draws = shape[1]
    else:
        n_chains = 1
        n_draws = shape[0]

    if n_chains < min_chains:
        _raise_or_warn(
            f"Number of chains ({n_chains}) is less than min_chains ({min_chains})"
        )
    if n_draws < min_draws:
        _raise_or_warn(
            f"Number of draws ({n_draws}) is less than min_draws ({min_draws})"
        )

    if not allow_inf:
        abs_max = np.max(np.abs(x[np.isfinite(x)]))
        if abs_max > 1e38:
            _raise_or_warn(
                f"Array contains very large values (max abs: {abs_max}). "
                "This may cause numerical instability."
            )

    return x


def reshape_draws(
    x: np.ndarray, chain_ids: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
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


def get_log_likelihood(idata, var_name=None, single_var=True):
    """Retrieve the log likelihood dataarray of a given variable."""
    if (
        not hasattr(idata, "log_likelihood")
        and hasattr(idata, "sample_stats")
        and hasattr(idata.sample_stats, "log_likelihood")
    ):
        warnings.warn(
            "Storing the log_likelihood in sample_stats groups has been deprecated",
            DeprecationWarning,
            stacklevel=2,
        )
        return idata.sample_stats.log_likelihood
    if not hasattr(idata, "log_likelihood"):
        raise TypeError("log likelihood not found in inference data object")
    if var_name is None:
        var_names = list(idata.log_likelihood.data_vars)
        if len(var_names) > 1:
            if single_var:
                raise TypeError(
                    f"Found several log likelihood arrays {var_names}, var_name cannot"
                    " be None"
                )
            return idata.log_likelihood[var_names]
        return idata.log_likelihood[var_names[0]]
    else:
        try:
            log_likelihood = idata.log_likelihood[var_name]
        except KeyError as err:
            raise TypeError(f"No log likelihood data named {var_name} found") from err
        return log_likelihood


def smooth_data(
    obs_vals: np.ndarray, pp_vals: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
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


def make_ufunc(
    func, n_dims=2, n_output=1, n_input=1, index=Ellipsis, ravel=True, check_shape=None
):
    """Make ufunc from a function taking 1D array input.

    Parameters
    ----------
    func : callable
    n_dims : int, optional
        Number of core dimensions not broadcasted. Dimensions are skipped from the end.
        At minimum n_dims > 0.
    n_output : int, optional
        Select number of results returned by `func`.
        If n_output > 1, ufunc returns a tuple of objects else returns an object.
    n_input : int, optional
        Number of **array** inputs to func, i.e. ``n_input=2`` means that func is called
        with ``func(ary1, ary2, *args, **kwargs)``
    index : int, optional
        Slice ndarray with `index`. Defaults to `Ellipsis`.
    ravel : bool, optional
        If true, ravel the ndarray before calling `func`.
    check_shape: bool, optional
        If false, do not check if the shape of the output is compatible with n_dims and
        n_output. By default, True only for n_input=1. If n_input is larger than 1, the last
        input array is used to check the shape, however, shape checking with multiple inputs
        may not be correct.

    Returns
    -------
    callable
        ufunc wrapper for `func`.
    """
    if n_dims < 1:
        raise TypeError("n_dims must be one or higher.")

    if n_input == 1 and check_shape is None:
        check_shape = True
    elif check_shape is None:
        check_shape = False

    def _ufunc(*args, out=None, out_shape=None, **kwargs):
        """General ufunc for single-output function."""
        arys = args[:n_input]
        n_dims_out = None
        if out is None:
            if out_shape is None:
                out = np.empty(arys[-1].shape[:-n_dims])
            else:
                out = np.empty((*arys[-1].shape[:-n_dims], *out_shape))
                n_dims_out = -len(out_shape)
        elif check_shape:
            if out.shape != arys[-1].shape[:-n_dims]:
                msg = f"Shape incorrect for `out`: {out.shape}."
                msg += f" Correct shape is {arys[-1].shape[:-n_dims]}"
                raise TypeError(msg)
        for idx in np.ndindex(out.shape[:n_dims_out]):
            arys_idx = [ary[idx].ravel() if ravel else ary[idx] for ary in arys]
            out_idx = np.asarray(func(*arys_idx, *args[n_input:], **kwargs))[index]
            if n_dims_out is None:
                out_idx = out_idx.item()
            out[idx] = out_idx
        return out

    def _multi_ufunc(*args, out=None, out_shape=None, **kwargs):
        """General ufunc for multi-output function."""
        arys = args[:n_input]
        element_shape = arys[-1].shape[:-n_dims]
        if out is None:
            if out_shape is None:
                out = tuple(np.empty(element_shape) for _ in range(n_output))
            else:
                out = tuple(
                    np.empty((*element_shape, *out_shape[i])) for i in range(n_output)
                )

        elif check_shape:
            raise_error = False
            correct_shape = tuple(element_shape for _ in range(n_output))
            if isinstance(out, tuple):
                out_shape = tuple(item.shape for item in out)
                if out_shape != correct_shape:
                    raise_error = True
            else:
                raise_error = True
                out_shape = "not tuple, type={type(out)}"
            if raise_error:
                msg = f"Shapes incorrect for `out`: {out_shape}."
                msg += f" Correct shapes are {correct_shape}"
                raise TypeError(msg)
        for idx in np.ndindex(element_shape):
            arys_idx = [ary[idx].ravel() if ravel else ary[idx] for ary in arys]
            results = func(*arys_idx, *args[n_input:], **kwargs)
            for i, res in enumerate(results):
                out[i][idx] = np.asarray(res)[index]
        return out

    if n_output > 1:
        ufunc = _multi_ufunc
    else:
        ufunc = _ufunc

    return ufunc


def wrap_xarray_ufunc(
    ufunc,
    *datasets,
    ufunc_kwargs=None,
    func_args=None,
    func_kwargs=None,
    **kwargs,
):
    """Wrap make_ufunc with xarray.apply_ufunc.

    Parameters
    ----------
    ufunc : callable
    *datasets : xarray.Dataset
    ufunc_kwargs : dict
        Keyword arguments passed to `make_ufunc`.
            - 'n_dims', int, by default 2
            - 'n_output', int, by default 1
            - 'n_input', int, by default len(datasets)
            - 'index', slice, by default Ellipsis
            - 'ravel', bool, by default True
    func_args : literal
        Arguments passed to 'ufunc'.
    func_kwargs : dict
        Keyword arguments passed to 'ufunc'.
            - 'out_shape', int, by default None
    **kwargs
        Passed to :func:`xarray.apply_ufunc`.

    Returns
    -------
    xarray.Dataset
    """
    if ufunc_kwargs is None:
        ufunc_kwargs = {}
    ufunc_kwargs.setdefault("n_input", len(datasets))
    if func_args is None:
        func_args = ()
    if func_kwargs is None:
        func_kwargs = {}

    kwargs.setdefault(
        "input_core_dims",
        tuple(("chain", "draw") for _ in range(len(func_args) + len(datasets))),
    )
    ufunc_kwargs.setdefault("n_dims", len(kwargs["input_core_dims"][-1]))
    kwargs.setdefault(
        "output_core_dims", tuple([] for _ in range(ufunc_kwargs.get("n_output", 1)))
    )

    callable_ufunc = make_ufunc(ufunc, **ufunc_kwargs)

    return apply_ufunc(
        callable_ufunc, *datasets, *func_args, kwargs=func_kwargs, **kwargs
    )


def _logsumexp(
    ary, *, b=None, b_inv=None, axis=None, keepdims=False, out=None, copy=True
):
    """Stable logsumexp when b >= 0 and b is scalar.

    b_inv overwrites b unless b_inv is None.
    """
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
                ()
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
