"""Utility functions."""

import warnings
from collections.abc import Sequence
from typing import Any, Optional, Tuple

import arviz as az
import numpy as np
from arviz import InferenceData
from xarray import apply_ufunc

__all__ = [
    "to_inference_data",
    "reshape_draws",
    "get_log_likelihood",
    "wrap_xarray_ufunc",
    "_logsumexp",
]


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


def get_log_likelihood(idata, var_name=None, single_var=True):
    """Retrieve the log likelihood dataarray of a given variable.

    Parameters
    ----------
    idata : InferenceData
        InferenceData object
    var_name : str, optional
        Name of the variable to retrieve
    single_var : bool, optional
        If True, return a single log likelihood array. If False, return a list of log
        likelihood arrays.

    Returns
    -------
    log_likelihood : xarray.DataArray
    """
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
