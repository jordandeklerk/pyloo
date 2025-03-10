"""Utility functions for the PyMCWrapper."""

import logging
import warnings
from typing import Any, Callable, Sequence

import numpy as np
import xarray as xr

from ..utils import wrap_xarray_ufunc

__all__ = [
    "_apply_ufunc",
    "_transform_to_unconstrained",
    "_transform_to_constrained",
    "_process_and_validate_indices",
    "_format_log_likelihood_result",
    "_create_selection_mask",
    "PyMCWrapperError",
    "_validate_model_state",
    "_extract_model_components",
    "_get_coords",
    "_validate_coords",
]

logger = logging.getLogger(__name__)


class PyMCWrapperError(Exception):
    """Exception raised for errors in the PyMCWrapper."""

    pass


def _apply_ufunc(
    self,
    func: Callable[..., Any],
    var_name: str | None = None,
    input_core_dims: list[list[str]] | None = None,
    output_core_dims: list[list[str]] | None = None,
    func_kwargs: dict | None = None,
    **kwargs: Any,
) -> xr.DataArray:
    """This is a utility method that applies a function to posterior samples using
    ``wrap_xarray_ufunc``."""
    if not hasattr(self.idata, "posterior"):
        raise PyMCWrapperError(
            "InferenceData object must contain posterior samples. "
            "The model does not appear to be fitted."
        )

    if input_core_dims is None:
        input_core_dims = [["chain", "draw"]]
    if output_core_dims is None:
        output_core_dims = [["chain", "draw"]]

    if var_name is not None:
        if var_name not in self.idata.posterior:
            raise PyMCWrapperError(
                f"Variable '{var_name}' not found in posterior. Available"
                f" variables: {list(self.idata.posterior.data_vars.keys())}"
            )
        data = self.idata.posterior[var_name]
    else:
        input_vars = kwargs.pop("input_vars", None)
        if input_vars is not None:
            for var in input_vars:
                if var not in self.idata.posterior:
                    raise PyMCWrapperError(
                        f"Variable '{var}' not found in posterior. Available"
                        f" variables: {list(self.idata.posterior.data_vars.keys())}"
                    )
                data = [self.idata.posterior[var] for var in input_vars]
        else:
            data = self.idata.posterior

    result = wrap_xarray_ufunc(
        func,
        data if isinstance(data, list) else [data],
        input_core_dims=input_core_dims,
        output_core_dims=output_core_dims,
        func_kwargs=func_kwargs,
        **kwargs,
    )
    return result


def _transform_to_unconstrained(self, values, transform):
    """Transform values from constrained to unconstrained space."""
    try:
        result = transform.backward(values)
        if hasattr(result, "eval"):
            result = result.eval()
        return np.asarray(result)
    except Exception as e:
        logger.warning("Backward transform failed: %s", str(e))
        raise


def _transform_to_constrained(self, values, transform):
    """Transform values from unconstrained to constrained space."""
    try:
        result = transform.forward(values)
        if hasattr(result, "eval"):
            result = result.eval()
        return np.asarray(result)
    except Exception as e:
        logger.warning("Forward transform failed: %s", str(e))
        raise


def _process_and_validate_indices(
    idx: int | np.ndarray | slice, n_obs: int
) -> tuple[np.ndarray, bool]:
    """Process and validate indices for log likelihood computation."""
    if isinstance(idx, (int, np.integer)):
        if idx < 0 or idx >= n_obs:
            raise IndexError(
                f"Index {idx} is out of bounds for axis 0 with size {n_obs}"
            )
        indices = np.array([idx], dtype=int)
        single_idx = True
    elif isinstance(idx, slice):
        start, stop, step = idx.indices(n_obs)

        if stop > n_obs:
            warnings.warn(
                f"Slice end index {idx.stop} is out of bounds for axis 0 with size"
                f" {n_obs}. Only indices up to {n_obs - 1} will be used.",
                UserWarning,
                stacklevel=2,
            )

        indices = np.arange(start, min(stop, n_obs), step, dtype=int)
        single_idx = False
    elif isinstance(idx, np.ndarray):
        if idx.dtype == bool:
            if len(idx) != n_obs:
                raise IndexError(
                    f"Boolean mask length {len(idx)} does not match "
                    f"data shape {n_obs} along axis 0"
                )
            indices = np.where(idx)[0]
            single_idx = False
        else:
            if len(idx) == 0:
                raise IndexError("Empty index array provided")

            invalid_indices = (idx < 0) | (idx >= n_obs)
            if np.any(invalid_indices):
                out_of_bounds = idx[invalid_indices]
                warnings.warn(
                    f"Some indices {out_of_bounds} are out of bounds for axis 0"
                    f" with size {n_obs}. These indices will be ignored.",
                    UserWarning,
                    stacklevel=2,
                )
                indices = idx[~invalid_indices].astype(int)
                if len(indices) == 0:
                    raise IndexError(
                        f"All indices {idx} are out of bounds for axis 0 with size"
                        f" {n_obs}"
                    )
            else:
                indices = idx.astype(int)

            single_idx = len(indices) == 1 and isinstance(idx[0], (int, np.integer))
    else:
        raise PyMCWrapperError(f"Unsupported index type: {type(idx)}")
    return indices, single_idx


def _format_log_likelihood_result(
    self,
    log_like_i: xr.DataArray,
    indices: np.ndarray,
    single_idx: bool,
    original_idx: int | np.ndarray | slice,
    holdout_data: np.ndarray,
    n_obs: int,
    var_name: str,
) -> xr.DataArray:
    """Format log likelihood results based on index type."""
    if single_idx and isinstance(original_idx, int):
        obs_dims = [dim for dim in log_like_i.dims if dim not in ("chain", "draw")]

        for dim in obs_dims:
            log_like_i = log_like_i.isel({dim: 0})

        if obs_dims and obs_dims[0] in log_like_i.coords:
            log_like_i.coords[obs_dims[0]] = original_idx

        log_like_i.attrs["observation_index"] = original_idx
        return log_like_i

    chains = log_like_i.sizes.get("chain", 1)
    draws = log_like_i.sizes.get("draw", 1)

    obs_dims = [dim for dim in log_like_i.dims if dim not in ("chain", "draw")]

    if not obs_dims:
        values = np.zeros((chains, draws, len(indices)))

        for i in range(len(indices)):
            values[:, :, i] = log_like_i.values

        result = xr.DataArray(
            values,
            dims=["chain", "draw", "obs_idx"],
            coords={
                "chain": log_like_i.coords.get("chain", np.arange(chains)),
                "draw": log_like_i.coords.get("draw", np.arange(draws)),
                "obs_idx": indices,
            },
            name=var_name,
        )
    else:
        obs_dim = obs_dims[0]
        obs_size = log_like_i.sizes[obs_dim]
        values = np.zeros((chains, draws, len(indices)))

        for i, idx_val in enumerate(indices):
            if obs_size == len(holdout_data):
                index_to_use = min(i, obs_size - 1)
            elif obs_size == n_obs:
                index_to_use = min(idx_val, obs_size - 1)
            else:
                index_to_use = 0

            values[:, :, i] = log_like_i.isel({obs_dim: index_to_use}).values

        result = xr.DataArray(
            values,
            dims=["chain", "draw", "obs_idx"],
            coords={
                "chain": log_like_i.coords.get("chain", np.arange(chains)),
                "draw": log_like_i.coords.get("draw", np.arange(draws)),
                "obs_idx": indices,
            },
            name=var_name,
        )

    result.attrs["observation_indices"] = indices.tolist()
    return result


def _create_selection_mask(
    indices: np.ndarray | slice, length: int, axis: int
) -> np.ndarray:
    """Create a boolean mask for selecting observations."""
    # Boolean mask input
    if isinstance(indices, np.ndarray) and indices.dtype == bool:
        if indices.shape[0] != length:
            raise PyMCWrapperError(
                f"Boolean mask shape {indices.shape[0]} does not match "
                f"data shape {length} along axis {axis}"
            )
        return indices

    # Slice input
    if isinstance(indices, slice):
        mask = np.zeros(length, dtype=bool)
        idx_range = range(*indices.indices(length))
        mask[idx_range] = True
        return mask

    # Integer array input
    indices = np.asarray(indices, dtype=np.int64)

    if indices.size == 0:
        raise IndexError("Empty index array provided")

    mask = np.zeros(length, dtype=bool)
    valid_mask = (indices >= 0) & (indices < length)

    if not np.any(valid_mask):
        raise IndexError(
            f"All indices are out of bounds for axis {axis} with size {length}"
        )

    if not np.all(valid_mask):
        invalid_indices = indices[~valid_mask]
        warnings.warn(
            f"Some indices {invalid_indices} are out of bounds for axis {axis}"
            f" with size {length}. These indices will be ignored.",
            UserWarning,
            stacklevel=2,
        )

    valid_indices = indices[valid_mask]
    mask[valid_indices] = True
    return mask


def _validate_model_state(self) -> None:
    """Validate that the model is properly fitted and ready for use."""
    # Check that posterior samples exist
    if not hasattr(self.idata, "posterior"):
        raise PyMCWrapperError(
            "InferenceData object must contain posterior samples. "
            "The model does not appear to be fitted."
        )

    # Check that all free variables have posterior samples
    posterior_vars = set(self.idata.posterior.data_vars.keys())
    model_vars = {rv.name for rv in self.model.free_RVs}
    missing_vars = model_vars - posterior_vars
    if missing_vars:
        raise PyMCWrapperError(
            f"Missing posterior samples for variables: {missing_vars}. "
            "The model may not be fully fitted."
        )

    # Check that observed data exists for observed variables
    for obs_rv in self.model.observed_RVs:
        if not hasattr(obs_rv.tag, "observations"):
            raise PyMCWrapperError(
                f"Missing observed data for variable {obs_rv.name}. "
                "The model is not properly initialized."
            )

    # Validate posterior sample shapes against model structure
    for var_name in model_vars:
        var = self.model.named_vars[var_name]
        expected_shape = tuple(d.eval() if hasattr(d, "eval") else d for d in var.shape)
        posterior_shape = tuple(self.idata.posterior[var_name].shape[2:])
        if expected_shape != posterior_shape:
            raise PyMCWrapperError(
                f"Shape mismatch for variable {var_name}. Model expects shape"
                f" {expected_shape}, but posterior has shape {posterior_shape}."
                " This may indicate an issue with the model specification or"
                " fitting process."
            )


def _extract_model_components(self) -> None:
    """Extract and organize the model's components."""
    self.observed_data = {}
    self.observed_dims = {}

    if hasattr(self.idata, "observed_data"):
        for var_name, data_array in self.idata.observed_data.items():
            if self.var_names is None or var_name in self.var_names:
                self.observed_data[var_name] = data_array.values.copy()
                self.observed_dims[var_name] = data_array.dims

    self.constant_data = {}
    for data_var in self.model.data_vars:
        data_name = data_var.name
        if hasattr(data_var, "get_value"):
            self.constant_data[data_name] = data_var.get_value()

    self.free_vars = [rv.name for rv in self.model.free_RVs]
    self.deterministic_vars = [det.name for det in self.model.deterministics]
    self.value_vars = [rv.name for rv in self.model.value_vars]
    self.unobserved_value_vars = [rv.name for rv in self.model.unobserved_value_vars]
    self.basic_RVs = [rv.name for rv in self.model.basic_RVs]
    self.unobserved_vars = [rv.name for rv in self.model.unobserved_RVs]
    self.continuous_value_vars = [rv.name for rv in self.model.continuous_value_vars]
    self.discrete_value_vars = [rv.name for rv in self.model.discrete_value_vars]


def _get_coords(self, var_name: str) -> dict[str, Sequence[int]] | None:
    """Get the coordinates for a variable."""
    dims = self.get_dims()
    if dims is None:
        return None

    shape = self.get_shape(var_name)
    if shape is None:
        return None

    coords: dict[str, Sequence[int]] = {}
    for dim, size in zip(dims, shape):
        if dim is not None:
            coords[dim] = list(range(size))
    return coords


def _validate_coords(
    self,
    var_name: str,
    coords: dict[str, Sequence],
) -> None:
    """Validate coordinate values against variable dimensions."""
    dims = self.get_dims()
    if dims is None:
        return

    shape = self.get_shape(var_name)
    if shape is None:
        return

    missing_coords = {d for d in dims if d is not None} - set(coords.keys())
    if missing_coords:
        raise ValueError(
            f"Missing coordinates for dimensions {missing_coords} of variable"
            f" {var_name}"
        )

    for dim, size in zip(dims, shape):
        if dim is not None and len(coords[dim]) != size:
            raise ValueError(
                f"Coordinate length {len(coords[dim])} for dimension {dim} "
                f"does not match variable shape {size} for {var_name}"
            )
