"""Helper functions for moment matching."""

import warnings
from dataclasses import dataclass
from typing import Any, TypedDict

import arviz as az
import numpy as np
import pymc as pm
import xarray as xr
from arviz.stats.diagnostics import ess

from .wrapper.pymc import PyMCWrapper

__all__ = [
    "log_lik_i_upars",
    "log_prob_upars",
    "_initialize_array",
    "ShiftResult",
    "ShiftAndScaleResult",
    "ShiftAndCovResult",
    "UpdateQuantitiesResult",
    "compute_updated_r_eff",
    "extract_log_likelihood_for_observation",
    "ParameterConverter",
]


class SplitMomentMatchResult(TypedDict):
    """Result of split moment matching."""

    lwi: np.ndarray
    lwfi: np.ndarray
    log_liki: np.ndarray
    r_eff_i: float


class UpdateQuantitiesResult(TypedDict):
    """Result of updating quantities for observation i."""

    lwi: np.ndarray
    lwfi: np.ndarray
    ki: float
    kfi: int
    log_liki: np.ndarray


class ShiftResult(TypedDict):
    """Result of shift transformation."""

    upars: np.ndarray
    shift: np.ndarray


class ShiftAndScaleResult(TypedDict):
    """Result of shift and scale transformation."""

    upars: np.ndarray
    shift: np.ndarray
    scaling: np.ndarray


class ShiftAndCovResult(TypedDict):
    """Result of shift and covariance transformation."""

    upars: np.ndarray
    shift: np.ndarray
    mapping: np.ndarray


@dataclass
class ParameterInfo:
    """Information about a parameter's shape and dimensions."""

    name: str
    original_shape: tuple[int, ...]
    flattened_size: int
    start_idx: int
    end_idx: int
    dims: list[str]
    coords: dict[str, np.ndarray]


class ParameterConverter:
    """Handle conversions between matrix and dictionary parameter formats.

    Parameters
    ----------
    wrapper : PyMCWrapper
        PyMC model wrapper instance containing parameter information
    """

    def __init__(self, wrapper: PyMCWrapper):
        """Initialize converter with parameter information from wrapper."""
        self.wrapper = wrapper
        self._param_info = {}
        self._total_size = 0

        unconstrained = wrapper.get_unconstrained_parameters()

        current_idx = 0
        for name, param in unconstrained.items():
            param_dims = [d for d in param.dims if d not in ("chain", "draw")]
            param_coords = {d: param.coords[d].values for d in param_dims}

            if param_dims:
                original_shape = tuple(param.sizes[d] for d in param_dims)
            else:
                original_shape = ()
                param_dims = []
                param_coords = {}

            flattened_size = int(np.prod(original_shape)) if original_shape else 1
            end_idx = current_idx + flattened_size

            self._param_info[name] = ParameterInfo(
                name=name,
                original_shape=original_shape,
                flattened_size=flattened_size,
                start_idx=current_idx,
                end_idx=end_idx,
                dims=param_dims,
                coords=param_coords,
            )
            current_idx = end_idx

        self._total_size = current_idx
        self._param_names = list(self._param_info.keys())

    def dict_to_matrix(self, params_dict: dict[str, xr.DataArray]) -> np.ndarray:
        """Convert dictionary of parameters to matrix format.

        Parameters
        ----------
        params_dict : dict[str, xr.DataArray]
            Dictionary mapping parameter names to their values

        Returns
        -------
        np.ndarray
            Matrix of shape (n_samples, total_params) containing flattened parameters
        """
        first_param = next(iter(params_dict.values()))
        n_samples = first_param.shape[0] * first_param.shape[1]

        result = np.zeros((n_samples, self._total_size))

        for name, param in params_dict.items():
            if name not in self._param_info:
                continue

            info = self._param_info[name]
            values = param.values

            if values.ndim > 2:
                values = values.reshape(values.shape[0], values.shape[1], -1)
            values = values.reshape(-1, info.flattened_size)

            result[:, info.start_idx : info.end_idx] = values

        return result

    def matrix_to_dict(self, matrix: np.ndarray) -> dict[str, xr.DataArray]:
        """Convert matrix of parameters to dictionary format.

        Parameters
        ----------
        matrix : np.ndarray
            Matrix of shape (n_samples, total_params) containing flattened parameters

        Returns
        -------
        dict[str, xr.DataArray]
            Dictionary mapping parameter names to their reshaped values

        Raises
        ------
        ValueError
            If matrix has incorrect number of columns
        """
        if matrix.shape[1] != self._total_size:
            raise ValueError(
                f"Matrix has {matrix.shape[1]} columns but expected {self._total_size}"
            )

        first_param = next(iter(self.wrapper.get_unconstrained_parameters().values()))
        n_chains = first_param.shape[0]
        n_draws = first_param.shape[1]

        result = {}
        for name in self._param_names:
            info = self._param_info[name]
            values = matrix[:, info.start_idx : info.end_idx]

            if info.original_shape:
                values = values.reshape(n_chains, n_draws, *info.original_shape)
            else:
                values = values.reshape(n_chains, n_draws)

            dims = ["chain", "draw"] + info.dims

            coords = {
                "chain": np.arange(n_chains),
                "draw": np.arange(n_draws),
                **info.coords,
            }

            result[name] = xr.DataArray(values, dims=dims, coords=coords)

        return result

    @property
    def param_names(self) -> list[str]:
        """Get list of parameter names in order."""
        return self._param_names

    @property
    def total_size(self) -> int:
        """Get total number of parameters when flattened."""
        return self._total_size

    def get_param_info(self, name: str) -> ParameterInfo:
        """Get information about a specific parameter.

        Parameters
        ----------
        name : str
            Name of the parameter

        Returns
        -------
        ParameterInfo
            Information about the parameter's shape and position in matrix
        """
        return self._param_info[name]


def log_lik_i_upars(model, upars: dict[str, xr.DataArray], pointwise: bool = False):
    """Compute log likelihood for unconstrained parameters.

    Parameters
    ----------
    model : PyMC model or PyMCWrapper
        Model object
    upars : dict of xarray.DataArray
        Unconstrained parameters dictionary from get_unconstrained_parameters()
    pointwise : bool, default False
        If True, returns pointwise log likelihood values as a DataArray
        If False, returns the full InferenceData object

    Returns
    -------
    arviz.InferenceData or xarray.DataArray
        If pointwise=False, returns InferenceData object with log_likelihood group
        If pointwise=True, returns DataArray with pointwise log likelihood values
    """
    model = getattr(model, "model", model)
    idata = az.InferenceData(posterior=xr.Dataset(upars))

    idata_with_loglik = pm.compute_log_likelihood(
        idata=idata, model=model, progressbar=False
    )

    if not pointwise:
        return idata_with_loglik

    log_lik = idata_with_loglik.log_likelihood
    observed_vars = list(log_lik.data_vars)

    if len(observed_vars) > 1:
        print(
            f"Warning: Multiple observed variables found: {observed_vars}. Using the"
            " first one."
        )

    var_loglik = log_lik[observed_vars[0]]
    obs_dims = [dim for dim in var_loglik.dims if dim not in ["chain", "draw"]]

    if len(obs_dims) > 1:
        return var_loglik.stack(__obs__=obs_dims)
    else:
        return var_loglik.rename({obs_dims[0]: "__obs__"})


def log_prob_upars(
    model, upars: dict[str, xr.DataArray], sum_params: bool = True
) -> np.ndarray:
    """Compute log probability for unconstrained parameters.

    Parameters
    ----------
    model : PyMC model
        PyMC model object that provides access to model variables
    upars : dict of xarray.DataArray
        Unconstrained parameters from model.get_unconstrained_parameters()
    sum_params : bool, default True
        If True, returns the sum of log probabilities across parameters for each sample.
        If False, returns the matrix of log probabilities for each parameter and sample.

    Returns
    -------
    np.ndarray
        If sum_params=True, returns a vector of summed log probability values with shape (n_samples,)
        If sum_params=False, returns a matrix of log probability values with shape (n_samples, n_variables)
        where n_samples = n_chains * n_draws and n_variables is the number of parameters in upars
    """
    model = getattr(model, "model", model)
    var_names = list(upars.keys())
    first_var = upars[var_names[0]]
    n_samples = first_var.shape[0] * first_var.shape[1]

    result_matrix = np.zeros((n_samples, len(var_names)))

    for i, name in enumerate(var_names):
        param = upars[name].stack(__sample__=("chain", "draw"))

        with model:
            var = model[name]
            value_var = model.rvs_to_values.get(var)

            if value_var is None:
                result_matrix[:, i] = np.nan
                continue

            logp_fn = model.compile_fn(model.logp(vars=var, sum=True, jacobian=True))

        # Calculate logp for all samples of this variable
        for j in range(len(param.__sample__)):
            try:
                value = (
                    param.isel(__sample__=j).values
                    if len(upars[name].shape) > 2
                    else param.isel(__sample__=j).item()
                )
                result_matrix[j, i] = logp_fn({value_var.name: value})
            except Exception:
                result_matrix[j, i] = np.nan

    if sum_params:
        return np.sum(result_matrix, axis=1)
    else:
        return result_matrix


def compute_updated_r_eff(
    wrapper: PyMCWrapper,
    i: int,
    log_liki_half: np.ndarray,
    S_half: int,
    r_eff_i: float,
) -> float:
    """Compute updated relative effective sample size.

    Parameters
    ----------
    wrapper : PyMCWrapper
        PyMC model wrapper instance
    i : int
        Index of the observation
    log_liki_half : np.ndarray
        Log likelihood values for observation i, shape (n_samples,)
    S_half : int
        Half the number of samples
    r_eff_i : float
        Current relative effective sample size for observation i

    Returns
    -------
    float
        Updated relative effective sample size (min of the two halves)
    """
    log_liki_half_1 = log_liki_half[S_half:]
    log_liki_half_2 = log_liki_half[:S_half]

    r_eff_i1 = r_eff_i2 = r_eff_i

    posterior = wrapper.idata.posterior
    n_chains = len(posterior.chain)

    if n_chains <= 1:
        r_eff_i1 = r_eff_i2 = 1.0
    else:
        try:
            upars_dict = wrapper.get_unconstrained_parameters()
            log_lik_result = log_lik_i_upars(wrapper, upars_dict, pointwise=True)

            if isinstance(log_lik_result, xr.DataArray):
                log_liki_chains = extract_log_likelihood_for_observation(
                    log_lik_result, i
                )

                n_draws = posterior.draw.size
                log_liki_chains = log_liki_chains.reshape(n_chains, n_draws)

                # Calculate ESS for first half
                if log_liki_chains[:, S_half:].size > 0:
                    ess_i1 = ess(log_liki_chains[:, S_half:], method="mean")
                    if isinstance(ess_i1, xr.DataArray):
                        ess_i1 = ess_i1.values
                    if ess_i1.size > 0:
                        r_eff_i1 = float(ess_i1 / max(1, len(log_liki_half_1)))

                # Calculate ESS for second half
                if log_liki_chains[:, :S_half].size > 0:
                    ess_i2 = ess(log_liki_chains[:, :S_half], method="mean")
                    if isinstance(ess_i2, xr.DataArray):
                        ess_i2 = ess_i2.values
                    if ess_i2.size > 0:
                        r_eff_i2 = float(ess_i2 / max(1, len(log_liki_half_2)))
            else:
                warnings.warn(
                    "Expected xarray.DataArray from log_lik_i_upars for"
                    f" observation {i}",
                    stacklevel=2,
                )
        except Exception as e:
            warnings.warn(
                f"Error calculating ESS for observation {i}, using original"
                f" r_eff_i: {e}",
                stacklevel=2,
            )
            return r_eff_i

    return min(r_eff_i1, r_eff_i2)


def extract_log_likelihood_for_observation(
    log_lik_result: xr.DataArray, i: int | Any
) -> np.ndarray:
    """Extract log likelihood values for a specific observation.

    Parameters
    ----------
    log_lik_result : xr.DataArray
        Log likelihood values from log_lik_i_upars
    i : Union[int, Any]
        Observation index, which is treated as:
        - An integer position when i is an integer
        - A direct coordinate value when i is any other type

    Returns
    -------
    np.ndarray
        Log likelihood values for observation i

    Raises
    ------
    ValueError
        If log_lik_result is not a DataArray
    IndexError
        If the integer index is out of bounds or no observation dimension is found
    """
    non_obs_dims = {"chain", "draw", "__sample__"}
    obs_dims = [dim for dim in log_lik_result.dims if dim not in non_obs_dims]

    if not obs_dims:
        return log_lik_result.values.flatten()

    # Try __obs__ first if it exists
    if "__obs__" in obs_dims:
        obs_dim = "__obs__"
    else:
        obs_dim = obs_dims[0]

    num_obs = log_lik_result.sizes[obs_dim]

    # Integer index
    if isinstance(i, (int, np.integer)):
        i_pos = int(i)

        if i_pos < 0 or i_pos >= num_obs:
            raise IndexError(
                f"Observation index {i_pos} out of bounds [0, {num_obs - 1}] for"
                f" dimension '{obs_dim}'"
            )

        return log_lik_result.isel({obs_dim: i_pos}).values.flatten()

    # Non-integer index
    try:
        return log_lik_result.sel({obs_dim: i}).values.flatten()
    except (KeyError, ValueError):
        raise ValueError(
            f"Observation {i} not found in dimension '{obs_dim}'. For positional"
            f" access, use an integer index between 0 and {num_obs - 1}."
        )


def _initialize_array(arr, default_func, dim):
    """Initialize array with default values if shape doesn't match."""
    return arr if arr.shape[0] == dim else default_func(dim)
