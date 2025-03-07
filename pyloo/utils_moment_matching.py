"""Utility functions for moment matching."""

from typing import TypedDict

import arviz as az
import numpy as np
import pymc as pm
import xarray as xr

__all__ = [
    "log_lik_i_upars",
    "log_prob_upars",
    "_initialize_array",
    "ShiftResult",
    "ShiftAndScaleResult",
    "ShiftAndCovResult",
    "UpdateQuantitiesResult",
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


def log_prob_upars(model, upars: dict[str, xr.DataArray]) -> np.ndarray:
    """Compute log probability for unconstrained parameters.

    This function computes the log probability for each unconstrained parameter
    directly, without converting to constrained space.

    Parameters
    ----------
    model : PyMC model
        PyMC model object that provides access to model variables
    upars : dict of xarray.DataArray
        Unconstrained parameters from model.get_unconstrained_parameters()

    Returns
    -------
    np.ndarray
        Matrix of log probability values with shape (n_samples, n_variables)
        where n_samples = n_chains * n_draws and n_variables is the number of
        parameters in upars
    """
    model = getattr(model, "model", model)
    var_names = list(upars.keys())
    first_var = upars[var_names[0]]
    n_samples = first_var.shape[0] * first_var.shape[1]

    result_matrix = np.zeros((n_samples, len(var_names)))

    for i, name in enumerate(var_names):
        # Stack chains and draws
        param = upars[name].stack(__sample__=("chain", "draw"))

        # Get variable and its transformed name
        with model:
            var = model[name]
            value_var = model.rvs_to_values.get(var)

            if value_var is None:
                result_matrix[:, i] = np.nan
                continue

            # Compile logp function once for this variable
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

    return result_matrix


def _initialize_array(arr, default_func, dim):
    """Initialize array with default values if shape doesn't match."""
    return arr if arr.shape[0] == dim else default_func(dim)
