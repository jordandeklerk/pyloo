"""Wrapper for fitted PyMC models to support LOO-CV computations."""

import copy
import warnings
from typing import Any, Dict, Sequence

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from arviz import InferenceData
from pymc.model import Model
from pymc.model.transform.conditioning import remove_value_transforms


class PyMCWrapper:
    """Wrapper for fitted PyMC models providing standardized access to model components.

    Extracts and organizes all relevant information from a fitted PyMC model,
    including variables, data, and sampling functionality. Provides a consistent interface
    for computing log likelihoods and generating predictions, which is particularly useful
    for cross-validation procedures.

    Parameters
    ----------
    model : Model
        A fitted PyMC model containing the model structure and relationships
    idata : InferenceData
        ArviZ InferenceData object containing the model's posterior samples
    var_names : Sequence[str] | None
        Names of specific variables to focus on. If None, all variables are included

    Attributes
    ----------
    model : Model
        The underlying PyMC model
    idata : InferenceData
        ArviZ InferenceData object containing the model's posterior samples
    var_names : List[str]
        Names of variables being tracked
    observed_data : Dict[str, np.ndarray]
        Mapping of observed variable names to their data
    constant_data : Dict[str, np.ndarray]
        Mapping of constant data names to their values
    """

    def __init__(
        self,
        model: Model,
        idata: InferenceData,
        var_names: Sequence[str] | None = None,
    ):
        self.model = model
        self.idata = idata
        self.var_names = list(var_names) if var_names is not None else None
        self._untransformed_model = remove_value_transforms(copy.deepcopy(model))
        self._validate_model_state()
        self._extract_model_components()

    def select_observations(
        self,
        indices: np.ndarray | slice,
        var_name: str | None = None,
        axis: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Select and partition observations from a variable's data.

        This method partitions the data into two sets: selected observations and remaining
        observations. This is particularly useful for leave-one-out cross-validation where
        you need both the held-out data and the training data. If no variable name is provided,
        uses the first observed variable in the model.

        Parameters
        ----------
        indices : array-like or slice
            Indices of observations to select for the held-out set
        var_name : Optional[str]
            Name of the variable. If None, uses the first observed variable.
        axis : Optional[int]
            Axis along which to select observations.
            If None, assumes the first axis is the observation axis.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            - Selected (held-out) observations
            - Remaining (training) observations

        Raises
        ------
        ValueError
            If no observed variables exist, the specified variable name is not found,
            or indices are invalid
        IndexError
            If indices are out of bounds or negative
        """
        if not self.observed_data:
            raise ValueError("No observed variables found in the model")

        if var_name is None:
            var_name = self.get_observed_name()
        elif var_name not in self.observed_data:
            raise ValueError(f"Variable {var_name} not found in observed data")

        data = self.observed_data[var_name]
        if axis is None:
            axis = 0

        if np.any(self.get_missing_mask(var_name)):
            warnings.warn(
                f"Missing values detected in {var_name}. This may affect the results.",
                UserWarning,
                stacklevel=2,
            )

        try:
            if isinstance(indices, np.ndarray) and indices.dtype == bool:
                if indices.shape[0] != data.shape[axis]:
                    raise ValueError(
                        f"Boolean mask shape {indices.shape[0]} does not match "
                        f"data shape {data.shape[axis]} along axis {axis}"
                    )
                selected_indices = np.where(indices)[0]
                remaining_indices = np.where(~indices)[0]
            else:
                if not isinstance(indices, slice):
                    indices = np.asarray(indices, dtype=int)
                    if indices.size > 0:
                        if np.any(indices < 0):
                            raise IndexError("Negative indices are not allowed")
                        if np.any(indices >= data.shape[axis]):
                            raise IndexError("Index out of bounds")

                mask = np.zeros(data.shape[axis], dtype=bool)
                if isinstance(indices, slice):
                    idx_range = range(*indices.indices(data.shape[axis]))
                    mask[idx_range] = True
                else:
                    mask[indices] = True

                selected_indices = np.where(mask)[0]
                remaining_indices = np.where(~mask)[0]

            selected = np.take(data, selected_indices, axis=axis)
            remaining = np.take(data, remaining_indices, axis=axis)

            return selected, remaining
        except Exception as e:
            if isinstance(e, IndexError):
                raise
            raise ValueError(f"Failed to select observations: {str(e)}")

    def log_likelihood_i(
        self,
        var_name: str,
        idx: int,
        refitted_idata: InferenceData,
    ) -> xr.DataArray:
        """Compute pointwise log likelihood for a single held-out observation using a refitted model.

        This method is specifically designed for leave-one-out cross-validation (LOO-CV) where
        we need to compute the log likelihood of a held-out observation using a model that was
        refitted without that observation. This is different from the regular log_likelihood method
        which uses the original model fit.

        Parameters
        ----------
        var_name : str
            Name of the variable to compute log likelihood for
        idx : int
            Index of the single observation to compute log likelihood for
        refitted_idata : InferenceData
            InferenceData object from a model that was refit without the observation at idx

        Returns
        -------
        xr.DataArray
            Log likelihood values for the single held-out observation with dimensions (chain, draw)

        Raises
        ------
        ValueError
            If the variable has missing values or if the data is invalid
        """
        if self.get_variable(var_name) is None:
            raise ValueError(f"Variable {var_name} not found in model")

        if var_name not in self.observed_data:
            raise ValueError(f"No observed data found for variable {var_name}")

        if np.any(self.get_missing_mask(var_name)):
            raise ValueError(f"Missing values found in {var_name}")

        if not hasattr(refitted_idata, "posterior"):
            raise ValueError("refitted_idata must contain posterior samples")

        data_shape = self.get_shape(var_name)
        if data_shape is None:
            raise ValueError(f"Could not determine shape for variable {var_name}")

        if idx < 0 or idx >= data_shape[0]:
            raise IndexError(
                f"Index {idx} is out of bounds for axis 0 with size {data_shape[0]}"
            )

        dims = self.get_dims(var_name)
        if dims is None:
            warnings.warn(
                f"Could not determine dimensions for variable {var_name}. "
                "This may affect coordinate handling.",
                UserWarning,
                stacklevel=2,
            )

        try:
            holdout_data, _ = self.select_observations(
                np.array([idx], dtype=int), var_name=var_name
            )
            original_data = self.observed_data[var_name].copy()
            self.set_data({var_name: holdout_data})

            log_like = pm.compute_log_likelihood(
                refitted_idata,
                var_names=[var_name],
                model=self._untransformed_model,
                extend_inferencedata=False,
            )

            if var_name not in log_like:
                raise ValueError(
                    f"Failed to compute log likelihood for variable {var_name}"
                )

            log_like_i = log_like[var_name]
            obs_dims = [dim for dim in log_like_i.dims if dim not in ("chain", "draw")]

            for dim in obs_dims:
                log_like_i = log_like_i.isel({dim: 0})

            return log_like_i

        except Exception as e:
            if isinstance(e, (ValueError, IndexError)):
                raise
            raise ValueError(f"Failed to compute log likelihood: {str(e)}")

        finally:
            self.set_data({var_name: original_data})

    def sample_posterior(
        self,
        draws: int = 1000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.8,
        random_seed: int | None = None,
        progressbar: bool = True,
        **kwargs: Any,
    ) -> InferenceData:
        """Sample from the model's posterior distribution.

        Parameters
        ----------
        draws : int
            Number of posterior samples to draw
        tune : int
            Number of tuning steps
        chains : int
            Number of chains to sample
        target_accept : float
            Target acceptance rate for the sampler
        random_seed : Optional[int]
            Random seed for reproducibility
        progressbar : bool
            Whether to display a progress bar
        **kwargs : Any
            Additional arguments passed to pm.sample()

        Returns
        -------
        InferenceData
            ArviZ InferenceData object containing the posterior samples and
            log likelihood values

        Raises
        ------
        ValueError
            If draws or chains are not positive integers

        Notes
        -----
        Log likelihood computation is always enabled as it is required for
        LOO-CV computations.
        """
        if draws <= 0:
            raise ValueError("Number of draws must be positive")
        if chains <= 0:
            raise ValueError("Number of chains must be positive")

        idata_kwargs = kwargs.get("idata_kwargs", {})
        if isinstance(idata_kwargs, dict):
            if not idata_kwargs.get("log_likelihood", False):
                warnings.warn(
                    "Automatically enabling log likelihood computation as it is "
                    "required for LOO-CV.",
                    UserWarning,
                    stacklevel=2,
                )
                idata_kwargs["log_likelihood"] = True
                kwargs["idata_kwargs"] = idata_kwargs
        else:
            kwargs["idata_kwargs"] = {"log_likelihood": True}

        with self._untransformed_model:
            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                random_seed=random_seed,
                progressbar=progressbar,
                **kwargs,
            )
        return idata

    def sample_posterior_predictive(
        self,
        var_names: Sequence[str] | None = None,
        progressbar: bool = True,
        **kwargs: Any,
    ) -> InferenceData:
        """Generate posterior predictions.

        Parameters
        ----------
        var_names : Optional[Sequence[str]]
            Names of variables to predict.
            If None, predicts for all observed variables.
        progressbar : bool
            Whether to display a progress bar
        **kwargs : Any
            Additional arguments passed to pm.sample_posterior_predictive()

        Returns
        -------
        InferenceData
            ArviZ InferenceData object containing the predictions
        """
        if var_names is None:
            var_names = list(self.observed_data.keys())
        else:
            for var_name in var_names:
                if self.get_variable(var_name) is None:
                    raise ValueError(f"Variable {var_name} not found in model")

        if not hasattr(self.idata, "posterior"):
            raise ValueError("No posterior samples found in InferenceData object")

        for var_name in var_names:
            if var_name in self.observed_data and np.any(
                self.get_missing_mask(var_name)
            ):
                warnings.warn(
                    f"Missing values detected in {var_name}. This may affect"
                    " predictions.",
                    UserWarning,
                    stacklevel=2,
                )

        try:
            with self._untransformed_model:
                predictions = pm.sample_posterior_predictive(
                    self.idata,
                    var_names=var_names,
                    progressbar=progressbar,
                    **kwargs,
                )
            return predictions
        except Exception as e:
            raise ValueError(f"Failed to generate posterior predictions: {str(e)}")

    def get_unconstrained_parameters(self) -> Dict[str, xr.DataArray]:
        """Get unconstrained parameters from posterior samples.

        This method transforms the parameters from the constrained space (where they
        follow their specified prior distributions) to the unconstrained space where
        they can be treated as approximately normal for various computations like
        moment matching in LOO-CV.

        Returns
        -------
        Dict[str, xr.DataArray]
            Dictionary mapping parameter names to their unconstrained values
            with dimensions (chain, draw) and any parameter-specific dimensions

        Notes
        -----
        This uses PyMC's internal transformation infrastructure to properly
        handle different parameter types (e.g. positive parameters become log
        transformed, simplex parameters use stick breaking, etc.)
        """
        unconstrained_params = {}

        for var in self.model.free_RVs:
            var_name = var.name
            if var_name not in self.idata.posterior:
                continue

            param_samples = self.idata.posterior[var_name]
            untransformed_var = self._untransformed_model.named_vars[var_name]
            dist = untransformed_var.owner.op
            transform = getattr(dist, "transform", None)

            if transform is not None:
                # Apply backward transformation to get unconstrained space
                samples_np = param_samples.values
                unconstrained_np = transform.backward(
                    samples_np, *var.owner.inputs[1:]
                ).eval()
                unconstrained_samples = xr.DataArray(
                    unconstrained_np,
                    dims=param_samples.dims,
                    coords=param_samples.coords,
                    name=param_samples.name,
                )
            else:
                unconstrained_samples = param_samples.copy()

            unconstrained_params[var_name] = unconstrained_samples

        return unconstrained_params

    def constrain_parameters(
        self, unconstrained_params: Dict[str, xr.DataArray]
    ) -> Dict[str, xr.DataArray]:
        """Transform parameters from unconstrained to constrained space.

        This method transforms parameters from the unconstrained space back to
        their original constrained space where they follow their specified
        prior distributions.

        Parameters
        ----------
        unconstrained_params : Dict[str, xr.DataArray]
            Dictionary mapping parameter names to their unconstrained values
            with dimensions (chain, draw) and any parameter-specific dimensions

        Returns
        -------
        Dict[str, xr.DataArray]
            Dictionary mapping parameter names to their constrained values
            with dimensions (chain, draw) and any parameter-specific dimensions

        Notes
        -----
        This is the inverse operation of get_unconstrained_parameters()
        """
        constrained_params = {}

        for var in self.model.free_RVs:
            var_name = var.name
            if var_name not in unconstrained_params:
                continue

            unconstrained = unconstrained_params[var_name]

            # Get the distribution and its transform from the untransformed model
            untransformed_var = self._untransformed_model.named_vars[var_name]
            dist = untransformed_var.owner.op
            transform = getattr(dist, "transform", None)

            if transform is not None:
                # Apply forward transformation to get constrained space
                unconstrained_np = unconstrained.values
                constrained_np = transform.forward(
                    unconstrained_np, *var.owner.inputs[1:]
                ).eval()
                constrained = xr.DataArray(
                    constrained_np,
                    dims=unconstrained.dims,
                    coords=unconstrained.coords,
                    name=unconstrained.name,
                )
            else:
                constrained = unconstrained.copy()

            constrained_params[var_name] = constrained

        return constrained_params

    def log_likelihood(
        self,
        var_name: str | None = None,
        indices: np.ndarray | slice | None = None,
        axis: int | None = None,
    ) -> xr.DataArray:
        """Compute pointwise log likelihoods using the original model fit.

        Main method for accessing pre-computed log likelihood values from the
        original model fit. It provides flexible indexing to access log likelihoods for
        specific observations. Unlike log_likelihood__i, this uses the original model fit
        and doesn't require refitting.

        Parameters
        ----------
        var_name : str | None
            Name of the variable to compute log likelihoods for.
            If None, uses the first observed variable.
        indices : np.ndarray | slice | None
            Indices for selecting specific observations.
            If None, returns log likelihoods for all observations.
        axis : int | None
            Axis along which to select observations.
            If None, assumes the first axis.

        Returns
        -------
        xr.DataArray
            Log likelihood values with dimensions (chain, draw) and any observation
            dimensions from the original data.

        See Also
        --------
        compute_pointwise_log_likelihood : Internal method for log likelihood computation
        """
        if var_name is None:
            var_name = self.get_observed_name()

        if (
            not hasattr(self.idata, "log_likelihood")
            or var_name not in self.idata.log_likelihood
        ):
            raise ValueError(f"No log likelihood values found for variable {var_name}")

        log_like = self.idata.log_likelihood[var_name]

        rename_dict = {}
        for dim in log_like.dims:
            if dim not in ("chain", "draw"):
                prefix = f"{var_name}_dim_"
                if dim.startswith(prefix):
                    new_name = dim[len(prefix) :]
                    new_name = dim[len(prefix) :]
                    rename_dict[dim] = f"dim_{new_name}"

        if rename_dict:
            log_like = log_like.rename(rename_dict)

        if indices is None and axis is None:
            return log_like

        indices_dict = {var_name: indices} if indices is not None else None
        axis_dict = {var_name: axis} if axis is not None else None

        return self._compute_log_likelihood([var_name], indices_dict, axis_dict)[
            var_name
        ]

    def _compute_log_likelihood(
        self,
        var_names: Sequence[str],
        indices: dict[str, np.ndarray | slice] | None = None,
        axis: dict[str, int] | None = None,
    ) -> dict[str, xr.DataArray]:
        """Internal method to compute pointwise log likelihoods for multiple variables.

        This is a lower-level helper method that handles the computation of log likelihoods
        for multiple variables simultaneously. It's used internally by log_likelihood() to
        handle the actual computations.

        Parameters
        ----------
        var_names : Sequence[str]
            Names of variables to compute log likelihoods for
        indices : dict[str, np.ndarray | slice] | None
            Dictionary mapping variable names to indices for selecting specific
            observations. If None, uses all observations.
        axis : dict[str, int] | None
            Dictionary mapping variable names to axes along which to select
            observations. If None for a variable, assumes the first axis.

        Returns
        -------
        dict[str, xr.DataArray]
            Dictionary mapping variable names to their pointwise log likelihoods
            as xarray DataArrays with dimensions (chain, draw) and any observation
            dimensions from the original data
        """
        if indices is None:
            indices = {}
        if axis is None:
            axis = {}

        log_likes = {}
        for var_name in var_names:
            log_like = self.log_likelihood(var_name=var_name)

            if var_name in indices:
                idx = indices[var_name]
                ax = axis.get(var_name, 0)
                dim_name = [d for d in log_like.dims if d not in ("chain", "draw")][ax]

                if isinstance(idx, slice):
                    log_like = log_like.isel({dim_name: idx})
                else:
                    log_like = log_like.isel({dim_name: idx})

            log_likes[var_name] = log_like

        return log_likes

    def set_data(
        self,
        new_data: dict[str, np.ndarray],
        coords: dict[str, Sequence] | None = None,
        mask: dict[str, np.ndarray] | None = None,
    ) -> None:
        """Update the observed data in the model.

        Parameters
        ----------
        new_data : Dict[str, np.ndarray]
            Dictionary mapping variable names to new observed values
        coords : Optional[Dict[str, Sequence]]
            Optional coordinates for the new data dimensions
        mask : Optional[Dict[str, np.ndarray]]
            Optional boolean masks for each variable to handle missing data or
            to select specific observations. True values indicate valid data points.

        Raises
        ------
        ValueError
            If the provided data has incompatible dimensions with the model,
            if the variable name is not found in the model,
            if the coordinates are invalid,
            or if the data violates distribution constraints (e.g., negative values for Poisson)
        """
        for var_name, values in new_data.items():
            var = self.get_variable(var_name)
            if var is None:
                raise ValueError(f"Variable {var_name} not found in model")

            expected_shape = self.get_shape(var_name)
            if expected_shape is None:
                raise ValueError(f"Could not determine shape for variable {var_name}")

            if mask is not None and var_name in mask:
                mask_array = mask[var_name]
                if mask_array.shape != values.shape:
                    raise ValueError(
                        f"Mask shape {mask_array.shape} does not match data shape"
                        f" {values.shape} for variable {var_name}"
                    )
                values = np.ma.masked_array(values, mask=~mask_array)

            if len(values.shape) != len(expected_shape):
                raise ValueError(
                    f"Incompatible dimensions for {var_name}. "
                    f"Expected {len(expected_shape)} dims, got {len(values.shape)}"
                )

            self.observed_data[var_name] = values
            if coords is not None:
                self._validate_coords(var_name, coords)

    def get_missing_mask(
        self,
        var_name: str,
        axis: int | None = None,
    ) -> np.ndarray:
        """Get boolean mask indicating missing values in the data.

        Parameters
        ----------
        var_name : str
            Name of the variable
        axis : Optional[int]
            Axis along which to check for missing values.
            If None, returns mask for all dimensions.

        Returns
        -------
        np.ndarray
            Boolean mask where True indicates missing values

        Raises
        ------
        ValueError
            If the variable name is not found
        """
        if var_name not in self.observed_data:
            raise ValueError(f"Variable {var_name} not found in observed data")

        data = self.observed_data[var_name]
        if isinstance(data, np.ma.MaskedArray):
            mask = data.mask
            if axis is not None:
                mask = np.any(
                    mask, axis=tuple(i for i in range(mask.ndim) if i != axis)
                )
            return mask
        else:
            mask = np.isnan(data)
            if axis is not None:
                mask = np.any(
                    mask, axis=tuple(i for i in range(mask.ndim) if i != axis)
                )
            return mask

    def get_observed_name(self) -> str:
        """Get the name of the first (and typically only) observed variable.

        Returns
        -------
        str
            Name of the first observed variable

        Raises
        ------
        ValueError
            If no observed variables exist in the model
        """
        if not self.observed_data:
            raise ValueError("No observed variables found in the model")
        return next(iter(self.observed_data))

    def get_observed_data(self) -> np.ndarray:
        """Get the data of the first (and typically only) observed variable.

        Returns
        -------
        np.ndarray
            Data of the first observed variable

        Raises
        ------
        ValueError
            If no observed variables exist in the model
        """
        return self.observed_data[self.get_observed_name()].copy()

    def get_variable(self, var_name: str) -> pt.TensorVariable | None:
        """Retrieve a variable from the model by name.

        Parameters
        ----------
        var_name : str
            Name of the variable to retrieve

        Returns
        -------
        Optional[pt.TensorVariable]
            The requested variable or None if not found
        """
        return self._untransformed_model.named_vars.get(var_name)

    def get_dims(self, var_name: str) -> tuple[str, ...] | None:
        """Get the dimension names for a variable.

        Parameters
        ----------
        var_name : str
            Name of the variable

        Returns
        -------
        Optional[Tuple[str, ...]]
            Tuple of dimension names or None if variable not found
        """
        return self._untransformed_model.named_vars_to_dims.get(var_name)

    def get_shape(self, var_name: str) -> tuple[int, ...] | None:
        """Get the shape of a variable.

        Parameters
        ----------
        var_name : str
            Name of the variable

        Returns
        -------
        Optional[Tuple[int, ...]]
            Shape of the variable or None if variable not found

        Notes
        -----
        For observed variables, returns the shape of the observed data.
        For other variables, returns the shape from the model definition.
        """
        if var_name in self.observed_data:
            return tuple(self.observed_data[var_name].shape)
        elif var_name in self._untransformed_model.named_vars:
            var = self._untransformed_model.named_vars[var_name]
            return tuple(d.eval() if hasattr(d, "eval") else d for d in var.shape)
        return None

    def _validate_model_state(self) -> None:
        """Validate that the model is properly fitted and ready for use."""
        # Check that posterior samples exist
        if not hasattr(self.idata, "posterior"):
            raise ValueError(
                "InferenceData object must contain posterior samples. "
                "The model does not appear to be fitted."
            )

        # Check that all free variables have posterior samples
        posterior_vars = set(self.idata.posterior.data_vars.keys())
        model_vars = {rv.name for rv in self.model.free_RVs}
        missing_vars = model_vars - posterior_vars
        if missing_vars:
            raise ValueError(
                f"Missing posterior samples for variables: {missing_vars}. "
                "The model may not be fully fitted."
            )

        # Check that observed data exists for observed variables
        for obs_rv in self.model.observed_RVs:
            if not hasattr(obs_rv.tag, "observations"):
                raise ValueError(
                    f"Missing observed data for variable {obs_rv.name}. "
                    "The model is not properly initialized."
                )

        # Validate posterior sample shapes against model structure
        for var_name in model_vars:
            var = self.model.named_vars[var_name]
            expected_shape = tuple(
                d.eval() if hasattr(d, "eval") else d for d in var.shape
            )
            posterior_shape = tuple(self.idata.posterior[var_name].shape[2:])
            if expected_shape != posterior_shape:
                raise ValueError(
                    f"Shape mismatch for variable {var_name}. Model expects shape"
                    f" {expected_shape}, but posterior has shape {posterior_shape}"
                )

    def _extract_model_components(self) -> None:
        """Extract and organize the model's components."""
        self.observed_data = {}
        if hasattr(self.idata, "observed_data"):
            for var_name, data_array in self.idata.observed_data.items():
                if self.var_names is None or var_name in self.var_names:
                    self.observed_data[var_name] = data_array.values.copy()

        self.constant_data = {}
        for data_var in self._untransformed_model.data_vars:
            data_name = data_var.name
            if hasattr(data_var, "get_value"):
                self.constant_data[data_name] = data_var.get_value()

        self.free_vars = [rv.name for rv in self._untransformed_model.free_RVs]
        self.deterministic_vars = [
            det.name for det in self._untransformed_model.deterministics
        ]

    def _validate_coords(
        self,
        var_name: str,
        coords: dict[str, Sequence],
    ) -> None:
        """Validate coordinate values against variable dimensions."""
        dims = self.get_dims(var_name)
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
