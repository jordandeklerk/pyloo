"""Wrapper for fitted PyMC models to support LOO-CV computations."""

import copy
import logging
import warnings
from typing import Any, Sequence

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from arviz import InferenceData
from pymc.model import Model
from pymc.model.transform.conditioning import remove_value_transforms

__all__ = ["PyMCWrapper"]

logger = logging.getLogger(__name__)


class PyMCWrapperError(Exception):
    """Base exception class for PyMC wrapper errors."""

    pass


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
    var_names : list[str]
        Names of variables being tracked
    observed_data : dict[str, np.ndarray]
        Mapping of observed variable names to their data
    constant_data : dict[str, np.ndarray]
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

        for var_name, data in self.observed_data.items():
            if isinstance(data, np.ndarray):
                self.observed_data[var_name] = data.copy()
                self.observed_data[var_name].flags.writeable = False

    def check_implemented_methods(self, required_methods: Sequence[str]) -> list[str]:
        """Check if all required methods are implemented.

        Parameters
        ----------
        required_methods : Sequence[str]
            Names of methods that should be implemented

        Returns
        -------
        list[str]
            Names of methods that are not implemented
        """
        return [
            method
            for method in required_methods
            if not hasattr(self, method) or not callable(getattr(self, method))
        ]

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
        var_name : str | None
            Name of the variable. If None, uses the first observed variable.
        axis : int | None
            Axis along which to select observations.
            If None, assumes the first axis is the observation axis.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            - Selected (held-out) observations
            - Remaining (training) observations

        Raises
        ------
        PyMCWrapperError
            If no observed variables exist or the specified variable name is not found
        IndexError
            If indices are out of bounds or negative
        """
        if not self.observed_data:
            raise PyMCWrapperError("No observed variables found in the model")

        if var_name is None:
            var_name = self.get_observed_name()
        elif var_name not in self.observed_data:
            raise PyMCWrapperError(
                f"Variable '{var_name}' not found in observed data. "
                f"Available variables: {list(self.observed_data.keys())}"
            )

        data = self.observed_data[var_name]
        if axis is None:
            axis = 0

        if np.any(self.get_missing_mask(var_name)):
            logger.warning(
                "Missing values detected in %s. This may affect the results.", var_name
            )

        try:
            mask = self._create_selection_mask(indices, data.shape[axis], axis)
            idx = [slice(None)] * data.ndim

            # Select observations
            idx[axis] = mask
            selected = data[tuple(idx)]

            # Remaining observations
            idx[axis] = ~mask
            remaining = data[tuple(idx)]

            return selected, remaining

        except Exception as e:
            if isinstance(e, (IndexError, PyMCWrapperError)):
                raise
            raise PyMCWrapperError(f"Failed to select observations: {str(e)}")

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
        PyMCWrapperError
            If the variable has missing values, if the data is invalid, or if computation fails

        See Also
        --------
        log_likelihood : Compute pointwise log likelihood for all observations
        """
        if self.get_variable(var_name) is None:
            raise PyMCWrapperError(
                f"Variable '{var_name}' not found in model. Available variables:"
                f" {list(self._untransformed_model.named_vars.keys())}"
            )

        if var_name not in self.observed_data:
            raise PyMCWrapperError(
                f"No observed data found for variable '{var_name}'. "
                f"Available observed variables: {list(self.observed_data.keys())}"
            )

        if np.any(self.get_missing_mask(var_name)):
            raise PyMCWrapperError(f"Missing values found in {var_name}")

        if not hasattr(refitted_idata, "posterior"):
            raise PyMCWrapperError(
                "refitted_idata must contain posterior samples. "
                "Check that the model was properly refit."
            )

        data_shape = self.get_shape(var_name)
        if data_shape is None:
            raise PyMCWrapperError(f"Could not determine shape for variable {var_name}")

        if idx < 0 or idx >= data_shape[0]:
            raise IndexError(
                f"Index {idx} is out of bounds for axis 0 with size {data_shape[0]}"
            )

        dims = self.get_dims(var_name)
        if dims is None:
            logger.warning(
                f"Could not determine dimensions for variable {var_name}. "
                "This may affect coordinate handling.",
                stacklevel=2,
            )

        try:
            holdout_data, _ = self.select_observations(
                np.array([idx], dtype=int), var_name=var_name
            )
            original_data = self.observed_data[var_name].copy()

            orig_coords = None
            if dims is not None:
                orig_coords = self._get_coords(var_name)

            # Set holdout data without coordinate validation
            self.observed_data[var_name] = holdout_data

            log_like = pm.compute_log_likelihood(
                refitted_idata,
                var_names=[var_name],
                model=self._untransformed_model,
                extend_inferencedata=False,
            )

            if var_name not in log_like:
                raise PyMCWrapperError(
                    f"Failed to compute log likelihood for variable {var_name}. "
                    "Check that the model specification matches the data."
                )

            log_like_i = log_like[var_name]
            obs_dims = [dim for dim in log_like_i.dims if dim not in ("chain", "draw")]

            for dim in obs_dims:
                log_like_i = log_like_i.isel({dim: 0})

            return log_like_i

        except Exception as e:
            if isinstance(e, (IndexError, PyMCWrapperError)):
                raise
            raise PyMCWrapperError(f"Failed to compute log likelihood: {str(e)}")
        finally:
            if orig_coords is not None:
                self.set_data({var_name: original_data}, coords=orig_coords)
            else:
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
        random_seed : int | None
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
        PyMCWrapperError
            If sampling parameters are invalid or sampling fails
        """
        if draws <= 0:
            raise PyMCWrapperError(f"Number of draws must be positive, got {draws}")
        if chains <= 0:
            raise PyMCWrapperError(f"Number of chains must be positive, got {chains}")

        idata_kwargs = kwargs.get("idata_kwargs", {})
        if isinstance(idata_kwargs, dict):
            if not idata_kwargs.get("log_likelihood", False):
                logger.info(
                    "Automatically enabling log likelihood computation as it is "
                    "required for LOO-CV."
                )
                idata_kwargs["log_likelihood"] = True
                kwargs["idata_kwargs"] = idata_kwargs
        else:
            kwargs["idata_kwargs"] = {"log_likelihood": True}

        try:
            with self.model:
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
        except Exception as e:
            raise PyMCWrapperError(f"Sampling failed: {str(e)}")

    def sample_posterior_predictive(
        self,
        var_names: Sequence[str] | None = None,
        progressbar: bool = True,
        **kwargs: Any,
    ) -> InferenceData:
        """Generate posterior predictions.

        Parameters
        ----------
        var_names : Sequence[str] | None
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

        Raises
        ------
        PyMCWrapperError
            If variables are invalid or prediction fails

        See Also
        --------
        sample_posterior : Sample from the model's posterior distribution
        """
        if var_names is None:
            var_names = list(self.observed_data.keys())
        else:
            for var_name in var_names:
                if self.get_variable(var_name) is None:
                    raise PyMCWrapperError(
                        f"Variable '{var_name}' not found in model. Available"
                        " variables:"
                        f" {list(self.model.named_vars.keys())}"
                    )

        if not hasattr(self.idata, "posterior"):
            raise PyMCWrapperError(
                "No posterior samples found in InferenceData object. "
                "The model must be fitted before generating predictions."
            )

        for var_name in var_names:
            if var_name in self.observed_data and np.any(
                self.get_missing_mask(var_name)
            ):
                logger.warning(
                    "Missing values detected in %s. This may affect predictions.",
                    var_name,
                )

        try:
            with self.model:
                predictions = pm.sample_posterior_predictive(
                    self.idata,
                    var_names=var_names,
                    progressbar=progressbar,
                    **kwargs,
                )
            return predictions
        except Exception as e:
            raise PyMCWrapperError(
                f"Failed to generate posterior predictions: {str(e)}"
            )

    def get_unconstrained_parameters(self) -> dict[str, xr.DataArray]:
        """Get unconstrained parameters from posterior samples.

        This method transforms the parameters from the constrained space (where they
        follow their specified prior distributions) to the unconstrained space where
        they can be treated as approximately normal for various computations like
        moment matching in LOO-CV.

        Returns
        -------
        dict[str, xr.DataArray]
            Dictionary mapping parameter names to their unconstrained values
            with dimensions (chain, draw) and any parameter-specific dimensions

        Notes
        -----
        This uses PyMC's internal transformation infrastructure to properly
        handle different parameter types (e.g. positive parameters become log
        transformed, simplex parameters use stick breaking, etc.)

        If a distribution does not provide a transform attribute, or if the
        transformation fails, the original parameter values are returned with
        a warning.

        See Also
        --------
        constrain_parameters : Inverse operation to get_unconstrained_parameters
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

            try:
                if transform is None:
                    raise ValueError("No transform available")

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
            except Exception as e:
                logger.warning(
                    "Failed to transform %s: %s",
                    var_name,
                    str(e),
                )
                unconstrained_samples = param_samples.copy()

            unconstrained_params[var_name] = unconstrained_samples

        return unconstrained_params

    def constrain_parameters(
        self, unconstrained_params: dict[str, xr.DataArray]
    ) -> dict[str, xr.DataArray]:
        """Transform parameters from unconstrained to constrained space.

        This method transforms parameters from the unconstrained space back to
        their original constrained space where they follow their specified
        prior distributions.

        Parameters
        ----------
        unconstrained_params : dict[str, xr.DataArray]
            Dictionary mapping parameter names to their unconstrained values
            with dimensions (chain, draw) and any parameter-specific dimensions

        Returns
        -------
        dict[str, xr.DataArray]
            Dictionary mapping parameter names to their constrained values
            with dimensions (chain, draw) and any parameter-specific dimensions

        Notes
        -----
        This is the inverse operation of get_unconstrained_parameters()

        See Also
        --------
        get_unconstrained_parameters : Inverse operation to constrain_parameters
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

            if transform is None:
                logger.warning(
                    "No transform found for variable %s. Using original values.",
                    var_name,
                )
                constrained = unconstrained.copy()
            else:
                try:
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
                except Exception as e:
                    logger.warning(
                        "Failed to transform %s to constrained space: %s. "
                        "Using original values.",
                        var_name,
                        str(e),
                    )
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
        log_likelihood_i : Compute pointwise log likelihood for a single held-out observation
        compute_pointwise_log_likelihood : Internal method for log likelihood computation
        """
        if var_name is None:
            var_name = self.get_observed_name()

        if (
            not hasattr(self.idata, "log_likelihood")
            or var_name not in self.idata.log_likelihood
        ):
            raise PyMCWrapperError(
                f"No log likelihood values found for variable {var_name}. "
                "Check that log_likelihood=True was set during model fitting."
            )

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
        update_coords: bool = True,
    ) -> None:
        """Update the observed data in the model.

        Parameters
        ----------
        new_data : dict[str, np.ndarray]
            Dictionary mapping variable names to new observed values
        coords : dict[str, Sequence] | None
            Optional coordinates for the new data dimensions
        mask : dict[str, np.ndarray] | None
            Optional boolean masks for each variable to handle missing data or
            to select specific observations. True values indicate valid data points.
        update_coords : bool
            If True, automatically update coordinates when data dimensions change.
            If False, raise an error when new data would break coordinate consistency.

        Returns
        -------
        None

        Raises
        ------
        PyMCWrapperError
            If the provided data has incompatible dimensions with the model,
            if the variable name is not found in the model
        ValueError
            If coordinates are invalid or missing required dimensions
        """
        for var_name, values in new_data.items():
            var = self.get_variable(var_name)
            if var is None:
                raise PyMCWrapperError(
                    f"Variable '{var_name}' not found in model. Available variables:"
                    f" {list(self._untransformed_model.named_vars.keys())}"
                )

            expected_shape = self.get_shape(var_name)
            if expected_shape is None:
                raise PyMCWrapperError(
                    f"Could not determine shape for variable {var_name}"
                )

            if len(values.shape) != len(expected_shape):
                raise PyMCWrapperError(
                    f"New data for {var_name} has {len(values.shape)} dimensions but"
                    f" model expects {len(expected_shape)} dimensions. Expected shape:"
                    f" {expected_shape}, got: {values.shape}"
                )

            # Create a copy of the data to ensure independence
            values = values.copy()

            if mask is not None and var_name in mask:
                mask_array = mask[var_name]
                if mask_array.shape != values.shape:
                    raise PyMCWrapperError(
                        f"Mask shape {mask_array.shape} does not match data shape "
                        f"{values.shape} for variable {var_name}"
                    )
                values = np.ma.masked_array(values, mask=~mask_array)

            orig_dims = self.get_dims(var_name)
            if orig_dims is None:
                self.observed_data[var_name] = values
                # Make data immutable
                if isinstance(values, np.ndarray):
                    values.flags.writeable = False
                continue

            working_coords = {} if coords is None else coords.copy()
            required_dims = {d for d in orig_dims if d is not None}
            missing_coords = required_dims - set(working_coords.keys())

            if not update_coords and missing_coords:
                raise ValueError(
                    f"Missing coordinates for dimensions {missing_coords} of variable"
                    f" {var_name}"
                )

            for dim, size in zip(orig_dims, values.shape):
                if dim is not None:
                    if dim not in working_coords:
                        if update_coords:
                            working_coords[dim] = list(range(size))
                            warnings.warn(
                                "Automatically created coordinates for dimension"
                                f" {dim}",
                                UserWarning,
                                stacklevel=2,
                            )
                        else:
                            raise ValueError(
                                f"Missing coordinates for dimension {dim} of variable"
                                f" {var_name}"
                            )
                    elif len(working_coords[dim]) != size:
                        if update_coords:
                            original_len = len(working_coords[dim])
                            working_coords[dim] = list(range(size))
                            warnings.warn(
                                f"Coordinate length changed from {original_len} to"
                                f" {size}",
                                UserWarning,
                                stacklevel=2,
                            )
                        else:
                            raise ValueError(
                                f"Coordinate length {len(working_coords[dim])} for"
                                f" dimension {dim} does not match variable shape"
                                f" {size} for {var_name}"
                            )

            self.observed_data[var_name] = values
            # Make data immutable
            if isinstance(values, np.ndarray):
                values.flags.writeable = False

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
        axis : int | None
            Axis along which to check for missing values.
            If None, returns mask for all dimensions.

        Returns
        -------
        np.ndarray
            Boolean mask where True indicates missing values

        Raises
        ------
        PyMCWrapperError
            If the variable name is not found
        """
        if var_name not in self.observed_data:
            raise PyMCWrapperError(
                f"Variable '{var_name}' not found in observed data. "
                f"Available variables: {list(self.observed_data.keys())}"
            )

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
        PyMCWrapperError
            If no observed variables exist in the model
        """
        if not self.observed_data:
            raise PyMCWrapperError("No observed variables found in the model")
        return next(iter(self.observed_data))

    def get_observed_data(self) -> np.ndarray:
        """Get the data of the first (and typically only) observed variable.

        Returns
        -------
        np.ndarray
            Data of the first observed variable

        Raises
        ------
        PyMCWrapperError
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
        pt.TensorVariable | None
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
        tuple[str, ...] | None
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
        tuple[int, ...] | None
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

    def _create_selection_mask(
        self, indices: np.ndarray | slice, length: int, axis: int
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
        if indices.size > 0:
            if np.any(indices < 0):
                raise IndexError(
                    "Negative indices are not allowed. Found indices:"
                    f" {indices[indices < 0]}"
                )
            if np.any(indices >= length):
                raise IndexError(
                    f"Index {max(indices)} is out of bounds for axis {axis}"
                    f" with size {length}"
                )

        mask = np.zeros(length, dtype=bool)
        np.put(mask, indices, True)
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
            expected_shape = tuple(
                d.eval() if hasattr(d, "eval") else d for d in var.shape
            )
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

    def _get_coords(self, var_name: str) -> dict[str, Sequence[int]] | None:
        """Get the coordinates for a variable."""
        dims = self.get_dims(var_name)
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
