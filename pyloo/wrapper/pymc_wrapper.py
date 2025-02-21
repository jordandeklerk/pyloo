"""Wrapper for fitted PyMC models to support LOO-CV computations."""

import copy
import warnings
from typing import Any, Sequence

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

    See Also
    --------
    get_log_likelihood : Compute pointwise log likelihoods
    sample_posterior_predictive : Generate posterior predictions
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
        # Work with a deep copy of the original model to avoid issues with extra dimensions
        self._untransformed_model = remove_value_transforms(copy.deepcopy(model))
        self._validate_model_state()
        self._extract_model_components()

    def log_likelihood(
        self,
        var_names: Sequence[str] | None = None,
        indices: dict[str, np.ndarray | slice] | None = None,
        axis: dict[str, int] | None = None,
    ) -> dict[str, np.ndarray]:
        """Compute pointwise log likelihoods for specified variables.

        Calculates the log likelihood values for each observation point in the specified
        variables using the model's posterior samples. Supports computing log likelihoods
        for specific subsets of observations through indexing.

        Parameters
        ----------
        var_names : Sequence[str] | None
            Names of variables to compute log likelihoods for.
            If None, computes for all observed variables
        indices : dict[str, np.ndarray | slice] | None
            Dictionary mapping variable names to indices for selecting specific
            observations. If None, uses all observations
        axis : dict[str, int] | None
            Dictionary mapping variable names to axes along which to select
            observations. If None for a variable, assumes the first axis
        progressbar : bool
            Whether to display a progress bar during computation

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary mapping variable names to their pointwise log likelihoods
            with shape (n_chains, n_draws, n_points)

        See Also
        --------
        compute_pointwise_log_likelihood : Internal method for log likelihood computation
        """
        return self._compute_log_likelihood(var_names, indices, axis)

    def _compute_log_likelihood(
        self,
        var_names: Sequence[str] | None = None,
        indices: dict[str, np.ndarray | slice] | None = None,
        axis: dict[str, int] | None = None,
    ) -> dict[str, np.ndarray]:
        """Internal method to compute pointwise log likelihoods.

        Parameters
        ----------
        var_names : Sequence[str] | None
            Names of variables to compute log likelihoods for.
            If None, computes for all observed variables
        indices : dict[str, np.ndarray | slice] | None
            Dictionary mapping variable names to indices for selecting specific
            observations. If None, uses all observations
        axis : dict[str, int] | None
            Dictionary mapping variable names to axes along which to select
            observations. If None for a variable, assumes the first axis

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary mapping variable names to their pointwise log likelihoods
            with shape (n_samples, n_points) where n_samples = n_chains * n_draws
        """
        if var_names is None:
            var_names = list(self.observed_data.keys())

        if indices is None:
            indices = {}
        if axis is None:
            axis = {}

        log_likes = {}
        for var_name in var_names:
            log_like = (
                self.idata.log_likelihood[var_name]
                .stack(__sample__=("chain", "draw"))
                .values
            )

            n_dims = len(log_like.shape)
            sample_dim = n_dims - 1
            log_like = np.moveaxis(log_like, sample_dim, 0)

            if var_name in indices:
                idx = indices[var_name]
                ax = axis.get(var_name, 0)
                ax += 1  # Account for sample dimension at front
                if isinstance(idx, slice):
                    idx_range = range(*idx.indices(log_like.shape[ax]))
                    log_like = np.take(log_like, idx_range, axis=ax)
                else:
                    log_like = np.take(log_like, idx, axis=ax)
            log_likes[var_name] = log_like

        return log_likes

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

        Notes
        -----
        Log likelihood computation is always enabled as it is required for
        LOO-CV computations.
        """
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

        with self._untransformed_model:
            predictions = pm.sample_posterior_predictive(
                self.idata,
                var_names=var_names,
                progressbar=progressbar,
                **kwargs,
            )
        return predictions

    def select_observations(
        self,
        var_name: str,
        indices: np.ndarray | slice,
        axis: int | None = None,
    ) -> tuple[np.ndarray, dict[str, Sequence] | None]:
        """Select specific observations from a variable's data.

        Parameters
        ----------
        var_name : str
            Name of the variable
        indices : array-like or slice
            Indices of observations to select
        axis : Optional[int]
            Axis along which to select observations.
            If None, assumes the first axis is the observation axis.

        Returns
        -------
        Tuple[np.ndarray, Optional[Dict[str, Sequence]]]
            - Selected observations
            - Dictionary of coordinates for the selected data (if available)
              or None if no coordinates are associated with the variable

        Raises
        ------
        ValueError
            If the variable name is not found or indices are invalid
        """
        if var_name not in self.observed_data:
            raise ValueError(f"Variable {var_name} not found in observed data")

        data = self.observed_data[var_name]
        if axis is None:
            axis = 0

        dims = self.get_dims(var_name)
        selected_coords: dict[str, Sequence] | None = None

        try:
            if isinstance(indices, slice):
                idx_range = range(*indices.indices(data.shape[axis]))
                selected = np.take(data, idx_range, axis=axis)

                if dims is not None:
                    selected_dim = dims[axis] if axis < len(dims) else None
                    if (
                        selected_dim is not None
                        and selected_dim in self._untransformed_model.coords
                    ):
                        selected_coords = {}
                        for i, dim in enumerate(dims):
                            if dim is None:
                                continue
                            if i == axis:
                                selected_coords[dim] = [
                                    list(self._untransformed_model.coords[dim])[i]
                                    for i in idx_range
                                ]
                            else:
                                selected_coords[dim] = list(
                                    self._untransformed_model.coords[dim]
                                )
            else:
                selected = np.take(data, indices, axis=axis)

                if dims is not None:
                    selected_dim = dims[axis] if axis < len(dims) else None
                    if (
                        selected_dim is not None
                        and selected_dim in self._untransformed_model.coords
                    ):
                        selected_coords = {}
                        for i, dim in enumerate(dims):
                            if dim is None:
                                continue
                            if i == axis:
                                selected_coords[dim] = [
                                    list(self._untransformed_model.coords[dim])[i]
                                    for i in indices
                                ]
                            else:
                                selected_coords[dim] = list(
                                    self._untransformed_model.coords[dim]
                                )

            return selected, selected_coords
        except Exception as e:
            raise ValueError(f"Failed to select observations: {str(e)}")

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
            or if the coordinates are invalid.
        """
        for var_name, values in new_data.items():
            if var_name not in self._untransformed_model.named_vars:
                raise ValueError(f"Variable {var_name} not found in model")

            var = self._untransformed_model.named_vars[var_name]
            expected_shape = tuple(
                d.eval() if hasattr(d, "eval") else d for d in var.shape
            )

            if mask is not None and var_name in mask:
                mask_array = mask[var_name]
                if mask_array.shape != values.shape:
                    raise ValueError(
                        f"Mask shape {mask_array.shape} does not match data shape"
                        f" {values.shape} for variable {var_name}"
                    )
                values = np.ma.masked_array(values, mask=~mask_array)

            self.observed_data[var_name] = values

            if len(values.shape) != len(expected_shape):
                raise ValueError(
                    f"Incompatible dimensions for {var_name}. "
                    f"Expected {len(expected_shape)} dims, got {len(values.shape)}"
                )

            if coords is not None:
                self._validate_coords(var_name, coords)

    def log_likelihood__i(
        self,
        var_name: str,
        idx: int,
        refitted_idata: InferenceData,
    ) -> xr.DataArray:
        """Compute pointwise log likelihood for a single held-out observation.

        Handles multidimensional observations and coordinate systems by properly
        managing dimension mappings and coordinate selections when computing
        log likelihoods for held-out data.

        Parameters
        ----------
        var_name : str
            Name of the variable to compute log likelihood for
        idx : int
            Index of the observation to compute log likelihood for
        refitted_idata : InferenceData
            InferenceData object from a model refit without the observation

        Returns
        -------
        xr.DataArray
            Log likelihood values for the held-out observation with dimensions (chain, draw)
        """
        holdout_data, holdout_coords = self.select_observations(
            var_name, np.array([idx])
        )
        # dims = self.get_dims(var_name)
        original_data = self.observed_data[var_name].copy()

        try:
            # Set just the held-out observation as the observed data
            self.set_data({var_name: holdout_data}, coords=holdout_coords)

            log_like = pm.compute_log_likelihood(
                refitted_idata,
                var_names=[var_name],
                model=self._untransformed_model,
                extend_inferencedata=False,
            )

            log_like_i = log_like[var_name]

            # Get all observation-related dimensions (excluding chain and draw)
            obs_dims = [dim for dim in log_like_i.dims if dim not in ("chain", "draw")]

            # For each observation dimension, select the first (and only) element
            # since we only provided one observation
            for dim in obs_dims:
                log_like_i = log_like_i.isel({dim: 0})

            return log_like_i

        finally:
            self.set_data({var_name: original_data})

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
        """Validate that the model is properly fitted and ready for use.

        Raises
        ------
        ValueError
            If the model state is invalid or inconsistent
        """
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
        """Validate coordinate values against variable dimensions.

        Parameters
        ----------
        var_name : str
            Name of the variable
        coords : Dict[str, Sequence]
            Coordinate values for each dimension

        Raises
        ------
        ValueError
            If coordinates are invalid or incompatible with the variable
        """
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
