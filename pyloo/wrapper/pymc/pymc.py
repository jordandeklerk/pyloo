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
from pymc.variational.approximations import Approximation

from .utils import (
    PyMCWrapperError,
    _create_selection_mask,
    _extract_model_components,
    _format_log_likelihood_result,
    _get_coords,
    _process_and_validate_indices,
    _validate_model_state,
)

__all__ = ["PyMCWrapper"]

_log = logging.getLogger(__name__)


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
    observed_dims : dict[str, tuple[str, ...]]
        Mapping of observed variable names to their dimension names
    """

    model: Model
    idata: InferenceData
    var_names: list[str] | None
    observed_data: dict[str, np.ndarray]
    constant_data: dict[str, np.ndarray]
    observed_dims: dict[str, tuple[str, ...]]
    _untransformed_model: Model
    approximation: Approximation | None

    def __init__(
        self,
        model: Model,
        idata: InferenceData,
        var_names: Sequence[str] | None = None,
        approximation: Approximation | None = None,
    ):
        """Initialize a PyMCWrapper.

        Parameters
        ----------
        model : Model
            A fitted PyMC model containing the model structure and relationships
        idata : InferenceData
            ArviZ InferenceData object containing the model's posterior samples
        var_names : Sequence[str] | None
            Names of specific variables to focus on. If None, all variables are included
        approximation : Approximation | None, optional
            A PyMC variational approximation object (e.g., from `pm.fit()`). If provided,
            it can be used for approximate posterior calculations.
        """
        self.model = model
        self.idata = idata
        self.var_names = list(var_names) if var_names is not None else None
        self.observed_data = {}
        self.constant_data = {}
        self.observed_dims = {}
        self.approximation = approximation

        try:
            self._untransformed_model = remove_value_transforms(copy.deepcopy(model))
        except Exception as e:
            _log.warning(
                "Failed during model cloning and transform removal: %s. Using original"
                " model. This might affect parameter transformations if value"
                " transforms were present.",
                str(e),
            )
            self._untransformed_model = model

        _validate_model_state(self)
        _extract_model_components(self)

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

        Partitions the data into two sets: selected observations and remaining
        observations. If no variable name is provided, uses the first observed
        variable in the model.

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
        """
        if not self.observed_data:
            raise PyMCWrapperError("No observed variables found in the model")

        if var_name is None:
            var_name = self.get_observed_name()
        else:
            self._validate_observed_var(var_name)

        data = self.observed_data[var_name]
        if axis is None:
            axis = 0

        if np.any(self.get_missing_mask(var_name)):
            _log.warning(
                "Missing values detected in %s. This may affect the results.", var_name
            )

        try:
            mask = _create_selection_mask(indices, data.shape[axis], axis)
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

            values = values.copy()

            if mask is not None and var_name in mask:
                mask_array = mask[var_name]
                if mask_array.shape != values.shape:
                    raise PyMCWrapperError(
                        f"Mask shape {mask_array.shape} does not match data shape "
                        f"{values.shape} for variable {var_name}"
                    )
                values = np.ma.masked_array(values, mask=~mask_array)

            self._coords_update(var_name, values, coords, update_coords)

            self.observed_data[var_name] = values
            # Make data immutable after potential updates
            if isinstance(values, np.ndarray):
                values.flags.writeable = False

    def log_likelihood_i(
        self,
        idx: int | np.ndarray | slice,
        idata: InferenceData,
        var_name: str | None = None,
    ) -> xr.DataArray:
        r"""Compute pointwise log likelihood for one or more observations using a fitted model.

        Parameters
        ----------
        idx : int | np.ndarray | slice
            Index or indices of the held-out observation(s).
        idata : InferenceData
            InferenceData object containing posterior samples from a model that was refitted
            without the observation(s) specified by `idx`.
        var_name : str | None
            Name of the variable for which to compute the log likelihood.
            If None, uses the first observed variable.

        Returns
        -------
        xr.DataArray
            Log likelihood values for the held-out observation(s).

            For a single index (int), returns a DataArray with dimensions (chain, draw) and
            with an attribute 'observation_index' containing the index. The coordinate for the
            observation dimension is preserved but set to the index value.

            For multiple indices (array, slice), returns a DataArray with dimensions (chain, draw, obs_idx)
            where obs_idx is the dimension for the observations, with an attribute 'observation_indices'
            containing the list of indices.

        Examples
        --------
        Create a synthetic dataset for a simple linear regression model:

        .. code-block:: python

            import pymc as pm
            import arviz as az
            import numpy as np
            from pyloo.wrapper import PyMCWrapper

            x = np.random.normal(0, 1, size=100)
            true_alpha = 1.0
            true_beta = 2.5
            true_sigma = 1.0
            y = true_alpha + true_beta * x + np.random.normal(0, true_sigma, size=100)

        Now, let's define a PyMC model for the linear regression and sample from the posterior:

        .. code-block:: python

            with pm.Model() as model:
                alpha = pm.Normal("alpha", mu=0, sigma=10)
                beta = pm.Normal("beta", mu=0, sigma=10)
                sigma = pm.HalfNormal("sigma", sigma=10)
                mu = alpha + beta * x
                obs = pm.Normal("y", mu=mu, sigma=sigma, observed=y)
                idata = pm.sample(1000, chains=2)

        Finally, we can create a PyMCWrapper instance and compute the log-likelihood for a single held-out observation:

        .. code-block:: python

            wrapper = PyMCWrapper(model, idata)
            idata = wrapper.sample_posterior(draws=1000, tune=1000, chains=2, target_accept=0.9)

            # Single observation
            log_like_i = wrapper.log_likelihood_i(10, idata)

            # Multiple observations
            indices = np.array([10, 20, 30])
            log_like_multiple = wrapper.log_likelihood_i(indices, idata)

            # Using a slice
            log_like_slice = wrapper.log_likelihood_i(slice(10, 15), idata)
        """
        if var_name is None:
            var_name = self.get_observed_name()

        if self.get_variable(var_name) is None:
            raise PyMCWrapperError(
                f"Variable '{var_name}' not found in model. Available variables: "
                f"{list(self.model.named_vars.keys())}"
            )

        if var_name not in self.observed_data:
            raise PyMCWrapperError(
                f"No observed data found for variable '{var_name}'. "
                f"Available observed variables: {list(self.observed_data.keys())}"
            )

        self._validate_observed_var(var_name)
        self._check_missing_values(var_name)

        if not hasattr(idata, "posterior"):
            raise PyMCWrapperError(
                "refitted_idata must contain posterior samples. "
                "Check that the model was properly refit."
            )

        data_shape = self.get_shape(var_name)
        if data_shape is None:
            raise PyMCWrapperError(f"Could not determine shape for variable {var_name}")

        n_obs = data_shape[0]
        original_data = None
        orig_coords = None

        indices, single_idx = _process_and_validate_indices(idx, n_obs)

        try:
            original_data = self.observed_data[var_name].copy()
            holdout_data, _ = self.select_observations(indices, var_name=var_name)

            if hasattr(self, "observed_dims") and var_name in self.observed_dims:
                orig_coords = _get_coords(self, var_name)

            self.observed_data[var_name] = holdout_data

            log_like = pm.compute_log_likelihood(
                idata,
                var_names=[var_name],
                model=self.model,
                extend_inferencedata=False,
            )

            if var_name not in log_like:
                raise PyMCWrapperError(
                    f"Failed to compute log likelihood for variable {var_name}. "
                    "Check that the model specification matches the data."
                )

            log_like_i = log_like[var_name]
            log_like_i = _format_log_likelihood_result(
                log_like_i,
                indices,
                single_idx,
                idx,
                holdout_data,
                n_obs,
                var_name,
            )

            return log_like_i

        except Exception as e:
            if isinstance(e, (IndexError, PyMCWrapperError)):
                raise
            raise PyMCWrapperError(f"Failed to compute log likelihood: {str(e)}")
        finally:
            if original_data is not None:
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
        r"""Sample from the model's posterior distribution.

        Parameters
        ----------
        draws : int
            Number of posterior samples to draw.
        tune : int
            Number of tuning steps.
        chains : int
            Number of chains to sample.
        target_accept : float
            Target acceptance rate for the sampler.
        random_seed : int | None
            Random seed for reproducibility.
        progressbar : bool
            Whether to display a progress bar.
        **kwargs : Any
            Additional keyword arguments to pass to `pm.sample()`.

        Returns
        -------
        InferenceData
            ArviZ InferenceData object containing the posterior samples and log likelihood values.
        """
        if draws <= 0:
            raise PyMCWrapperError(f"Number of draws must be positive, got {draws}")
        if chains <= 0:
            raise PyMCWrapperError(f"Number of chains must be positive, got {chains}")

        idata_kwargs = kwargs.get("idata_kwargs", {})
        if isinstance(idata_kwargs, dict):
            if not idata_kwargs.get("log_likelihood", False):
                _log.info(
                    "Automatically enabling log likelihood computation as it is "
                    "required for LOO-CV."
                )
                idata_kwargs["log_likelihood"] = True
                kwargs["idata_kwargs"] = idata_kwargs
        else:
            kwargs["idata_kwargs"] = {"log_likelihood": True}

        try:
            for var in self.model.free_RVs:
                if var not in self.model.rvs_to_transforms:
                    _log.warning(
                        "Variable %s missing from rvs_to_transforms, adding identity"
                        " transform.",
                        var.name,
                    )
                    self.model.rvs_to_transforms[var] = None

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

    def get_unconstrained_parameters(self) -> dict[str, xr.DataArray]:
        r"""Convert posterior samples from the constrained to the unconstrained space.

        This method transforms each free parameter's posterior samples from its native, constrained
        domain to an unconstrained space where optimization and other operations can be performed
        more effectively.

        When transforming random variables, we must account for the change in probability density
        using the Jacobian adjustment. For a transformation :math:`\theta' = g(\theta)`, the probability
        densities are related by:

        .. math::
            p(\theta') = p(\theta) \left| \frac{d}{d\theta'} g^{-1}(\theta') \right|

        The mathematical transformations applied depend on the parameter's domain constraints.
        For example, for positive variables (e.g., HalfNormal, Gamma), a logarithmic transformation is applied:

        .. math::
            \theta' = g(\theta) = \log(\theta), \quad \theta \in (0, \infty) \mapsto
            \theta' \in (-\infty, \infty)

        With Jacobian determinant:

        .. math::
            \left| \frac{d}{d\theta'} g^{-1}(\theta') \right| =
            \left| \frac{d}{d\theta'} \exp(\theta') \right| = \exp(\theta').

        Returns
        -------
        dict[str, xr.DataArray]
            A dictionary mapping parameter names to their posterior samples in the unconstrained space.
            Each array retains its original dimensions (chain, draw, and parameter-specific dimensions).

        Notes
        -----
        The method applies the `backward` transform from each parameter's transform object in the
        model's `rvs_to_transforms` dictionary. If a transform is unavailable or fails, the original
        constrained samples are returned instead. Jacobian adjustments are automatically handled
        by PyMC's transform objects.

        Examples
        --------
        Let's first import the necessary packages and create a simple linear regression dataset:

        .. code-block:: python

            import pymc as pm
            import arviz as az
            import numpy as np
            from pyloo.wrapper import PyMCWrapper

            x = np.random.normal(0, 1, size=100)
            true_alpha = 1.0
            true_beta = 2.5
            true_sigma = 1.0
            y = true_alpha + true_beta * x + np.random.normal(0, true_sigma, size=100)

        Now, let's create a simple Bayesian linear regression model with PyMC and sample from its posterior:

        .. code-block:: python

            with pm.Model() as model:
                alpha = pm.Normal("alpha", mu=0, sigma=10)
                beta = pm.Normal("beta", mu=0, sigma=10)
                sigma = pm.HalfNormal("sigma", sigma=10)
                mu = alpha + beta * x
                obs = pm.Normal("y", mu=mu, sigma=sigma, observed=y)
                idata = pm.sample(1000, chains=2)

        Finally, we can use the PyMCWrapper to transform our parameters to the unconstrained space:

        .. code-block:: python

            wrapper = PyMCWrapper(model, idata)
            unconstrained_params = wrapper.get_unconstrained_parameters()
        """
        unconstrained_params = {}

        for var in self.model.free_RVs:
            var_name = var.name
            if var_name not in self.idata.posterior:
                continue

            transform = self.model.rvs_to_transforms.get(var, None)

            if transform is None:
                # No transformation needed
                unconstrained_samples = self.idata.posterior[var_name].copy()
            else:
                try:
                    # Apply transformation
                    data = self.idata.posterior[var_name].values
                    transformed_data = self._transform_to_unconstrained(data, transform)

                    unconstrained_samples = xr.DataArray(
                        transformed_data,
                        dims=self.idata.posterior[var_name].dims,
                        coords=self.idata.posterior[var_name].coords,
                        name=var_name,
                    )
                except Exception as e:
                    _log.warning(
                        "Failed to transform %s: %s. Using original values.",
                        var_name,
                        str(e),
                    )
                    unconstrained_samples = self.idata.posterior[var_name].copy()

            unconstrained_params[var_name] = unconstrained_samples

        return unconstrained_params

    def constrain_parameters(
        self, unconstrained_params: dict[str, xr.DataArray]
    ) -> dict[str, xr.DataArray]:
        r"""Convert parameters from the unconstrained back to the constrained space.

        This method transforms parameter values from an unconstrained representation back to
        their original constrained domain as specified by their prior distributions.

        When transforming random variables from unconstrained to constrained space, we apply the
        inverse transformation. For a transformation :math:`\theta' = g(\theta)`, we apply
        :math:`\theta = g^{-1}(\theta')`. The probability densities are related by:

        .. math::
            p(\theta) = p(\theta') \left| \frac{d}{d\theta} g(\theta) \right|^{-1} =
            p(\theta') \left| \frac{1}{\frac{d}{d\theta} g(\theta)} \right|

        The mathematical inverse transformations applied depend on the parameter's original constraints.
        For example, for positive variables, the inverse of the logarithmic transform is applied:

        .. math::
            \theta = g^{-1}(\theta') = \exp(\theta'), \quad \theta' \in (-\infty, \infty) \mapsto
            \theta \in (0, \infty)

        With Jacobian determinant:

        .. math::
            \left| \frac{d}{d\theta'} g^{-1}(\theta') \right| =
            \left| \frac{d}{d\theta'} \exp(\theta') \right| = \exp(\theta').

        Parameters
        ----------
        unconstrained_params : dict[str, xr.DataArray]
            A dictionary mapping parameter names to their values in the unconstrained space,
            with dimensions (chain, draw) and any additional parameter-specific dimensions.

        Returns
        -------
        dict[str, xr.DataArray]
            A dictionary mapping parameter names to their values transformed back to the constrained
            space, preserving the original dimensions.

        Notes
        -----
        The method applies the `forward` transform from each parameter's transform object in the
        model's `rvs_to_transforms` dictionary. If a transform is unavailable or fails, the original
        unconstrained values are returned unchanged. Jacobian adjustments are automatically handled
        by PyMC's transform objects. This operation is the inverse of `get_unconstrained_parameters`.

        Examples
        --------
        Let's begin by importing necessary packages and creating a linear regression dataset:

        .. code-block:: python

            import pymc as pm
            import arviz as az
            import numpy as np
            from pyloo.wrapper import PyMCWrapper

            x = np.random.normal(0, 1, size=100)
            true_alpha = 1.0
            true_beta = 2.5
            true_sigma = 1.0
            y = true_alpha + true_beta * x + np.random.normal(0, true_sigma, size=100)

        Next, we'll create a simple linear regression model and sample from its posterior:

        .. code-block:: python

            with pm.Model() as model:
                alpha = pm.Normal("alpha", mu=0, sigma=10)
                beta = pm.Normal("beta", mu=0, sigma=10)
                sigma = pm.HalfNormal("sigma", sigma=10)
                mu = alpha + beta * x
                obs = pm.Normal("y", mu=mu, sigma=sigma, observed=y)
                idata = pm.sample(1000, chains=2)

        Let's create a PyMCWrapper instance and get our parameters in the unconstrained space:

        .. code-block:: python

            wrapper = PyMCWrapper(model, idata)
            unconstrained_params = wrapper.get_unconstrained_parameters()

        Now we can modify the unconstrained parameters and then transform them back to the constrained space:

        .. code-block:: python

            modified_unconstrained = {name: param + 0.1 for name, param in unconstrained_params.items()}
            constrained_params = wrapper.constrain_parameters(modified_unconstrained)
        """
        constrained_params = {}

        for var in self.model.free_RVs:
            var_name = var.name
            if var_name not in unconstrained_params:
                continue

            unconstrained = unconstrained_params[var_name]
            transform = self.model.rvs_to_transforms.get(var, None)

            if not isinstance(unconstrained, xr.DataArray):
                raise TypeError(
                    f"Parameter value for {var_name} must be an xarray.DataArray, "
                    f"got {type(unconstrained).__name__}"
                )

            if transform is None:
                # No transformation needed
                _log.info(
                    "No transform found for variable %s. Using original values.",
                    var_name,
                )
                constrained = unconstrained.copy()
            else:
                try:
                    # Apply transformation
                    data = unconstrained.values
                    transformed_data = self._transform_to_constrained(data, transform)

                    constrained = xr.DataArray(
                        transformed_data,
                        dims=unconstrained.dims,
                        coords=unconstrained.coords,
                        name=var_name,
                    )
                except Exception as e:
                    _log.warning(
                        "Failed to transform %s to constrained space: %s. Using"
                        " original values.",
                        var_name,
                        str(e),
                    )
                    constrained = unconstrained.copy()

            constrained_params[var_name] = constrained

        return constrained_params

    def get_draws(self) -> np.ndarray:
        """Get all draws from the posterior.

        Returns
        -------
        np.ndarray
            All draws from the posterior
        """
        if not hasattr(self.idata, "posterior"):
            raise PyMCWrapperError(
                "InferenceData object must contain posterior samples. "
                "The model does not appear to be fitted."
            )

        draws = self.idata.posterior
        return draws

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
        """
        self._validate_observed_var(var_name)

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

    def get_dims(self, var_name: str = None) -> tuple[str, ...] | None:
        """Get the dimension names for a variable.

        Parameters
        ----------
        var_name : str, optional
            Name of the variable. If None, returns dimensions for the first observed variable.

        Returns
        -------
        tuple[str, ...] | None
            Tuple of dimension names or None if variable not found
        """
        if var_name is None:
            try:
                var_name = self.get_observed_name()
            except PyMCWrapperError:
                return None

        if (
            hasattr(self.idata, "observed_data")
            and var_name in self.idata.observed_data
        ):
            dims = self.idata.observed_data[var_name].dims
            if dims:
                return tuple(dims)

        if hasattr(self.idata, "posterior") and var_name in self.idata.posterior:
            dims = self.idata.posterior[var_name].dims
            param_dims = tuple(d for d in dims if d not in ("chain", "draw"))
            if param_dims:
                return param_dims

        if hasattr(self.model, "named_vars_to_dims") and self.model.named_vars_to_dims:
            if var_name in self.model.named_vars_to_dims:
                return tuple(self.model.named_vars_to_dims[var_name])

        shape = self.get_shape(var_name)
        if shape is not None and len(shape) > 0:
            return tuple(f"{var_name}_dim_{i}" for i in range(len(shape)))

        return None

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
        """
        if var_name in self.observed_data:
            return tuple(self.observed_data[var_name].shape)
        elif var_name in self._untransformed_model.named_vars:
            var = self._untransformed_model.named_vars[var_name]
            return tuple(d.eval() if hasattr(d, "eval") else d for d in var.shape)
        return None

    def _transform_to_unconstrained(self, values, transform):
        """Transform values from constrained to unconstrained space."""
        try:
            result = transform.backward(values)
            if hasattr(result, "eval"):
                result = result.eval()
            return np.asarray(result)
        except Exception as e:
            _log.warning("Backward transform failed: %s", str(e))
            raise

    def _transform_to_constrained(self, values, transform):
        """Transform values from unconstrained to constrained space."""
        try:
            result = transform.forward(values)
            if hasattr(result, "eval"):
                result = result.eval()
            return np.asarray(result)
        except Exception as e:
            _log.warning("Forward transform failed: %s", str(e))
            raise

    def _coords_update(
        self,
        var_name: str,
        values: np.ndarray,
        coords: dict[str, Sequence] | None,
        update_coords: bool,
    ) -> None:
        """Modifies 'working_coords' but doesn't directly set self.observed_data or
        self.observed_dims. The caller (`set_data`) is responsible for setting
        self.observed_data. Coordinate info is primarily managed via InferenceData
        or model attributes elsewhere.
        """
        orig_dims = self.get_dims(var_name)
        if orig_dims is None:
            return

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
                            f"Automatically created coordinates for dimension {dim}",
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
                            f"Coordinate length for dimension {dim} changed from"
                            f" {original_len} to {size}",
                            UserWarning,
                            stacklevel=2,
                        )
                    else:
                        raise ValueError(
                            f"Coordinate length {len(working_coords[dim])} for"
                            f" dimension {dim} does not match variable shape {size}"
                            f" for {var_name}"
                        )

    def _validate_observed_var(self, var_name: str) -> None:
        """Check if the variable name exists in observed_data."""
        if var_name not in self.observed_data:
            raise PyMCWrapperError(
                f"Variable '{var_name}' not found in observed data. "
                f"Available variables: {list(self.observed_data.keys())}"
            )

    def _check_missing_values(self, var_name: str) -> None:
        """Check if the observed variable contains missing values."""
        if np.any(self.get_missing_mask(var_name)):
            raise PyMCWrapperError(f"Missing values found in {var_name}")
