"""Wrapper for fitted PyMC models to support LOO-CV computations."""

import copy
import logging
import warnings
from typing import Any, Callable, Sequence

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from arviz import InferenceData
from pymc.model import Model
from pymc.model.transform.conditioning import remove_value_transforms

from ..utils import wrap_xarray_ufunc

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

        try:
            self._untransformed_model = remove_value_transforms(copy.deepcopy(model))
        except KeyError as e:
            logger.warning(
                "KeyError during model cloning: %s. Using original model.", str(e)
            )
            self._untransformed_model = model
        except Exception as e:
            logger.warning("Failed to clone model: %s. Using original model.", str(e))
            self._untransformed_model = model

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

            values = values.copy()

            if mask is not None and var_name in mask:
                mask_array = mask[var_name]
                if mask_array.shape != values.shape:
                    raise PyMCWrapperError(
                        f"Mask shape {mask_array.shape} does not match data shape "
                        f"{values.shape} for variable {var_name}"
                    )
                values = np.ma.masked_array(values, mask=~mask_array)

            orig_dims = self.get_dims()
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
            if isinstance(values, np.ndarray):
                values.flags.writeable = False

    def log_likelihood_i(
        self,
        idx: int | np.ndarray | slice,
        idata: InferenceData,
        var_name: str | None = None,
    ) -> xr.DataArray:
        r"""Compute pointwise log likelihood for one or more observations using a fitted model.

        This method computes the log likelihood for one or more observations intended to be used
        in cross-validation procedures like leave-one-out cross-validation (LOO-CV) or k-fold CV.
        It uses a fitted model (i.e. one that has been refitted without the observation(s) at the given
        index/indices) to evaluate the likelihood of the observation(s).

        Parameters
        ----------
        idx : int | np.ndarray | slice
            Index or indices of the held-out observation(s).
        idata : InferenceData
            InferenceData object from a model that was refitted without the observation(s) at index `idx`.
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

        Raises
        ------
        PyMCWrapperError
            If the variable is not found, observed data is missing, the variable contains missing values,
            or if the log likelihood computation fails.
        IndexError
            If `idx` is out of bounds for the observed data when providing a single index.
            For slices or arrays, out-of-bounds indices will be clipped with a warning.

        Examples
        --------
        First, let's import the necessary libraries and create a synthetic dataset for a simple linear regression model:

        .. code-block:: python

            import pymc as pm
            import arviz as az
            import numpy as np
            from pyloo.wrapper import PyMCWrapper

            # Generate some example data
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

        if np.any(self.get_missing_mask(var_name)):
            raise PyMCWrapperError(f"Missing values found in {var_name}")

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

        indices, single_idx = self._process_and_validate_indices(idx, n_obs)

        try:
            original_data = self.observed_data[var_name].copy()
            holdout_data, _ = self.select_observations(indices, var_name=var_name)

            if hasattr(self, "observed_dims") and var_name in self.observed_dims:
                orig_coords = self._get_coords(var_name)

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
            log_like_i = self._format_log_likelihood_result(
                log_like_i, indices, single_idx, idx, holdout_data, n_obs, var_name
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

        Raises
        ------
        PyMCWrapperError
            If sampling parameters are invalid or if the sampling process fails.

        Notes
        -----
        The method first checks that your sampling parameters are valid, like making sure you have a
        positive number of draws and chains. It automatically enables log likelihood computation if you
        haven't already specified it, since this is necessary for LOO-CV. Also, if any of your model's
        free random variables are missing transforms, the method adds an identity transform to prevent
        sampling errors.
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
            for var in self.model.free_RVs:
                if var not in self.model.rvs_to_transforms:
                    logger.warning(
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
                    logger.warning(
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

            if transform is None:
                # No transformation needed
                logger.info(
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
                    logger.warning(
                        "Failed to transform %s to constrained space: %s. Using"
                        " original values.",
                        var_name,
                        str(e),
                    )
                    constrained = unconstrained.copy()

            constrained_params[var_name] = constrained

        return constrained_params

    def get_log_likelihood(
        self,
        var_name: str | None = None,
        draws: dict[str, np.ndarray] | None = None,
    ) -> xr.DataArray:
        """Compute the log likelihood for a set of posterior draws.

        This method calculates the pointwise log likelihood for an observed variable
        in the model. It can either use the current parameter values in the model or
        a specific set of parameter draws.

        Parameters
        ----------
        var_name : str | None, default=None
            Name of the observed variable for which to compute the log likelihood.
            If None, the method will use the first observed variable in the model,
            as determined by `get_observed_name()`.

        draws : dict[str, np.ndarray] | None, default=None
            Optional dictionary mapping parameter names to parameter values.
            If provided, the method will temporarily set these parameter values
            in the model before computing the log likelihood. The original values
            will be restored after computation.

        Returns
        -------
        xr.DataArray
            An xarray DataArray containing the pointwise log likelihood values.
            The dimensions and coordinates of the DataArray will match those of
            the observed variable, with the name format "log_likelihood_{var_name}".

        Raises
        ------
        PyMCWrapperError
            If the variable is not found in the model, if no observed data exists
            for the variable, if the variable contains missing values, or if the
            log likelihood computation fails for any other reason.
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

        if np.any(self.get_missing_mask(var_name)):
            raise PyMCWrapperError(f"Missing values found in {var_name}")

        try:
            # If draws are provided, temporarily set the model's parameters
            original_values = {}
            if draws is not None:
                for param_name, param_draws in draws.items():
                    if param_name in self.model.named_vars:
                        var = self.model.named_vars[param_name]
                        if hasattr(var, "get_value"):
                            original_values[param_name] = var.get_value()
                            var.set_value(param_draws)
                        else:
                            logger.warning(
                                f"Could not set value for parameter {param_name}"
                            )

            observed_rv = None
            for rv in self.model.observed_RVs:
                if rv.name == var_name:
                    observed_rv = rv
                    break

            if observed_rv is None:
                raise PyMCWrapperError(f"Could not find observed RV for {var_name}")

            # Compute the pointwise log likelihood
            elemwise_logp = self.model.logp([observed_rv], sum=False)

            if hasattr(self, "observed_dims") and var_name in self.observed_dims:
                dims = self.observed_dims[var_name]
                coords = self._get_coords(var_name)
                if coords is None:
                    coords = {}
                    for i, dim in enumerate(dims):
                        if dim is not None:
                            shape = self.get_shape(var_name)
                            if shape is not None:
                                coords[dim] = np.arange(shape[i])

                log_like = xr.DataArray(
                    elemwise_logp,
                    dims=dims,
                    coords=coords,
                    name=f"log_likelihood_{var_name}",
                )
            else:
                shape = self.get_shape(var_name)
                if shape is None:
                    raise PyMCWrapperError(f"Could not determine shape for {var_name}")

                dims = [f"dim_{i}" for i in range(len(shape))]
                coords = {dim: np.arange(size) for dim, size in zip(dims, shape)}

                log_like = xr.DataArray(
                    elemwise_logp,
                    dims=dims,
                    coords=coords,
                    name=f"log_likelihood_{var_name}",
                )

            return log_like

        except Exception as e:
            if isinstance(e, PyMCWrapperError):
                raise
            raise PyMCWrapperError(f"Failed to compute log likelihood: {str(e)}")
        finally:
            # Restore original parameter values if they were changed
            if draws is not None:
                for param_name, original_value in original_values.items():
                    var = self.model.named_vars[param_name]
                    if hasattr(var, "set_value"):
                        var.set_value(original_value)

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

    def get_dims(self) -> tuple[str, ...] | None:
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
        if hasattr(self.idata, "observed_data"):
            dims = self.idata.observed_data[self.get_observed_name()].dims
            if dims:
                return tuple(dims)
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
        self, idx: int | np.ndarray | slice, n_obs: int
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
        self.unobserved_value_vars = [
            rv.name for rv in self.model.unobserved_value_vars
        ]
        self.basic_RVs = [rv.name for rv in self.model.basic_RVs]
        self.unobserved_vars = [rv.name for rv in self.model.unobserved_RVs]
        self.continuous_value_vars = [
            rv.name for rv in self.model.continuous_value_vars
        ]
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
