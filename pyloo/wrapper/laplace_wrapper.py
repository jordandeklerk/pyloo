"""Laplace Variational Inference wrapper for PyMC models."""

import warnings
from dataclasses import dataclass, field
from typing import Literal

import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from arviz import InferenceData
from better_optimize.constants import minimize_method
from pymc.blocking import RaveledVars
from scipy import stats

try:
    from pymc_extras.inference.find_map import find_MAP
    from pymc_extras.inference.laplace import (
        add_fit_to_inferencedata,
        fit_mvn_at_MAP,
        sample_laplace_posterior,
    )

    PYMC_EXTRAS_AVAILABLE = True
except ImportError:
    PYMC_EXTRAS_AVAILABLE = False
    warnings.warn(
        "pymc-extras is not installed. The LaplaceWrapper requires pymc-extras for "
        "Laplace approximation. "
        "Install it with: pip install pymc-extras",
        UserWarning,
        stacklevel=2,
    )


class LaplaceWrapperError(Exception):
    """Exception raised for errors in the LaplaceWrapper."""

    pass


@dataclass
class LaplaceVIResult:
    """Container for Laplace Variational Inference results."""

    idata: InferenceData
    mu: RaveledVars
    H_inv: np.ndarray
    model: pm.Model
    importance_sampled: bool = False
    importance_sampling_method: str | None = None
    pareto_k: float | None = None
    warnings: list[str] = field(default_factory=list)
    log_weights: np.ndarray | None = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


# Mainly based on https://github.com/pymc-devs/pymc-extras/blob/main/pymc_extras/inference/laplace.py
# with some modifications
class LaplaceWrapper:
    """Laplace Variational Inference wrapper for PyMC models.

    This class provides Laplace approximation functionality for PyMC models,
    including importance resampling to improve the quality of the approximation.

    Parameters
    ----------
    model : pm.Model
        PyMC model to perform inference on
    idata : InferenceData, optional
        ArviZ InferenceData object containing the model's posterior samples.
        If None, the model will be fit using Laplace approximation.
    var_names : Sequence[str], optional
        Names of specific variables to focus on. If None, all variables are included.
    random_seed : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    model : pm.Model
        The underlying PyMC model
    idata : InferenceData
        ArviZ InferenceData object containing the model's posterior samples
    var_names : list[str]
        Names of variables being tracked
    result : LaplaceVIResult
        Container with the results of the Laplace approximation
    """

    def __init__(
        self,
        model: pm.Model,
        idata: InferenceData | None = None,
        var_names: list[str] | None = None,
        random_seed: int | None = None,
    ):
        """Initialize a LaplaceWrapper.

        Parameters
        ----------
        model : pm.Model
            PyMC model to perform inference on
        idata : InferenceData
            ArviZ InferenceData object containing the model's posterior samples.
        var_names : Sequence[str], optional
            Names of specific variables to focus on. If None, all variables are included.
        random_seed : int, optional
            Random seed for reproducibility.

        Notes
        -----
        The Laplace approximation assumes that the posterior distribution is approximately
        Gaussian. This assumption may not hold for posteriors with significant skewness,
        multimodality, or heavy tails. In such cases, the approximation may be poor and
        lead to unreliable inference.

        Currently, this wrapper only implements Laplace approximation. Future versions
        will include additional approximation methods for more complex posteriors.
        """
        self.model = model
        self.idata = idata
        self.var_names = list(var_names) if var_names is not None else None
        self.random_seed = random_seed
        self.result: LaplaceVIResult

    def fit(
        self,
        optimize_method: minimize_method = "BFGS",
        use_grad: bool | None = None,
        use_hessp: bool | None = None,
        use_hess: bool | None = None,
        initvals: dict | None = None,
        jitter_rvs: list[pt.TensorVariable] | None = None,
        progressbar: bool = True,
        include_transformed: bool = True,
        gradient_backend: Literal["pytensor", "jax"] = "pytensor",
        chains: int = 2,
        draws: int = 500,
        on_bad_cov: Literal["warn", "error", "ignore", "regularize"] = "regularize",
        fit_in_unconstrained_space: bool = False,
        zero_tol: float = 1e-8,
        diag_jitter: float | None = 1e-8,
        optimizer_kwargs: dict | None = None,
        compile_kwargs: dict | None = None,
        regularization_min_eigval: float = 1e-8,
        regularization_max_attempts: int = 10,
        compute_log_likelihood: bool = True,
    ) -> LaplaceVIResult:
        """Fit the model using Laplace approximation.

        Parameters
        ----------
        optimize_method : str
            The optimization method to use for finding MAP. See scipy.optimize.minimize documentation.
        use_grad : Optional[bool]
            Whether to use gradients in the optimization.
        use_hessp : Optional[bool]
            Whether to use Hessian-vector products in the optimization.
        use_hess : Optional[bool]
            Whether to use the Hessian matrix in the optimization.
        initvals : Optional[Dict]
            Initial values for the model parameters.
        jitter_rvs : Optional[List[pt.TensorVariable]]
            Variables whose initial values should be jittered.
        progressbar : bool
            Whether to display a progress bar during optimization.
        include_transformed : bool
            Whether to include transformed variable values in the returned dictionary.
        gradient_backend : str
            The backend to use for gradient computations. Must be one of "pytensor" or "jax".
        chains : int
            The number of chain dimensions to sample.
        draws : int
            The number of samples to draw from the approximated posterior.
        on_bad_cov : str
            What to do when the inverse Hessian is not positive semi-definite.
            "regularize" (default): Attempt to regularize the matrix to make it positive definite.
            "warn": Issue a warning and continue.
            "error": Raise an error.
            "ignore": Proceed without taking any action.
        fit_in_unconstrained_space : bool
            Whether to fit the Laplace approximation in the unconstrained parameter space.
        zero_tol : float
            Value below which an element of the Hessian matrix is counted as 0.
        diag_jitter : Optional[float]
            A small value added to the diagonal of the inverse Hessian matrix.
        optimizer_kwargs : Optional[Dict]
            Additional keyword arguments to pass to scipy.minimize.
        compile_kwargs : Optional[Dict]
            Additional keyword arguments to pass to pytensor.function.
        regularization_min_eigval : float
            Minimum eigenvalue threshold for matrix regularization.
        regularization_max_attempts : int
            Maximum number of regularization attempts.

        Returns
        -------
        LaplaceVIResult
            Container with the results of the Laplace approximation.
        """
        if not PYMC_EXTRAS_AVAILABLE:
            raise ImportError(
                "pymc-extras is required for Laplace approximation. "
                "Install it with: pip install pymc-extras"
            )

        optimizer_kwargs = {} if optimizer_kwargs is None else optimizer_kwargs
        compile_kwargs = {} if compile_kwargs is None else compile_kwargs
        warnings_list = []

        with self.model:
            optimized_point = find_MAP(
                method=optimize_method,
                model=self.model,
                use_grad=use_grad,
                use_hessp=use_hessp,
                use_hess=use_hess,
                initvals=initvals,
                random_seed=self.random_seed,
                jitter_rvs=jitter_rvs,
                progressbar=progressbar,
                include_transformed=include_transformed,
                gradient_backend=gradient_backend,
                compile_kwargs=compile_kwargs,
                **optimizer_kwargs,
            )

            # Fit multivariate normal at MAP
            try:
                mu, H_inv = fit_mvn_at_MAP(
                    optimized_point=optimized_point,
                    model=self.model,
                    on_bad_cov=on_bad_cov if on_bad_cov != "regularize" else "ignore",
                    transform_samples=fit_in_unconstrained_space,
                    gradient_backend=gradient_backend,
                    zero_tol=zero_tol,
                    diag_jitter=diag_jitter,
                    compile_kwargs=compile_kwargs,
                )

                if on_bad_cov == "regularize":
                    try:
                        eigvals = np.linalg.eigvalsh(H_inv)
                        min_eigval = np.min(eigvals)

                        if min_eigval <= regularization_min_eigval:
                            H_inv_regularized, orig_min, final_min, attempts = (
                                _regularize_matrix(
                                    H_inv,
                                    min_eigenvalue=regularization_min_eigval,
                                    max_attempts=regularization_max_attempts,
                                )
                            )

                            warning_msg = (
                                "Inverse Hessian matrix required regularization. Min"
                                f" eigenvalue before: {orig_min}, after: {final_min},"
                                f" attempts: {attempts}"
                            )
                            warnings.warn(warning_msg, UserWarning, stacklevel=2)
                            warnings_list.append(warning_msg)

                            # Use the regularized matrix
                            H_inv = H_inv_regularized

                    except np.linalg.LinAlgError as e:
                        error_msg = (
                            "Matrix regularization failed during model fitting:"
                            f" {str(e)}"
                        )
                        warnings.warn(error_msg, UserWarning, stacklevel=2)
                        raise LaplaceWrapperError(error_msg) from e

                idata = sample_laplace_posterior(
                    mu=mu,
                    H_inv=H_inv,
                    model=self.model,
                    chains=chains,
                    draws=draws,
                    transform_samples=fit_in_unconstrained_space,
                    progressbar=progressbar,
                    random_seed=self.random_seed,
                    compile_kwargs=compile_kwargs,
                )

            except (np.linalg.LinAlgError, ValueError) as e:
                error_msg = f"Error during Laplace approximation: {str(e)}"
                warnings.warn(error_msg, UserWarning, stacklevel=2)
                raise LaplaceWrapperError(error_msg) from e

        result = LaplaceVIResult(
            idata=idata,
            mu=mu,
            H_inv=H_inv,
            model=self.model,
            warnings=warnings_list,
        )
        self.idata = idata
        self.result = result

        if compute_log_likelihood:
            pm.compute_log_likelihood(
                result.idata, model=self.model, extend_inferencedata=True
            )

        return result

    def _compute_log_prob_target(self) -> np.ndarray:
        """Compute log probability of samples under the target distribution (true posterior).

        Parameters
        ----------
        posterior : xr.Dataset
            Posterior samples from InferenceData

        Returns
        -------
        np.ndarray
            Log probability values with shape (n_chains, n_draws)
        """
        model = self.model
        posterior = self.result.idata.posterior

        n_chains = len(posterior.chain)
        n_draws = len(posterior.draw)

        logp_values = np.zeros((n_chains, n_draws))
        var_names = list(posterior.data_vars)

        for name in var_names:
            if name not in model.named_vars:
                continue

            var = model[name]
            value_var = model.rvs_to_values.get(var)

            if value_var is None:
                continue

            with model:
                try:
                    logp_fn = model.compile_fn(
                        model.logp(vars=[var], sum=True, jacobian=True)
                    )
                except Exception as e:
                    warnings.warn(f"Error compiling logp for {name}: {e}", stacklevel=2)
                    continue

            for c in range(n_chains):
                for d in range(n_draws):
                    try:
                        value = posterior[name].values[c, d]

                        if hasattr(value, "shape") and len(value.shape) > 0:
                            pass
                        # Add to the joint log probability
                        logp_values[c, d] += logp_fn({value_var.name: value})
                    except Exception as e:
                        # If there's an error, set to NaN and warn
                        logp_values[c, d] = np.nan
                        warnings.warn(
                            f"Error computing log probability for {name} at sample"
                            f" {c},{d}: {e}",
                            stacklevel=2,
                        )
                        break

        return logp_values

    def _compute_log_prob_proposal(self) -> np.ndarray:
        """Compute log probability of samples under the proposal distribution (Laplace approximation).

        Parameters
        ----------
        posterior : xr.Dataset
            Posterior samples from InferenceData
        mu : RaveledVars
            Mean vector of the multivariate normal distribution
        H_inv : np.ndarray
            Covariance matrix of the multivariate normal distribution

        Returns
        -------
        np.ndarray
            Log probability values with shape (n_chains, n_draws)
        """
        posterior = self.result.idata.posterior
        mu = self.result.mu
        H_inv = self.result.H_inv

        try:
            H_inv_regularized, orig_min, final_min, attempts = _regularize_matrix(
                H_inv, min_eigenvalue=1e-8, max_attempts=10
            )

            if attempts > 0:
                warning_msg = (
                    "Inverse Hessian matrix required regularization. "
                    f"Min eigenvalue before: {orig_min}, after: {final_min}, "
                    f"attempts: {attempts}"
                )
                warnings.warn(warning_msg, UserWarning, stacklevel=2)

                if hasattr(self, "result") and self.result is not None:
                    self.result.warnings.append(warning_msg)

        except np.linalg.LinAlgError as e:
            warning_msg = (
                f"Matrix regularization failed: {str(e)}. Using allow_singular=True as"
                " fallback."
            )
            warnings.warn(warning_msg, UserWarning, stacklevel=2)
            if hasattr(self, "result") and self.result is not None:
                self.result.warnings.append(warning_msg)
            H_inv_regularized = H_inv

        mvn = stats.multivariate_normal(
            mean=mu.data, cov=H_inv_regularized, allow_singular=True
        )

        n_chains = len(posterior.chain)
        n_draws = len(posterior.draw)
        logp_values = np.zeros((n_chains, n_draws))

        # Get the point map info from mu
        info = mu.point_map_info

        for c in range(n_chains):
            for d in range(n_draws):
                flattened = []
                for name, _, _, _ in info:
                    if name in posterior:
                        flat_sample = posterior[name].values[c, d].flatten()
                        flattened.append(flat_sample)

                x = np.concatenate(flattened)
                try:
                    logp_values[c, d] = mvn.logpdf(x)
                except ValueError as e:
                    warnings.warn(
                        f"Error computing log probability for sample {c},{d}: {e}. "
                        "Setting to NaN.",
                        stacklevel=2,
                    )
                    logp_values[c, d] = np.nan

        if np.any(np.isnan(logp_values)):
            warnings.warn(
                f"Found {np.sum(np.isnan(logp_values))} NaN values in log"
                " probabilities. This may indicate problems with the model or"
                " regularization.",
                UserWarning,
                stacklevel=2,
            )

        return logp_values

    def _reshape_posterior_for_importance_sampling(
        self, posterior: xr.Dataset
    ) -> np.ndarray:
        """Reshape posterior samples for importance sampling.

        Parameters
        ----------
        posterior : xr.Dataset
            Posterior samples from InferenceData

        Returns
        -------
        np.ndarray
            Reshaped samples with shape (n_chains, n_draws, n_params)
        """
        var_names = list(posterior.data_vars)
        n_chains = len(posterior.chain)
        n_draws = len(posterior.draw)

        n_params = 0
        for name in var_names:
            var_shape = posterior[name].shape[2:]
            n_params += np.prod(var_shape)

        samples = np.zeros((n_chains, n_draws, int(n_params)))

        param_idx = 0
        for name in var_names:
            var_shape = posterior[name].shape[2:]
            flat_size = np.prod(var_shape)

            for c in range(n_chains):
                for d in range(n_draws):
                    samples[c, d, param_idx : param_idx + int(flat_size)] = (
                        posterior[name].values[c, d].flatten()
                    )

            param_idx += int(flat_size)

        return samples

    def _convert_resampled_to_inferencedata(
        self, resampled_samples: np.ndarray
    ) -> InferenceData:
        """Convert resampled samples to InferenceData format."""
        if self.result is None:
            raise LaplaceWrapperError(
                "Model must be fit before converting resampled samples"
            )

        original_idata = self.result.idata  # type: ignore[union-attr]
        info = self.result.mu.point_map_info

        n_draws = resampled_samples.shape[0]
        n_chains = 1

        posterior_dict = {}

        param_idx = 0
        for name, shape, _, _ in info:
            if name in original_idata.posterior:
                flat_size = int(np.prod(shape))

                var_samples = resampled_samples[:, param_idx : param_idx + flat_size]
                var_samples = var_samples.reshape((n_draws,) + shape)
                var_samples = np.expand_dims(var_samples, axis=0)

                posterior_dict[name] = var_samples
                param_idx += flat_size

        coords = {
            "chain": np.arange(n_chains),
            "draw": np.arange(n_draws),
        }

        for coord_name, coord_values in original_idata.posterior.coords.items():
            if coord_name not in ["chain", "draw"]:
                coords[coord_name] = coord_values.values

        posterior_dataset = xr.Dataset(
            {
                name: (
                    ["chain", "draw"] + list(original_idata.posterior[name].dims[2:]),
                    data,
                )
                for name, data in posterior_dict.items()
            },
            coords=coords,
        )

        new_idata = az.InferenceData(posterior=posterior_dataset)

        for group in original_idata.groups():
            if group != "posterior" and group != "sample_stats" and group != "fit":
                new_idata.add_groups({group: getattr(original_idata, group)})

        if PYMC_EXTRAS_AVAILABLE:
            # type: ignore[union-attr]
            new_idata = add_fit_to_inferencedata(
                new_idata, self.result.mu, self.result.H_inv, self.model
            )

        return new_idata


def _regularize_matrix(
    matrix: np.ndarray, min_eigenvalue: float = 1e-8, max_attempts: int = 10
) -> tuple[np.ndarray, float, float, int]:
    """Regularize a matrix to ensure it is positive definite.

    Parameters
    ----------
    matrix : np.ndarray
        The input matrix to regularize
    min_eigenvalue : float, default 1e-8
        Minimum acceptable eigenvalue for positive definiteness
    max_attempts : int, default 10
        Maximum number of regularization attempts

    Returns
    -------
    tuple[np.ndarray, float, float, int]
        Tuple containing:
        - Regularized matrix
        - Original minimum eigenvalue
        - Final minimum eigenvalue
        - Number of attempts taken

    Raises
    ------
    np.linalg.LinAlgError
        If eigenvalue computation fails or regularization is unsuccessful after max_attempts
    """
    matrix_copy = matrix.copy()

    eigvals = np.linalg.eigvalsh(matrix_copy)
    orig_min_eigval = np.min(eigvals)
    min_eigval = orig_min_eigval

    if min_eigval > min_eigenvalue:
        return matrix_copy, orig_min_eigval, min_eigval, 0

    jitter_scale = min_eigenvalue
    attempts = 0

    while min_eigval <= min_eigenvalue and attempts < max_attempts:
        jitter = jitter_scale * 10**attempts
        matrix_copy = matrix + np.eye(matrix.shape[0]) * jitter

        # Check if PSD
        eigvals = np.linalg.eigvalsh(matrix_copy)
        min_eigval = np.min(eigvals)
        attempts += 1

    if min_eigval <= min_eigenvalue:
        raise np.linalg.LinAlgError(
            f"Failed to make matrix positive definite after {max_attempts} attempts. "
            f"Min eigenvalue: {min_eigval}, min required: {min_eigenvalue}"
        )

    return matrix_copy, orig_min_eigval, min_eigval, attempts
