"""Laplace Variational Inference wrapper for PyMC models."""

import warnings
from dataclasses import dataclass, field
from typing import Any, Literal

import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from arviz import InferenceData
from better_optimize.constants import minimize_method
from pymc.blocking import RaveledVars
from scipy import stats

from ..psis import vi_psis_sampling
from ..wrapper.pymc_wrapper import PyMCWrapper
from ..wrapper.utils import PyMCWrapperError

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


class LaplaceWrapper(PyMCWrapper):
    """Laplace Variational Inference wrapper for PyMC models.

    This class extends the PyMCWrapper to provide Laplace approximation
    functionality for PyMC models, including importance resampling to improve
    the quality of the approximation.

    Parameters
    ----------
    model : pm.Model
        PyMC model to perform inference on
    idata : InferenceData, optional
        ArviZ InferenceData object containing the model's posterior samples.
        If None, the model will be fit using Laplace approximation.
    var_names : Sequence[str], optional
        Names of specific variables to focus on. If None, all variables are included.
    approximation : Any, optional
        A PyMC approximation object. If provided, it can be used for approximate
        posterior calculations without passing it explicitly each time.
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
    observed_data : dict[str, np.ndarray]
        Mapping of observed variable names to their data
    constant_data : dict[str, np.ndarray]
        Mapping of constant data names to their values
    observed_dims : dict[str, tuple[str, ...]]
        Mapping of observed variable names to their dimension names
    result : LaplaceVIResult
        Container with the results of the Laplace approximation
    """

    def __init__(
        self,
        model: pm.Model,
        idata: InferenceData,
        var_names: list[str] | None = None,
        approximation: Any | None = None,
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
        approximation : Any, optional
            A PyMC approximation object. If provided, it can be used for approximate
            posterior calculations without passing it explicitly each time.
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
        # Initialize with the parent class
        super().__init__(model, idata, var_names, approximation)

        # Store additional attributes
        self.random_seed = random_seed
        self.result: LaplaceVIResult | None = None

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
        on_bad_cov: Literal["warn", "error", "ignore"] = "ignore",
        fit_in_unconstrained_space: bool = False,
        zero_tol: float = 1e-8,
        diag_jitter: float | None = 1e-8,
        optimizer_kwargs: dict | None = None,
        compile_kwargs: dict | None = None,
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
            mu, H_inv = fit_mvn_at_MAP(
                optimized_point=optimized_point,
                model=self.model,
                on_bad_cov=on_bad_cov,
                transform_samples=fit_in_unconstrained_space,
                gradient_backend=gradient_backend,
                zero_tol=zero_tol,
                diag_jitter=diag_jitter,
                compile_kwargs=compile_kwargs,
            )

            # Sample from the posterior
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

        result = LaplaceVIResult(
            idata=idata,
            mu=mu,
            H_inv=H_inv,
            model=self.model,
        )
        self.idata = idata
        self.result = result
        return result

    def importance_resample(
        self,
        num_draws: int | None = None,
        method: Literal["psis", "psir", "identity"] = "psis",
        random_seed: int | None = None,
    ) -> LaplaceVIResult:
        """Apply importance resampling to the Laplace approximation samples.

        Parameters
        ----------
        num_draws : Optional[int]
            Number of draws to return. If None, uses the same number as in the original samples.
        method : Literal["psis", "psir", "identity"]
            Method to apply for importance resampling:
            - "psis": Pareto Smoothed Importance Sampling (default)
            - "psir": Pareto Smoothed Importance Resampling
            - "identity": Applies log importance weights directly without resampling
        random_seed : Optional[int]
            Random seed for reproducibility. If None, uses the instance's random_seed.

        Returns
        -------
        LaplaceVIResult
            Container with the results after importance resampling.
        """
        if self.result is None:
            raise PyMCWrapperError(
                "Model must be fit before applying importance resampling"
            )

        if not PYMC_EXTRAS_AVAILABLE:
            raise ImportError(
                "pymc-extras is required for importance resampling. "
                "Install it with: pip install pymc-extras"
            )

        if random_seed is None:
            random_seed = self.random_seed

        # type: ignore[union-attr]
        posterior = self.result.idata.posterior
        n_chains = len(posterior.chain)
        n_draws_per_chain = len(posterior.draw)

        if num_draws is None:
            num_draws = n_chains * n_draws_per_chain

        # Compute log probabilities for the target (true posterior) and proposal (Laplace approximation)
        logP = self._compute_log_prob_target(posterior)
        # type: ignore[union-attr]
        logQ = self._compute_log_prob_proposal(
            posterior, self.result.mu, self.result.H_inv
        )
        samples = self._reshape_posterior_for_importance_sampling(posterior)

        is_result = vi_psis_sampling(
            samples=samples,
            logP=logP,
            logQ=logQ,
            num_draws=num_draws,
            method=method,
            random_seed=random_seed,
        )

        resampled_idata = self._convert_resampled_to_inferencedata(is_result.samples)

        # type: ignore[union-attr]
        new_result = LaplaceVIResult(
            idata=resampled_idata,
            mu=self.result.mu,
            H_inv=self.result.H_inv,
            model=self.result.model,
            importance_sampled=True,
            importance_sampling_method=method,
            pareto_k=is_result.pareto_k,
            log_weights=is_result.log_weights,
            warnings=self.result.warnings + is_result.warnings,
        )
        self.result = new_result
        return new_result

    def _compute_log_prob_target(self, posterior: xr.Dataset) -> np.ndarray:
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

    def _compute_log_prob_proposal(
        self, posterior: xr.Dataset, mu: RaveledVars, H_inv: np.ndarray
    ) -> np.ndarray:
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
        mvn = stats.multivariate_normal(mean=mu.data, cov=H_inv)

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
                logp_values[c, d] = mvn.logpdf(x)

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
            raise PyMCWrapperError(
                "Model must be fit before converting resampled samples"
            )

        # type: ignore[union-attr]
        original_idata = self.result.idata
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
