"""Base class for expected log pointwise predictive density (ELPD) calculations."""
from abc import ABC, abstractmethod
from typing import Dict, Optional, Union

import numpy as np
from arviz import InferenceData

from .utils import (
    compute_estimates,
    extract_log_likelihood,
    get_log_likelihood,
    validate_data,
)


class ELPD(ABC):
    """Abstract base class for ELPD calculations.

    This class provides a framework for computing the expected log pointwise
    predictive density (ELPD) using various methods such as LOO-CV, WAIC, etc.
    Specific implementations should inherit from this class and implement the
    required methods.

    Parameters
    ----------
    data : Union[InferenceData, np.ndarray]
        Input data containing log likelihood values. Can be:
        - ArviZ InferenceData object
        - numpy array of log-likelihood values
        - Any object that can be converted to InferenceData
    var_name : Optional[str]
        Name of variable in log_likelihood group (only used if data is InferenceData)
    """

    def __init__(
        self,
        data: Union[InferenceData, np.ndarray],
        var_name: Optional[str] = "obs",
    ) -> None:
        self.data = validate_data(data, var_name=var_name)
        self.var_name = var_name
        self._log_likelihood: Optional[np.ndarray] = None
        self._chain_ids: Optional[np.ndarray] = None
        self._pointwise: Optional[Dict[str, np.ndarray]] = None
        self._n_samples: Optional[int] = None
        self._n_observations: Optional[int] = None
        self._dims: Optional[Dict[str, int]] = None

    @property
    def log_likelihood(self) -> np.ndarray:
        """Get log likelihood values.

        For InferenceData objects with multiple dimensions beyond chain and draw,
        all non-chain/draw dimensions are automatically combined into observations.
        The resulting matrix has shape (n_samples, n_observations) where n_observations
        is the product of the sizes of all dimensions after chain and draw.

        Returns
        -------
        np.ndarray
            Log likelihood matrix with shape (n_samples, n_observations)
        """
        if self._log_likelihood is None:
            if isinstance(self.data, InferenceData):
                log_lik = get_log_likelihood(self.data, self.var_name)

                dims = list(log_lik.dims)
                if "chain" in dims:
                    dims.remove("chain")
                if "draw" in dims:
                    dims.remove("draw")

                self._dims = {dim: log_lik.sizes[dim] for dim in dims}
                stacked = log_lik.stack(__sample__=("chain", "draw"))

                if dims:
                    stacked = stacked.stack(__obs__=dims)
                values = stacked.values

                if values.ndim > 2:
                    n_samples = values.shape[0]
                    n_obs = np.prod(values.shape[1:])
                    values = values.reshape(n_samples, n_obs)

                self._log_likelihood = values
                self._chain_ids = np.repeat(np.arange(log_lik.sizes["chain"]), log_lik.sizes["draw"])
            else:
                if self.data.ndim == 3:
                    n_chains, n_draws, n_obs = self.data.shape
                    self._log_likelihood = self.data.reshape(n_chains * n_draws, n_obs)
                    self._chain_ids = np.repeat(np.arange(n_chains), n_draws)
                elif self.data.ndim > 3:
                    # For numpy arrays with more than 3 dimensions,
                    # combine all dimensions after chains and draws
                    n_chains, n_draws = self.data.shape[:2]
                    remaining_dims = self.data.shape[2:]
                    n_obs = np.prod(remaining_dims)
                    self._log_likelihood = self.data.reshape(n_chains * n_draws, n_obs)
                    self._chain_ids = np.repeat(np.arange(n_chains), n_draws)
                else:
                    self._log_likelihood = np.asarray(self.data)

        return self._log_likelihood

    @property
    def chain_ids(self) -> Optional[np.ndarray]:
        """Get chain IDs if available."""
        if self._chain_ids is None and isinstance(self.data, InferenceData):
            if self.var_name is None:
                raise ValueError("var_name must be provided for InferenceData objects")
            _, self._chain_ids = extract_log_likelihood(self.data, self.var_name)
        return self._chain_ids

    @property
    def n_samples(self) -> int:
        """Get number of posterior samples."""
        if self._n_samples is None:
            self._n_samples = self.log_likelihood.shape[0]
        return self._n_samples

    @property
    def n_observations(self) -> int:
        """Get number of observations."""
        if self._n_observations is None:
            self._n_observations = self.log_likelihood.shape[1]
        return self._n_observations

    @property
    def dims(self) -> Optional[Dict[str, int]]:
        """Get dimension information if available.

        Returns
        -------
        Optional[Dict[str, int]]
            Dictionary mapping dimension names to their sizes,
            excluding chain and draw dimensions. Returns None for
            non-InferenceData inputs.
        """
        if self._dims is None and isinstance(self.data, InferenceData):
            _ = self.log_likelihood  # This will populate self._dims
        return self._dims

    @abstractmethod
    def pointwise_estimates(self) -> Dict[str, np.ndarray]:
        """Compute pointwise estimates.

        This method must be implemented by subclasses to compute pointwise
        estimates specific to their ELPD calculation method.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing pointwise estimates
        """
        pass

    def aggregate_estimates(self) -> Dict[str, Dict[str, float]]:
        """Compute aggregate estimates from pointwise values.

        Returns
        -------
        Dict[str, Dict[str, float]]
            Dictionary containing aggregate estimates and standard errors
        """
        pointwise = self.pointwise_estimates()
        estimates = {}
        for key, values in pointwise.items():
            estimates[key] = compute_estimates(values.reshape(-1, 1))
        return estimates

    def __str__(self) -> str:
        """String representation of ELPD object."""
        estimates = self.aggregate_estimates()
        output = [f"ELPD estimates computed from {self.n_samples} by {self.n_observations} log-likelihood matrix"]
        if self.dims:
            output.append("\nOriginal dimensions (excluding chain and draw):")
            for dim, size in self.dims.items():
                output.append(f"    {dim}: {size}")
            output.append("\nNote: All dimensions have been combined into a single observation dimension")
        for metric, values in estimates.items():
            output.append(f"\n{metric}:")
            output.append(f"    Estimate: {float(values['estimate']):.2f}")
            output.append(f"    SE: {float(values['se']):.2f}")
        return "\n".join(output)

    def __repr__(self) -> str:
        """Object representation."""
        return f"{self.__class__.__name__}(n_samples={self.n_samples}, n_observations={self.n_observations})"
