"""Base functionality for LOO approximation methods."""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import xarray as xr


class LooApproximation(ABC):
    """Abstract base class for LOO approximation methods."""

    @abstractmethod
    def compute_approximation(
        self,
        log_likelihood: xr.DataArray,
        n_draws: Optional[int] = None,
    ) -> np.ndarray:
        """Compute approximation of LOO values.

        Parameters
        ----------
        log_likelihood : xr.DataArray
            Log likelihood values with shape (n_obs, n_samples)
        n_draws : Optional[int]
            Number of draws to use for approximation, if applicable.
            If None, uses all available draws.

        Returns
        -------
        np.ndarray
            Array of approximated LOO values with shape (n_obs,)
        """
        pass


def thin_draws(
    data: xr.DataArray,
    n_draws: Optional[int] = None,
) -> xr.DataArray:
    """Thin data to specified size.

    Parameters
    ----------
    data : xr.DataArray
        Data to thin
    n_draws : Optional[int]
        Target number of draws. If None, returns original data.

    Returns
    -------
    xr.DataArray
        Thinned data with shape (n_draws, ...)

    Raises
    ------
    ValueError
        If n_draws exceeds available draws
    """
    if n_draws is None:
        return data

    if isinstance(data, xr.Dataset):
        if "chain" in data.dims and "draw" in data.dims:
            available_draws = data.dims["chain"] * data.dims["draw"]
        else:
            available_draws = data.dims.get("__sample__", data.dims.get("sample", 0))
    else:
        available_draws = data.sizes.get("__sample__", 0)

    if n_draws is not None and n_draws > available_draws:
        raise ValueError(
            f"Requested {n_draws} draws but only {available_draws} are available"
        )

    if isinstance(data, xr.Dataset):
        sample_dims = [dim for dim in data.dims if dim in ["__sample__", "sample"]]
        if sample_dims:
            sample_dim = sample_dims[0]
            n_samples = data.dims[sample_dim]
        else:
            if "chain" in data.dims and "draw" in data.dims:
                n_samples = data.dims["chain"] * data.dims["draw"]
                data = data.stack(__sample__=("chain", "draw"))
                sample_dim = "__sample__"
            else:
                raise ValueError("No sample dimension found and cannot create one")
    else:
        if "__sample__" in data.dims:
            n_samples = data.sizes["__sample__"]
            sample_dim = "__sample__"
        else:
            raise ValueError("No sample dimension found in DataArray")

    if n_draws > n_samples:
        raise ValueError(
            f"Target number of draws ({n_draws}) cannot exceed "
            f"current number of draws ({n_samples})"
        )

    idx = np.linspace(0, n_samples - 1, n_draws, dtype=int)
    return data.isel(__sample__=idx)
