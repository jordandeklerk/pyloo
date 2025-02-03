"""Helper functions for pyloo package."""

from typing import Any, Dict, Optional, Tuple, Union

import arviz as az
import numpy as np
from arviz import InferenceData

from .psis import _logsumexp

_FLOAT_EPS = np.finfo(float).eps


def to_inference_data(obj: Any) -> InferenceData:
    """Convert supported objects to ArviZ InferenceData.

    Parameters
    ----------
    obj : Any
        Object to convert. Supported types include:
        - ArviZ InferenceData
        - PyMC model/trace
        - CmdStanPy fit
        - NumPyro mcmc
        - Pyro MCMC
        - TensorFlow Probability
        - xarray Dataset
        - numpy array
        - dict with array-like values

    Returns
    -------
    InferenceData
        ArviZ InferenceData object

    Raises
    ------
    ValueError
        If conversion fails or object type is not supported
    """
    if isinstance(obj, InferenceData):
        return obj

    if isinstance(obj, (list, tuple)):
        raise ValueError("Lists and tuples cannot be converted to InferenceData directly")

    if isinstance(obj, dict) and not all(isinstance(v, (np.ndarray, list)) for v in obj.values()):
        raise ValueError("Dictionary values must be array-like")

    try:
        return az.convert_to_inference_data(obj)
    except Exception as e:
        allowable_types = (
            "xarray Dataset",
            "xarray DataArray",
            "dict with array-like values",
            "numpy array",
            "pystan fit",
            "emcee fit",
            "pyro mcmc fit",
            "numpyro mcmc fit",
            "cmdstan fit",
            "cmdstanpy fit",
        )
        raise ValueError(
            f'Can only convert {", ".join(allowable_types)} to InferenceData, ' f"not {obj.__class__.__name__}"
        ) from e


def extract_log_likelihood(idata: InferenceData, var_name: str = "obs") -> Tuple[np.ndarray, np.ndarray]:
    """Extract log likelihood values and chain IDs from InferenceData.

    Parameters
    ----------
    idata : InferenceData
        ArviZ InferenceData object
    var_name : str, optional
        Name of log likelihood variable in log_likelihood group
        Default is "obs" as it's commonly used in ArviZ

    Returns
    -------
    tuple
        (log_likelihood_array, chain_ids)
        - log_likelihood_array: Array with shape (chains*draws, observations)
        - chain_ids: Array of chain IDs for each draw

    Raises
    ------
    ValueError
        If log likelihood values cannot be extracted
    """
    if not hasattr(idata, "log_likelihood"):
        raise ValueError("InferenceData object must have log_likelihood group")

    if var_name not in idata.log_likelihood.data_vars:
        raise ValueError(f"Variable '{var_name}' not found in log_likelihood group")

    log_lik = idata.log_likelihood[var_name]

    if log_lik.ndim == 2:
        return log_lik.values, np.ones(len(log_lik))

    n_chains = log_lik.shape[0]  # First dimension is chain
    n_draws = log_lik.shape[1]  # Second dimension is draw
    chain_ids = np.repeat(np.arange(1, n_chains + 1), n_draws)

    # Reshape to (chains*draws, observations)
    reshaped_values = log_lik.values.reshape(-1, log_lik.shape[-1])
    return reshaped_values, chain_ids


def compute_log_mean_exp(x: np.ndarray, axis: Optional[int] = None) -> Union[float, np.ndarray]:
    """Compute log(mean(exp(x))) in a numerically stable way.

    Parameters
    ----------
    x : np.ndarray
        Input array
    axis : int, optional
        Axis along which to compute the log mean exp

    Returns
    -------
    Union[float, np.ndarray]
        log(mean(exp(x))) computed stably
    """
    if axis is None:
        log_S = np.log(x.size)
        return _logsumexp(x.ravel()) - log_S
    else:
        log_S = np.log(x.shape[axis])
        return _logsumexp(x, axis=axis) - log_S


def compute_estimates(x: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute point estimates and standard errors.

    Parameters
    ----------
    x : np.ndarray
        Input matrix of pointwise values

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with keys:
        - estimate: Point estimates
        - se: Standard errors
    """
    estimate = np.sum(x, axis=0)
    se = np.sqrt(x.shape[0] * np.var(x, axis=0, ddof=1))
    return {"estimate": estimate, "se": se}


def validate_data(
    x: Union[np.ndarray, InferenceData, Any],
    var_name: Optional[str] = None,
    check_shape: Optional[Tuple[int, ...]] = None,
    allow_inf: bool = False,
) -> Union[np.ndarray, InferenceData]:
    """Validate input data for PSIS-LOO-CV computations.

    This function performs comprehensive validation of input data to ensure it meets
    the requirements for PSIS-LOO-CV computations. It handles both raw arrays and
    InferenceData objects.

    Parameters
    ----------
    x : Union[np.ndarray, InferenceData, Any]
        Data to validate. Can be:
        - numpy array of log-likelihood values
        - ArviZ InferenceData object
        - Any object that can be converted to InferenceData
    var_name : str, optional
        For InferenceData objects, the name of the log-likelihood variable
        Default is None
    check_shape : tuple of int, optional
        Expected shape of the array. If provided, validates the array shape.
        Default is None
    allow_inf : bool, optional
        Whether to allow infinite values (useful for log-likelihoods)
        Default is False

    Returns
    -------
    Union[np.ndarray, InferenceData]
        Validated data object

    Raises
    ------
    TypeError
        If input type is not supported or conversion fails
    ValueError
        If validation fails for any of these reasons:
        - Array contains NaN values
        - Array contains infinite values when not allowed
        - Array shape doesn't match expected shape
        - InferenceData missing required groups or variables
        - Numerical instability detected
    """
    if isinstance(x, InferenceData) or not isinstance(x, np.ndarray):
        try:
            if not isinstance(x, InferenceData):
                x = to_inference_data(x)

            if not hasattr(x, "log_likelihood"):
                raise ValueError("InferenceData object must have a log_likelihood group for PSIS-LOO-CV")

            if var_name is not None and var_name not in x.log_likelihood.data_vars:
                available_vars = list(x.log_likelihood.data_vars.keys())
                raise ValueError(
                    f"Variable '{var_name}' not found in log_likelihood group. "
                    f"Available variables are: {available_vars}"
                )

            return x

        except Exception as e:
            raise TypeError(f"Failed to validate or convert input: {str(e)}")

    if not isinstance(x, np.ndarray):
        raise TypeError(f"Expected numpy array or InferenceData, got {type(x).__name__}")

    if np.any(np.isnan(x)):
        raise ValueError("Input contains NaN values.")

    if not allow_inf and np.any(np.isinf(x)):
        raise ValueError("Input contains infinite values. If these are log-likelihoods, " "set allow_inf=True")

    if check_shape is not None and x.shape != check_shape:
        raise ValueError(f"Array has incorrect shape. Expected {check_shape}, got {x.shape}")

    if not allow_inf:
        abs_max = np.max(np.abs(x[np.isfinite(x)]))
        if abs_max > 1e38:  # Close to float32 limit
            raise ValueError(
                f"Array contains very large values (max abs: {abs_max}). " "This may cause numerical instability."
            )

    return x


def reshape_draws(x: np.ndarray, chain_ids: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Reshape MCMC draws between matrix and array formats.

    Parameters
    ----------
    x : np.ndarray
        Array of MCMC draws
    chain_ids : np.ndarray, optional
        Chain ID for each draw

    Returns
    -------
    Tuple[np.ndarray, Optional[np.ndarray]]
        Reshaped array and updated chain IDs
    """
    if x.ndim == 3:
        # Convert (iterations, chains, variables) to (iterations*chains, variables)
        return x.reshape(-1, x.shape[2]), None
    elif x.ndim == 2 and chain_ids is not None:
        # Convert (iterations*chains, variables) to (iterations, chains, variables)
        n_chains = len(np.unique(chain_ids))
        n_iter = len(x) // n_chains
        return x.reshape(n_iter, n_chains, -1), chain_ids
    else:
        return x, chain_ids


def is_constant(x: np.ndarray, tol: float = _FLOAT_EPS) -> bool:
    """Check if array is constant within tolerance.

    Parameters
    ----------
    x : np.ndarray
        Input array
    tol : float, optional
        Tolerance for comparison (default: machine epsilon for float)

    Returns
    -------
    bool
        True if max difference between elements is less than tolerance
    """
    return np.abs(np.max(x) - np.min(x)) < tol
