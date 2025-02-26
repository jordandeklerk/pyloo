"""Helper function to compute LOO for a single observation using importance sampling methods."""

import warnings
from typing import Any, Literal

import numpy as np
from arviz.data import InferenceData
from arviz.stats.diagnostics import ess

from .elpd import ELPDData
from .importance_sampling import ISMethod, compute_importance_weights
from .rcparams import rcParams
from .utils import _logsumexp, get_log_likelihood, to_inference_data, wrap_xarray_ufunc


def loo_i(
    i: int,
    data: InferenceData | Any,
    pointwise: bool | None = None,
    var_name: str | None = None,
    reff: float | None = None,
    scale: str | None = None,
    method: Literal["psis", "sis", "tis"] | ISMethod = "psis",
) -> ELPDData:
    """Compute leave-one-out cross-validation (LOO-CV) for a single observation using importance sampling.

    This function computes the expected log pointwise predictive density (elpd) for a single observation
    using importance sampling leave-one-out cross-validation. By default, uses Pareto-smoothed
    importance sampling (PSIS), which is the recommended method.

    Parameters
    ----------
    i : int
        The index of the observation to process. Must be a valid index for the data.
    data: obj
        Any object that can be converted to an :class:`arviz.InferenceData` object.
        Refer to documentation of :func:`arviz.convert_to_dataset` for details.
    pointwise: bool, optional
        If True the pointwise predictive accuracy will be returned. Defaults to
        ``stats.ic_pointwise`` rcParam.
    var_name : str, optional
        The name of the variable in log_likelihood groups storing the pointwise log
        likelihood data to use for loo computation.
    reff: float, optional
        Relative MCMC efficiency, ``ess / n`` i.e. number of effective samples divided by the number
        of actual samples. Computed from trace by default.
    scale: str
        Output scale for loo. Available options are:
    method: {'psis', 'sis', 'tis'}, default 'psis'
        The importance sampling method to use:
        - 'psis': Pareto Smoothed Importance Sampling (recommended)
        - 'sis': Standard Importance Sampling
        - 'tis': Truncated Importance Sampling
            Output scale for loo. Available options are:

            - ``log`` : (default) log-score
            - ``negative_log`` : -1 * log-score
            - ``deviance`` : -2 * log-score

            A higher log-score (or a lower deviance or negative log_score) indicates a model with
            better predictive accuracy.

    Returns
    -------
    ELPDData object (inherits from :class:`pandas.Series`) with the following row/attributes:
    elpd_loo: approximated expected log pointwise predictive density (elpd)
    se: standard error of the elpd
    p_loo: effective number of parameters
    n_samples: number of samples
    n_data_points: number of data points (always 1 for loo_i)
    warning: bool
        True if using PSIS and the estimated shape parameter of Pareto distribution
        is greater than ``good_k``.
    loo_i: :class:`~xarray.DataArray` with the pointwise predictive accuracy,
            only if pointwise=True
    diagnostic: array of diagnostic values, only if pointwise=True
        - For PSIS: Pareto shape parameter (pareto_k)
        - For SIS/TIS: Effective sample size (ess)
    scale: scale of the elpd
    good_k: For PSIS method and sample size S, threshold computed as min(1 - 1/log10(S), 0.7)

    Examples
    --------
    Calculate LOO for a single observation to assess its influence on the model:

    .. code-block:: python

        import pyloo as pl
        import arviz as az
        data = az.load_arviz_data("centered_eight")
        loo_single = pl.loo_i(0, data)
        print(f"p_loo: {loo_single.p_loo:.2f}, warning: {loo_single.warning}")

    Calculate LOO for a single observation and return the pointwise values:

    .. code-block:: python

        data_loo = pl.loo_i(0, data, pointwise=True)
        data_loo.loo_i
    """
    inference_data = to_inference_data(data)
    log_likelihood = get_log_likelihood(inference_data, var_name=var_name)
    pointwise = rcParams["stats.ic_pointwise"] if pointwise is None else pointwise

    log_likelihood = log_likelihood.stack(__sample__=("chain", "draw"))
    shape = log_likelihood.shape
    n_samples = shape[-1]
    n_data_points = 1  # Always 1 for loo_i since we're processing a single observation

    # Extract the i-th observation's log likelihood
    if isinstance(i, (list, tuple, np.ndarray)):
        raise ValueError("loo_i only accepts a single integer index")
    try:
        i = int(i)
    except (TypeError, ValueError):
        raise TypeError("Index i must be an integer")

    total_obs = np.prod(shape[:-1])
    if i >= total_obs or i < 0:
        raise IndexError(
            f"Index {i} is out of bounds for log likelihood array with"
            f" {total_obs} observations"
        )

    try:
        # Handle different possible shapes of log_likelihood
        if len(shape) == 2:  # (observation, sample)
            log_likelihood_i = log_likelihood[i : i + 1]
        else:  # Handle multi-dimensional cases
            # Convert flat index to multi-dimensional index
            idx = np.unravel_index(i, shape[:-1])
            log_likelihood_i = log_likelihood.isel(
                {dim: idx[n] for n, dim in enumerate(log_likelihood.dims[:-1])}
            )
            log_likelihood_i = log_likelihood_i.expand_dims("_dummy", axis=0)

    except IndexError:
        raise IndexError(
            f"Index {i} is out of bounds for log likelihood array with shape {shape}"
        )

    scale = rcParams["stats.ic_scale"] if scale is None else scale.lower()

    if scale == "deviance":
        scale_value = -2
    elif scale == "log":
        scale_value = 1
    elif scale == "negative_log":
        scale_value = -1
    else:
        raise TypeError('Valid scale values are "deviance", "log", "negative_log"')

    if reff is None:
        if not hasattr(inference_data, "posterior"):
            raise TypeError("Must be able to extract a posterior group from data.")
        posterior = inference_data.posterior
        n_chains = len(posterior.chain)
        if n_chains == 1:
            reff = 1.0
        else:
            ess_p = ess(posterior, method="mean")
            reff = (
                np.hstack([ess_p[v].values.flatten() for v in ess_p.data_vars]).mean()
                / n_samples
            )

    has_nan = np.any(np.isnan(log_likelihood_i.values))
    has_inf = np.any(np.isinf(log_likelihood_i.values))

    if has_nan:
        warnings.warn(
            "NaN values detected in log-likelihood. These will be ignored in the LOO"
            " calculation.",
            UserWarning,
            stacklevel=2,
        )
        log_likelihood_i = log_likelihood_i.where(~np.isnan(log_likelihood_i), -1e10)

    if has_inf:
        warnings.warn(
            "Infinite values detected in log-likelihood. These will be ignored in the"
            " LOO calculation.",
            UserWarning,
            stacklevel=2,
        )
        log_likelihood_i = log_likelihood_i.where(
            ~np.isinf(log_likelihood_i),
            np.where(np.isinf(log_likelihood_i) & (log_likelihood_i > 0), 1e10, -1e10),
        )

    try:
        method = method if isinstance(method, ISMethod) else ISMethod(method.lower())
    except ValueError:
        valid_methods = ", ".join(m.value for m in ISMethod)
        raise ValueError(f"Invalid method '{method}'. Must be one of: {valid_methods}")

    if method != ISMethod.PSIS:
        method_name = (
            method.value.upper() if isinstance(method, ISMethod) else method.upper()
        )
        warnings.warn(
            f"Using {method_name} for LOO computation. Note that PSIS is the"
            " recommended method as it is typically more efficient and reliable.",
            UserWarning,
            stacklevel=2,
        )

    log_weights, diagnostic = compute_importance_weights(
        -log_likelihood_i, method=method, reff=reff
    )
    log_weights += log_likelihood_i

    warn_mg = False
    good_k = min(1 - 1 / np.log10(n_samples), 0.7)

    if method == ISMethod.PSIS:
        if np.any(diagnostic > good_k):
            warnings.warn(
                "Estimated shape parameter of Pareto distribution is greater than"
                f" {good_k:.2f} for the observation. You should consider using a more"
                " robust model, this is because importance sampling is less likely to"
                " work well if the marginal posterior and LOO posterior are very"
                " different. This is more likely to happen with a non-robust model and"
                " highly influential observations.",
                UserWarning,
                stacklevel=2,
            )
            warn_mg = True
    else:
        # For SIS/TIS, warn if effective sample size is too low
        min_ess = np.min(diagnostic)
        if min_ess < n_samples * 0.1:
            warnings.warn(
                f"Low effective sample size detected (minimum ESS: {min_ess:.1f}). This"
                " indicates that the importance sampling approximation may be"
                " unreliable. Consider using PSIS which is more robust to such cases.",
                UserWarning,
                stacklevel=2,
            )
            warn_mg = True

    ufunc_kwargs = {"n_dims": 1, "ravel": False}
    kwargs = {"input_core_dims": [["__sample__"]]}
    loo_lppd_i = scale_value * wrap_xarray_ufunc(
        _logsumexp, log_weights, ufunc_kwargs=ufunc_kwargs, **kwargs
    )
    loo_lppd = loo_lppd_i.values.sum()

    # Convert to linear scale for variance calculation
    weights = np.exp(
        log_weights.values - np.max(log_weights.values, axis=-1, keepdims=True)
    )
    weights /= np.sum(weights, axis=-1, keepdims=True)
    w2 = weights**2
    lik = np.exp(log_likelihood_i.values)
    E_epd = np.exp(loo_lppd)

    # Variance in linear scale (Equation 6 in Vehtari et al. 2024)
    var_epd = np.sum(w2 * (lik - E_epd) ** 2) / reff

    # Convert to log scale using log-normal approximation
    loo_lppd_se = np.sqrt(np.log1p(var_epd / E_epd**2))

    lppd = np.sum(
        wrap_xarray_ufunc(
            _logsumexp,
            log_likelihood_i,
            func_kwargs={"b_inv": n_samples},
            ufunc_kwargs=ufunc_kwargs,
            **kwargs,
        ).values
    )
    p_loo = lppd - loo_lppd / scale_value

    if not pointwise:
        data = [loo_lppd, loo_lppd_se, p_loo, n_samples, n_data_points, warn_mg, scale]
        index = [
            "elpd_loo",
            "se",
            "p_loo",
            "n_samples",
            "n_data_points",
            "warning",
            "scale",
        ]

        if method == ISMethod.PSIS:
            data.append(good_k)
            index.append("good_k")

        return ELPDData(data=data, index=index)

    data = [
        loo_lppd,
        loo_lppd_se,
        p_loo,
        n_samples,
        n_data_points,
        warn_mg,
        loo_lppd_i.rename("loo_i"),
        scale,
    ]
    index = [
        "elpd_loo",
        "se",
        "p_loo",
        "n_samples",
        "n_data_points",
        "warning",
        "loo_i",
        "scale",
    ]

    if method == ISMethod.PSIS:
        data.extend([diagnostic, good_k])
        index.extend(["pareto_k", "good_k"])
    else:
        data.append(diagnostic)
        index.append("ess")

    return ELPDData(data=data, index=index)
