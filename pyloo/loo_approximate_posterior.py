"""Efficient approximate leave-one-out cross-validation (LOO) for posterior approximations."""

import warnings
from typing import Any, Literal

import numpy as np
import xarray as xr
from arviz.data import InferenceData

from .base import ISMethod, compute_importance_weights
from .elpd import ELPDData
from .rcparams import rcParams
from .utils import _logsumexp, get_log_likelihood, to_inference_data, wrap_xarray_ufunc

__all__ = ["loo_approximate_posterior"]


def loo_approximate_posterior(
    data: InferenceData | Any,
    log_p: np.ndarray,
    log_g: np.ndarray,
    pointwise: bool | None = None,
    var_name: str | None = None,
    reff: float | None = None,
    scale: str | None = None,
    method: Literal["psis", "sis", "tis", "psir", "identity"] | ISMethod = "psis",
    variational: bool | None = None,
    samples: np.ndarray | None = None,
    num_draws: int | None = None,
    random_seed: int | None = None,
) -> ELPDData:
    """Efficient approximate leave-one-out cross-validation (LOO) for posterior approximations.

    This function computes LOO-CV for posterior approximations where it is possible to compute
    the log density for the posterior approximation.

    Parameters
    ----------
    data : InferenceData or Any
        Any object that can be converted to an :class:`arviz.InferenceData` object.
        Refer to documentation of :func:`arviz.convert_to_dataset` for details.
    log_p : array-like
        The log-posterior (target) evaluated at S samples from the proposal distribution (g).
        A vector of length S where S is the number of samples.
    log_g : array-like
        The log-density (proposal) evaluated at S samples from the proposal distribution (g).
        A vector of length S.
    pointwise : bool, optional
        If True, returns pointwise values. Defaults to rcParams["stats.ic_pointwise"].
    var_name : str, optional
        The name of the variable in log_likelihood groups storing the pointwise log
        likelihood data to use for loo computation.
    reff : float, optional
        Relative MCMC efficiency, ``ess / n`` i.e. number of effective samples divided by the number
        of actual samples. Computed from trace by default.
    scale : str, optional
        Output scale for LOO. Available options are:
        - "log": (default) log-score
        - "negative_log": -1 * log-score
        - "deviance": -2 * log-score
    method : {'psis', 'sis', 'tis', 'psir', 'identity'}, default 'psis'
        The importance sampling method to use:
        - 'psis': Pareto Smoothed Importance Sampling (recommended)
        - 'sis': Standard Importance Sampling
        - 'tis': Truncated Importance Sampling
        - 'psir': Pareto Smoothed Importance Resampling (for variational inference)
        - 'identity': Apply log importance weights directly (for variational inference)
    variational : bool, default False
        Whether to use variational inference specific importance sampling.
        If True, samples, logP, logQ, and num_draws must be provided.
    samples : array-like, optional
        Samples from proposal distribution, shape (L, M, N) where:
        - L is the number of chains/paths
        - M is the number of draws per chain
        - N is the number of parameters
        Required when variational=True.
    num_draws : int, optional
        Number of draws to return where num_draws <= samples.shape[0] * samples.shape[1]
        Required when variational=True.
    random_seed : int, optional
        Random seed for reproducibility in variational inference sampling.

    Returns
    -------
    ELPDData object (inherits from :class:`pandas.Series`) with the following row/attributes:
        elpd_loo: approximated expected log pointwise predictive density (elpd)
        se: standard error of the elpd
        p_loo: effective number of parameters
        n_samples: number of samples
        n_data_points: number of data points
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
        approximate_posterior: dictionary with log_p and log_g values

        The returned object has a custom print method that overrides pd.Series method.

    Examples
    --------
    Variational inference with Laplace approximation for large models

    .. code-block:: python

        import pymc as pm
        import numpy as np
        import pyloo as pl
        from pyloo.wrapper import LaplaceWrapper

        true_mu = 0.5
        true_sigma = 1.0
        y = np.random.normal(true_mu, true_sigma, size=10000)

        with pm.Model() as model:
            mu = pm.Normal('mu', mu=0, sigma=10)
            sigma = pm.HalfNormal('sigma', sigma=10)
            pm.Normal('y', mu=mu, sigma=sigma, observed=y)

            wrapper = LaplaceWrapper(model)
            result = wrapper.fit()

            # Extract log_p and log_g
            log_p = wrapper._compute_log_prob_target()
            log_g = wrapper._compute_log_prob_proposal()
            samples = wrapper._reshape_posterior_for_importance_sampling(result.idata.posterior)
            n_samples = len(log_p)

            # Compute approximate LOO with PSIS
            loo_result = pl.loo_approximate_posterior(
                result.idata,
                log_p,
                log_g,
                method="psis",
                pointwise=True,
                variational=True,
                samples=samples,
                num_draws=n_samples,
                random_seed=42
            )

            # Compute approximate LOO with PSIR
            loo_result_psir = pl.loo_approximate_posterior(
                result.idata,
                log_p,
                log_g,
                method="psir",
                pointwise=True,
                variational=True,
                samples=samples,
                num_draws=n_samples,
                random_seed=42,
            )

    See Also
    --------
    loo : Standard LOO-CV computation
    loo_subsample : Subsample-based LOO-CV computation
    loo_kfold : K-fold LOO-CV computation
    loo_moment_match : Moment-matching LOO-CV computation
    """
    inference_data = to_inference_data(data)
    log_likelihood = get_log_likelihood(inference_data, var_name=var_name)

    pointwise = rcParams["stats.ic_pointwise"] if pointwise is None else pointwise

    log_likelihood = log_likelihood.stack(__sample__=("chain", "draw"))
    shape = log_likelihood.shape
    n_samples = shape[-1]
    n_data_points = np.prod(shape[:-1])

    scale = rcParams["stats.ic_scale"] if scale is None else scale.lower()

    if scale == "deviance":
        scale_value = -2
    elif scale == "log":
        scale_value = 1
    elif scale == "negative_log":
        scale_value = -1
    else:
        raise TypeError('Valid scale values are "deviance", "log", "negative_log"')

    if not isinstance(log_p, np.ndarray):
        log_p = np.asarray(log_p)
    if not isinstance(log_g, np.ndarray):
        log_g = np.asarray(log_g)

    if len(log_p) != n_samples:
        raise ValueError(
            f"log_p must have length equal to number of samples ({n_samples}), "
            f"got {len(log_p)}"
        )
    if len(log_g) != n_samples:
        raise ValueError(
            f"log_g must have length equal to number of samples ({n_samples}), "
            f"got {len(log_g)}"
        )

    if reff is None:
        if not hasattr(inference_data, "posterior"):
            raise TypeError("Must be able to extract a posterior group from data.")
        posterior = inference_data.posterior
        n_chains = len(posterior.chain)
        if n_chains == 1:
            reff = 1.0
        else:
            from arviz.stats.diagnostics import ess as az_ess

            ess_p = az_ess(posterior, method="mean")
            reff = (
                np.hstack([ess_p[v].values.flatten() for v in ess_p.data_vars]).mean()
                / n_samples
            )

    has_nan = np.any(np.isnan(log_likelihood.values))
    if has_nan:
        warnings.warn(
            "NaN values detected in log-likelihood. These will be ignored in the LOO"
            " calculation.",
            UserWarning,
            stacklevel=2,
        )
        log_likelihood = log_likelihood.where(~np.isnan(log_likelihood), -1e10)

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
            f"Using {method_name} for approximate LOO computation. Note that PSIS is"
            " highly recommended for approximate LOO computations.",
            UserWarning,
            stacklevel=2,
        )

    if variational:
        if method not in [ISMethod.PSIS, ISMethod.PSIR, ISMethod.IDENTITY]:
            raise ValueError(
                f"Method {method} is not supported for variational inference. "
                "Use 'psis', 'psir', or 'identity' instead."
            )
        if samples is None or num_draws is None:
            raise ValueError(
                "When variational=True, samples and num_draws must be provided"
            )

    # Apply correction for posterior approximation
    approx_correction = log_p - log_g - np.max(log_p - log_g)

    approx_correction_xr = xr.DataArray(
        approx_correction,
        dims=["__sample__"],
        coords={"__sample__": log_likelihood.__sample__},
    )

    log_ratios_xr = (-log_likelihood + approx_correction_xr) - (
        -log_likelihood + approx_correction_xr
    ).max(dim="__sample__")

    if variational:
        log_weights, diagnostic = compute_importance_weights(
            method=method,
            variational=True,
            samples=samples,
            logP=log_p,
            logQ=log_g,
            num_draws=num_draws,
            random_seed=random_seed,
        )

        log_likelihood_subset = log_likelihood.isel(
            __sample__=slice(0, len(log_weights))
        )
        broadcasted_weights = xr.DataArray(
            log_weights,
            dims=["__sample__"],
            coords={"__sample__": log_likelihood_subset.__sample__},
        )

        log_weights = broadcasted_weights + log_likelihood_subset
    else:
        log_weights, diagnostic = compute_importance_weights(
            log_ratios_xr, method=method, reff=reff
        )
        log_weights += log_likelihood

    warn_mg = False
    good_k = min(1 - 1 / np.log10(n_samples), 0.7)

    if method == ISMethod.PSIS:
        if np.any(diagnostic > good_k):
            n_high_k = np.sum(diagnostic > good_k)
            warnings.warn(
                "Estimated shape parameter of Pareto distribution is greater than"
                f" {good_k:.2f} for {n_high_k} observations. This indicates that"
                " importance sampling may be unreliable because the marginal posterior"
                " and LOO posterior are very different.",
                UserWarning,
                stacklevel=2,
            )
            warn_mg = True
    else:
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
    xarray_kwargs = {"input_core_dims": [["__sample__"]]}

    loo_lppd_i = scale_value * wrap_xarray_ufunc(
        _logsumexp,
        log_weights,
        ufunc_kwargs=ufunc_kwargs,
        **xarray_kwargs,
    )

    loo_lppd = loo_lppd_i.values.sum()
    loo_lppd_se = (n_data_points * np.var(loo_lppd_i.values)) ** 0.5

    lppd = np.sum(
        wrap_xarray_ufunc(
            _logsumexp,
            log_likelihood,
            func_kwargs={"b_inv": n_samples},
            ufunc_kwargs=ufunc_kwargs,
            **xarray_kwargs,
        ).values
    )

    p_loo = lppd - loo_lppd / scale_value
    looic = -2 * loo_lppd
    looic_se = 2 * loo_lppd_se

    result_data: list[Any] = []
    result_index: list[str] = []

    if not pointwise:
        result_data = [
            loo_lppd,
            loo_lppd_se,
            p_loo,
            n_samples,
            n_data_points,
            warn_mg,
            scale,
            looic,
            looic_se,
        ]
        result_index = [
            "elpd_loo",
            "se",
            "p_loo",
            "n_samples",
            "n_data_points",
            "warning",
            "scale",
            "looic",
            "looic_se",
        ]

        if method == ISMethod.PSIS:
            result_data.append(good_k)
            result_index.append("good_k")

        result_data.append(n_data_points)
        result_index.append("subsample_size")

        result = ELPDData(data=result_data, index=result_index)
        result.approximate_posterior = {"log_p": log_p, "log_g": log_g}

        return result

    if np.allclose(loo_lppd_i, loo_lppd_i[0]):
        warnings.warn(
            "The point-wise LOO is the same with the sum LOO, please double check "
            "the Observed RV in your model to make sure it returns element-wise logp.",
            stacklevel=2,
        )

    result_data = [
        loo_lppd,
        loo_lppd_se,
        p_loo,
        n_samples,
        n_data_points,
        warn_mg,
        loo_lppd_i.rename("loo_i"),
        scale,
        looic,
        looic_se,
    ]
    result_index = [
        "elpd_loo",
        "se",
        "p_loo",
        "n_samples",
        "n_data_points",
        "warning",
        "loo_i",
        "scale",
        "looic",
        "looic_se",
    ]

    if method == ISMethod.PSIS or method == ISMethod.PSIR:
        result_data.append(diagnostic)
        result_index.append("pareto_k")
        result_data.append(good_k)
        result_index.append("good_k")
    else:
        result_data.append(diagnostic)
        result_index.append("ess")

    result_data.append(n_data_points)
    result_index.append("subsample_size")

    result = ELPDData(data=result_data, index=result_index)
    result.approximate_posterior = {"log_p": log_p, "log_g": log_g}
    result.__class__ = ELPDData

    return result
