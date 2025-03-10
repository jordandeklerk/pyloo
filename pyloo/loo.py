"""Leave-one-out cross-validation (LOO-CV) using importance sampling methods based on ArviZ."""

import warnings
from typing import Any, Literal

import numpy as np
from arviz.data import InferenceData
from arviz.stats.diagnostics import ess

from .base import ISMethod, compute_importance_weights
from .elpd import ELPDData
from .loo_moment_match import loo_moment_match
from .rcparams import rcParams
from .utils import _logsumexp, get_log_likelihood, to_inference_data, wrap_xarray_ufunc

__all__ = ["loo"]


def loo(
    data: InferenceData | Any,
    pointwise: bool | None = None,
    var_name: str | None = None,
    reff: float | None = None,
    scale: str | None = None,
    method: Literal["psis", "sis", "tis"] | ISMethod = "psis",
    moment_match: bool = False,
    **kwargs,
) -> ELPDData:
    """Compute leave-one-out cross-validation (LOO-CV) using various importance sampling methods.

    Estimates the expected log pointwise predictive density (elpd) using importance sampling
    leave-one-out cross-validation. By default, uses Pareto-smoothed importance sampling (PSIS),
    which is the recommended method. Also calculates LOO's standard error and the effective
    number of parameters.

    For observations with high Pareto k diagnostics, moment matching can be used to transform
    posterior draws to better approximate leave-one-out posteriors. This approach is computationally
    more efficient than exact refitting ('reloo') while still providing improved estimates.

    Parameters
    ----------
    data: obj
        Any object that can be converted to an :class:`arviz.InferenceData` object.
        Refer to documentation of
        :func:`arviz.convert_to_dataset` for details.
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
    moment_match: bool, default False
        Whether to perform moment matching to improve the LOO estimates for observations with
        high Pareto k values. If True, the `wrapper` parameter must be provided in kwargs.
    **kwargs:
        Additional keyword arguments for moment matching. These include:
        - wrapper: PyMCWrapper, required if moment_match=True
            PyMC model wrapper instance
        - max_iters: int, default 30
            Maximum number of moment matching iterations
        - k_threshold: float, optional
            Threshold value for Pareto k values above which moment matching is used.
            If None, uses min(1 - 1/log10(n_samples), 0.7)
        - split: bool, default True
            Whether to do the split transformation at the end of moment matching
        - cov: bool, default True
            Whether to match the covariance matrix of the samples

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

        The returned object has a custom print method that overrides pd.Series method.

        Examples
        --------
        Calculate the Leave-One-Out Cross-Validation (LOO-CV) for a model by providing InferenceData:

        .. code-block:: python

            import pyloo as pl
            import arviz as az

            data = az.load_arviz_data("centered_eight")
            loo_results = pl.loo(data)
            print(loo_results)

        Calculate LOO with pointwise values to examine individual observation contributions:

        .. code-block:: python

            import pyloo as pl
            import arviz as az

            data = az.load_arviz_data("centered_eight")
            data_loo = pl.loo(data, pointwise=True)
            print(data_loo.loo_i)

            if hasattr(data_loo, "pareto_k"):
                print(data_loo.pareto_k)

        Calculate LOO with moment matching to improve estimates for observations with high Pareto k values:

        .. code-block:: python

            import pyloo as pl
            import arviz as az
            import pymc as pm

            with pm.Model() as model:
                mu = pm.Normal('mu', mu=0, sigma=10)
                sigma = pm.HalfNormal('sigma', sigma=10)
                y = pm.Normal('y', mu=mu, sigma=sigma, observed=data)
                idata = pm.sample(1000, tune=1000, idata_kwargs={"log_likelihood": True})

            wrapper = pl.PyMCWrapper(model, idata)

            # Calculate LOO with moment matching
            loo_results = pl.loo(
                idata,
                pointwise=True,
                moment_match=True,
                wrapper=wrapper,
                max_iters=30,
                split=True,
                cov=True
            )

            print(loo_results.pareto_k)

    See Also
    --------
    loo_subsample : Subsampled LOO-CV computation
    reloo : Exact LOO-CV computation for PyMC models
    loo_moment_match : LOO-CV computation using moment matching
    loo_kfold : K-fold cross-validation
    waic : Compute WAIC
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
            f"Using {method_name} for LOO computation. Note that PSIS is the"
            " recommended method as it is typically more efficient and reliable.",
            UserWarning,
            stacklevel=2,
        )

    log_weights, diagnostic = compute_importance_weights(
        -log_likelihood, method=method, reff=reff
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
                " and LOO posterior are very different. If you're using a PyMC model,"
                " consider using reloo() to compute exact LOO for these problematic"
                " observations, or moment matching to improve the estimates.",
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

        # Add subsample_size if needed
        result_data.append(n_data_points)
        result_index.append("subsample_size")

        result = ELPDData(data=result_data, index=result_index)

        # We can't do moment matching without pointwise values
        if moment_match:
            raise ValueError(
                "Moment matching requires pointwise LOO results. "
                "Please set pointwise=True when using moment_match=True."
            )

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

    if method == ISMethod.PSIS:
        result_data.append(diagnostic)
        result_index.append("pareto_k")
        result_data.append(good_k)
        result_index.append("good_k")
    else:
        result_data.append(diagnostic)
        result_index.append("ess")

    # Add subsample_size if needed
    result_data.append(n_data_points)
    result_index.append("subsample_size")

    result = ELPDData(data=result_data, index=result_index)

    # Moment matching
    if moment_match:
        wrapper = kwargs.get("wrapper", None)
        if wrapper is None:
            raise ValueError(
                "PyMC model wrapper must be provided when moment_match=True"
            )

        mm_kwargs = {
            "max_iters": kwargs.get("max_iters", 30),
            "k_threshold": kwargs.get("k_threshold", None),
            "split": kwargs.get("split", True),
            "cov": kwargs.get("cov", True),
            "method": method,
        }

        result = loo_moment_match(wrapper, result, **mm_kwargs)

    return result
