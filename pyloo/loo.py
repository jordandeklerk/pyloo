"""Leave-one-out cross-validation (LOO-CV) using importance sampling methods."""

import warnings
from typing import Any, Literal

import numpy as np
import xarray as xr
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
    jacobian: np.ndarray | None = None,
    mixture: bool = False,
    **kwargs,
) -> ELPDData:
    r"""Compute leave-one-out cross-validation (LOO-CV) using various importance sampling methods.

    Estimates the expected log pointwise predictive density (elpd) using importance sampling
    leave-one-out cross-validation. By default, uses Pareto-smoothed importance sampling (PSIS),
    which is the recommended method. Also calculates LOO's standard error and the effective
    number of parameters.

    Parameters
    ----------
    data: InferenceData | Any
        Any object that can be converted to an :class:`arviz.InferenceData` object.
        Refer to documentation of
        :func:`arviz.convert_to_dataset` for details.
    pointwise: bool | None
        If True the pointwise predictive accuracy will be returned. Defaults to
        ``stats.ic_pointwise`` rcParam.
    var_name : str | None
        The name of the variable in log_likelihood groups storing the pointwise log
        likelihood data to use for loo computation.
    reff: float | None
        Relative MCMC efficiency, ``ess / n`` i.e. number of effective samples divided by the number
        of actual samples. Computed from trace by default.
    scale: str | None
        Output scale for loo. Available options are:
    method: Literal['psis', 'sis', 'tis'] | ISMethod
        The importance sampling method to use:
        - 'psis': Pareto Smoothed Importance Sampling (recommended)
        - 'sis': Standard Importance Sampling
        - 'tis': Truncated Importance Sampling
    scale : str | None
        Output scale for LOO. Available options are:
        - "log": (default) log-score
        - "negative_log": -1 * log-score
        - "deviance": -2 * log-score
    moment_match: bool
        Whether to perform moment matching to improve the LOO estimates for observations with
        high Pareto k values. If True, the `wrapper` parameter must be provided in kwargs.
    jacobian: array-like | None
        Adjustment for the Jacobian of a transformation applied to the response variable.
    mixture: bool
        Whether the log likelihood is from a mixture posterior. If True, the Mix-IS-LOO
        ``elpd_loo`` is calculated using the numerically stable mixture importance sampling approach
        outlined in Appendix A.2 of Silva and Zanella (2022).
    **kwargs:
        Additional keyword arguments for moment matching.

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
    good_k: For PSIS method and sample size :math:`S`, threshold computed as :math:`\\min(1 - 1/\\log_{10}(S), 0.7)`

    Notes
    -----
    For problematic observations, it is recommended to first use moment matching through `loo`,
    or directly through `loo_moment_match`. Moment matching through `loo` can be achieved by
    either passing a model wrapper for PyMC models and specifying the `moment_match` parameter
    to `True`, or by passing additional keyword arguments via the `**kwargs` parameter.
    When using a custom model, you can pass the model object via the `model_obj` keyword argument
    and the required functions via the other keyword arguments such as `post_draws`, `log_lik_i`,
    `unconstrain_pars`, `log_prob_upars_fn`, and `log_lik_i_upars_fn`.

    Moment matching can fail to improve LOO estimates for several reasons. If moment matching does
    not improve the LOO estimates and there are a moderate number of problematic observations,
    we suggest using the `reloo` function to compute exact LOO-CV if the model is a PyMC model, or
    using k-fold cross-validation for a more robust estimate.

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

    Calculate LOO with moment matching for a PyMC model to potentially improve estimates for
    observations with high Pareto k values:

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
    loo_subsample : Leave-one-out cross-validation with subsampling
    loo_moment_match : Leave-one-out cross-validation with moment matching
    loo_kfold : K-fold cross-validation
    loo_approximate_posterior : Leave-one-out cross-validation for posterior approximations
    loo_score : Compute LOO score for continuous ranked probability score
    loo_group : Leave-one-group-out cross-validation
    loo_nonfactor : Leave-one-out cross-validation for non-factorized models
    waic : Compute WAIC
    """
    inference_data = to_inference_data(data)
    log_likelihood = get_log_likelihood(inference_data, var_name=var_name)
    pointwise = rcParams["stats.ic_pointwise"] if pointwise is None else pointwise

    if jacobian is not None and not pointwise:
        raise ValueError(
            "Jacobian adjustment requires pointwise LOO results. "
            "Please set pointwise=True when using jacobian_adjustment."
        )

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

    ufunc_kwargs = {"n_dims": 1, "ravel": False}
    xarray_kwargs = {"input_core_dims": [["__sample__"]]}

    good_k = min(1 - 1 / np.log10(n_samples), 0.7)
    warn_mg = False

    if mixture:
        warnings.warn(
            "Mix-IS-LOO requires a model that is sampled from a mixture of"
            " leave-one-out posteriors. Ensure the inference data passed to the `loo`"
            " function comes from a model that is sampled from such a distribution.",
            UserWarning,
            stacklevel=2,
        )

        l_common_mix = wrap_xarray_ufunc(
            _logsumexp,
            -log_likelihood,
            ufunc_kwargs=ufunc_kwargs,
            input_core_dims=[["__sample__"]],
        )

        neg_log_likelihood = -log_likelihood
        log_weights = neg_log_likelihood.copy()

        log_weights = neg_log_likelihood - l_common_mix.values[:, np.newaxis]
        log_norm_const = _logsumexp(-l_common_mix.values)

        log_obs_weights = np.array([_logsumexp(row.values) for row in log_weights])
        elpd_mixis = log_norm_const - log_obs_weights

        diagnostic = np.zeros(log_likelihood.shape[0])
        log_weights = np.zeros_like(log_likelihood)

        loo_lppd_i = scale_value * xr.DataArray(
            elpd_mixis,
            dims=log_likelihood.dims[0],
            coords={log_likelihood.dims[0]: log_likelihood[log_likelihood.dims[0]]},
        )
    else:
        log_weights, diagnostic = compute_importance_weights(
            -log_likelihood, method=method, reff=reff
        )
        log_weights += log_likelihood

        if method == ISMethod.PSIS:
            if np.any(diagnostic > good_k):
                n_high_k = np.sum(diagnostic > good_k)

                warnings.warn(
                    "Estimated shape parameter of Pareto distribution is greater than"
                    f" {good_k:.2f} for {n_high_k} observations. This indicates that"
                    " importance sampling may be unreliable because the marginal"
                    " posterior and LOO posterior are very different.",
                    UserWarning,
                    stacklevel=2,
                )

                warn_mg = True
        else:
            min_ess = np.min(diagnostic)
            if min_ess < n_samples * 0.1:
                warnings.warn(
                    f"Low effective sample size detected (minimum ESS: {min_ess:.1f})."
                    " This indicates that the importance sampling approximation may be"
                    " unreliable. Consider using PSIS which is more robust to such"
                    " cases.",
                    UserWarning,
                    stacklevel=2,
                )

                warn_mg = True

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
    p_loo_se = np.sqrt(np.sum(np.var(loo_lppd_i.values)))
    looic = -2 * loo_lppd
    looic_se = 2 * loo_lppd_se

    if not pointwise:
        result_data, result_index = _get_result_data_and_index(
            mixture=mixture,
            loo_lppd=loo_lppd,
            loo_lppd_se=loo_lppd_se,
            p_loo=p_loo,
            p_loo_se=p_loo_se,
            n_samples=n_samples,
            n_data_points=n_data_points,
            warn_mg=warn_mg,
            scale=scale,
            looic=looic,
            looic_se=looic_se,
            pointwise=False,
        )

        if method == ISMethod.PSIS:
            result_data.append(good_k)
            result_index.append("good_k")

        result_data.append(n_data_points)
        result_index.append("subsample_size")

        result = ELPDData(data=result_data, index=result_index)

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

    result_data, result_index = _get_result_data_and_index(
        mixture=mixture,
        loo_lppd=loo_lppd,
        loo_lppd_se=loo_lppd_se,
        p_loo=p_loo,
        p_loo_se=p_loo_se,
        n_samples=n_samples,
        n_data_points=n_data_points,
        warn_mg=warn_mg,
        scale=scale,
        looic=looic,
        looic_se=looic_se,
        loo_lppd_i=loo_lppd_i,
        pointwise=True,
    )

    if method == ISMethod.PSIS:
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

    if jacobian is not None:
        jacobian_adj = np.asarray(jacobian)

        if jacobian_adj.shape != result.loo_i.shape:
            raise ValueError(
                f"Jacobian adjustment shape {jacobian_adj.shape} does not match "
                f"loo_i shape {result.loo_i.shape}"
            )

        result.loo_i.values = result.loo_i.values + jacobian_adj

        loo_lppd = result.loo_i.values.sum()
        loo_lppd_se = (n_data_points * np.var(result.loo_i.values)) ** 0.5

        p_loo = lppd - loo_lppd / scale_value
        p_loo_se = np.sqrt(np.sum(np.var(result.loo_i.values)))

        looic = -2 * loo_lppd
        looic_se = 2 * loo_lppd_se

        result["elpd_loo"] = loo_lppd
        result["se"] = loo_lppd_se
        result["p_loo"] = p_loo
        result["p_loo_se"] = p_loo_se
        result["looic"] = looic
        result["looic_se"] = looic_se

    if moment_match:
        from .wrapper.pymc.pymc import PyMCWrapper

        wrapper = kwargs.get("wrapper", None)
        model_obj = wrapper

        mm_kwargs = {
            "max_iters": kwargs.get("max_iters", 30),
            "k_threshold": kwargs.get("k_threshold", None),
            "split": kwargs.get("split", True),
            "cov": kwargs.get("cov", True),
            "method": method,
            "verbose": kwargs.get("verbose", False),
        }

        if wrapper is None:
            # If no wrapper, expect the custom model object via `model_obj` kwarg
            # and extract required custom functions from kwargs
            model_obj = kwargs.get("model_obj", None)
            if model_obj is None:
                raise ValueError(
                    "When moment_match=True and no `wrapper` is provided, the custom "
                    "model object must be passed via the `model_obj` keyword argument."
                )

            custom_funcs = {
                "post_draws": kwargs.get("post_draws", None),
                "log_lik_i": kwargs.get("log_lik_i", None),
                "unconstrain_pars": kwargs.get("unconstrain_pars", None),
                "log_prob_upars_fn": kwargs.get("log_prob_upars_fn", None),
                "log_lik_i_upars_fn": kwargs.get("log_lik_i_upars_fn", None),
            }
            mm_kwargs.update(custom_funcs)

            missing_funcs = [
                name for name, func in custom_funcs.items() if func is None
            ]
            if missing_funcs:
                raise ValueError(
                    "When moment_match=True and no `wrapper` is provided, the"
                    " following functions must be passed via kwargs:"
                    f" {', '.join(missing_funcs)}"
                )
        elif not isinstance(wrapper, PyMCWrapper):
            raise TypeError(
                "The `wrapper` argument must be an instance of PyMCWrapper when"
                " provided."
            )

        handled_kwargs = set(mm_kwargs.keys())
        handled_kwargs.update({
            "wrapper",
            "pointwise",
            "var_name",
            "reff",
            "scale",
            "method",
            "moment_match",
            "jacobian",
            "mixture",
            "model_obj",
            "post_draws",
            "log_lik_i",
            "unconstrain_pars",
            "log_prob_upars_fn",
            "log_lik_i_upars_fn",
        })
        additional_kwargs = {k: v for k, v in kwargs.items() if k not in handled_kwargs}
        mm_kwargs.update(additional_kwargs)

        result = loo_moment_match(model_obj, result, **mm_kwargs)

    return result


def _get_result_data_and_index(
    mixture: bool,
    loo_lppd: float,
    loo_lppd_se: float,
    p_loo: float | None = None,
    p_loo_se: float | None = None,
    n_samples: int | None = None,
    n_data_points: int | None = None,
    warn_mg: bool | None = None,
    scale: str | None = None,
    looic: float | None = None,
    looic_se: float | None = None,
    loo_lppd_i: xr.DataArray | None = None,
    pointwise: bool = False,
) -> tuple[list[Any], list[str]]:
    """Helper function to create result data and index based on mixture flag."""
    result_data: list[Any] = []
    result_index: list[str] = []

    if not pointwise:
        if mixture:
            result_data = [
                loo_lppd,
                loo_lppd_se,
                n_samples,
                n_data_points,
                warn_mg,
                scale,
            ]
            result_index = [
                "elpd_loo",
                "se",
                "n_samples",
                "n_data_points",
                "warning",
                "scale",
            ]
        else:
            result_data = [
                loo_lppd,
                loo_lppd_se,
                p_loo,
                p_loo_se,
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
                "p_loo_se",
                "n_samples",
                "n_data_points",
                "warning",
                "scale",
                "looic",
                "looic_se",
            ]
    else:
        if mixture:
            result_data = [
                loo_lppd,
                loo_lppd_se,
                n_samples,
                n_data_points,
                warn_mg,
                loo_lppd_i.rename("loo_i") if loo_lppd_i is not None else None,
                scale,
            ]
            result_index = [
                "elpd_loo",
                "se",
                "n_samples",
                "n_data_points",
                "warning",
                "loo_i",
                "scale",
            ]
        else:
            result_data = [
                loo_lppd,
                loo_lppd_se,
                p_loo,
                p_loo_se,
                n_samples,
                n_data_points,
                warn_mg,
                loo_lppd_i.rename("loo_i") if loo_lppd_i is not None else None,
                scale,
                looic,
                looic_se,
            ]
            result_index = [
                "elpd_loo",
                "se",
                "p_loo",
                "p_loo_se",
                "n_samples",
                "n_data_points",
                "warning",
                "loo_i",
                "scale",
                "looic",
                "looic_se",
            ]

    return result_data, result_index
