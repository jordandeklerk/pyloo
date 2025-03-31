"""Leave-one-group-out cross-validation (LOGO-CV) using importance sampling methods."""

import warnings
from typing import Any, Literal

import numpy as np
import xarray as xr
from arviz.data import InferenceData
from arviz.stats.diagnostics import ess

from .base import ISMethod, compute_importance_weights
from .elpd import ELPDData
from .rcparams import rcParams
from .utils import _logsumexp, get_log_likelihood, to_inference_data, wrap_xarray_ufunc

__all__ = ["loo_group"]


def loo_group(
    data: InferenceData | Any,
    group_ids: np.ndarray,
    pointwise: bool | None = None,
    var_name: str | None = None,
    reff: float | None = None,
    scale: str | None = None,
    method: ISMethod | Literal["psis", "sis", "tis"] = "psis",
) -> ELPDData:
    """Compute leave-one-group-out cross-validation (LOGO-CV) using importance sampling methods.

    Parameters
    ----------
    data: InferenceData | Any
        Any object that can be converted to an :class:`arviz.InferenceData` object.
        Refer to documentation of :func:`arviz.convert_to_dataset` for details.
    group_ids: np.ndarray
        Array of group identifiers for each observation. Must have the same length as
        the number of observations in the log likelihood.
    pointwise: bool | None, optional
        If True the pointwise predictive accuracy will be returned. Defaults to
        ``stats.ic_pointwise`` rcParam.
    var_name : str, optional
        The name of the variable in log_likelihood groups storing the pointwise log
        likelihood data to use for loo computation.
    reff: float, optional
        Relative MCMC efficiency, ``ess / n`` i.e. number of effective samples divided by the number
        of actual samples. Computed from trace by default.
    scale : str | None, default None
        Output scale for LOO_group. Available options are:
        - "log": (default) log-score
        - "negative_log": -1 * log-score
        - "deviance": -2 * log-score
    method: Literal['psis', 'sis', 'tis'] | ISMethod, default 'psis'
        The importance sampling method to use:
        - 'psis': Pareto Smoothed Importance Sampling (recommended)
        - 'sis': Standard Importance Sampling
        - 'tis': Truncated Importance Sampling

    Returns
    -------
    ELPDData object (inherits from :class:`pandas.Series`) with the following row/attributes:
    elpd_logo: approximated expected log pointwise predictive density (elpd)
    se: standard error of the elpd
    p_logo: effective number of parameters
    n_samples: number of samples
    n_groups: number of groups
    warning: bool
        True if using PSIS and the estimated shape parameter of Pareto distribution
        is greater than ``good_k``.
    logo_i: :class:`~xarray.DataArray` with the group-wise predictive accuracy,
            only if pointwise=True
    diagnostic: array of diagnostic values, only if pointwise=True
        - For PSIS: Pareto shape parameter (pareto_k)
        - For SIS/TIS: Effective sample size (ess)
    scale: scale of the elpd
    good_k: For PSIS method and sample size :math:`S`, threshold computed as :math:`\\min(1 - 1/\\log_{10}(S), 0.7)`

    Notes
    -----
    This function should be used with caution. In practice, leaving out several observations
    from a group may change the posterior too much for importance sampling to work. In this case,
    it is recommended to use quadrature integration to compute the exact LOO-CV.

    Examples
    --------
    Calculate the Leave-One-Group-Out Cross-Validation (LOGO-CV) for a model by providing InferenceData
    and group identifiers

    .. code-block:: python

        import pyloo as pl
        import arviz as az
        import numpy as np

        data = az.load_arviz_data("radon")
        group_ids = data.observed_data.county_idx.values

        logo_results = pl.loo_group(data, group_ids)
        print(logo_results)

    Calculate LOGO with pointwise values to examine individual group contributions

    .. code-block:: python

        import pyloo as pl
        import arviz as az
        import numpy as np

        data = az.load_arviz_data("radon")
        group_ids = data.observed_data.county_idx.values

        logo_results = pl.loo_group(data, group_ids, pointwise=True)
        print(logo_results.logo_i)
        print(logo_results.pareto_k)

    See Also
    --------
    loo_subsample : Leave-one-out cross-validation with subsampling
    loo_moment_match : Leave-one-out cross-validation with moment matching
    loo_kfold : K-fold cross-validation
    loo_approximate_posterior : Leave-one-out cross-validation for posterior approximations
    loo_score : Compute LOO score for continuous ranked probability score
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

    if len(group_ids) != n_data_points:
        raise ValueError(
            f"Length of group_ids ({len(group_ids)}) must match the number of "
            f"observations in log_likelihood ({n_data_points})."
        )

    unique_groups = np.unique(group_ids)
    n_groups = len(unique_groups)

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
            "NaN values detected in log-likelihood. These will be ignored in the LOGO "
            "calculation.",
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
            f"Using {method_name} for LOGO computation. Note that PSIS is the "
            "recommended method as it is typically more efficient and reliable.",
            UserWarning,
            stacklevel=2,
        )

    # Aggregate log-likelihoods by group
    group_log_liks = []
    for group in unique_groups:
        group_mask = group_ids == group
        # Sum log-likelihoods within each group
        group_log_lik = log_likelihood.values[group_mask].sum(axis=0)
        group_log_liks.append(group_log_lik)

    group_log_liks = np.array(group_log_liks)
    log_weights_list = []
    diagnostic_list = []

    for _, group_log_lik in enumerate(group_log_liks):
        # Compute importance weights for leaving out this group
        log_weights, diagnostic = compute_importance_weights(
            -group_log_lik, method=method, reff=reff
        )
        log_weights_list.append(log_weights + group_log_lik)
        diagnostic_list.append(diagnostic)

    log_weights = np.array(log_weights_list)
    diagnostics = np.array(diagnostic_list)

    warn_mg = False
    good_k = min(1 - 1 / np.log10(n_samples), 0.7)

    if method == ISMethod.PSIS:
        if np.any(diagnostics > good_k):
            n_high_k = np.sum(diagnostics > good_k)

            warnings.warn(
                "Estimated shape parameter of Pareto distribution is greater than "
                f"{good_k:.2f} for {n_high_k} groups. This indicates that "
                "importance sampling may be unreliable because the marginal posterior "
                "and LOGO posterior are very different.",
                UserWarning,
                stacklevel=2,
            )

            warn_mg = True
    else:
        min_ess = np.min(diagnostics)
        if min_ess < n_samples * 0.1:
            warnings.warn(
                f"Low effective sample size detected (minimum ESS: {min_ess:.1f}). This"
                " indicates that the importance sampling approximation may be"
                " unreliable. Consider using PSIS which is more robust to such cases.",
                UserWarning,
                stacklevel=2,
            )

            warn_mg = True

    log_weights_xr = xr.DataArray(
        log_weights, dims=["group", "__sample__"], coords={"group": unique_groups}
    )

    ufunc_kwargs = {"n_dims": 1, "ravel": False}
    xarray_kwargs = {"input_core_dims": [["__sample__"]]}

    logo_lppd_i_xr = scale_value * wrap_xarray_ufunc(
        _logsumexp,
        log_weights_xr,
        ufunc_kwargs=ufunc_kwargs,
        **xarray_kwargs,
    )
    logo_lppd_i_xr = logo_lppd_i_xr.rename("logo_i")
    logo_lppd = logo_lppd_i_xr.values.sum()
    logo_lppd_se = (n_groups * np.var(logo_lppd_i_xr.values)) ** 0.5

    group_log_liks_xr = xr.DataArray(
        group_log_liks, dims=["group", "__sample__"], coords={"group": unique_groups}
    )

    group_lppd_xr = wrap_xarray_ufunc(
        _logsumexp,
        group_log_liks_xr,
        func_kwargs={"b_inv": n_samples},
        ufunc_kwargs=ufunc_kwargs,
        **xarray_kwargs,
    )

    lppd = group_lppd_xr.values.sum()

    p_logo = lppd - logo_lppd / scale_value
    p_logo_se = np.sqrt(np.sum(np.var(logo_lppd_i_xr.values)))
    logoic = -2 * logo_lppd
    logoic_se = 2 * logo_lppd_se

    result_data: list[Any] = []
    result_index: list[str] = []

    if not pointwise:
        result_data = [
            logo_lppd,
            logo_lppd_se,
            p_logo,
            p_logo_se,
            n_samples,
            n_groups,
            warn_mg,
            scale,
            logoic,
            logoic_se,
        ]
        result_index = [
            "elpd_logo",
            "se",
            "p_logo",
            "p_logo_se",
            "n_samples",
            "n_groups",
            "warning",
            "scale",
            "logoic",
            "logoic_se",
        ]

        if method == ISMethod.PSIS:
            result_data.append(good_k)
            result_index.append("good_k")

        result = ELPDData(data=result_data, index=result_index)

        return result

    result_data = [
        logo_lppd,
        logo_lppd_se,
        p_logo,
        p_logo_se,
        n_samples,
        n_groups,
        warn_mg,
        logo_lppd_i_xr,
        scale,
        logoic,
        logoic_se,
    ]
    result_index = [
        "elpd_logo",
        "se",
        "p_logo",
        "p_logo_se",
        "n_samples",
        "n_groups",
        "warning",
        "logo_i",
        "scale",
        "logoic",
        "logoic_se",
    ]

    if method == ISMethod.PSIS:
        result_data.append(diagnostics)
        result_index.append("pareto_k")
        result_data.append(good_k)
        result_index.append("good_k")
    else:
        result_data.append(diagnostics)
        result_index.append("ess")

    result = ELPDData(data=result_data, index=result_index)

    return result
