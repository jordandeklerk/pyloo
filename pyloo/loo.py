"""Leave-one-out cross-validation (LOO-CV) using importance sampling methods with modifications from ArviZ."""

import warnings

import numpy as np
from arviz.data import convert_to_inference_data
from arviz.stats.diagnostics import ess

from .elpd import ELPDData
from .importance_sampling import ISMethod, compute_importance_weights
from .rcparams import rcParams
from .utils import _logsumexp, get_log_likelihood, wrap_xarray_ufunc


def loo(data, pointwise=None, var_name=None, reff=None, scale=None, method="psis"):
    """Compute leave-one-out cross-validation (LOO-CV) using various importance sampling methods.

        Estimates the expected log pointwise predictive density (elpd) using importance sampling
        leave-one-out cross-validation. By default, uses Pareto-smoothed importance sampling (PSIS),
        which is the recommended method. Also calculates LOO's standard error and the effective
        number of parameters. Read more theory here https://arxiv.org/abs/1507.04544 and here
        https://arxiv.org/abs/1507.02646
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
        Calculate LOO of a model:

        .. ipython::

            In [1]: import pyloo as pl
               ...: import arviz as az
               ...: data = az.load_arviz_data("centered_eight")
               ...: pl.loo(data)

        Calculate LOO of a model and return the pointwise values:

        .. ipython::

            In [2]: data_loo = pl.loo(data, pointwise=True)
               ...: data_loo.loo_i
    """
    inference_data = convert_to_inference_data(data)
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
            reff = np.hstack([ess_p[v].values.flatten() for v in ess_p.data_vars]).mean() / n_samples

    has_nan = np.any(np.isnan(log_likelihood.values))
    has_inf = np.any(np.isinf(log_likelihood.values))

    if has_nan:
        warnings.warn(
            "NaN values detected in log-likelihood. These will be ignored in the LOO calculation.",
            UserWarning,
            stacklevel=2,
        )
        log_likelihood = log_likelihood.where(~np.isnan(log_likelihood), -1e10)

    if has_inf:
        warnings.warn(
            "Infinite values detected in log-likelihood. These will be ignored in the LOO calculation.",
            UserWarning,
            stacklevel=2,
        )
        log_likelihood = log_likelihood.where(
            ~np.isinf(log_likelihood), np.where(np.isinf(log_likelihood) & (log_likelihood > 0), 1e10, -1e10)
        )

    try:
        method = ISMethod(method.lower())
    except ValueError:
        raise ValueError(f"Invalid method '{method}'. Must be one of: {', '.join(m.value for m in ISMethod)}")

    if method != ISMethod.PSIS:
        warnings.warn(
            f"Using {method.value.upper()} for LOO computation. Note that PSIS is the recommended "
            "method as it is typically more efficient and reliable.",
            UserWarning,
            stacklevel=2,
        )

    log_weights, diagnostic = compute_importance_weights(-log_likelihood, method=method, reff=reff)
    log_weights += log_likelihood

    warn_mg = False
    good_k = min(1 - 1 / np.log10(n_samples), 0.7)

    if method == ISMethod.PSIS:
        if np.any(diagnostic > good_k):
            warnings.warn(
                f"Estimated shape parameter of Pareto distribution is greater than {good_k:.2f} "
                "for one or more samples. You should consider using a more robust model, this is "
                "because importance sampling is less likely to work well if the marginal posterior "
                "and LOO posterior are very different. This is more likely to happen with a "
                "non-robust model and highly influential observations.",
                UserWarning,
                stacklevel=2,
            )
            warn_mg = True
    else:
        # For SIS/TIS, warn if effective sample size is too low
        min_ess = np.min(diagnostic)
        if min_ess < n_samples * 0.1:
            warnings.warn(
                f"Low effective sample size detected (minimum ESS: {min_ess:.1f}). "
                "This indicates that the importance sampling approximation may be unreliable. "
                "Consider using PSIS which is more robust to such cases.",
                UserWarning,
                stacklevel=2,
            )
            warn_mg = True

    ufunc_kwargs = {"n_dims": 1, "ravel": False}
    kwargs = {"input_core_dims": [["__sample__"]]}
    loo_lppd_i = scale_value * wrap_xarray_ufunc(_logsumexp, log_weights, ufunc_kwargs=ufunc_kwargs, **kwargs)
    loo_lppd = loo_lppd_i.values.sum()
    loo_lppd_se = (n_data_points * np.var(loo_lppd_i.values)) ** 0.5

    lppd = np.sum(
        wrap_xarray_ufunc(
            _logsumexp,
            log_likelihood,
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

    if np.allclose(loo_lppd_i, loo_lppd_i[0]):
        warnings.warn(
            "The point-wise LOO is the same with the sum LOO, please double check "
            "the Observed RV in your model to make sure it returns element-wise logp.",
            stacklevel=2,
        )

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
