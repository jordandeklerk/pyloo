"""Widely applicable information criterion (WAIC) based on Arviz."""

import warnings
from typing import Any

import numpy as np
from arviz.data import InferenceData

from .elpd import ELPDData
from .rcparams import rcParams
from .utils import _logsumexp, get_log_likelihood, to_inference_data, wrap_xarray_ufunc


def waic(
    data: InferenceData | Any,
    pointwise: bool | None = None,
    var_name: str | None = None,
    scale: str | None = None,
) -> ELPDData:
    """Compute the widely applicable information criterion (WAIC).

    Estimates the expected log pointwise predictive density (elpd) using WAIC.
    Also calculates the WAIC's standard error and the effective number of parameters.

    Parameters
    ----------
    data: obj
        Any object that can be converted to an :class:`arviz.InferenceData` object.
        Refer to documentation of arviz.convert_to_dataset for details.
    pointwise: bool, optional
        If True the pointwise predictive accuracy will be returned. Defaults to
        stats.ic_pointwise rcParam.
    var_name : str, optional
        The name of the variable in log_likelihood groups storing the pointwise log
        likelihood data to use for waic computation.
    scale: str, optional
        Output scale for WAIC. Available options are:
        - 'log' : (default) log-score
        - 'negative_log' : -1 * log-score
        - 'deviance' : -2 * log-score
        A higher log-score (or a lower deviance) indicates better predictive accuracy.

    Returns
    -------
    ELPDData object (inherits from pandas.Series) with the following row/attributes:
        elpd_waic: approximated expected log pointwise predictive density (elpd)
        se: standard error of the elpd
        p_waic: effective number of parameters
        n_samples: number of samples
        n_data_points: number of data points
        warning: bool
            True if posterior variance of the log predictive densities exceeds 0.4
        waic_i: xarray.DataArray with the pointwise predictive accuracy,
                only if pointwise=True
        scale: scale of the elpd

    See Also
    --------
    loo : Compute Pareto-smoothed importance sampling leave-one-out cross-validation.
    loo_subsample : Compute approximate LOO-CV using subsampling.
    compare : Compare models based on PSIS-LOO-CV or WAIC cross-validation.

    Examples
    --------
    Calculate WAIC of a model:

    .. ipython::

        In [1]: import arviz as az
           ...: import pyloo as pl
           ...: data = az.load_arviz_data("centered_eight")
           ...: pl.waic(data)

    Calculate WAIC of a model and return the pointwise values:

    .. ipython::

        In [2]: data_waic = pl.waic(data, pointwise=True)
           ...: data_waic.waic_i

    References
    ----------
    .. [1] Watanabe, S. (2010). Asymptotic equivalence of Bayes cross validation and
        widely application information criterion in singular learning theory.
        Journal of Machine Learning Research 11, 3571-3594.
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

    has_nan = np.any(np.isnan(log_likelihood.values))
    has_inf = np.any(np.isinf(log_likelihood.values))

    if has_nan:
        warnings.warn(
            "NaN values detected in log-likelihood. These will be ignored in the WAIC"
            " calculation.",
            UserWarning,
            stacklevel=2,
        )
        log_likelihood = log_likelihood.where(~np.isnan(log_likelihood), -1e10)

    if has_inf:
        warnings.warn(
            "Infinite values detected in log-likelihood. These will be ignored in the"
            " WAIC calculation.",
            UserWarning,
            stacklevel=2,
        )
        log_likelihood = log_likelihood.where(
            ~np.isinf(log_likelihood),
            np.where(np.isinf(log_likelihood) & (log_likelihood > 0), 1e10, -1e10),
        )

    ufunc_kwargs = {"n_dims": 1, "ravel": False}
    kwargs = {"input_core_dims": [["__sample__"]]}

    lppd_i = wrap_xarray_ufunc(
        _logsumexp,
        log_likelihood,
        func_kwargs={"b_inv": n_samples},
        ufunc_kwargs=ufunc_kwargs,
        **kwargs,
    )

    vars_lpd = log_likelihood.var(dim="__sample__")

    warn_mg = np.any(vars_lpd.values > 0.4)
    if warn_mg:
        warnings.warn(
            "For one or more samples the posterior variance of the log predictive "
            "densities exceeds 0.4. This could be indication of WAIC starting to fail.",
            UserWarning,
            stacklevel=2,
        )

    # Calculate WAIC components
    waic_i = scale_value * (lppd_i - vars_lpd)
    waic_se = (n_data_points * np.var(waic_i.values)) ** 0.5
    waic_sum = np.sum(waic_i.values)
    p_waic = np.sum(vars_lpd.values)

    if not pointwise:
        return ELPDData(
            data=[waic_sum, waic_se, p_waic, n_samples, n_data_points, warn_mg, scale],
            index=[
                "elpd_waic",
                "se",
                "p_waic",
                "n_samples",
                "n_data_points",
                "warning",
                "scale",
            ],
        )

    if np.allclose(waic_i, waic_i[0]):
        warnings.warn(
            "The point-wise WAIC is the same with the sum WAIC, please double check "
            "the Observed RV in your model to make sure it returns element-wise logp.",
            UserWarning,
            stacklevel=2,
        )

    waic_i_renamed = waic_i.rename("waic_i")

    return ELPDData(
        data=[
            waic_sum,
            waic_se,
            p_waic,
            n_samples,
            n_data_points,
            warn_mg,
            waic_i_renamed,
            scale,
        ],
        index=[
            "elpd_waic",
            "se",
            "p_waic",
            "n_samples",
            "n_data_points",
            "warning",
            "waic_i",
            "scale",
        ],
    )
