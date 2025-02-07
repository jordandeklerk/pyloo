"""Leave-one-out cross-validation (LOO-CV) using importance sampling methods with modifications from ArviZ."""

import warnings
from typing import Literal, Optional

import numpy as np
from arviz import convert_to_inference_data, ess, mcse

from .diagnostics import ParetokTable, k_cut
from .elpd import ELPDData
from .psis import psislw
from .utils import _logsumexp, get_log_likelihood


# Mainly taken from https://github.com/arviz-devs/arviz/blob/main/arviz/stats/stats.py
def loo(
    data,
    pointwise: Optional[bool] = None,
    var_name: Optional[str] = None,
    reff: Optional[float] = None,
    scale: Optional[str] = None,
    method: Literal["psis", "tis", "sis"] = "psis",
    verbose: bool = False,
) -> ELPDData:
    """Compute leave-one-out cross-validation using importance sampling.

    Estimates the expected log pointwise predictive density (elpd) using importance
    sampling leave-one-out cross-validation. By default, uses Pareto smoothed importance
    sampling (PSIS-LOO-CV), which is recommended for most cases. Also calculates LOO's
    standard error and the effective number of parameters.

    Parameters
    ----------
    data : obj
        Any object that can be converted to an ArviZ InferenceData object.
        Refer to documentation of arviz.convert_to_dataset for details.
    pointwise : bool, optional
        If True the pointwise predictive accuracy will be returned.
        Defaults to False.
    var_name : str, optional
        The name of the variable in log_likelihood groups storing the pointwise log
        likelihood data to use for loo computation.
    reff : float, optional
        Relative MCMC efficiency, ``ess / n`` i.e. number of effective samples divided by the number
        of actual samples. Computed from trace by default.
    scale : str, optional
        Output scale for loo. Available options are:
        - "log" : (default) log-score
        - "negative_log" : -1 * log-score
        - "deviance" : -2 * log-score
        A higher log-score (or a lower deviance or negative log_score) indicates a model with
        better predictive accuracy.
    method : {"psis", "tis", "sis"}, optional
        The importance sampling method to use:
        - "psis" (default): Pareto Smoothed Importance Sampling (recommended)
        - "tis": Truncated Importance Sampling
        - "sis": Standard Importance Sampling
        PSIS is generally more robust and efficient than other methods.
    verbose : bool, optional
        If True, includes additional PSIS diagnostic information in the output,
        including counts and proportions of observations in different Pareto k
        categories and minimum effective sample sizes. Defaults to False.

    Returns
    -------
    ELPDData
        Object containing:
        - elpd_loo: approximated expected log pointwise predictive density (elpd)
        - se: standard error of the elpd
        - p_loo: effective number of parameters
        - n_samples: number of samples
        - n_data_points: number of data points
        - warning: bool indicating if any Pareto k values were too high
        - loo_i: pointwise predictive accuracy (if pointwise=True)
        - pareto_k: array of Pareto shape values (if pointwise=True)
        - scale: scale of the elpd
        - good_k: threshold for Pareto k values (only for PSIS)
        - mcse_elpd: Monte Carlo standard errors for each observation
        - influence_pareto_k: Pareto k values indicating observation influence
        - pareto_table: detailed PSIS diagnostics table (if verbose=True)

    Notes
    -----
    PSIS is the recommended method as it provides:
    1. Better variance reduction through Pareto smoothing
    2. Diagnostic information via Pareto k values
    3. More reliable estimates in most cases

    Other methods (TIS, SIS) are provided for comparison and special cases,
    but should be used with caution.

    Returns
    -------
    ELPDData
        Object containing:
        - elpd_loo: approximated expected log pointwise predictive density (elpd)
        - se: standard error of the elpd
        - p_loo: effective number of parameters
        - n_samples: number of samples
        - n_data_points: number of data points
        - warning: bool indicating if any Pareto k values were too high
        - loo_i: pointwise predictive accuracy (if pointwise=True)
        - pareto_k: array of Pareto shape values (if pointwise=True)
        - scale: scale of the elpd
        - good_k: threshold for Pareto k values

    Examples
    --------
    Calculate LOO using PSIS (recommended default):

    .. ipython::

        In [1]: import arviz as az
           ...: import pyloo as pl
           ...: data = az.load_arviz_data("centered_eight")
           ...: loo_psis = pl.loo(data)  # Uses PSIS by default
           ...: print(loo_psis)

    Calculate LOO with pointwise values:

    .. ipython::

        In [2]: loo_with_pointwise = pl.loo(data, pointwise=True)
           ...: print(loo_with_pointwise.loo_i)  # Show pointwise contributions
           ...: print(loo_with_pointwise.pareto_k)  # Show Pareto k diagnostics

    Using different importance sampling methods:

    .. ipython::

        In [3]: # Compare different methods (PSIS recommended for most cases)
           ...: loo_psis = pl.loo(data, method="psis")
           ...: loo_tis = pl.loo(data, method="tis")
           ...: loo_sis = pl.loo(data, method="sis")
           ...: print("PSIS:", loo_psis.elpd_loo)
           ...: print("TIS:", loo_tis.elpd_loo)
           ...: print("SIS:", loo_sis.elpd_loo)

    Using different scales:

    .. ipython::

        In [4]: # Compare different scales
           ...: loo_log = pl.loo(data, scale="log")  # default
           ...: loo_dev = pl.loo(data, scale="deviance")
           ...: print("Log scale:", loo_log.elpd_loo)
           ...: print("Deviance:", loo_dev.elpd_loo)

    See Also
    --------
    psis : Pareto Smoothed Importance Sampling implementation.
    tis : Truncated Importance Sampling implementation.
    sis : Standard Importance Sampling implementation.

    References
    ----------
    .. [1] Vehtari, A., Gelman, A., & Gabry, J. (2017). Practical Bayesian model
        evaluation using leave-one-out cross-validation and WAIC. Statistics and
        Computing, 27(5), 1413-1432.
    .. [2] Vehtari, A., Simpson, D., Gelman, A., Yao, Y., & Gabry, J. (2024).
        Pareto smoothed importance sampling. Journal of Machine Learning Research.
    """
    inference_data = convert_to_inference_data(data)
    log_likelihood = get_log_likelihood(inference_data, var_name=var_name)
    log_likelihood = log_likelihood.stack(__sample__=("chain", "draw"))

    shape = log_likelihood.shape
    n_samples = shape[-1]
    n_data_points = np.prod(shape[:-1])

    scale = scale.lower() if scale is not None else "log"
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

    log_weights, pareto_shape, _ = psislw(-log_likelihood, reff)
    log_weights += log_likelihood

    warn_mg = False
    good_k = None

    if method == "psis":
        good_k = min(1 - 1 / np.log10(n_samples), 0.7)
        if np.any(pareto_shape > good_k):
            warnings.warn(
                f"Estimated shape parameter of Pareto distribution is greater than {good_k:.2f} "
                "for one or more samples. You should consider using a more robust model, this is "
                "because importance sampling is less likely to work well if the marginal posterior "
                "and LOO posterior are very different. This is more likely to happen with a "
                "non-robust model and highly influential observations.",
                stacklevel=2,
            )
            warn_mg = True
    else:
        good_k = 0.0

    loo_lppd_i = scale_value * _logsumexp(log_weights)
    loo_lppd = loo_lppd_i.values.sum()
    loo_lppd_se = (n_data_points * np.var(loo_lppd_i.values)) ** 0.5

    lppd = np.sum(_logsumexp(log_likelihood).values)
    p_loo = lppd - loo_lppd / scale_value

    mcse_elpd_i = mcse(log_likelihood, method="mean").values
    influence_k = pareto_shape.copy()

    pareto_table = None
    if verbose and method == "psis":
        ess_values = np.exp(2 * _logsumexp(log_weights))
        k_categories = k_cut(pareto_shape, good_k)

        counts = np.zeros(3, dtype=int)
        proportions = np.zeros(3)
        min_ess = np.full(3, np.nan)

        for cat in range(3):
            mask = k_categories == cat
            if np.any(mask):
                counts[cat] = np.sum(mask)
                proportions[cat] = np.sum(mask) / len(pareto_shape)
                min_ess[cat] = np.nanmin(ess_values[mask])

        pareto_table = ParetokTable(counts=counts, proportions=proportions, min_ess=min_ess, k_threshold=good_k)

    if not pointwise:
        data = [
            loo_lppd,
            loo_lppd_se,
            p_loo,
            n_samples,
            n_data_points,
            warn_mg,
            scale,
            good_k,
            mcse_elpd_i,
            influence_k,
        ]
        index = [
            "elpd_loo",
            "se",
            "p_loo",
            "n_samples",
            "n_data_points",
            "warning",
            "scale",
            "good_k",
            "mcse_elpd",
            "influence_pareto_k",
        ]
        if verbose and pareto_table is not None:
            data.append(pareto_table)
            index.append("pareto_table")

        return ELPDData(data=data, index=index)

    if np.equal(loo_lppd, loo_lppd_i).all():
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
        pareto_shape,
        scale,
        good_k,
        mcse_elpd_i,
        influence_k,
    ]
    index = [
        "elpd_loo",
        "se",
        "p_loo",
        "n_samples",
        "n_data_points",
        "warning",
        "loo_i",
        "pareto_k",
        "scale",
        "good_k",
        "mcse_elpd",
        "influence_pareto_k",
    ]

    if verbose and pareto_table is not None:
        data.append(pareto_table)
        index.append("pareto_table")

    return ELPDData(data=data, index=index)
