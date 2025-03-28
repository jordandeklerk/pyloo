"""Leave-one-out cross-validation with score functions using importance sampling."""

import warnings
from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np
import xarray as xr
from arviz.data import InferenceData

from .e_loo import e_loo
from .psis import psislw
from .rcparams import rcParams
from .utils import get_log_likelihood, to_inference_data

__all__ = ["loo_score", "LooScoreResult"]


@dataclass
class LooScoreResult:
    """Container for results from LOO-CRPS and LOO-SCRPS calculations.

    Attributes
    ----------
    estimates : np.ndarray
        A numpy array with named fields 'Estimate' and 'SE' containing the
        mean score and its standard error.
    pointwise : np.ndarray
        A numpy array with the pointwise score values.
    pareto_k : xr.DataArray, optional
        Pareto shape parameter diagnostic values. Only present if pointwise=True
        was specified in the loo_score function call.
    good_k : float, optional
        Threshold for good Pareto k values. Only present if pointwise=True
        was specified in the loo_score function call.
    warning : bool, optional
        Whether any Pareto k values exceed good_k. Only present if pointwise=True
        was specified in the loo_score function call.
    """

    estimates: np.ndarray
    pointwise: np.ndarray
    pareto_k: xr.DataArray | None = None
    good_k: float | None = None
    warning: bool | None = None


def loo_score(
    data: InferenceData | Any,
    x_group: str = "posterior_predictive",
    x_var: str | None = None,
    x2_group: str | None = None,
    x2_var: str | None = None,
    y_group: str = "observed_data",
    y_var: str | None = None,
    var_name: str | None = None,
    pointwise: bool | None = None,
    permutations: int = 1,
    reff: float | None = None,
    scale: bool = False,
    **kwargs,
) -> LooScoreResult:
    r"""Compute the leave-one-out continuously ranked probability score (LOO-CRPS) or
    leave-one-out scaled continuously ranked probability score (LOO-SCRPS).

    Parameters
    ----------
    data : InferenceData | Any
        Any object that can be converted to an :class:`arviz.InferenceData` object.
    x_group : str, default "posterior_predictive"
        Name of the InferenceData group containing the first set of predictive samples.
    x_var : str | None, optional
        Name of the variable in x_group to use. If None and there is only one variable,
        that variable will be used.
    x2_group : str | None, optional
        Name of the InferenceData group containing the second set of predictive samples.
        If None, uses the same group as x_group.
    x2_var : str | None, optional
        Name of the variable in x2_group to use. If None, uses the same variable as x_var.
    y_group : str, default "observed_data"
        Name of the InferenceData group containing the observed data.
    y_var : str | None, optional
        Name of the variable in y_group to use. If None and there is only one variable,
        that variable will be used.
    var_name : str | None, optional
        The name of the variable in log_likelihood groups storing the pointwise log
        likelihood data to use for LOO computation.
    pointwise : bool | None, optional
        If True, the pointwise values will be returned. Defaults to
        ``stats.ic_pointwise`` rcParam.
    permutations : int, default 1
        An integer specifying how many times the expected value of ::math:|X - X'|
        (::math:|x - x2|) is computed. The row order of x2 is shuffled as elements x and x2 are typically
        drawn given the same values of parameters. Generating more permutations is
        expected to decrease the variance of the computed expected value.
    reff : float | None, optional
        Relative MCMC efficiency, ``ess / n`` i.e. number of effective samples divided by the number
        of actual samples. Computed from trace by default.
    scale : bool, default False
        Whether to compute LOO-SCRPS (True) or LOO-CRPS (False).
    **kwargs
        Additional arguments passed to e_loo().

    Returns
    -------
    LooScoreResult
        Container with results from LOO-CRPS or LOO-SCRPS calculations:
        - estimates: A numpy array with named fields 'Estimate' and 'SE' containing the
          mean score and its standard error.
        - pointwise: A numpy array with the pointwise score values.
        - pareto_k: (Only if pointwise=True) Pareto shape parameter diagnostic values.
        - good_k: (Only if pointwise=True) Threshold for good Pareto k values.
        - warning: (Only if pointwise=True) Whether any Pareto k values exceed good_k.

    Notes
    -----
    To compute LOO-CRPS or LOO-SCRPS, the user needs to provide two sets of draws from the predictive
    distribution, along with the log-likelihood data.

    Examples
    --------
    Calculate LOO-CRPS for a model using InferenceData:

    .. code-block:: python

        import pyloo as pl
        import arviz as az
        import numpy as np

        idata = az.load_arviz_data("centered_eight")

        # For this example, we need two sets of predictive samples
        # In practice, these would typically be different
        pp_data = idata.posterior_predictive.copy()
        pp_data["obs2"] = pp_data["obs"] + np.random.normal(0, 0.1, size=pp_data["obs"].shape)

        # Create a new InferenceData object with both sets of predictive samples
        idata_with_two_pp = az.InferenceData(
            posterior=idata.posterior,
            posterior_predictive=pp_data,
            log_likelihood=idata.log_likelihood,
            observed_data=idata.observed_data
        )

        # LOO-CRPS
        result = pl.loo_score(
            idata_with_two_pp,
            x_group="posterior_predictive",
            x_var="obs",
            x2_group="posterior_predictive",
            x2_var="obs2",
            y_group="observed_data",
            y_var="obs"
        )

    Using multiple permutations to reduce variance:

    .. code-block:: python

        result = pl.loo_score(
            idata_with_two_pp,
            x_group="posterior_predictive",
            x_var="obs",
            x2_group="posterior_predictive",
            x2_var="obs2",
            y_group="observed_data",
            y_var="obs",
            permutations=5
        )

    See Also
    --------
    e_loo : Compute the expected value of a function using importance sampling.
    psislw : Compute importance sampling weights.

    References
    ----------
    Bolin, D., & Wallin, J. (2023). Local scale invariance and robustness of
    proper scoring rules. Statistical Science, 38(1):140-159.

    Gneiting, T., & Raftery, A. E. (2007). Strictly Proper Scoring Rules,
    Prediction, and Estimation. Journal of the American Statistical Association,
    102(477), 359–378.
    """
    inference_data = to_inference_data(data)
    log_likelihood = get_log_likelihood(inference_data, var_name=var_name)
    pointwise = rcParams["stats.ic_pointwise"] if pointwise is None else pointwise

    x_data, x2_data, y_data, log_likelihood = _get_data(
        inference_data,
        x_group=x_group,
        x_var=x_var,
        x2_group=x2_group,
        x2_var=x2_var,
        y_group=y_group,
        y_var=y_var,
        log_likelihood=log_likelihood,
    )

    _validate_crps_input(x_data, x2_data, y_data, log_likelihood)

    if reff is None:
        if not hasattr(inference_data, "posterior"):
            raise TypeError("Must be able to extract a posterior group from data.")
        posterior = inference_data.posterior
        n_chains = len(posterior.chain)
        n_samples = x_data.sizes["__sample__"]
        if n_chains == 1:
            reff = 1.0
        else:
            from arviz.stats.diagnostics import ess

            ess_p = ess(posterior, method="mean")
            reff = (
                np.hstack([ess_p[v].values.flatten() for v in ess_p.data_vars]).mean()
                / n_samples
            )

    repeats = [
        _compute_expected_diff_loo(
            x_data, x2_data, log_likelihood, r_eff=reff, **kwargs
        )
        for _ in range(permutations)
    ]
    EXX = sum(repeats) / permutations

    log_weights, pareto_k = psislw(-log_likelihood, reff=reff)

    # Compute EXy using e_loo
    abs_diff = abs(x_data - y_data)

    EXy = e_loo(
        abs_diff,
        log_weights=log_weights,
        log_ratios=-log_likelihood,
        **kwargs,
    ).value

    score_pw = _crps(EXX, EXy, scale=scale)

    score_value = float(score_pw.mean())
    score_se = float(score_pw.std() / np.sqrt(score_pw.size))

    estimates = np.array([score_value, score_se])
    estimates.setflags(write=True)
    estimates.dtype = np.dtype([("Estimate", float), ("SE", float)])

    result = LooScoreResult(
        estimates=estimates,
        pointwise=score_pw.values,
    )

    if pointwise:
        n_samples = x_data.sizes["__sample__"]
        good_k = min(1 - 1 / np.log10(n_samples), 0.7)

        result.pareto_k = pareto_k
        result.good_k = good_k

        if np.any(pareto_k > good_k):
            n_high_k = np.sum(pareto_k > good_k)
            warnings.warn(
                "Estimated shape parameter of Pareto distribution is greater than"
                f" {good_k:.2f} for {n_high_k} observations. This indicates that"
                " importance sampling may be unreliable because the marginal posterior"
                " and LOO posterior are very different.",
                UserWarning,
                stacklevel=2,
            )
            result.warning = True
        else:
            result.warning = False

    return result


def _compute_expected_diff_loo(
    x: xr.DataArray,
    x2: xr.DataArray,
    log_lik: xr.DataArray,
    r_eff: float = 1,
    **kwargs,
) -> xr.DataArray:
    """Compute the expected value of |X - X'| using LOO weights.

    Parameters
    ----------
    x : xr.DataArray
        First set of draws from the predictive distribution.
    x2 : xr.DataArray
        Second set of draws from the predictive distribution.
    log_lik : xr.DataArray
        Log-likelihood array.
    r_eff : float, default 1
        Relative effective sample size estimates.
    **kwargs
        Additional arguments passed to e_loo().

    Returns
    -------
    xr.DataArray
        Expected value of ::math: |X - X'| for each observation using LOO weights.
    """
    S = x.sizes["__sample__"]

    shuffle = np.random.permutation(S)

    x2_shuffled = x2.isel(__sample__=shuffle)
    log_lik2_shuffled = log_lik.isel(__sample__=shuffle)

    joint_log_lik = -log_lik - log_lik2_shuffled
    log_weights, _ = psislw(joint_log_lik, reff=r_eff)

    abs_diff = abs(x - x2_shuffled)

    result = e_loo(
        abs_diff,
        log_weights=log_weights,
        log_ratios=joint_log_lik,
        **kwargs,
    ).value

    return result


def _crps(EXX: xr.DataArray, EXy: xr.DataArray, scale: bool = False) -> xr.DataArray:
    r"""Function to compute CRPS and SCRPS.

    Parameters
    ----------
    EXX : xr.DataArray
        Expected value of ::math: |X - X'|.
    EXy : xr.DataArray
        Expected value of ::math: |X - y|.
    scale : bool, default False
        Whether to compute SCRPS (True) or CRPS (False).

    Returns
    -------
    xr.DataArray
        CRPS or SCRPS values.
    """
    if scale:
        return -EXy / EXX - 0.5 * np.log(EXX)
    else:
        return 0.5 * EXX - EXy


def _validate_crps_input(
    x: xr.DataArray, x2: xr.DataArray, y: xr.DataArray, log_lik: xr.DataArray = None
) -> None:
    """Validate input of CRPS functions.

    Check that predictive draws and observed data are of compatible shape.

    Parameters
    ----------
    x : xr.DataArray
        First set of predictive draws.
    x2 : xr.DataArray
        Second set of predictive draws.
    y : xr.DataArray
        Observed data.
    log_lik : xr.DataArray, optional
        Log-likelihood array.
    """
    if x.dims != x2.dims:
        raise ValueError("x and x2 must have the same dimensions")

    if x.shape != x2.shape:
        raise ValueError("x and x2 must have the same shape")

    if (
        np.isnan(x.values).any()
        or np.isnan(x2.values).any()
        or np.isnan(y.values).any()
    ):
        warnings.warn(
            "NaN values detected in input data. These may lead to unreliable results.",
            UserWarning,
            stacklevel=2,
        )

    if (
        np.isinf(x.values).any()
        or np.isinf(x2.values).any()
        or np.isinf(y.values).any()
    ):
        warnings.warn(
            "Infinite values detected in input data. These may lead to unreliable"
            " results.",
            UserWarning,
            stacklevel=2,
        )

    # We expect y to have all the same dimensions as x except for __sample__
    x_dims_without_sample = [d for d in x.dims if d != "__sample__"]
    y_dims = list(y.dims)

    if set(x_dims_without_sample) != set(y_dims):
        raise ValueError(
            f"y dimensions {y_dims} are not compatible with x dimensions {x.dims}"
        )

    if log_lik is not None:
        if "__sample__" not in log_lik.dims:
            raise ValueError("log_lik must have '__sample__' dimension")

        log_lik_dims_without_sample = [d for d in log_lik.dims if d != "__sample__"]
        if set(log_lik_dims_without_sample) != set(x_dims_without_sample):
            raise ValueError(
                f"log_lik dimensions {log_lik.dims} are not compatible with x"
                f" dimensions {x.dims}"
            )


def _get_data(
    inference_data: InferenceData,
    x_group: str = "posterior_predictive",
    x_var: str | None = None,
    x2_group: str | None = None,
    x2_var: str | None = None,
    y_group: str = "observed_data",
    y_var: str | None = None,
    log_likelihood: xr.DataArray | None = None,
) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray | None]:
    """Extract and prepare data from InferenceData object for CRPS calculations.

    Parameters
    ----------
    inference_data : InferenceData
        ArviZ InferenceData object containing the data.
    x_group : str, default "posterior_predictive"
        Name of the InferenceData group containing the first set of predictive samples.
    x_var : str, optional
        Name of the variable in x_group to use. If None and there is only one variable,
        that variable will be used.
    x2_group : str, optional
        Name of the InferenceData group containing the second set of predictive samples.
        If None, uses the same group as x_group.
    x2_var : str, optional
        Name of the variable in x2_group to use. If None, uses the same variable as x_var.
    y_group : str, default "observed_data"
        Name of the InferenceData group containing the observed data.
    y_var : str, optional
        Name of the variable in y_group to use. If None and there is only one variable,
        that variable will be used.
    log_likelihood : xr.DataArray, optional
        Log-likelihood array if already extracted.

    Returns
    -------
    Tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray | None]
        Tuple containing (x_data, x2_data, y_data, log_likelihood)
    """
    if not hasattr(inference_data, x_group):
        raise ValueError(f"InferenceData object does not have a {x_group} group")

    x_data_group = getattr(inference_data, x_group)

    if x_var is None:
        x_vars = list(x_data_group.data_vars)
        if len(x_vars) == 1:
            x_var = x_vars[0]
        else:
            raise ValueError(
                f"Multiple variables found in {x_group} group. Please specify x_var"
                f" from: {x_vars}"
            )
    elif x_var not in x_data_group.data_vars:
        raise ValueError(
            f"Variable '{x_var}' not found in {x_group} group. Available variables:"
            f" {list(x_data_group.data_vars)}"
        )

    x_data = x_data_group[x_var]

    if x2_group is None:
        x2_group = x_group

    if not hasattr(inference_data, x2_group):
        raise ValueError(f"InferenceData object does not have a {x2_group} group")

    x2_data_group = getattr(inference_data, x2_group)

    if x2_var is None:
        x2_var = x_var

    if x2_var not in x2_data_group.data_vars:
        raise ValueError(
            f"Variable '{x2_var}' not found in {x2_group} group. Available variables:"
            f" {list(x2_data_group.data_vars)}"
        )

    x2_data = x2_data_group[x2_var]

    if not hasattr(inference_data, y_group):
        raise ValueError(f"InferenceData object does not have a {y_group} group")

    y_data_group = getattr(inference_data, y_group)

    if y_var is None:
        y_vars = list(y_data_group.data_vars)
        if len(y_vars) == 1:
            y_var = y_vars[0]
        else:
            raise ValueError(
                f"Multiple variables found in {y_group} group. Please specify y_var"
                f" from: {y_vars}"
            )
    elif y_var not in y_data_group.data_vars:
        raise ValueError(
            f"Variable '{y_var}' not found in {y_group} group. Available variables:"
            f" {list(y_data_group.data_vars)}"
        )

    y_data = y_data_group[y_var]

    if "chain" in x_data.dims and "draw" in x_data.dims:
        x_data = x_data.stack(__sample__=("chain", "draw"))

    if "chain" in x2_data.dims and "draw" in x2_data.dims:
        x2_data = x2_data.stack(__sample__=("chain", "draw"))

    if (
        log_likelihood is not None
        and "chain" in log_likelihood.dims
        and "draw" in log_likelihood.dims
    ):
        log_likelihood = log_likelihood.stack(__sample__=("chain", "draw"))

    return x_data, x2_data, y_data, log_likelihood
