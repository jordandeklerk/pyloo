"""K-fold cross-validation for PyMC models."""

import logging
import warnings
from copy import deepcopy
from typing import Any

import numpy as np
import pymc as pm
import xarray as xr

from .elpd import ELPDData
from .rcparams import rcParams
from .utils import _logsumexp, get_log_likelihood, wrap_xarray_ufunc
from .wrapper.pymc import PyMCWrapper

__all__ = ["loo_kfold"]

_log = logging.getLogger(__name__)


def loo_kfold(
    data: PyMCWrapper,
    K: int = 10,
    pointwise: bool | None = None,
    folds: np.ndarray | None = None,
    var_name: str | None = None,
    scale: str | None = None,
    save_fits: bool = False,
    progressbar: bool = False,
    stratify: np.ndarray | None = None,
    groups: np.ndarray | None = None,
    random_seed: int | None = None,
    **kwargs: Any,
) -> ELPDData:
    r"""Perform exact K-fold cross-validation for PyMC models.

    This function first randomly partitions the data into ::math:`K` subsets
    of equal size (or as close to equal as possible), or the user can specify the
    ``folds`` argument. If ::math:`K` is equal to the total number of observations,
    this is equivalent to leave-one-out cross-validation (LOO-CV).

    For each subset, it fits a model using the remaining data, and then computes
    the log-likelihood of the held-out data. The sum of the log-likelihoods is then
    converted to an estimate of the expected log pointwise predictive density (ELPD).

    Parameters
    ----------
    data : PyMCWrapper
        A PyMCWrapper object containing the fitted model and data
    K : int
        Number of folds for cross-validation (default: 10)
    pointwise : bool, optional
        If True, return the pointwise predictive accuracy values. Defaults to
        ``stats.ic_pointwise`` rcParam.
    folds : np.ndarray | None
        Optional pre-specified fold assignments. If None, folds are created randomly.
        Fold indices must be integers from 1 to K.
        If provided, this overrides the stratify parameter.
    var_name : str | None
        Name of the observed variable to use. If None, uses the first observed variable.
    scale : str | None
        Scale for the returned statistics: "log" (default), "negative_log", or "deviance"
    save_fits : bool
        Whether to save the fitted models for each fold (default: False)
    progressbar : bool
        Whether to display a progress bar during fitting (default: True)
    stratify : np.ndarray | None
        Array of values to use for stratified k-fold cross-validation. If provided,
        folds will be created to preserve the distribution of these values across folds.
        Ignored if folds is not None.
    groups : np.ndarray | None
        Array of group identifiers for each observation. If provided, folds will be created
        such that all observations from the same group are assigned to the same fold.
        Ignored if folds is not None. Takes precedence over stratify if both
        are provided.
    random_seed : int | None
        Random seed for reproducibility when creating folds. Ignored if folds is not None.
    **kwargs : Any
        Additional arguments passed to the sampling function

    Returns
    -------
    ELPDData object (inherits from :class:`pandas.Series`) with the following row/attributes:
    elpd_kfold: expected log pointwise predictive density (elpd)
    se: standard error of the elpd
    p_kfold: effective number of parameters
    n_samples: number of samples
    n_data_points: number of data points
    warning: boolean indicating whether any warnings occurred during fitting
    scale: scale of the elpd
    K: number of folds
    kfoldic: K-fold information criterion
    kfoldic_se: standard error of the kfoldic
    kfold_i: :class:`~xarray.DataArray` with the pointwise predictive accuracy,
        only if pointwise=True

    Raises
    ------
    TypeError
        If data is not a PyMCWrapper instance
    ValueError
        If K is invalid, folds are improperly specified, or scale is invalid

    Examples
    --------
    Let's consider using K-fold cross-validation with a simple linear regression model:

    .. code-block:: python

        import pymc as pm
        import numpy as np
        import pyloo as pl
        from pyloo.wrapper import PyMCWrapper

        np.random.seed(42)
        x = np.random.normal(0, 1, size=100)
        true_alpha = 1.0
        true_beta = 2.5
        true_sigma = 1.0
        y = true_alpha + true_beta * x + np.random.normal(0, true_sigma, size=100)

    Let's create a PyMC model that represents our linear regression problem.
    We'll use weakly informative priors for all parameters:

    .. code-block:: python

        with pm.Model() as model:
            alpha = pm.Normal("alpha", mu=0, sigma=10)
            beta = pm.Normal("beta", mu=0, sigma=10)
            sigma = pm.HalfNormal("sigma", sigma=10)

            mu = alpha + beta * x
            obs = pm.Normal("y", mu=mu, sigma=sigma, observed=y)

            idata = pm.sample(
                1000,
                chains=4,
                return_inferencedata=True,
                idata_kwargs={"log_likelihood": True},
            )

    With our model fitted, we can now perform K-fold cross-validation to assess its predictive performance.
    First, we'll create a PyMCWrapper object, which provides a standardized interface for working with
    PyMC models in cross-validation:

    .. code-block:: python

        wrapper = PyMCWrapper(model, idata)
        kfold_result = pl.loo_kfold(wrapper, K=5)

    The result contains various statistics about the model's predictive performance, including
    the expected log pointwise predictive density (ELPD) and its standard error.

    For datasets with imbalanced features or outcomes, stratified K-fold cross-validation can provide
    more reliable performance estimates. You can use the stratify parameter to ensure each fold
    has a similar distribution of values

    .. code-block:: python

        np.random.seed(42)
        n_samples = 200
        y = np.random.binomial(1, 0.3, size=n_samples)

        # Create a feature that's correlated with the outcome
        x1 = np.random.normal(y, 1.0)
        # Add another independent feature
        x2 = np.random.normal(0, 1.0, size=n_samples)

        X = np.column_stack((x1, x2))

        with pm.Model() as model:
            alpha = pm.Normal("alpha", mu=0, sigma=2)
            beta = pm.Normal("beta", mu=0, sigma=2, shape=2)

            logit_p = alpha + pm.math.dot(X, beta)
            obs = pm.Bernoulli("y", logit_p=logit_p, observed=y)

            idata = pm.sample(
                1000,
                chains=2,
                return_inferencedata=True,
                idata_kwargs={"log_likelihood": True},
            )

        wrapper = PyMCWrapper(model, idata)
        kfold_result = pl.loo_kfold(wrapper, K=5, stratify=wrapper.get_observed_data(), random_seed=123)

    For hierarchical or grouped data, it's important to ensure that all observations from the same
    group are assigned to the same fold. This can be achieved using the groups parameter:

    .. code-block:: python

        import pymc as pm
        import numpy as np
        import pyloo as pl
        import arviz as az
        from pyloo.wrapper import PyMCWrapper

        data = az.load_arviz_data("radon")

        group_ids = data.constant_data.county_idx.values
        wrapper = PyMCWrapper.from_arviz(data)

        kfold_result = pl.loo_kfold(wrapper, K=5, groups=group_ids, random_seed=123)
        print(kfold_result)

    See Also
    --------
    loo_subsample : Leave-one-out cross-validation with subsampling
    loo_moment_match : Leave-one-out cross-validation with moment matching
    loo_approximate_posterior : Leave-one-out cross-validation for posterior approximations
    loo_score : Compute LOO score for continuous ranked probability score
    loo_group : Leave-one-group-out cross-validation
    waic : Compute WAIC
    """
    if not isinstance(data, PyMCWrapper):
        raise TypeError(f"Expected PyMCWrapper, got {type(data).__name__}")

    wrapper = data
    original_idata = wrapper.idata
    pointwise = rcParams["stats.ic_pointwise"] if pointwise is None else pointwise

    if var_name is None:
        var_name = wrapper.get_observed_name()

    observed_data = wrapper.get_observed_data()
    n_obs = len(observed_data)

    scale = "log" if scale is None else scale.lower()
    if scale not in ["log", "negative_log", "deviance"]:
        raise ValueError("Scale must be 'log', 'negative_log', or 'deviance'")

    if scale == "deviance":
        scale_factor = -2
    elif scale == "negative_log":
        scale_factor = -1
    else:
        scale_factor = 1

    folds, K = _prepare_folds(folds, K, n_obs, stratify, groups, random_seed)
    log_lik_full = get_log_likelihood(original_idata, var_name=var_name)

    obs_dims = [dim for dim in log_lik_full.dims if dim not in ("chain", "draw")]
    if not obs_dims:
        raise ValueError("Could not identify observation dimension in log_likelihood")

    has_nan = np.any(np.isnan(log_lik_full.values))

    if has_nan:
        warnings.warn(
            "NaN values detected in log-likelihood. These will be ignored in the LOGO "
            "calculation.",
            UserWarning,
            stacklevel=2,
        )
        log_lik_full = log_lik_full.where(~np.isnan(log_lik_full), -1e10)

    lpds_full = _compute_lpds_full(log_lik_full)
    elpds = np.zeros(n_obs)
    fits: list[Any] | None = [] if save_fits else None

    for k in range(1, K + 1):
        if progressbar:
            _log.info(f"Fitting model {k} out of {K}")

        val_indices = np.where(folds == k)[0]
        if len(val_indices) == 0:
            _log.warning(f"Fold {k} is empty, skipping")
            continue

        fold_fits, fold_elpds = _process_fold(
            wrapper=wrapper,
            var_name=var_name,
            observed_data=observed_data,
            train_indices=np.where(folds != k)[0],
            val_indices=val_indices,
            progressbar=progressbar,
            save_fits=save_fits,
            **kwargs,
        )

        for idx, val in zip(val_indices, fold_elpds):
            elpds[idx] = val

        if save_fits and fold_fits and fits is not None:
            fits.append(fold_fits)

    p_kfold = lpds_full - elpds
    p_kfold_se = np.sqrt(n_obs * np.var(p_kfold))
    elpds = scale_factor * elpds

    elpd_kfold = np.sum(elpds)
    se = np.sqrt(n_obs * np.var(elpds))
    p_kfold_sum = np.sum(p_kfold)
    kfoldic = -2 * elpd_kfold / scale_factor
    kfoldic_se = 2 * se

    n_samples = log_lik_full.sizes.get("chain", 1) * log_lik_full.sizes.get("draw", 1)
    is_stratified = stratify is not None and folds is None
    is_grouped = groups is not None and folds is None

    result_data: list[Any] = []
    index: list[str] = []

    if pointwise:
        pointwise_metrics = xr.DataArray(
            np.column_stack((elpds, p_kfold, -2 * elpds / scale_factor)),
            dims=("observation", "metric"),
            coords={
                "observation": np.arange(n_obs),
                "metric": ["elpd_kfold", "p_kfold", "kfoldic"],
            },
            name="kfold_metrics",
        )
        kfold_i = pointwise_metrics.sel(metric="elpd_kfold").rename("kfold_i")

        result_data = [
            elpd_kfold,
            se,
            p_kfold_sum,
            p_kfold_se,
            n_samples,
            n_obs,
            False,
            kfold_i,
            scale,
            K,
            kfoldic,
            kfoldic_se,
            is_stratified,
            is_grouped,
        ]
        index = [
            "elpd_kfold",
            "se",
            "p_kfold",
            "p_kfold_se",
            "n_samples",
            "n_data_points",
            "warning",
            "kfold_i",
            "scale",
            "K",
            "kfoldic",
            "kfoldic_se",
            "stratified",
            "grouped",
        ]
    else:
        result_data = [
            elpd_kfold,
            se,
            p_kfold_sum,
            p_kfold_se,
            n_samples,
            n_obs,
            False,  # warning
            scale,
            K,
            kfoldic,
            kfoldic_se,
            is_stratified,
            is_grouped,
        ]
        index = [
            "elpd_kfold",
            "se",
            "p_kfold",
            "p_kfold_se",
            "n_samples",
            "n_data_points",
            "warning",
            "scale",
            "K",
            "kfoldic",
            "kfoldic_se",
            "stratified",
            "grouped",
        ]

    if fits is not None:
        result_data.append(fits)
        index.append("fits")

    elpd_data = ELPDData(data=result_data, index=index)
    elpd_data.method = "kfold"
    elpd_data.K = K
    elpd_data.stratified = is_stratified
    elpd_data.grouped = is_grouped

    return elpd_data


def _prepare_folds(
    folds: np.ndarray | None,
    K: int,
    n_obs: int,
    stratify: np.ndarray | None,
    groups: np.ndarray | None,
    random_seed: int | None,
) -> tuple[np.ndarray, int]:
    """Prepare or validate fold assignments for K-fold cross-validation.

    Returns
    -------
    tuple
        A tuple containing (folds, adjusted_K)
    """
    if K <= 0:
        raise ValueError(f"K must be positive, got {K}")

    if K > n_obs:
        _log.warning(f"K ({K}) is greater than N ({n_obs}), setting K=N")
        K = min(K, n_obs)

    if folds is not None:
        if stratify is not None:
            _log.warning(
                "Both folds and stratify were provided. Using the provided folds and"
                " ignoring stratify."
            )
        folds = np.asarray(folds)

        if len(folds) != n_obs:
            raise ValueError(
                f"Length of folds ({len(folds)}) must match observations ({n_obs})"
            )

        unique_folds = np.unique(folds)

        if len(unique_folds) < 2:
            raise ValueError(
                f"Need at least 2 unique fold values, got {len(unique_folds)}"
            )

        if 0 in unique_folds:
            raise ValueError("Fold indices must be >= 1")
        return folds, len(unique_folds)

    if groups is not None:
        if not isinstance(groups, np.ndarray):
            groups = np.asarray(groups)

        if len(groups) != n_obs:
            raise ValueError(
                f"Length of groups ({len(groups)}) must match observations ({n_obs})"
            )

        _log.info(f"Creating group-based folds with K={K}")
        try:
            return _kfold_split_grouped(K=K, groups=groups, seed=random_seed), K
        except Exception as e:
            raise ValueError(f"Failed to create group-based folds: {str(e)}")

    if stratify is not None:
        if not isinstance(stratify, np.ndarray):
            stratify = np.asarray(stratify)

        if len(stratify) != n_obs:
            raise ValueError(
                f"Length of stratify ({len(stratify)}) must match observations"
                f" ({n_obs})"
            )

        _log.info(f"Creating stratified folds with K={K}")
        try:
            return _kfold_split_stratified(K=K, x=stratify, seed=random_seed), K
        except Exception as e:
            raise ValueError(f"Failed to create stratified folds: {str(e)}")

    _log.info(f"Creating random folds with K={K}")

    return _kfold_split_random(K=K, N=n_obs, seed=random_seed), K


def _kfold_split_grouped(
    K: int, groups: np.ndarray, seed: int | None = None
) -> np.ndarray:
    """Create group-based folds for K-fold cross-validation.

    Parameters
    ----------
    K : int
        Number of folds
    groups : np.ndarray
        Array of group identifiers for each observation
    seed : int | None
        Random seed for reproducibility

    Returns
    -------
    np.ndarray
        Array of fold assignments (integers from 1 to K)
    """
    if seed is not None:
        np.random.seed(seed)

    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)

    if n_groups < K:
        _log.warning(
            f"Number of groups ({n_groups}) is less than K ({K}). Setting K={n_groups}"
        )
        K = n_groups

    if K <= 1:
        raise ValueError(f"K must be > 1 for group-based folds, got {K}")

    group_to_fold = {}

    group_indices = np.random.permutation(n_groups)
    fold_sizes = np.zeros(K, dtype=int)

    for _, group_idx in enumerate(group_indices):
        fold = np.argmin(fold_sizes) + 1
        group_to_fold[unique_groups[group_idx]] = fold
        fold_sizes[fold - 1] += 1

    folds = np.zeros(len(groups), dtype=int)
    for i, group in enumerate(groups):
        folds[i] = group_to_fold[group]

    return folds


def _kfold_split_random(K: int, N: int, seed: int | None = None) -> np.ndarray:
    """Create random folds for K-fold cross-validation."""
    if seed is not None:
        np.random.seed(seed)

    folds = np.zeros(N, dtype=int)

    fold_sizes = np.full(K, N // K, dtype=int)
    fold_sizes[: N % K] += 1

    fold_indices = np.random.permutation(N)
    start = 0
    for i in range(K):
        end = start + fold_sizes[i]
        folds[fold_indices[start:end]] = i + 1
        start = end

    return folds


def _kfold_split_stratified(
    K: int, x: np.ndarray, seed: int | None = None
) -> np.ndarray:
    """Create stratified folds for K-fold cross-validation."""
    if seed is not None:
        np.random.seed(seed)

    x = np.asarray(x)
    N = len(x)

    if K <= 1:
        raise ValueError(f"K must be > 1 for stratified folds, got {K}")

    if np.issubdtype(x.dtype, np.number) and np.any(np.isnan(x)):
        raise ValueError("Stratification variable contains NaN values")

    if np.issubdtype(x.dtype, np.number) and len(np.unique(x)) > K:
        try:
            bins = np.percentile(x, np.linspace(0, 100, K + 1))
            bins = np.unique(bins)
            x_binned = np.digitize(x, bins[:-1])
        except Exception as e:
            raise ValueError(f"Failed to bin continuous variable: {str(e)}")
    else:
        x_binned = x

    unique_values, counts = np.unique(x_binned, return_counts=True)

    if len(unique_values) == 1 and K > 1:
        _log.warning(
            f"Only {len(unique_values)} unique value in stratification variable, using"
            " random folds instead"
        )
        return _kfold_split_random(K=K, N=N, seed=seed)

    folds = np.zeros(N, dtype=int)

    for val, count in zip(unique_values, counts):
        val_indices = np.where(x_binned == val)[0]
        val_indices = np.random.permutation(val_indices)

        val_fold_sizes = np.full(K, count // K, dtype=int)
        val_fold_sizes[: count % K] += 1

        start = 0
        for k in range(K):
            end = start + val_fold_sizes[k]
            folds[val_indices[start:end]] = k + 1
            start = end

    if not np.all((folds >= 1) & (folds <= K)):
        raise ValueError(f"Generated fold values outside range 1-{K}")

    return folds


def _process_fold(
    wrapper: PyMCWrapper,
    var_name: str,
    observed_data: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    progressbar: bool,
    save_fits: bool,
    **kwargs: Any,
) -> tuple[tuple | None, list]:
    """Process a single fold for K-fold cross-validation."""
    fold_result = None
    fold_elpds = [0.0] * len(val_indices)

    try:
        train_data = observed_data[train_indices]

        fold_wrapper = deepcopy(wrapper)
        fold_wrapper.set_data({var_name: train_data})
        idata_k = fold_wrapper.sample_posterior(progressbar=progressbar, **kwargs)

        val_data = observed_data[val_indices]

        val_wrapper = deepcopy(fold_wrapper)
        val_wrapper.set_data({var_name: val_data})
        ll_k = pm.compute_log_likelihood(
            idata_k,
            var_names=[var_name],
            model=val_wrapper.model,
            extend_inferencedata=False,
        )[var_name]

        if save_fits:
            fold_result = (idata_k, val_indices)

        if "chain" in ll_k.dims and "draw" in ll_k.dims:
            ll_k_stacked = ll_k.stack(__sample__=("chain", "draw"))
        else:
            ll_k_stacked = ll_k

        ufunc_kwargs = {"n_dims": 1, "ravel": False}
        kwargs = {"input_core_dims": [["__sample__"]]}

        elpds_k_xr = wrap_xarray_ufunc(
            _logsumexp,
            ll_k_stacked,
            func_kwargs={"b_inv": ll_k_stacked.sizes.get("__sample__", 1)},
            ufunc_kwargs=ufunc_kwargs,
            **kwargs,
        )

        k_dims = list(ll_k.dims)
        k_obs_dims = [dim for dim in k_dims if dim not in ("chain", "draw")]

        if len(k_obs_dims) > 0:
            k_obs_dim = k_obs_dims[0]
            for i in range(len(val_indices)):
                fold_elpds[i] = elpds_k_xr.isel({k_obs_dim: i}).values.item()
        else:
            fold_elpds[0] = elpds_k_xr.values.item()

    except Exception as e:
        _log.warning(f"Error processing fold: {e}")

    return fold_result, fold_elpds


def _compute_lpds_full(log_lik_full: xr.DataArray) -> np.ndarray:
    """Compute log pointwise predictive density for full dataset."""
    if "chain" in log_lik_full.dims and "draw" in log_lik_full.dims:
        log_lik_stacked = log_lik_full.stack(__sample__=("chain", "draw"))
    else:
        log_lik_stacked = log_lik_full

    ufunc_kwargs = {"n_dims": 1, "ravel": False}
    kwargs = {"input_core_dims": [["__sample__"]]}

    lpds_full_xr = wrap_xarray_ufunc(
        _logsumexp,
        log_lik_stacked,
        func_kwargs={"b_inv": log_lik_stacked.sizes.get("__sample__", 1)},
        ufunc_kwargs=ufunc_kwargs,
        **kwargs,
    )

    return lpds_full_xr.values
