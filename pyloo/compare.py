"""Model comparison utilities for LOO-CV, WAIC, and K-fold CV based on Arviz."""

import warnings
from copy import deepcopy
from typing import Callable, Literal

import numpy as np
import pandas as pd
import scipy.stats as st
from arviz.data import InferenceData
from scipy import optimize

from .elpd import ELPDData
from .loo import loo
from .loo_kfold import loo_kfold
from .loo_subsample import loo_subsample
from .rcparams import _validate_scale
from .waic import waic

__all__ = ["loo_compare"]


def loo_compare(
    compare_dict: dict[str, InferenceData | ELPDData],
    ic: str = "loo",
    method: Literal["stacking", "bb-pseudo-bma", "pseudo-bma"] = "stacking",
    b_samples: int = 1000,
    alpha: float = 1,
    seed: int | np.random.RandomState | None = None,
    scale: str | None = None,
    var_name: str | None = None,
    observations: int | np.ndarray | None = None,
    estimator: Literal["diff_srs", "srs", "hh_pps"] | None = None,
    K: int | None = None,
    folds: np.ndarray | None = None,
    stratify: np.ndarray | None = None,
    random_seed: int | None = None,
) -> pd.DataFrame:
    """Compare models based on their expected log pointwise predictive density (ELPD).

    The ELPD is estimated either by Pareto smoothed importance sampling leave-one-out
    cross-validation (PSIS-LOO), using the widely applicable information criterion (WAIC),
    or using K-fold cross-validation. We recommend PSIS-LOO over WAIC for most cases.
    For large datasets, PSIS-LOO can be approximated using subsampling. For models with
    influential observations, K-fold cross-validation may provide more stable estimates.
    K-fold cross-validation is only supported for PyMC models.

    Parameters
    ----------
    compare_dict : dict of {str: InferenceData or ELPDData}
        A dictionary of model names and InferenceData or ELPDData objects
    ic : str, optional
        Information Criterion (LOO, WAIC, or kfold) used to compare models.
        Default is "loo". We recommend using LOO over WAIC for most cases.
    method : str, optional
        Method used to estimate the weights for each model. Available options are:
        - 'stacking' : stacking of predictive distributions.
        - 'BB-pseudo-BMA' : pseudo-Bayesian Model averaging using Akaike-type
          weighting. The weights are stabilized using the Bayesian bootstrap.
        - 'pseudo-BMA': pseudo-Bayesian Model averaging using Akaike-type
          weighting, without Bootstrap stabilization (not recommended).
        Default is "stacking".
    b_samples : int, optional
        Number of samples taken by the Bayesian bootstrap estimation.
        Only useful when method = 'BB-pseudo-BMA'.
    alpha : float, optional
        The shape parameter in the Dirichlet distribution used for the Bayesian bootstrap.
        Only useful when method = 'BB-pseudo-BMA'. When alpha=1 (default), the distribution
        is uniform on the simplex. A smaller alpha will keeps the final weights more away
        from 0 and 1.
    seed : int or np.random.RandomState, optional
        If int or RandomState, use it for seeding Bayesian bootstrap.
        Only useful when method = 'BB-pseudo-BMA'.
    scale : str, optional
        Output scale for the ELPD. Available options are:
        - 'log' : (default) log-score
        - 'negative_log' : -1 * log-score
        - 'deviance' : -2 * log-score
        A higher log-score (or a lower deviance) indicates better predictive accuracy.
    var_name : str, optional
        Name of the variable in log_likelihood groups storing the pointwise log
        likelihood data to use for loo computation.
    observations : int or array-like, optional
        Number of observations to use for subsampling. If None, use full LOO-CV.
    estimator : str, optional
        Estimator to use for subsampling. Only used if observations is not None.
        Options are "diff_srs", "srs", or "hh_pps".
    K : int, optional
        Number of folds for K-fold cross-validation. Only used if ic="kfold".
        Default is 10 if not specified.
    folds : array-like, optional
        Pre-specified fold assignments for K-fold cross-validation. Only used if ic="kfold".
        If provided, this overrides the K parameter.
    stratify : array-like, optional
        Array of values to use for stratified K-fold cross-validation. Only used if ic="kfold".
        If provided, folds will be created to preserve the distribution of these values.
    random_seed : int, optional
        Random seed for reproducibility when creating folds for K-fold cross-validation.
        Only used if ic="kfold" and folds is None.

    Returns
    -------
    pandas.DataFrame
        A DataFrame, ordered from best to worst model (measured by the ELPD).
        The index reflects the key with which the models are passed to this function.
        The columns are:
        - rank: The rank-order of the models. 0 is the best.
        - elpd: ELPD estimated either using LOO-CV or WAIC.
        - p_loo: Effective number of parameters.
        - elpd_diff: The difference in ELPD between two models.
        - weight: Relative weight for each model.
        - se: Standard error of the ELPD estimate.
        - dse: Standard error of the difference in ELPD between each model.
        - warning: A value of 1 indicates that the computation of the ELPD may not be reliable.
        - scale: Scale used for the ELPD.

    Examples
    --------
    Compare models using PSIS-LOO:

    .. code-block:: python

        import arviz as az
        import pyloo as pl

        data1 = az.load_arviz_data("centered_eight")
        data2 = az.load_arviz_data("non_centered_eight")

        compare_dict = {"centered": data1, "non_centered": data2}
        pl.loo_compare(compare_dict)

    Compare models using K-fold cross-validation:

    .. code-block:: python

        import arviz as az
        import pyloo as pl
        from pyloo import PyMCWrapper
        import pymc as pm
        import numpy as np

        # Create two PyMC models with different priors
        x = np.random.normal(0, 1, size=100)
        y = 2 * x + np.random.normal(0, 1, size=100)

        # Model 1 with informative priors
        with pm.Model() as model1:
            alpha = pm.Normal("alpha", mu=0, sigma=1)
            beta = pm.Normal("beta", mu=2, sigma=1)
            sigma = pm.HalfNormal("sigma", sigma=1)
            mu = alpha + beta * x
            obs = pm.Normal("y", mu=mu, sigma=sigma, observed=y)
            idata1 = pm.sample(1000, return_inferencedata=True)

        # Model 2 with less informative priors
        with pm.Model() as model2:
            alpha = pm.Normal("alpha", mu=0, sigma=10)
            beta = pm.Normal("beta", mu=0, sigma=10)
            sigma = pm.HalfNormal("sigma", sigma=10)
            mu = alpha + beta * x
            obs = pm.Normal("y", mu=mu, sigma=sigma, observed=y)
            idata2 = pm.sample(1000, return_inferencedata=True)

        wrapper1 = PyMCWrapper(model1, idata1)
        wrapper2 = PyMCWrapper(model2, idata2)

        compare_dict = {"informative_priors": wrapper1, "weak_priors": wrapper2}
        pl.loo_compare(compare_dict, ic="kfold", K=5)
    """
    if not isinstance(compare_dict, dict):
        raise TypeError("compare_dict must be a dictionary")

    if len(compare_dict) < 2:
        raise ValueError("You must specify at least two models for comparison")

    if scale is None:
        scale = "log"
    scale = scale.lower()
    if scale not in ["log", "negative_log", "deviance"]:
        raise ValueError("Scale must be 'log', 'negative_log' or 'deviance'")

    method = method.lower()  # type: ignore
    if method not in ["stacking", "bb-pseudo-bma", "pseudo-bma"]:
        raise ValueError("Method must be 'stacking', 'BB-pseudo-BMA' or 'pseudo-BMA'")

    if ic not in ["loo", "waic", "kfold"]:
        raise ValueError("ic must be 'loo', 'waic', or 'kfold'")

    elpds, scale, ic = _calculate_ics(
        compare_dict,
        scale=scale,
        ic=ic,
        var_name=var_name,
        observations=observations,
        estimator=estimator,
        K=K,
        folds=folds,
        stratify=stratify,
        random_seed=random_seed,
    )

    ascending = scale != "log"

    model_names = list(elpds.keys())
    elpd_values = np.array([elpds[name][f"elpd_{ic}"] for name in model_names])
    order = np.argsort(elpd_values) if ascending else np.argsort(-elpd_values)
    ordered_names = [model_names[i] for i in order]

    best_model = ordered_names[0]
    diffs = []
    ses = []
    dses = []

    for name in ordered_names:
        if name == best_model:
            diff = 0
            dse = 0
        else:
            diff = elpds[name][f"elpd_{ic}"] - elpds[best_model][f"elpd_{ic}"]
            if scale == "negative_log":
                diff *= -1
            elif scale == "deviance":
                diff *= -2

            ic_i = f"{ic}_i"
            pointwise_diff = elpds[name][ic_i].values - elpds[best_model][ic_i].values
            dse = np.sqrt(len(pointwise_diff) * np.var(pointwise_diff))

        diffs.append(diff)
        ses.append(elpds[name]["se"])
        dses.append(dse)

    weights_result = _compute_weights(
        elpds=elpds,
        ic=ic,
        method=method,
        b_samples=b_samples,
        alpha=alpha,
        seed=seed,
        scale=scale,
    )

    if method == "bb-pseudo-bma":
        weights, computed_ses = weights_result
        ses = [computed_ses[name] for name in ordered_names]
    else:
        weights = dict(weights_result)  # type: ignore

    df = pd.DataFrame(
        {
            "rank": range(len(ordered_names)),
            f"elpd_{ic}": [elpds[name][f"elpd_{ic}"] for name in ordered_names],
            f"p_{ic}": [elpds[name][f"p_{ic}"] for name in ordered_names],
            "elpd_diff": diffs,
            "weight": [weights[name] for name in ordered_names],
            "se": ses,
            "dse": dses,
            "warning": [elpds[name]["warning"] for name in ordered_names],
            "scale": scale,
        },
        index=ordered_names,
    )

    return df


def _ic_matrix(elpds: dict[str, ELPDData], ic_i: str) -> tuple[int, int, np.ndarray]:
    """Store the previously computed pointwise predictive accuracy values (elpds) in a 2D matrix."""
    model_names = list(elpds.keys())
    cols = len(model_names)
    rows = len(elpds[model_names[0]][ic_i].values)
    ic_i_val = np.zeros((rows, cols))

    for idx, name in enumerate(model_names):
        ic = elpds[name][ic_i].values
        if len(ic) != rows:
            raise ValueError(
                "The number of observations should be the same across all models"
            )
        ic_i_val[:, idx] = ic

    return rows, cols, ic_i_val


def _calculate_ics(
    compare_dict: dict[str, InferenceData | ELPDData],
    scale: str | None = None,
    ic: str | None = None,
    var_name: str | None = None,
    observations: int | np.ndarray | None = None,
    estimator: Literal["diff_srs", "srs", "hh_pps"] | None = None,
    K: int | None = None,
    folds: np.ndarray | None = None,
    stratify: np.ndarray | None = None,
    random_seed: int | None = None,
) -> tuple[dict[str, ELPDData], str, str]:
    """Calculate LOO, WAIC, K-fold CV, or subsampled LOO.

    Parameters
    ----------
    compare_dict : dict of {str : InferenceData or ELPDData}
        A dictionary of model names and InferenceData or ELPDData objects
    scale : str, optional
        Output scale for IC. Available options are:
        - 'log' : (default) log-score
        - 'negative_log' : -1 * log-score
        - 'deviance' : -2 * log-score
        A higher log-score (or a lower deviance) indicates better predictive accuracy.
    ic : str, optional
        Information Criterion (LOO, WAIC, or kfold) used to compare models.
        Default is "loo".
    var_name : str, optional
        Name of the variable storing pointwise log likelihood values.
    observations : int or array-like, optional
        Number of observations to use for subsampling. If None, use full LOO-CV.
    estimator : str, optional
        Estimator to use for subsampling. Only used if observations is not None.
        Options are "diff_srs", "srs", or "hh_pps".
    K : int, optional
        Number of folds for K-fold cross-validation. Only used if ic="kfold".
        Default is 10 if not specified.
    folds : array-like, optional
        Pre-specified fold assignments for K-fold cross-validation. Only used if ic="kfold".
        If provided, this overrides the K parameter.
    stratify : array-like, optional
        Array of values to use for stratified K-fold cross-validation. Only used if ic="kfold".
        If provided, folds will be created to preserve the distribution of these values.
    random_seed : int, optional
        Random seed for reproducibility when creating folds for K-fold cross-validation.
        Only used if ic="kfold" and folds is None.

    Returns
    -------
    compare_dict : dict of ELPDData
    scale : str
    ic : str
    """
    precomputed_elpds = {
        name: elpd_data
        for name, elpd_data in compare_dict.items()
        if isinstance(elpd_data, ELPDData)
    }
    precomputed_ic = None
    precomputed_scale = None

    if precomputed_elpds:
        _, arbitrary_elpd = precomputed_elpds.popitem()
        precomputed_ic = arbitrary_elpd.index[0].split("_")[1]
        precomputed_scale = arbitrary_elpd["scale"]
        raise_non_pointwise = f"{precomputed_ic}_i" not in arbitrary_elpd

        if any(
            elpd_data.index[0].split("_")[1] != precomputed_ic
            for elpd_data in precomputed_elpds.values()
        ):
            raise ValueError("All information criteria to be compared must be the same")

        if any(
            elpd_data["scale"] != precomputed_scale
            for elpd_data in precomputed_elpds.values()
        ):
            raise ValueError(
                "All information criteria to be compared must use the same scale"
            )

        if (
            any(
                f"{precomputed_ic}_i" not in elpd_data
                for elpd_data in precomputed_elpds.values()
            )
            or raise_non_pointwise
        ):
            raise ValueError(
                "Not all provided ELPDData have been calculated with pointwise=True"
            )

        if ic is not None and ic.lower() != precomputed_ic.lower():
            warnings.warn(
                "Provided ic argument is incompatible with precomputed elpd data. "
                f"Using ic from precomputed elpddata: {precomputed_ic}",
                stacklevel=2,
            )
            ic = precomputed_ic

        if scale is not None and scale.lower() != precomputed_scale:
            warnings.warn(
                "Provided scale argument is incompatible with precomputed elpd data. "
                f"Using scale from precomputed elpddata: {precomputed_scale}",
                stacklevel=2,
            )
            scale = precomputed_scale

    if ic is None and precomputed_ic is None:
        ic = "loo"
    elif ic is None:
        ic = precomputed_ic
    else:
        ic = ic.lower()
    if scale is None and precomputed_scale is None:
        scale = "log"
    elif scale is None:
        scale = precomputed_scale
    else:
        scale = _validate_scale(scale)

    ic_func: Callable[..., ELPDData]
    if ic == "waic":
        ic_func = waic
    elif ic == "kfold":
        ic_func = loo_kfold
    elif observations is not None:
        ic_func = loo_subsample
    else:
        ic_func = loo

    compare_dict = dict(deepcopy(compare_dict))
    for name, dataset in compare_dict.items():
        if not isinstance(dataset, ELPDData):
            try:
                if ic == "kfold":
                    compare_dict[name] = ic_func(
                        dataset,
                        K=K if K is not None else 10,
                        folds=folds,
                        pointwise=True,
                        var_name=var_name,
                        scale=scale,
                        stratify=stratify,
                        random_seed=random_seed,
                        save_fits=False,
                        progressbar=True,
                    )
                elif observations is not None:
                    compare_dict[name] = ic_func(
                        dataset,
                        observations=observations,
                        estimator=estimator,
                        pointwise=True,
                        var_name=var_name,
                        scale=scale,
                    )
                else:
                    compare_dict[name] = ic_func(
                        dataset,
                        pointwise=True,
                        var_name=var_name,
                        scale=scale,
                    )
            except Exception as e:
                raise e.__class__(
                    f"Encountered error trying to compute {ic} from model {name}."
                ) from e

    if scale is None:
        scale = "log"
    return dict(compare_dict), scale, ic  # type: ignore


def _compute_weights(
    elpds: dict[str, ELPDData],
    ic: str,
    method: Literal["stacking", "bb-pseudo-bma", "pseudo-bma"],
    b_samples: int,
    alpha: float,
    seed: int | np.random.RandomState | None,
    scale: str,
) -> dict[str, float] | tuple[dict[str, float], pd.Series]:
    """Compute model weights using the specified method."""
    if method == "stacking":
        return _stacking_weights(elpds, ic, scale)
    elif method == "bb-pseudo-bma":
        return _bb_pseudo_bma_weights(elpds, ic, b_samples, alpha, seed, scale)
    else:
        return _pseudo_bma_weights(elpds, ic, scale)


def _stacking_weights(
    elpds: dict[str, ELPDData], ic: str, scale: str
) -> dict[str, float]:
    """Compute stacking weights."""
    model_names = list(elpds.keys())
    n_models = len(model_names)

    ic_i = f"{ic}_i"
    pointwise_elpds = np.stack(
        [elpds[name][ic_i].values for name in model_names], axis=1
    )

    if scale == "deviance":
        pointwise_elpds /= -2
    elif scale == "negative_log":
        pointwise_elpds *= -1

    max_elpd = np.max(pointwise_elpds, axis=1, keepdims=True)
    exp_elpds = np.exp(pointwise_elpds - max_elpd)

    def objective(weights):
        """Compute negative log score of weight combination."""
        weights = np.concatenate((weights, [max(1.0 - np.sum(weights), 0.0)]))
        weights = np.maximum(weights, 0)
        weights = weights / np.sum(weights)
        score = np.sum(np.log(np.dot(exp_elpds, weights)))
        return -score

    def gradient(weights):
        """Compute gradient of objective."""
        weights = np.concatenate((weights, [max(1.0 - np.sum(weights), 0.0)]))
        weights = np.maximum(weights, 0)
        weights = weights / np.sum(weights)
        grad = np.zeros(n_models - 1)
        denom = np.dot(exp_elpds, weights)
        for k in range(n_models - 1):
            grad[k] = np.sum((exp_elpds[:, k] - exp_elpds[:, -1]) / denom)
        return -grad

    x0 = np.full(n_models - 1, 1.0 / n_models)
    bounds = [(0.0, 1.0)] * (n_models - 1)
    constraints = [
        {"type": "ineq", "fun": lambda x: 1.0 - np.sum(x)},
        {"type": "ineq", "fun": np.sum},
    ]

    result = optimize.minimize(
        objective,
        x0,
        jac=gradient,
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 2000},
    )

    weights = np.concatenate((result.x, [max(1.0 - np.sum(result.x), 0.0)]))
    weights = np.maximum(weights, 0)
    weights = weights / np.sum(weights)
    return dict(zip(model_names, weights))


def _bb_pseudo_bma_weights(
    elpds: dict[str, ELPDData],
    ic: str,
    b_samples: int,
    alpha: float,
    seed: int | np.random.RandomState | None,
    scale: str,
) -> tuple[dict[str, float], pd.Series]:
    """Compute Bayesian bootstrap pseudo-BMA weights."""
    if seed is not None:
        np.random.seed(seed)

    model_names = list(elpds.keys())

    rows, cols, ic_i_val = _ic_matrix(elpds, f"{ic}_i")
    ic_i_val = ic_i_val * rows

    if scale == "deviance":
        ic_i_val /= -2
    elif scale == "negative_log":
        ic_i_val *= -1

    rng = np.random.RandomState(seed) if isinstance(seed, int) else seed
    b_weighting = st.dirichlet.rvs(
        alpha=[alpha] * rows, size=b_samples, random_state=rng
    )
    weights = np.zeros((b_samples, cols))
    z_bs = np.zeros_like(weights)

    for i in range(b_samples):
        z_b = np.dot(b_weighting[i], ic_i_val)
        z_bs[i] = z_b
        rel_elpds = z_b - np.max(z_b)
        weights[i] = np.exp(rel_elpds)
        weights[i] = weights[i] / np.sum(weights[i])

    mean_weights = weights.mean(axis=0)
    ses = pd.Series(z_bs.std(axis=0), index=elpds.keys())
    return dict(zip(model_names, mean_weights)), ses


def _pseudo_bma_weights(
    elpds: dict[str, ELPDData], ic: str, scale: str
) -> dict[str, float]:
    """Compute pseudo-BMA weights."""
    model_names = list(elpds.keys())
    elpd_values = np.array([elpds[name][f"elpd_{ic}"] for name in model_names])

    if scale == "deviance":
        elpd_values /= -2
    elif scale == "negative_log":
        elpd_values *= -1

    rel_elpds = elpd_values - np.max(elpd_values)
    weights = np.exp(rel_elpds)
    weights = weights / np.sum(weights)

    return dict(zip(model_names, weights))
