"""Plot LOO difference diagnostics."""

import arviz as az
import numpy as np

from ..rcparams import rcParams
from .plot_utils import _scale_fig_size, get_plotting_function


def plot_loo_difference(
    x_values,
    loo_results_1,
    loo_results_2,
    group=None,
    outlier_thresh=None,
    size=1,
    alpha=1,
    jitter=0,
    sort_by_group=False,
    figsize=None,
    textsize=None,
    backend=None,
    backend_kwargs=None,
    show=None,
    ax=None,
    style="arviz-white",
    **kwargs,
):
    """Plot the difference in ELPD between two models across a domain.

    The LOO difference plot shows how the ELPD of two different models
    changes when a predictor is varied.

    Parameters
    ----------
    x_values : array-like
        Values for the x-axis. These could be a predictor variable, categorical
        variable, or an index (1:n).
    loo_results_1 : ELPDData
        LOO results from the first model.
    loo_results_2 : ELPDData
        LOO results from the second model.
    group : array-like, optional
        Grouping variable for coloring points. Must be the same length as x_values.
    outlier_thresh : float, optional
        Flag values when the difference in the ELPD exceeds this threshold.
    size : float, default 1
        Size of the points.
    alpha : float, default 1
        Transparency of the points.
    jitter : float or tuple, default 0
        Amount of jitter to add to points. If a single number, jitter is applied
        only to the x-axis. If a tuple, the first value is used for x-axis jitter
        and the second for y-axis jitter.
    sort_by_group : bool, default False
        Sort observations by group, then plot against an arbitrary index.
        Useful when categories have very different sample sizes.
    figsize : tuple, optional
        Figure size. If None it will be defined automatically.
    textsize : float, optional
        Text size scaling factor for labels, titles and lines. If None it will be autoscaled based
        on figsize.
    backend : {"matplotlib"}, default "matplotlib"
        Select plotting backend.
    backend_kwargs : dict, optional
        These are kwargs specific to the backend being used, passed to
        :func:`matplotlib.pyplot.subplots`.
    show : bool, optional
        Call backend show function.
    ax : matplotlib axes, optional
        Axes to draw the plot on.
    style : str, default "arviz-white"
        The name of the ArviZ style to use for the plot. Set to None to use the current style.
    **kwargs : dict, optional
        Additional keywords passed to the backend plotting function.

    Returns
    -------
    axes : matplotlib axes

    Examples
    --------

    Basic usage with pre-computed LOO results

    .. code-block:: python

        import pyloo as pl
        import matplotlib.pyplot as plt
        import numpy as np

        # Assuming we have already computed LOO for two models
        # and have a predictor variable X
        loo_model1 = pl.loo(idata_model1, pointwise=True)
        loo_model2 = pl.loo(idata_model2, pointwise=True)

        pl.plot_loo_difference(X, loo_model1, loo_model2,
                              jitter=(0.1, 0), alpha=0.7, size=2)
        plt.title("LOO Difference: Model 1 vs Model 2")
        plt.xlabel("Predictor X")

    Compare models with different robustness to outliers

    .. code-block:: python

        import pyloo as pl
        import pymc as pm
        import numpy as np
        import matplotlib.pyplot as plt

        rng = np.random.default_rng(42)
        n_obs = 500

        X = rng.normal(0, 1, size=n_obs)
        outlier_indices = np.arange(0, n_obs, 20)
        X[outlier_indices] = rng.normal(5, 2, size=len(outlier_indices))

        noise = np.zeros(n_obs)
        regular_indices = np.ones(n_obs, dtype=bool)
        regular_indices[outlier_indices] = False
        noise[regular_indices] = rng.normal(0, 0.1, size=regular_indices.sum())
        noise[outlier_indices] = rng.standard_t(df=2, size=len(outlier_indices))
        y = 1.0 + 2.0 * X + noise

        with pm.Model() as normal_model:
            alpha = pm.Normal("alpha", mu=0, sigma=10)
            beta = pm.Normal("beta", mu=0, sigma=10)
            sigma = pm.HalfNormal("sigma", sigma=10)
            pm.Normal("y", mu=alpha + beta * X, sigma=sigma, observed=y)
            idata_normal = pm.sample(1000, tune=1000,
                                    idata_kwargs={"log_likelihood": True})

        with pm.Model() as t_model:
            alpha = pm.Normal("alpha", mu=0, sigma=10)
            beta = pm.Normal("beta", mu=0, sigma=10)
            sigma = pm.HalfNormal("sigma", sigma=10)
            nu = pm.Exponential("nu", lam=1/30) + 2
            pm.StudentT("y", mu=alpha + beta * X, sigma=sigma,
                       nu=nu, observed=y)
            idata_t = pm.sample(1000, tune=1000,
                               idata_kwargs={"log_likelihood": True})

        loo_normal = pl.loo(idata_normal, pointwise=True)
        loo_t = pl.loo(idata_t, pointwise=True)

        # Positive values indicate t model performs better
        pl.plot_loo_difference(X, loo_t, loo_normal,
                              jitter=(0.1, 0.1), alpha=0.7, size=1.5)
        plt.title("LOO Difference Plot: t Model vs Normal Model")
        plt.xlabel("X")

    Using groups to color-code points by category

    .. code-block:: python

        groups = np.zeros(n_obs, dtype=str)
        groups[X < -1] = "Low"
        groups[(X >= -1) & (X < 1)] = "Medium"
        groups[X >= 1] = "High"
        groups[outlier_indices] = "Outlier"

        pl.plot_loo_difference(X, loo_t, loo_normal,
                              group=groups, jitter=(0.1, 0.1),
                              alpha=0.7, size=1.5)
        plt.title("LOO Difference Plot with Groups")
        plt.xlabel("X")

    Adding a smoothed trend line

    .. code-block:: python

        from scipy.ndimage import gaussian_filter1d

        ax = pl.plot_loo_difference(X, loo_t, loo_normal,
                                   jitter=(0.1, 0.1), alpha=0.5, size=1.5)

        sorted_indices = np.argsort(X)
        x_sorted = X[sorted_indices]
        elpd_diff_sorted = (loo_t.loo_i.values - loo_normal.loo_i.values)[sorted_indices]
        smoothed = gaussian_filter1d(elpd_diff_sorted, sigma=5)
        plt.plot(x_sorted, smoothed, 'r-', linewidth=2, label="Smoothed trend")

        plt.title("LOO Difference Plot with Smoothed Trend")
        plt.xlabel("X")
        plt.legend()
    """
    if backend is None:
        backend = rcParams["plot.backend"]
    backend = backend.lower()

    if style is not None:
        az.style.use(style)

    plot_loo_difference_backend = get_plotting_function(
        "plot_loo_difference", "loo_difference_plot", backend
    )

    if ax is None and figsize is None:
        figsize, *_ = _scale_fig_size(figsize, textsize)

    x_values = np.asarray(x_values)

    if hasattr(loo_results_1, "loo_i") and hasattr(loo_results_2, "loo_i"):
        elpd_diff = loo_results_1.loo_i.values - loo_results_2.loo_i.values
    else:
        raise ValueError(
            "Both loo_results_1 and loo_results_2 must have pointwise ELPD values"
            " (loo_i). Make sure to compute LOO with pointwise=True."
        )

    if len(x_values) != len(elpd_diff):
        raise ValueError(
            f"Length of x_values ({len(x_values)}) must match "
            f"length of ELPD differences ({len(elpd_diff)})"
        )

    if group is not None and len(group) != len(x_values):
        raise ValueError(
            f"Length of group ({len(group)}) must match "
            f"length of x_values ({len(x_values)})"
        )

    ax = plot_loo_difference_backend(
        ax=ax,
        x_values=x_values,
        elpd_diff=elpd_diff,
        group=group,
        outlier_thresh=outlier_thresh,
        size=size,
        alpha=alpha,
        jitter=jitter,
        sort_by_group=sort_by_group,
        figsize=figsize,
        textsize=textsize,
        backend_kwargs=backend_kwargs,
        show=show,
        **kwargs,
    )

    return ax
