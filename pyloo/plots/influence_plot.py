"""Plot observation influence diagnostics."""

import arviz as az

from ..rcparams import rcParams
from .plot_utils import _scale_fig_size, get_plotting_function


def plot_influence(
    loo_results,
    var_name=None,
    figsize=None,
    textsize=None,
    color="C0",
    threshold=None,
    sort=True,
    n_points=10,
    use_pareto_k=True,
    k_threshold=0.7,
    backend=None,
    backend_kwargs=None,
    show=None,
    ax=None,
    style="arviz-white",
    **kwargs,
):
    """Plot the influence of individual observations on the model.

    This plot visualizes the negative of the pointwise ELPD LOO values to identify
    observations with the greatest influence on model performance. Higher values indicate
    observations that, if removed, would have a larger impact on the model's predictive accuracy.

    Parameters
    ----------
    loo_results : ELPDData
        Object containing elpd and pareto_k data from LOO cross-validation
    var_name : str, optional
        If there are multiple observed variables in the model, specify which one to plot.
        If None and there are multiple variables, the first variable will be used.
    figsize : tuple, optional
        Figure size. If None it will be defined automatically.
    textsize : float, optional
        Text size scaling factor for labels, titles and lines. If None it will be autoscaled based
        on figsize.
    color : str, optional
        Color used for the bar plot. Defaults to "C0".
    threshold : float, optional
        Show a threshold line for influence values. If None, no threshold line is drawn.
    sort : bool, default True
        Whether to sort the observations by influence value (descending).
    n_points : int, default 10
        Number of points to display. If positive, shows the n_points most influential observations.
        If negative, shows the n_points least influential observations.
        If None, all points are shown.
    use_pareto_k : bool, default True
        Whether to include observations with Pareto k values above k_threshold,
        even if they are not among the top n_points most influential observations.
    k_threshold : float, default 0.7
        Threshold for Pareto k values above which observations are included when use_pareto_k=True.
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
    Plot the influence of observations from LOO cross-validation:

    .. code-block:: python

        import pyloo as pl
        import arviz as az

        centered_eight = az.load_arviz_data("centered_eight")
        loo_results = loo(centered_eight, pointwise=True)

        # Plot influence of all observations
        pl.plot_influence(loo_results)

        # Plot only the 10 most influential observations
        pl.plot_influence(loo_results, n_points=10, sort=True)
    """
    if backend is None:
        backend = rcParams["plot.backend"]
    backend = backend.lower()

    if style is not None:
        az.style.use(style)

    plot_influence_backend = get_plotting_function(
        "plot_influence", "influence_plot", backend
    )

    if ax is None and figsize is None:
        figsize, *_ = _scale_fig_size(figsize, textsize)

    ax = plot_influence_backend(
        ax=ax,
        loo_results=loo_results,
        var_name=var_name,
        figsize=figsize,
        textsize=textsize,
        color=color,
        threshold=threshold,
        sort=sort,
        n_points=n_points,
        use_pareto_k=use_pareto_k,
        k_threshold=k_threshold,
        backend_kwargs=backend_kwargs,
        show=show,
        **kwargs,
    )

    return ax
