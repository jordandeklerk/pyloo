"""Plot LOO diagnostics."""

import arviz as az

from ..rcparams import rcParams
from .plot_utils import _scale_fig_size, get_plotting_function


def plot_loo(
    loo_results,
    var_name=None,
    figsize=None,
    textsize=None,
    color="C0",
    threshold=None,
    show_pareto_k=True,
    show_elpd=False,
    backend=None,
    backend_kwargs=None,
    show=None,
    ax=None,
    style="arviz-white",
    **kwargs,
):
    """Plot Leave-One-Out (LOO) cross-validation results.

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
        Color used for the scatter plot. Defaults to "C0".
    threshold : float, optional
        Show the threshold for Pareto k values. If None, no threshold line is drawn.
    show_pareto_k : bool, default True
        Show Pareto k values in the plot.
    show_elpd : bool, default True
        Show ELPD values in the plot.
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
        Additional keywords passed to :func:`matplotlib.pyplot.scatter`.

    Returns
    -------
    axes : matplotlib axes

    Examples
    --------
    Plot Pareto k values from LOO cross-validation:

    .. code-block:: python

        import pyloo as pl
        import arviz as az

        centered_eight = az.load_arviz_data("centered_eight")
        loo_results = loo(centered_eight, pointwise=True)

        # Plot Pareto k values
        pl.loo_plot(loo_results, threshold=0.7)

        # Plot ELPD values
        pl.loo_plot(loo_results, show_elpd=True)
    """
    if backend is None:
        backend = rcParams["plot.backend"]
    backend = backend.lower()

    if style is not None:
        az.style.use(style)

    plot_loo_backend = get_plotting_function("plot_loo", "loo_plot", backend)

    if ax is None and figsize is None:
        figsize, *_ = _scale_fig_size(figsize, textsize)

    if show_elpd:
        show_pareto_k = False

    ax = plot_loo_backend(
        ax=ax,
        loo_results=loo_results,
        var_name=var_name,
        figsize=figsize,
        textsize=textsize,
        color=color,
        threshold=threshold,
        show_pareto_k=show_pareto_k,
        show_elpd=show_elpd,
        backend_kwargs=backend_kwargs,
        show=show,
        **kwargs,
    )

    return ax
