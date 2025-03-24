"""Matplotlib LOO plot."""

import matplotlib.pyplot as plt
import numpy as np


def plot_loo(
    ax,
    loo_results,
    var_name,
    figsize,
    textsize,
    color,
    threshold,
    show_pareto_k,
    show_elpd,
    backend_kwargs,
    show,
    **kwargs,
):
    """Matplotlib loo plot."""
    if ax is None:
        _, ax = plt.subplots(
            figsize=figsize, **({} if backend_kwargs is None else backend_kwargs)
        )

    if textsize is not None:
        ax.tick_params(labelsize=textsize)
        title_fontsize = textsize * 1.2
        label_fontsize = textsize
    else:
        title_fontsize = plt.rcParams["axes.titlesize"]
        label_fontsize = plt.rcParams["axes.labelsize"]

    if hasattr(loo_results, "pareto_k"):
        if var_name is not None:
            if (
                hasattr(loo_results.pareto_k, "sel")
                and var_name in loo_results.pareto_k.coords
            ):
                pareto_k = loo_results.pareto_k.sel({var_name: True}).values
            else:
                pareto_k = loo_results.pareto_k.values
        else:
            pareto_k = loo_results.pareto_k.values
    else:
        pareto_k = None
        show_pareto_k = False

    if hasattr(loo_results, "elpd_loo"):
        if hasattr(loo_results, "loo_i"):
            if var_name is not None:
                if (
                    hasattr(loo_results.loo_i, "sel")
                    and var_name in loo_results.loo_i.coords
                ):
                    elpd_loo = loo_results.loo_i.sel({var_name: True}).values
                else:
                    elpd_loo = loo_results.loo_i.values
            else:
                elpd_loo = loo_results.loo_i.values
        else:
            elpd_loo = None
            show_elpd = False
    else:
        elpd_loo = None
        show_elpd = False

    if show_elpd and elpd_loo is not None:
        x = np.arange(len(elpd_loo))
        ax.scatter(x, elpd_loo, color=color, alpha=0.5, **kwargs)
        ax.set_xlabel("Observations", fontsize=label_fontsize)
        ax.set_ylabel("ELPD LOO", fontsize=label_fontsize)

        if var_name is not None:
            ax.set_title(f"ELPD LOO values for {var_name}", fontsize=title_fontsize)
        else:
            ax.set_title("ELPD LOO values", fontsize=title_fontsize)
    elif show_pareto_k and pareto_k is not None:
        x = np.arange(len(pareto_k))
        ax.scatter(x, pareto_k, color=color, alpha=0.5, **kwargs)
        ax.set_xlabel("Observations", fontsize=label_fontsize)
        ax.set_ylabel("Pareto k", fontsize=label_fontsize)

        if threshold is not None:
            ax.axhline(y=threshold, color="r", linestyle="--")
            ax.text(
                0,
                threshold + 0.01,
                f"k = {threshold}",
                color="r",
                fontsize=label_fontsize,
            )

        if var_name is not None:
            ax.set_title(f"Pareto k values for {var_name}", fontsize=title_fontsize)
        else:
            ax.set_title("Pareto k values", fontsize=title_fontsize)

    if show:
        plt.show()

    return ax
