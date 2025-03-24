"""Matplotlib influence plot."""

import matplotlib.pyplot as plt
import numpy as np


def plot_influence(
    ax,
    loo_results,
    var_name,
    figsize,
    textsize,
    color,
    threshold,
    sort,
    n_points,
    use_pareto_k=True,
    k_threshold=0.7,
    backend_kwargs=None,
    show=None,
    **kwargs,
):
    """Influence plot."""
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
        raise ValueError(
            "loo_results object does not contain pointwise loo values (loo_i)"
        )

    # Calculate influence as the negative of elpd_loo
    # Higher values indicate more influential observations
    influence = -elpd_loo
    orig_indices = np.arange(len(influence))

    pareto_k_indices = []
    if use_pareto_k and hasattr(loo_results, "pareto_k"):
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

        high_k_mask = pareto_k > k_threshold
        if np.any(high_k_mask):
            pareto_k_indices = np.where(high_k_mask)[0]

    if sort:
        sorted_idx = np.argsort(influence)

        if n_points is not None and n_points < len(influence):
            if n_points > 0:
                # Most influential
                top_indices = sorted_idx[-n_points:]
            else:
                # Least influential
                top_indices = sorted_idx[: abs(n_points)]
        else:
            top_indices = sorted_idx

        if use_pareto_k and len(pareto_k_indices) > 0:
            selected_indices = np.unique(
                np.concatenate([top_indices, pareto_k_indices])
            )
        else:
            selected_indices = top_indices

        orig_indices = orig_indices[selected_indices]
        influence_selected = influence[selected_indices]

        display_sort_idx = np.argsort(influence_selected)
        orig_indices = orig_indices[display_sort_idx]
        influence = influence_selected[display_sort_idx]

        x = np.arange(len(influence))
        labels = [str(i) for i in orig_indices]

        if use_pareto_k and len(pareto_k_indices) > 0:
            high_k_mask = np.isin(orig_indices, pareto_k_indices)
        else:
            high_k_mask = np.zeros(len(influence), dtype=bool)
    else:
        if n_points is not None and n_points < len(influence):
            if n_points > 0:
                # First n_points
                selected_indices = np.arange(n_points)
            else:
                # Last n_points
                selected_indices = np.arange(len(influence) + n_points, len(influence))

            if use_pareto_k and len(pareto_k_indices) > 0:
                selected_indices = np.unique(
                    np.concatenate([selected_indices, pareto_k_indices])
                )

            orig_indices = orig_indices[selected_indices]
            influence = influence[selected_indices]

        x = np.arange(len(influence))
        labels = [str(i) for i in orig_indices]

        if use_pareto_k and len(pareto_k_indices) > 0:
            high_k_mask = np.isin(orig_indices, pareto_k_indices)
        else:
            high_k_mask = np.zeros(len(influence), dtype=bool)

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    if threshold is not None:
        ax.axhline(y=threshold, color="r", linestyle="--")
        ax.text(
            0,
            threshold + 0.01,
            f"threshold = {threshold:.2f}",
            color="r",
            fontsize=label_fontsize,
        )

    ax.set_xlabel("Index", fontsize=label_fontsize)
    ax.set_ylabel(r"$-\text{elpd}_{\text{loo}}$", fontsize=label_fontsize)

    if var_name is not None:
        ax.set_title(f"Influence for {var_name}", fontsize=title_fontsize)
    else:
        ax.set_title("Influence", fontsize=title_fontsize)

    if use_pareto_k and hasattr(loo_results, "pareto_k") and np.any(high_k_mask):
        colors = np.array([color] * len(influence), dtype=object)
        colors[high_k_mask] = "red"  # Highlight high Pareto k observations

        for _, (xi, yi, ci) in enumerate(zip(x, influence, colors)):
            line_width = 1.0 + 1.0 * (
                yi / influence.max() if influence.max() > 0 else 0
            )
            ax.plot([xi, xi], [0, yi], color=ci, alpha=0.5, linewidth=line_width)

        regular_points = ~high_k_mask
        if np.any(regular_points):
            ax.scatter(
                x[regular_points],
                influence[regular_points],
                color=color,
                s=50,
                alpha=0.8,
                zorder=3,
                label="Regular observation",
                **kwargs,
            )

        if np.any(high_k_mask):
            ax.scatter(
                x[high_k_mask],
                influence[high_k_mask],
                color="red",
                s=50,
                alpha=0.8,
                zorder=4,
                label=f"Pareto k > {k_threshold}",
                **kwargs,
            )

        ax.legend(loc="best", fontsize=label_fontsize * 0.8)
    else:
        for _, (xi, yi) in enumerate(zip(x, influence)):
            line_width = 1.0 + 1.0 * (
                yi / influence.max() if influence.max() > 0 else 0
            )
            ax.plot([xi, xi], [0, yi], color=color, alpha=0.5, linewidth=line_width)

        ax.scatter(x, influence, color=color, s=50, alpha=0.8, zorder=3, **kwargs)

    if sort and labels is not None:
        if len(labels) <= 20:
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right")
        else:
            step = max(1, len(labels) // 10)
            ax.set_xticks(x[::step])
            ax.set_xticklabels(labels[::step], rotation=45, ha="right")

    if show:
        plt.tight_layout()
        plt.show()

    return ax
