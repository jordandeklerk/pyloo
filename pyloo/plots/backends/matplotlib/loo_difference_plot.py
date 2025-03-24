"""Matplotlib LOO difference plot."""

import matplotlib.pyplot as plt
import numpy as np


def plot_loo_difference(
    ax,
    x_values,
    elpd_diff,
    group=None,
    outlier_thresh=None,
    size=1,
    alpha=1,
    jitter=None,
    sort_by_group=False,
    figsize=None,
    textsize=None,
    backend_kwargs=None,
    show=None,
    **kwargs,
):
    """LOO difference plot.

    Parameters
    ----------
    ax : matplotlib axes, optional
        Axes to draw the plot on.
    x_values : array-like
        Values for the x-axis.
    elpd_diff : array-like
        Difference in ELPD values between two models.
    group : array-like, optional
        Grouping variable for coloring points.
    outlier_thresh : float, optional
        Threshold for flagging outliers.
    size : float, default 1
        Size of the points.
    alpha : float, default 1
        Transparency of the points.
    jitter : float or tuple, optional
        Amount of jitter to add to points. If a tuple, the first value is used for
        x-axis jitter and the second for y-axis jitter.
    sort_by_group : bool, default False
        Whether to sort observations by group.
    figsize : tuple, optional
        Figure size.
    textsize : float, optional
        Text size.
    backend_kwargs : dict, optional
        Additional keyword arguments for the backend.
    show : bool, optional
        Whether to show the plot.
    **kwargs : dict, optional
        Additional keyword arguments passed to matplotlib's scatter function.

    Returns
    -------
    ax : matplotlib axes
    """
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

    if jitter is None:
        jitter = (0, 0)
    elif not isinstance(jitter, (list, tuple)):
        jitter = (jitter, 0)

    if sort_by_group:
        if group is None:
            raise ValueError("sort_by_group=True requires group to be specified")

        if not np.array_equal(x_values, np.arange(len(x_values))):
            raise ValueError(
                "sort_by_group should only be used with an index variable (e.g.,"
                " 1:len(data))"
            )

        ordering = np.argsort(group)
        x_values = x_values[ordering]
        elpd_diff = elpd_diff[ordering]
        if group is not None:
            group = group[ordering]

    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.5)

    if group is None or group is False:
        x_jittered = x_values + np.random.uniform(
            -jitter[0], jitter[0], size=len(x_values)
        )
        y_jittered = elpd_diff + np.random.uniform(
            -jitter[1], jitter[1], size=len(elpd_diff)
        )

        ax.scatter(x_jittered, y_jittered, s=size * 50, alpha=alpha, **kwargs)
    else:
        x_jittered = x_values + np.random.uniform(
            -jitter[0], jitter[0], size=len(x_values)
        )
        y_jittered = elpd_diff + np.random.uniform(
            -jitter[1], jitter[1], size=len(elpd_diff)
        )

        unique_groups = np.unique(group)
        group_indices = np.array([np.where(unique_groups == g)[0][0] for g in group])

        scatter = ax.scatter(
            x_jittered, y_jittered, c=group_indices, s=size * 50, alpha=alpha, **kwargs
        )

        if len(unique_groups) <= 10:
            legend = ax.legend(
                scatter.legend_elements()[0],
                unique_groups,
                title="Groups",
                fontsize=label_fontsize * 0.8,
                loc="best",
            )
            ax.add_artist(legend)

    if outlier_thresh is not None:
        outlier_indices = np.where(elpd_diff > outlier_thresh)[0]
        if len(outlier_indices) > 0:
            for idx in outlier_indices:
                ax.annotate(
                    str(idx),
                    (x_jittered[idx], y_jittered[idx]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=label_fontsize * 0.8,
                )

    ax.set_xlabel("x" if not sort_by_group else "Index", fontsize=label_fontsize)
    ax.set_ylabel(r"$\text{ELPD}_{i1} - \text{ELPD}_{i2}$", fontsize=label_fontsize)
    ax.set_title("LOO Difference Plot", fontsize=title_fontsize)

    if show:
        plt.tight_layout()
        plt.show()

    return ax
