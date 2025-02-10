"""Diagnostics for Pareto smoothed importance sampling (PSIS)."""

import warnings
from dataclasses import dataclass
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from arviz.stats.diagnostics import _mc_error


@dataclass
class ParetokTable:
    """Container for Pareto k diagnostic table results.

    Attributes
    ----------
    counts : np.ndarray
        Number of observations in each k category
    proportions : np.ndarray
        Proportion of observations in each k category
    min_ess : np.ndarray
        Minimum effective sample size in each k category
    k_threshold : float
        Threshold value used for k categorization
    """

    counts: np.ndarray
    proportions: np.ndarray
    min_ess: np.ndarray
    k_threshold: float

    def __str__(self) -> str:
        """Format table as string."""
        header = "Pareto k diagnostic values:\n"

        if sum(self.counts[1:]) == 0:
            return header + f"All Pareto k estimates are good (k < {self.k_threshold:.2f}).\n"

        rows = []
        labels = ["(good)", "(bad)", "(very bad)"]
        for _, (count, prop, ess, label) in enumerate(zip(self.counts, self.proportions, self.min_ess, labels)):
            rows.append(
                f"{label:<12} {int(count):>5d} {prop * 100:>6.1f}% {int(ess) if not np.isnan(ess) else 'NA':>8}"
            )

        table = "\n".join(rows)
        return header + table


def compute_ess(log_weights: Union[np.ndarray, xr.DataArray]) -> Union[np.ndarray, xr.DataArray]:
    """Compute effective sample size from importance weights.

    Uses the formula 1/sum(w^2) where w are normalized importance weights.
    This is the correct formula for importance sampling ESS.

    Parameters
    ----------
    log_weights : array-like
        Array of log weights

    Returns
    -------
    array-like
        Effective sample size estimates
    """
    if isinstance(log_weights, xr.DataArray):
        log_weights = log_weights.values
    weights = np.exp(log_weights)
    weights = weights / weights.sum()
    return 1.0 / np.sum(weights**2)


def compute_mcse(
    log_weights: Union[np.ndarray, xr.DataArray], pareto_k: Union[np.ndarray, xr.DataArray]
) -> Union[np.ndarray, xr.DataArray]:
    """Compute Monte Carlo standard error.

    Uses ArviZ's MCSE calculation.

    Parameters
    ----------
    log_weights : array-like
        Array of log weights
    pareto_k : array-like
        Array of Pareto k estimates

    Returns
    -------
    array-like
        Monte Carlo standard error estimates
    """
    if isinstance(log_weights, xr.DataArray):
        log_weights = log_weights.values
    if isinstance(pareto_k, xr.DataArray):
        pareto_k = pareto_k.values

    weights = np.exp(log_weights)
    mcse = _mc_error(weights)

    mcse[~np.isfinite(pareto_k)] = np.inf
    return mcse


def pareto_k_table(
    log_weights: Union[np.ndarray, xr.DataArray], pareto_k: Union[np.ndarray, xr.DataArray]
) -> ParetokTable:
    """Compute diagnostic table summarizing Pareto k estimates and effective sample sizes.

    Parameters
    ----------
    log_weights : array-like
        Array of log weights
    pareto_k : array-like
        Array of Pareto k estimates

    Returns
    -------
    ParetokTable
        Table containing counts, proportions and minimum effective sample sizes
        for different ranges of k values
    """
    if isinstance(log_weights, xr.DataArray):
        log_weights = log_weights.values
    if isinstance(pareto_k, xr.DataArray):
        pareto_k = pareto_k.values

    S = len(log_weights)
    k_threshold = ps_khat_threshold(S)
    k_categories = k_cut(pareto_k, k_threshold)

    ess_vals = compute_ess(log_weights)
    ess_vals[pareto_k > k_threshold] = np.nan

    all_counts = np.zeros(3, dtype=int)
    all_proportions = np.zeros(3)
    all_min_ess = np.full(3, np.nan)

    for cat in range(3):
        mask = k_categories == cat
        if np.any(mask):
            all_counts[cat] = np.sum(mask)
            all_proportions[cat] = np.sum(mask) / len(pareto_k)
            all_min_ess[cat] = np.nanmin(ess_vals[mask])

    return ParetokTable(counts=all_counts, proportions=all_proportions, min_ess=all_min_ess, k_threshold=k_threshold)


def pareto_k_ids(
    pareto_k: Union[np.ndarray, xr.DataArray], n_samples: int, threshold: Optional[float] = None
) -> np.ndarray:
    """Find indices of observations with Pareto k estimates above threshold.

    Parameters
    ----------
    pareto_k : array-like
        Array of Pareto k estimates
    n_samples : int
        Number of samples used to compute Pareto k estimates
    threshold : float, optional
        Threshold for flagging k values. If None, uses sample size dependent
        threshold min(1 - 1/log10(S), 0.7).

    Returns
    -------
    np.ndarray
        Indices of observations with k > threshold
    """
    if isinstance(pareto_k, xr.DataArray):
        pareto_k = pareto_k.values

    if threshold is None:
        threshold = ps_khat_threshold(n_samples)
    return np.where(pareto_k > threshold)[0]


def mcse_loo(
    log_weights: Union[np.ndarray, xr.DataArray],
    pareto_k: Union[np.ndarray, xr.DataArray],
    n_samples: int,
    threshold: Optional[float] = None,
) -> float:
    """Compute Monte Carlo standard error for PSIS-LOO.

    Parameters
    ----------
    log_weights : array-like
        Array of log weights
    pareto_k : array-like
        Array of Pareto k estimates
    n_samples : int
        Number of samples used to compute Pareto k estimates
    threshold : float, optional
        If any k values are above this threshold, returns NaN.
        If None, uses sample size dependent threshold.

    Returns
    -------
    float
        Monte Carlo standard error estimate, or NaN if any k values exceed threshold
    """
    if isinstance(pareto_k, xr.DataArray):
        pareto_k = pareto_k.values

    if threshold is None:
        threshold = ps_khat_threshold(n_samples)

    if np.any(pareto_k > threshold):
        return np.nan

    mcse = compute_mcse(log_weights, pareto_k)
    return np.sqrt(np.sum(mcse**2))


def plot_diagnostic(
    log_weights: Union[np.ndarray, xr.DataArray],
    pareto_k: Union[np.ndarray, xr.DataArray],
    diagnostic: str = "k",
    label_points: bool = False,
    title: str = "PSIS diagnostic plot",
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> plt.Axes:
    """Plot PSIS diagnostics.

    Parameters
    ----------
    log_weights : array-like
        Array of log weights
    pareto_k : array-like
        Array of Pareto k estimates
    diagnostic : str, optional
        Which diagnostic to plot:
        - "k": Pareto shape parameters (default)
        - "ess": Effective sample sizes
    label_points : bool, optional
        If True, label points with k values above threshold
    title : str, optional
        Plot title
    ax : plt.Axes, optional
        Matplotlib axes to plot on. If None, creates new figure.
    **kwargs
        Additional arguments passed to plt.text() for point labels

    Returns
    -------
    plt.Axes
        The matplotlib axes containing the plot
    """
    if ax is None:
        _, ax = plt.subplots()

    if isinstance(pareto_k, xr.DataArray):
        pareto_k = pareto_k.values
    k = pareto_k
    k[np.isnan(k)] = 0  # Replace NaN with 0 for plotting
    k_inf = ~np.isfinite(k)

    if np.any(k_inf):
        warnings.warn(
            f"{100 * np.mean(k_inf):.1f}% of Pareto k estimates are Inf/NA/NaN " "and not plotted.",
            stacklevel=2,
        )

    S = len(log_weights)
    k_threshold = ps_khat_threshold(S)

    if diagnostic.lower() in ("ess", "ess"):
        y = compute_ess(log_weights)
        ylabel = "PSIS ESS"
    else:
        y = k
        ylabel = "Pareto shape k"

    x_coords = np.arange(len(y))
    colors = np.where(k <= k_threshold, "#6497b1", np.where(k <= 1, "#005b96", "#03396c"))
    ax.scatter(x_coords, y, c=colors, marker="x", s=20, alpha=0.6)

    if diagnostic.lower() == "k":
        ymin, ymax = ax.get_ylim()
        for val, style in [(0, "darkgray"), (k_threshold, "#C79999"), (1.0, "#7C0000")]:
            if ymin <= val <= ymax:
                ax.axhline(val, color=style, linestyle="--", alpha=0.5)

    # Label points above threshold
    if label_points and not np.all(k < k_threshold):
        bad_idx = np.where(k > k_threshold)[0]
        text_kwargs = {"va": "center", "alpha": 0.7, "fontsize": 8}
        text_kwargs.update(kwargs)
        for idx in bad_idx:
            ax.text(idx, y[idx], str(idx), **text_kwargs)

    ax.set_xlabel("Data point")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    return ax


def k_cut(k: np.ndarray, threshold: float) -> np.ndarray:
    """Categorize Pareto k values into ranges.

    Parameters
    ----------
    k : np.ndarray
        Array of Pareto k values
    threshold : float
        Threshold for good/bad/very bad categorization

    Returns
    -------
    np.ndarray
        Array of same length as k with values:
        0: k <= threshold (good)
        1: threshold < k <= 1 (bad)
        2: k > 1 (very bad)
    """
    categories = np.zeros_like(k, dtype=int)
    categories[k > threshold] = 1
    categories[k > 1] = 2
    return categories


def ps_khat_threshold(S: int) -> float:
    """Compute sample size dependent threshold for Pareto k values.

    Parameters
    ----------
    S : int
        Sample size

    Returns
    -------
    float
        Threshold value min(1 - 1/log10(S), 0.7)

    Notes
    -----
    Given sample size S computes khat threshold for reliable Pareto smoothed estimate
    (to have small probability of large error). Sample sizes 100, 320, 1000, 2200,
    10000 correspond to thresholds 0.5, 0.6, 0.67, 0.7, 0.75. Although with bigger
    sample size S we can achieve estimates with small probability of large error, it
    is difficult to get accurate MCSE estimates as the bias starts to dominate when
    k > 0.7. Thus the sample size dependent k-hat threshold is capped at 0.7.
    """
    if S <= 0:
        raise ValueError("Sample size must be positive")
    return float(min(1 - 1 / np.log(S), 0.7))
