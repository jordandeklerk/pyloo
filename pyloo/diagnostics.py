"""Diagnostics for Pareto smoothed importance sampling (PSIS)."""

import warnings
from typing import Any, Dict, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .psis import PSISObject


class DiagnosticData(pd.Series):
    """Container for PSIS diagnostic data including Pareto k values and effective sample sizes.

    This class extends pandas.Series to provide a specialized container for PSIS diagnostic
    information, particularly Pareto k values and effective sample sizes. It includes methods
    for pretty printing, validation, and analysis of the results.

    Required Data
    ------------
    The Series must contain:
    - pareto_k: array of Pareto k values
    - n_eff: array of effective sample sizes
    - k_threshold: threshold for good Pareto k values (computed based on sample size)

    Notes
    -----
    The reliability of PSIS-based estimates can be assessed using the shape parameter k
    of the generalized Pareto distribution. The diagnostic threshold depends on sample
    size S:

    * If k < min(1 - 1/log10(S), 0.7), the PSIS estimate and Monte Carlo standard error
      are reliable.

    * If 1 - 1/log10(S) <= k < 0.7, estimates are not reliable, but increasing sample
      size S above 2200 may help (this increases the sample size specific threshold
      above 0.7).

    * If 0.7 <= k < 1, estimates have large bias and are not reliable. Increasing sample
      size may reduce variability in k estimate.

    * If k >= 1, the target distribution is estimated to have infinite mean. Estimates
      are not well defined.

    What to do when k exceeds the threshold:

    1. Transform MCMC draws using moment matching to obtain more reliable importance
       sampling estimates. See loo_moment_match().

    2. Use mixture estimators to improve stability. See loo_mixture().

    3. Use K-fold cross-validation which will generally be more stable.

    4. Consider using a more robust model that is less sensitive to influential
       observations.

    The effective sample size (ESS) estimates will be over-optimistic when k is greater
    than min(1-1/log10(S), 0.7).

    See Also
    --------
    pareto_k_table : Create diagnostic table from Pareto k values
    plot_khat : Plot Pareto k values with diagnostic thresholds
    """

    @property
    def k_threshold(self) -> float:
        """Get the Pareto k threshold value."""
        return self["k_threshold"]

    def has_warnings(self) -> bool:
        """Check if any k values exceed the threshold."""
        return np.any(self.pareto_k > self.k_threshold)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of Pareto k values.

        Returns
        -------
        dict
            Dictionary containing:
            - k_threshold: Threshold for reliable k values
            - counts: Number of values in each category (good, bad, very bad)
            - proportions: Proportion of values in each category
            - min_n_eff: Minimum effective sample size in each category
        """
        bins = np.array([-np.inf, self.k_threshold, 1, np.inf])
        counts = np.histogram(self.pareto_k, bins=bins)[0]
        proportions = counts / len(self.pareto_k)

        min_n_eff = np.full(3, np.nan)
        for i in range(3):
            mask = (self.pareto_k > bins[i]) & (self.pareto_k <= bins[i + 1])
            if np.any(mask):
                min_n_eff[i] = np.nanmin(self.n_eff[mask])

        return {
            "k_threshold": self.k_threshold,
            "counts": counts,
            "proportions": proportions,
            "min_n_eff": min_n_eff,
            "categories": ["good", "bad", "very bad"],
        }

    def __str__(self) -> str:
        """Format diagnostic data as string."""
        summary = self.get_summary()

        if sum(summary["counts"][1:]) == 0:
            return f"\nAll Pareto k estimates are good (k < {summary['k_threshold']:.2f}).\n"

        rows = []
        rows.append("Pareto k diagnostic values:")
        rows.append("                         Count    Pct.   Min. ESS")

        labels = ["(good)", "(bad)", "(very bad)"]
        ranges = [f"(-Inf, {summary['k_threshold']:.2f}]", f"({summary['k_threshold']:.2f}, 1.00]", "(1.00, Inf)"]

        for i in range(3):
            count = summary["counts"][i]
            pct = summary["proportions"][i] * 100
            n_eff = summary["min_n_eff"][i]
            n_eff_str = f"{n_eff:.0f}" if np.isfinite(n_eff) else "-"
            rows.append(f"{ranges[i]:>16} {labels[i]:<9} {count:4d}  {pct:5.1f}%  {n_eff_str:>8}")

        return "\n".join(rows)


def ps_khat_threshold(S: int) -> float:
    """Compute k-hat threshold for reliable Pareto smoothed estimate.

    Parameters
    ----------
    S : int
        Sample size

    Returns
    -------
    float
        Threshold value. Sample sizes 100, 320, 1000, 2200, 10000 correspond to
        thresholds 0.5, 0.6, 0.67, 0.7, 0.75. Although with bigger sample size S
        we can achieve estimates with small probability of large error, it is
        difficult to get accurate MCSE estimates as the bias starts to dominate
        when k > 0.7 (see Section 3.2.3). Thus the sample size dependent k-hat
        threshold is capped at 0.7.

    Notes
    -----
    The threshold is computed as min(1 - 1/log10(S), 0.7) where S is the sample size.
    This provides an optimistic threshold if the effective sample size is less than
    2200, but if MCMC-ESS > S/2 the difference is usually negligible. Thinning of
    MCMC draws can be used to improve the ratio ESS/S.
    """
    return min(1 - 1 / np.log10(S), 0.7)


def pareto_k_values(x: Union[dict, PSISObject]) -> np.ndarray:
    """Extract Pareto k values from diagnostics.

    Parameters
    ----------
    x : Union[dict, PSISObject]
        Object containing Pareto k values, either in diagnostics["pareto_k"]
        or as a PSISObject

    Returns
    -------
    np.ndarray
        Array of Pareto k values

    Raises
    ------
    ValueError
        If no Pareto k values found
    """
    if isinstance(x, PSISObject):
        return x.pareto_k
    elif isinstance(x, dict):
        k = x.get("diagnostics", {}).get("pareto_k")
        if k is None:
            raise ValueError("No Pareto k estimates found.")
        return np.asarray(k)
    else:
        raise TypeError("Input must be a dict or PSISObject")


def pareto_k_ids(x: Union[dict, PSISObject], threshold: Optional[float] = None) -> np.ndarray:
    """Find indices of observations with Pareto k above threshold.

    Parameters
    ----------
    x : Union[dict, PSISObject]
        Object containing Pareto k values
    threshold : Optional[float]
        Threshold value. If None, computed based on sample size.

    Returns
    -------
    np.ndarray
        Indices where k > threshold
    """
    k = pareto_k_values(x)
    if threshold is None:
        S = len(k)
        threshold = ps_khat_threshold(S)
    return np.where(k > threshold)[0]


def pareto_k_influence_values(x: Union[dict, PSISObject]) -> np.ndarray:
    """Extract Pareto k influence values.

    Parameters
    ----------
    x : Union[dict, PSISObject]
        Object containing influence values in pointwise["influence_pareto_k"]

    Returns
    -------
    np.ndarray
        Array of influence values

    Raises
    ------
    ValueError
        If no influence values found

    Notes
    -----
    The influence values measure each observation's impact on the posterior
    distribution. Higher values indicate observations with greater influence
    on the model fit.
    """
    if isinstance(x, PSISObject):
        raise ValueError("Influence values not available in PSISObject")
    elif isinstance(x, dict):
        if "pointwise" not in x or "influence_pareto_k" not in x["pointwise"]:
            raise ValueError("No Pareto k influence estimates found.")
        return np.asarray(x["pointwise"]["influence_pareto_k"])
    else:
        raise TypeError("Input must be a dict or PSISObject")


def psis_n_eff_values(x: Union[dict, PSISObject]) -> np.ndarray:
    """Extract PSIS effective sample sizes.

    Parameters
    ----------
    x : Union[dict, PSISObject]
        Object containing n_eff values in diagnostics["n_eff"]

    Returns
    -------
    np.ndarray
        Array of effective sample sizes

    Raises
    ------
    ValueError
        If no n_eff values found

    Notes
    -----
    The effective sample size estimates will be over-optimistic when the
    corresponding k value exceeds min(1-1/log10(S), 0.7). When using MCMC
    samples, these provide more accurate estimates than simple importance
    sampling or truncated importance sampling.
    """
    if isinstance(x, PSISObject):
        if x.n_eff is None:
            raise ValueError("No PSIS ESS estimates found.")
        return x.n_eff
    elif isinstance(x, dict):
        n_eff = x.get("diagnostics", {}).get("n_eff")
        if n_eff is None:
            raise ValueError("No PSIS ESS estimates found.")
        return np.asarray(n_eff)
    else:
        raise TypeError("Input must be a dict or PSISObject")


def mcse_loo(x: Union[dict, PSISObject], threshold: Optional[float] = None) -> float:
    """Compute Monte Carlo standard error for PSIS-LOO.

    Parameters
    ----------
    x : Union[dict, PSISObject]
        Object containing Pareto k values and mcse values
    threshold : Optional[float]
        Threshold for Pareto k values. If None, computed based on sample size.

    Returns
    -------
    float
        Monte Carlo standard error estimate, or np.nan if any k values are
        above threshold

    Notes
    -----
    The Monte Carlo standard error estimate is only reliable when all k values
    are below the threshold. When any k value exceeds the threshold, the
    estimate is returned as np.nan to indicate potential unreliability.
    """
    k = pareto_k_values(x)
    if threshold is None:
        S = len(k)
        threshold = ps_khat_threshold(S)

    if np.any(k > threshold):
        return np.nan

    if isinstance(x, PSISObject):
        raise ValueError("MCSE values not available in PSISObject")
    elif isinstance(x, dict):
        mc_var = x["pointwise"]["mcse_elpd_loo"] ** 2
        return float(np.sqrt(np.sum(mc_var)))
    else:
        raise TypeError("Input must be a dict or PSISObject")


def pareto_k_table(x: Union[dict, PSISObject]) -> DiagnosticData:
    """Create diagnostic table for Pareto k values.

    Creates a table summarizing the reliability of PSIS-LOO-CV estimates based on their
    Pareto k values. The k values are categorized into three groups:
    - good (k ≤ k_threshold): reliable estimates
    - bad (k_threshold < k ≤ 1): somewhat reliable, proceed with caution
    - very bad (k > 1): unreliable estimates

    Parameters
    ----------
    x : Union[dict, PSISObject]
        Object containing Pareto k values and n_eff values

    Returns
    -------
    DiagnosticData
        Table object containing diagnostic summary with counts, proportions, and minimum
        effective sample sizes for each k category

    Examples
    --------
    Create a diagnostic table from PSIS-LOO results:

    .. ipython::

        In [1]: import numpy as np
           ...: from pyloo import loo, pareto_k_table
           ...: # Example log-likelihood matrix (n_samples x n_observations)
           ...: log_lik = np.random.normal(size=(1000, 100))
           ...: loo_result = loo(log_lik)
           ...: diag_table = pareto_k_table(loo_result)
           ...: print(diag_table)

    See Also
    --------
    DiagnosticData : Container class for Pareto k diagnostics
    plot_khat : Plot Pareto k values with diagnostic thresholds
    """
    k = pareto_k_values(x)
    try:
        n_eff = psis_n_eff_values(x)
    except ValueError:
        n_eff = np.full_like(k, np.nan)

    S = len(k)
    k_threshold = ps_khat_threshold(S)

    data = pd.Series({"pareto_k": k, "n_eff": n_eff, "k_threshold": k_threshold})

    return DiagnosticData(data)


def plot_khat(
    x: Union[dict, PSISObject],
    diagnostic: str = "k",
    label_points: bool = False,
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> Optional[plt.Axes]:
    """Plot Pareto k values or effective sample sizes.

    Creates a diagnostic plot showing either Pareto k values or effective sample sizes
    for each data point. For k values, horizontal lines indicate important thresholds:
    - Dotted gray line at k = 0: minimum possible value
    - Dashed pink line at k = k_threshold: reliability threshold
    - Dashed red line at k = 1: maximum reliable value
    Points are colored based on their k value category (good, bad, very bad).

    Parameters
    ----------
    x : Union[dict, PSISObject]
        Object containing diagnostic values
    diagnostic : str
        Which diagnostic to plot:
        - "k": Pareto k values (default)
        - "n_eff" or "ESS": Effective sample sizes
    label_points : bool
        Whether to label points with high k values
    ax : Optional[plt.Axes]
        Matplotlib axes to plot on. If None, creates new figure.
    **kwargs
        Additional arguments passed to plt.text() for point labels

    Returns
    -------
    Optional[plt.Axes]
        Matplotlib axes object if ax was None

    Examples
    --------
    Plot Pareto k values from PSIS-LOO results:

    .. ipython::

        In [1]: import numpy as np
           ...: from pyloo import loo, plot_khat
           ...: # Example log-likelihood matrix (n_samples x n_observations)
           ...: log_lik = np.random.normal(size=(1000, 100))
           ...: loo_result = loo(log_lik)
           ...: plot_khat(loo_result)

    Plot effective sample sizes and label problematic points:

    .. ipython::

        In [1]: plot_khat(loo_result, diagnostic="n_eff", label_points=True)

    See Also
    --------
    pareto_k_table : Create diagnostic table from Pareto k values
    DiagnosticData : Container class for Pareto k diagnostics
    """
    if diagnostic in ("n_eff", "ESS"):
        y = psis_n_eff_values(x)
        ylabel = "PSIS ESS"
        use_n_eff = True
    else:
        y = pareto_k_values(x)
        ylabel = "Pareto shape k"
        use_n_eff = False

    S = len(y)
    k_threshold = ps_khat_threshold(S)

    y_inf = ~np.isfinite(y)
    if np.any(y_inf):
        pct = 100 * np.mean(y_inf)
        warnings.warn(f"{pct:.1f}% of values are Inf/NA/NaN and not plotted.", stacklevel=2)
        y = y[~y_inf]

    created_ax = False
    if ax is None:
        _, ax = plt.subplots()
        created_ax = True

    x_vals = np.arange(len(y))
    ax.set_xlabel("Data point")
    ax.set_ylabel(ylabel)

    if not use_n_eff:
        for val, style in [(0, "darkgray"), (k_threshold, "#C79999"), (1, "#7C0000")]:
            if val in (k_threshold, 1):
                ax.axhline(val, color=style, linestyle="--", alpha=0.7)
            else:
                ax.axhline(val, color=style, linestyle=":", alpha=0.5)

    colors = np.where(y < k_threshold, "#6497b1", np.where(y < 1, "#005b96", "#03396c"))
    ax.scatter(x_vals, y, c=colors, marker="x", s=20, alpha=0.7)

    if label_points and not use_n_eff:
        high_k = y > k_threshold
        if np.any(high_k):
            text_kwargs = {"va": "center", "alpha": 0.7}
            text_kwargs.update(kwargs)
            for i, yi in zip(x_vals[high_k], y[high_k]):
                ax.text(i, yi, str(i), **text_kwargs)

    if created_ax:
        return ax
    return None
