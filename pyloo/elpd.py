"""Expected Log Pointwise Predictive Density (ELPD) data container modified from Arviz."""

from copy import copy as _copy
from copy import deepcopy as _deepcopy

import numpy as np
import pandas as pd

# Standard LOO output
STD_BASE_FMT = """
Computed from {n_samples} posterior samples and {n_points} observations log-likelihood matrix.

         Estimate       SE
elpd_loo   {elpd:<8.2f}    {se:<.2f}
p_loo       {p_loo:<8.2f}    {p_loo_se:<.2f}
looic      {looic:<8.2f}    {looic_se:<.2f}"""


# Subsampled LOO output
SUBSAMPLE_BASE_FMT = """
Computed from {n_samples} by {subsample_size} subsampled log-likelihood
values from {n_data_points} total observations.

         Estimate       SE  subsampling SE
elpd_loo   {elpd_loo:<8.2f}    {elpd_loo_se:<.2f}         {elpd_loo_subsamp_se:<.2f}
p_loo       {p_loo:<8.2f}    {p_loo_se:<.2f}         {p_loo_subsamp_se:<.2f}
looic      {looic:<8.2f}    {looic_se:<.2f}         {looic_subsamp_se:<.2f}
{pareto_msg}"""


# LOO approximate posterior output
APPROX_POSTERIOR_FMT = """
Computed from {n_samples} posterior samples and {n_points} observations log-likelihood matrix.
Posterior approximation correction used.
------

         Estimate       SE
elpd_loo   {elpd:<8.2f}    {se:<.2f}
p_loo       {p_loo:<8.2f}    {p_loo_se:<.2f}
looic      {looic:<8.2f}    {looic_se:<.2f}"""


# k-fold cross-validation output
KFOLD_BASE_FMT = """
Computed from {n_samples} posterior samples using {K}-fold cross-validation
with {n_points} observations.{stratify_msg}

           Estimate       SE
elpd_kfold   {elpd:<8.2f}    {se:<.2f}
p_kfold       {p_kfold:<8.2f}    {p_kfold_se:<.2f}
kfoldic      {kfoldic:<8.2f}    {kfoldic_se:<.2f}
"""

# LOGO (Leave-One-Group-Out) output
LOGO_BASE_FMT = """
Computed from {n_samples} posterior samples and {n_groups} groups log-likelihood matrix.

         Estimate       SE
elpd_logo   {elpd:<8.2f}    {se:<.2f}
p_logo       {p_logo:<8.2f}    {p_logo_se:<.2f}
logoic      {logoic:<8.2f}    {logoic_se:<.2f}"""


POINTWISE_LOO_FMT = """
------

Pareto k diagnostic values:
                         Count   Pct.
(-Inf, {2:.2f}]   (good)      {3:d}   {6:.1f}%
   ({2:.2f}, 1]   (bad)         {4:d}    {7:.1f}%
   (1, Inf)   (very bad)    {5:d}    {8:.1f}%"""

SCALE_DICT = {
    "log": "Using log score",
    "negative_log": "Using negative log score",
    "deviance": "Using deviance score",
}


class ELPDData(pd.Series):
    """Class to contain ELPD information criterion data.

    This class extends pandas.Series to store and display expected log pointwise
    predictive density (ELPD) information, supporting multiple importance sampling
    methods (PSIS, TIS, SIS) and different scales.

    Parameters
    ----------
    data : array-like
        Input data containing ELPD values and diagnostics
    index : array-like
        Index labels for the data
    """

    def __str__(self):
        """Print ELPD data in a user friendly way.

        Returns
        -------
        str
            Formatted string containing ELPD information and diagnostics
        """
        kind = self.index[0].split("_")[1]

        if kind not in ("loo", "waic", "kfold", "logo"):
            raise ValueError("Invalid ELPDData object")

        is_subsampled = "subsampling_SE" in self

        if kind == "kfold":
            K = getattr(self, "K", None)

            elpd_kfold = self["elpd_kfold"]
            se = self["se"]
            p_kfold = self["p_kfold"]
            p_kfold_se = self["p_kfold_se"]
            kfoldic = -2 * elpd_kfold
            kfoldic_se = 2 * se

            stratify_msg = ""
            if self.stratified:
                stratify_msg = " Using stratified k-fold cross-validation"

            base = KFOLD_BASE_FMT.format(
                n_samples=self.n_samples,
                K=K,
                n_points=self.n_data_points,
                elpd=elpd_kfold,
                se=se,
                p_kfold=p_kfold,
                p_kfold_se=p_kfold_se,
                kfoldic=kfoldic,
                kfoldic_se=kfoldic_se,
                stratify_msg=stratify_msg,
            )

            if self.warning:
                base += (
                    "\n\nThere has been a warning during the calculation. Please check"
                    " the results."
                )

            return base

        elif kind == "logo":
            elpd_logo = self["elpd_logo"]
            se = self["se"]
            p_logo = self["p_logo"]
            p_logo_se = self.get("p_logo_se", float("nan"))
            logoic = self["logoic"]
            logoic_se = self["logoic_se"]

            base = LOGO_BASE_FMT.format(
                n_samples=self.n_samples,
                n_groups=self.n_groups,
                elpd=elpd_logo,
                se=se,
                p_logo=p_logo,
                p_logo_se=p_logo_se,
                logoic=logoic,
                logoic_se=logoic_se,
            )

            if self.warning:
                base += (
                    "\n\nThere has been a warning during the calculation. Please check"
                    " the results."
                )
            if (
                "pareto_k" in self
                and hasattr(self, "good_k")
                and self.good_k is not None
            ):
                bins = np.asarray([-np.inf, self.good_k, 1, np.inf])
                pareto_k_values = (
                    self.pareto_k.values
                    if hasattr(self.pareto_k, "values")
                    else self.pareto_k
                )
                counts, *_ = _histogram(pareto_k_values, bins)
                if counts[1] == 0 and counts[2] == 0:
                    base += (
                        f"\n\nAll Pareto k estimates are good (k < {self.good_k:.1f})."
                        "\nSee help('pareto-k-diagnostic') for details."
                    )
                else:
                    percentages = counts / np.sum(counts) * 100
                    base += POINTWISE_LOO_FMT.format(
                        "Count",
                        "Pct.",
                        self.good_k,
                        counts[0],
                        counts[1],
                        counts[2],
                        percentages[0],
                        percentages[1],
                        percentages[2],
                    )

            return base

        elif is_subsampled:
            subsample_size = self["subsample_size"]
            method = getattr(self, "method", "psis")
            default_good_k = 0.7
            pareto_msg = (
                f"\n\nAll Pareto k estimates are good (k < {default_good_k:.1f}).\nSee"
                " help('pareto-k-diagnostic') for details."
            )

            if (
                "pareto_k" in self
                and hasattr(self, "good_k")
                and self.good_k is not None
            ):
                bins = np.asarray([-np.inf, self.good_k, 1, np.inf])
                pareto_k_values = (
                    self.pareto_k.values
                    if hasattr(self.pareto_k, "values")
                    else self.pareto_k
                )
                counts, *_ = _histogram(pareto_k_values, bins)
                if counts[1] == 0 and counts[2] == 0:
                    pass
                else:
                    percentages = counts / np.sum(counts) * 100
                    pareto_msg = POINTWISE_LOO_FMT.format(
                        "Count",
                        "Pct.",
                        self.good_k,
                        counts[0],
                        counts[1],
                        counts[2],
                        percentages[0],
                        percentages[1],
                        percentages[2],
                    )

            elpd_loo = self["elpd_loo"]
            elpd_loo_se = self["se"]
            elpd_loo_subsamp_se = self["subsampling_SE"]

            p_loo = self["p_loo"]
            p_loo_se = self.get("p_loo_se", float("nan"))
            p_loo_subsamp_se = self.get("p_loo_subsampling_se", float("nan"))

            looic = -2 * elpd_loo
            looic_se = 2 * elpd_loo_se
            looic_subsamp_se = 2 * elpd_loo_subsamp_se

            base = SUBSAMPLE_BASE_FMT.format(
                elpd_loo=elpd_loo,
                elpd_loo_se=elpd_loo_se,
                elpd_loo_subsamp_se=elpd_loo_subsamp_se,
                p_loo=p_loo,
                p_loo_se=p_loo_se,
                p_loo_subsamp_se=p_loo_subsamp_se,
                looic=looic,
                looic_se=looic_se,
                looic_subsamp_se=looic_subsamp_se,
                n_samples=self.n_samples,
                subsample_size=subsample_size,
                n_data_points=self.n_data_points,
                pareto_msg=pareto_msg,
                r_eff=self.get("r_eff", 1.0),
            )

            if self.warning:
                base += (
                    "\n\nThere has been a warning during the calculation. Please check"
                    " the results."
                )

            return base
        else:
            method = getattr(self, "method", "psis")
            default_good_k = 0.7
            pareto_msg = ""

            if (
                "pareto_k" in self
                and hasattr(self, "good_k")
                and self.good_k is not None
            ):
                bins = np.asarray([-np.inf, self.good_k, 1, np.inf])
                pareto_k_values = (
                    self.pareto_k.values
                    if hasattr(self.pareto_k, "values")
                    else self.pareto_k
                )
                counts, *_ = _histogram(pareto_k_values, bins)
                if counts[1] == 0 and counts[2] == 0:
                    pareto_msg = (
                        "\n\nAll Pareto k estimates are good (k <"
                        f" {self.good_k:.1f}).\nSee help('pareto-k-diagnostic') for"
                        " details."
                    )
                else:
                    percentages = counts / np.sum(counts) * 100
                    pareto_msg = POINTWISE_LOO_FMT.format(
                        "Count",
                        "Pct.",
                        self.good_k,
                        counts[0],
                        counts[1],
                        counts[2],
                        percentages[0],
                        percentages[1],
                        percentages[2],
                    )
            elif kind == "loo" and method == "psis":
                if self.warning:
                    pareto_msg = (
                        "\n\nSome Pareto k diagnostic values are high (k >"
                        f" {default_good_k:.1f}), indicating that the importance"
                        " sampling approximation is unreliable. Consider using moment"
                        " matching or exact LOO for more accurate estimates. Use"
                        " pointwise=True to see detailed diagnostics."
                    )
                else:
                    pareto_msg = (
                        "\n\nAll Pareto k estimates are good (k <"
                        f" {default_good_k:.1f}).\nSee help('pareto-k-diagnostic') for"
                        " details."
                    )

            elpd_loo = self["elpd_loo"]
            se = self["se"]

            if hasattr(self, "approximate_posterior"):
                looic = self["looic"]
                looic_se = self["looic_se"]
                base = APPROX_POSTERIOR_FMT.format(
                    n_samples=self.n_samples,
                    n_points=self.n_data_points,
                    elpd=elpd_loo,
                    se=se,
                    p_loo=self["p_loo"],
                    p_loo_se=self["p_loo_se"],
                    looic=looic,
                    looic_se=looic_se,
                )
            else:
                if "p_loo" not in self:
                    base = """
Computed from {n_samples} posterior samples and {n_points} observations log-likelihood matrix with
mixture posterior.

         Estimate       SE
elpd_loo   {elpd:<8.2f}    -""".format(
                        n_samples=self.n_samples,
                        n_points=self.n_data_points,
                        elpd=elpd_loo,
                    )
                else:
                    looic = self["looic"]
                    looic_se = self["looic_se"]
                    base = STD_BASE_FMT.format(
                        n_samples=self.n_samples,
                        n_points=self.n_data_points,
                        elpd=elpd_loo,
                        se=se,
                        p_loo=self["p_loo"],
                        p_loo_se=self["p_loo_se"],
                        looic=looic,
                        looic_se=looic_se,
                    )

            if self.warning:
                base += (
                    "\n\nThere has been a warning during the calculation. Please check"
                    " the results."
                )
            base += pareto_msg

            return base

    def __repr__(self):
        """Return string representation."""
        return self.__str__()

    def copy(self, deep=True):
        """Create a copy of the ELPDData object."""
        copied_obj = pd.Series.copy(self)
        for key in copied_obj.keys():
            if deep:
                copied_obj[key] = _deepcopy(copied_obj[key])
            else:
                copied_obj[key] = _copy(copied_obj[key])
        return ELPDData(copied_obj)

    @property
    def n_samples(self):
        """Get number of samples."""
        return self["n_samples"]

    @property
    def n_data_points(self):
        """Get number of data points."""
        return self["n_data_points"]

    @property
    def n_groups(self):
        """Get number of groups."""
        return self.get("n_groups", None)

    @property
    def warning(self):
        """Get warning status."""
        return self["warning"]

    @property
    def method(self):
        """Get importance sampling method used."""
        return getattr(self, "_method", "psis")

    @method.setter
    def method(self, value):
        """Set importance sampling method."""
        self._method = value

    @property
    def estimates(self):
        """Get the estimation results."""
        return self._estimates

    @estimates.setter
    def estimates(self, value):
        """Set the estimation results."""
        self._estimates = value

    @property
    def K(self):
        """Get the number of folds for k-fold cross-validation."""
        return getattr(self, "_K", None)

    @K.setter
    def K(self, value):
        """Set the number of folds."""
        self._K = value

    @property
    def stratified(self):
        """Get whether stratified k-fold cross-validation was used."""
        return getattr(self, "_stratified", False)

    @stratified.setter
    def stratified(self, value):
        """Set whether stratified k-fold cross-validation was used."""
        self._stratified = value


def _histogram(data, bins, range_hist=None):
    """Histogram."""
    hist, bin_edges = np.histogram(data, bins=bins, range=range_hist)
    hist_dens = hist / (hist.sum() * np.diff(bin_edges))
    return hist, hist_dens, bin_edges
