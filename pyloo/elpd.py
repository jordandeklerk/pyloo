"""Expected Log Pointwise Predictive Density (ELPD) data container modified from Arviz."""

from copy import copy as _copy
from copy import deepcopy as _deepcopy

import numpy as np
import pandas as pd

# Format for standard LOO output
STD_BASE_FMT = """
Computed from {n_samples} posterior samples and {n_points} observations log-likelihood matrix.

         Estimate       SE
elpd_loo   {elpd:<8.2f}    {se:<.2f}
p_loo       {p_loo:<8.2f}        -
looic      {looic:<8.2f}    {looic_se:<.2f}

{pareto_msg}"""

# Format for k-fold cross-validation output
KFOLD_BASE_FMT = """
Computed from {n_samples} posterior samples using {K}-fold cross-validation
with {n_points} observations.{stratify_msg}

           Estimate       SE
elpd_kfold   {elpd:<8.2f}    {se:<.2f}
p_kfold       {p_kfold:<8.2f}        -
kfoldic      {kfoldic:<8.2f}    {kfoldic_se:<.2f}
"""

# Format for subsampled LOO output
SUBSAMPLE_BASE_FMT = """
Computed from {n_samples} by {subsample_size} subsampled log-likelihood
values from {n_data_points} total observations.

         Estimate       SE  subsampling SE
elpd_loo   {0:<8.2f}    {1:<.2f}         {2:<.2f}
p_loo       {3:<8.2f}        -                -
looic      {4:<8.2f}    {5:<.2f}         {6:<.2f}

{pareto_msg}"""

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

        if kind not in ("loo", "waic", "kfold"):
            raise ValueError("Invalid ELPDData object")

        is_subsampled = "subsampling_SE" in self

        if kind == "kfold":
            K = getattr(self, "K", None)

            elpd_kfold = self["elpd_kfold"]
            se = self["se"]
            p_kfold = self["p_kfold"]
            p_kfold_se = self.get("se_p_kfold", float("nan"))
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

            if hasattr(self, "scale") and self.scale in SCALE_DICT:
                base += f"\n{SCALE_DICT[self.scale]}"

            return base

        elif is_subsampled:
            subsample_size = self["subsample_size"]
            pareto_msg = ""
            method = getattr(self, "method", "psis")

            if (
                "pareto_k" in self
                and hasattr(self, "good_k")
                and self.good_k is not None
            ):
                pareto_msg = ""
            elif method == "psis":
                pareto_msg = ""
            else:
                pareto_msg = f"Using {method.upper()} importance sampling method."

            elpd_loo = self["elpd_loo"]
            elpd_loo_se = self["se"]
            elpd_loo_subsamp_se = self["subsampling_SE"]

            p_loo = self["p_loo"]

            looic = -2 * elpd_loo
            looic_se = 2 * elpd_loo_se
            looic_subsamp_se = 2 * elpd_loo_subsamp_se

            base = SUBSAMPLE_BASE_FMT.format(
                elpd_loo,
                elpd_loo_se,
                elpd_loo_subsamp_se,
                p_loo,
                looic,
                looic_se,
                looic_subsamp_se,
                n_samples=self.n_samples,
                subsample_size=subsample_size,
                n_data_points=self.n_data_points,
                pareto_msg=pareto_msg,
                r_eff=self.get("r_eff", 1.0),
            )
        else:
            pareto_msg = ""
            method = getattr(self, "method", "psis")

            if (
                "pareto_k" in self
                and hasattr(self, "good_k")
                and self.good_k is not None
            ):
                pareto_msg = ""
            elif kind == "loo" and method == "psis":
                pareto_msg = ""
            elif kind == "loo":
                pareto_msg = f"Using {method.upper()} importance sampling method."
            else:
                pareto_msg = ""

            elpd_loo = self["elpd_loo"]
            se = self["se"]
            looic = -2 * elpd_loo
            looic_se = 2 * se

            base = STD_BASE_FMT.format(
                n_samples=self.n_samples,
                n_points=self.n_data_points,
                elpd=elpd_loo,
                se=se,
                p_loo=self["p_loo"],
                looic=looic,
                looic_se=looic_se,
                pareto_msg=pareto_msg,
                r_eff=self.get("r_eff", 1.0),
            )

        if self.warning:
            base += (
                "\nThere has been a warning during the calculation. Please check the"
                " results."
            )

        if (
            kind == "loo"
            and "pareto_k" in self
            and hasattr(self, "good_k")
            and self.good_k is not None
        ):
            bins = np.asarray([-np.inf, self.good_k, 1, np.inf])
            counts, *_ = _histogram(self.pareto_k.values, bins)
            percentages = counts / np.sum(counts) * 100

            extended = POINTWISE_LOO_FMT.format(
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
            base = "\n".join([base, extended])

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
