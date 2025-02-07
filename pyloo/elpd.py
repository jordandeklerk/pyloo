"""Expected Log Pointwise Predictive Density (ELPD) data container with modifications from ArviZ."""

from copy import copy as _copy
from copy import deepcopy as _deepcopy

import numpy as np
import pandas as pd

BASE_FMT = """
Computed from {{:>{0}}} samples and {{:>{1}}} observations

         Estimate       SE
elpd_{kind:<{padding}} {0:<10.2f}  {1:<.2f}
p_{kind:<{padding}}    {2:<10.2f}  {3:<.2f}

{scale}"""

POINTWISE_LOO_FMT = """
Pareto k diagnostic values:
                         {{:>{0}}}    {{:>7}}
(-Inf, {:.2f})    {{:>{0}d}}    {{:>7.1f}}
[{:.2f}, 1)       {{:>{0}d}}    {{:>7.1f}}
[1, Inf)         {{:>{0}d}}    {{:>7.1f}}"""

SCALE_DICT = {
    "log": "Using log score",
    "negative_log": "Using negative log score",
    "deviance": "Using deviance score",
}


def _histogram(data, bins, range_hist=None):
    """Histogram."""
    hist, bin_edges = np.histogram(data, bins=bins, range=range_hist)
    hist_dens = hist / (hist.sum() * np.diff(bin_edges))
    return hist, hist_dens, bin_edges


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

        if kind not in ("loo", "waic"):
            raise ValueError("Invalid ELPDData object")

        scale_str = SCALE_DICT[self["scale"]]
        padding = len(scale_str) + len(kind) + 1
        base = BASE_FMT.format(padding, padding - 2)
        base = base.format(
            "", *self.values, kind=kind, scale=scale_str, n_samples=self.n_samples, n_points=self.n_data_points
        )

        if self.warning:
            base += "\n\nThere has been a warning during the calculation. Please check the results."

        if kind == "loo" and "pareto_k" in self and hasattr(self, "good_k") and self.good_k is not None:
            bins = np.asarray([-np.inf, self.good_k, 1, np.inf])
            counts, *_ = _histogram(self.pareto_k.values, bins)
            extended = POINTWISE_LOO_FMT.format(max(4, len(str(np.max(counts)))))
            extended = extended.format(
                "Count",
                "Pct.",
                *[*counts, *(counts / np.sum(counts) * 100)],
                self.good_k,
            )
            base = "\n".join([base, extended])
        elif kind == "loo" and "pareto_k" in self:
            # For non-PSIS methods, show simplified diagnostics
            method = getattr(self, "method", "unknown")
            base += f"\n\nUsing {method.upper()} importance sampling method"
            if method in ("tis", "sis"):
                base += " (no Pareto k diagnostics available)"

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
