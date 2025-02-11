"""Constants and enumerations for LOO-CV functionality."""

from enum import Enum
from typing import Literal


class LooApproximationMethod(str, Enum):
    """Enumeration of available LOO approximation methods.

    Attributes
    ----------
    PLPD : str
        Point estimate based approximation (default)
    LPD : str
        Log predictive density
    TIS : str
        Truncated importance sampling
    SIS : str
        Standard importance sampling
    """

    PLPD = "plpd"
    LPD = "lpd"
    TIS = "tis"
    SIS = "sis"


class EstimatorMethod(str, Enum):
    """Enumeration of available estimator methods.

    Attributes
    ----------
    DIFF_SRS : str
        Difference estimator with simple random sampling (default)
    HH_PPS : str
        Hansen-Hurwitz estimator with probability proportional to size
    SRS : str
        Simple random sampling
    """

    DIFF_SRS = "diff_srs"
    HH_PPS = "hh_pps"
    SRS = "srs"


# Type aliases for better type safety
LooApproximationMethodType = Literal["plpd", "lpd", "tis", "sis"]
EstimatorMethodType = Literal["diff_srs", "hh_pps", "srs"]

SCALE_OPTIONS = Literal["deviance", "log", "negative_log"]
