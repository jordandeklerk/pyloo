"""Exact refitting for problematic observations in LOO-CV for PyMC models."""

import logging

import numpy as np
import pymc as pm

from .elpd import ELPDData
from .loo import loo
from .loo_subsample import loo_subsample
from .utils import _logsumexp
from .wrapper.pymc_wrapper import PyMCWrapper

__all__ = ["reloo"]

_log = logging.getLogger(__name__)

# Required methods for reloo
RELOO_REQUIRED_METHODS: tuple[str, ...] = (
    "select_observations",
    "sample_posterior",
    "log_likelihood_i",
    "get_observed_name",
    "get_observed_data",
    "set_data",
)


def reloo(
    wrapper: PyMCWrapper,
    loo_orig: ELPDData | None = None,
    k_thresh: float = 0.7,
    scale: str | None = None,
    verbose: bool = True,
    use_subsample: bool = False,
    subsample_observations: int | np.ndarray | None = 400,
    subsample_approximation: str = "plpd",
    subsample_estimator: str = "diff_srs",
    subsample_draws: int | None = None,
) -> ELPDData:
    """Exact refitting for problematic observations in LOO-CV for PyMC models.

    This function recalculates Leave-One-Out (LOO) cross validation exactly for those observations
    where the Pareto Smoothed Importance Sampling (PSIS) approximation fails. PSIS, as implemented
    in ``pl.loo``, provides an efficient approximation to LOO-CV by comparing the full posterior to
    the leave-one-out posteriors (i.e., with the i-th observation omitted). In many cases, this works
    well; however, for a few highly influential observations, the PSIS approximation breaks down,
    indicated by large Pareto shape values.

    Instead of abandoning the approximation for the entire dataset, this function uses PSIS for
    observations with Pareto shape values below a specified threshold and performs a full model
    refit to obtain exact LOO-CV results for those few problematic cases. This targeted refitting
    approach typically requires far fewer refits than performing an exact LOO-CV for every observation,
    yielding a more accurate overall measure of model fit with substantially reduced computational cost.

    For large datasets where even standard LOO-CV might be computationally intensive, this function
    can utilize subsampling methods through the ``use_subsample`` parameter. When enabled, it uses
    the efficient subsampling approach from ``loo_subsample`` to compute the initial LOO estimates,
    then performs exact refits only for the problematic observations within the subsample.

    Parameters
    ----------
    wrapper : PyMCWrapper
        A PyMCWrapper instance that holds the fitted model and its data. The implementation is optimized
        specifically for PyMC models.
    loo_orig : ELPDData, optional
        An ELPDData instance containing initial pointwise LOO results. If provided, its Pareto shape values
        are used to identify problematic observations. If omitted, PSIS-LOO is computed first.
    k_thresh : float, default 0.7
        The threshold for the Pareto shape value. Observations with a Pareto shape above this value will
        trigger a model refit. The default threshold of 0.7 is based on simulation studies.
    scale : str, optional
        The output scale for LOO. Options are:
        - 'log' (default): log-score,
        - 'negative_log': -1 * log-score,
        - 'deviance': -2 * log-score.
    verbose : bool, default True
        If True, detailed information about the refitting process is logged, including which observations are
        being refitted.
    use_subsample : bool, default False
        If True, uses the subsampling approach from loo_subsample for the initial LOO computation.
    subsample_observations : int or numpy.ndarray or None, default 400
        The subsample observations to use when use_subsample is True:
        - An integer specifying the number of observations to subsample
        - An array of integers providing specific indices to use
        - None to use all observations (equivalent to standard LOO)
    subsample_approximation : str, default "plpd"
        The type of approximation to use for the loo_i values when use_subsample is True:
        - "plpd": Point estimate based approximation (default)
        - "lpd": Log predictive density
        - "tis": Truncated importance sampling
        - "sis": Standard importance sampling
    subsample_estimator : str, default "diff_srs"
        The estimation method to use when use_subsample is True:
        - "diff_srs": Difference estimator with simple random sampling (default)
        - "hh_pps": Hansen-Hurwitz estimator
        - "srs": Simple random sampling
    subsample_draws : int, optional
        The number of posterior draws to use for approximation methods that require integration
        over the posterior when use_subsample is True.

    Returns
    -------
    ELPDData
        An ELPDData instance that merges the PSIS-LOO approximations with the exact LOO-CV estimates obtained
        by refitting. For observations where exact LOO-CV was performed, the Pareto shape value is set to 0.
        This combined result provides a more reliable assessment of model fit when some observations are highly
        influential.

    Notes
    -----
    We recommend to first run ``loo()`` or ``loo_subsample()`` to check the number of observations with
    Pareto shape values above the threshold. If many observations exceed the threshold, the computational
    cost of refitting might become prohibitive, as each refit involves running the full MCMC sampling
    procedure without the problematic observation.

    For non-PyMC models, please refer to ArviZ's ``az.reloo`` implementation, which offers a more abstract
    interface through its sampling wrapper experimental feature. You will need to manually implement the
    methods required by ``az.reloo`` in your wrapper.

    Examples
    --------
    Create a simple normal model and calculate exact LOO for problematic observations

    .. code-block:: python

        import pyloo as pl
        import pymc as pm
        import numpy as np

        np.random.seed(0)
        N = 100
        true_mu = 1.0
        true_sigma = 2.0
        y = np.random.normal(true_mu, true_sigma, N)

        with pm.Model() as model:
            mu = pm.Normal('mu', mu=0, sigma=10)
            sigma = pm.HalfNormal('sigma', sigma=10)
            likelihood = pm.Normal('y', mu=mu, sigma=sigma, observed=y)
            idata = pm.sample(1000, tune=1000)

        wrapper = pl.PyMCWrapper(model, idata)
        loo_exact = pl.reloo(wrapper, k_thresh=0.7)

    Use subsampling for efficient computation with large datasets

    .. code-block:: python

        loo_exact_subsample = pl.reloo(
            wrapper,
            k_thresh=0.7,
            use_subsample=True,
            subsample_observations=50
        )

    See Also
    --------
    loo : Compute LOO-CV using importance sampling
    loo_i : Pointwise LOO-CV values
    loo_subsample : Subsampled LOO-CV computation
    loo_moment_match : Moment matching for problematic observations
    loo_kfold : K-fold cross-validation
    """
    if not isinstance(wrapper.model, pm.Model):
        raise TypeError(
            "The reloo functionality is currently only supported for PyMC models. "
            f"Got {type(wrapper.model)} instead."
        )

    not_implemented = wrapper.check_implemented_methods(RELOO_REQUIRED_METHODS)
    if not_implemented:
        raise TypeError(
            "Passed wrapper instance does not implement all methods required for"
            f" reloo. Check the documentation of PyMCWrapper. {not_implemented} must be"
            " implemented and were not found."
        )

    if loo_orig is None:
        if use_subsample:
            loo_orig = loo_subsample(
                wrapper.idata,
                observations=subsample_observations,
                loo_approximation=subsample_approximation,
                estimator=subsample_estimator,
                loo_approximation_draws=subsample_draws,
                pointwise=True,
                scale=scale,
            )
        else:
            loo_orig = loo(wrapper.idata, pointwise=True, scale=scale)

    loo_refitted = loo_orig.copy()
    khats = loo_refitted.pareto_k
    loo_i = loo_refitted.loo_i
    scale = loo_orig.scale if scale is None else scale

    if scale is None:
        scale = "log"

    if scale.lower() == "deviance":
        scale_value = -2
    elif scale.lower() == "log":
        scale_value = 1
    elif scale.lower() == "negative_log":
        scale_value = -1

    lppd_orig = loo_orig.p_loo + loo_orig.elpd_loo / scale_value
    n_data_points = loo_orig.n_data_points

    khats_values = khats.values if hasattr(khats, "values") else khats
    if np.any(khats_values > k_thresh):
        for idx in np.argwhere(khats_values > k_thresh):
            if verbose:
                _log.info("Refitting model excluding observation %d", idx.item())

            var_name = wrapper.get_observed_name()
            original_data = wrapper.get_observed_data().copy()

            try:
                if use_subsample:
                    # For subsampled LOO, we need to map the subsample index back to the original data
                    if isinstance(subsample_observations, np.ndarray):
                        # If specific indices were provided, use those
                        orig_idx = subsample_observations[idx.item()]
                    else:
                        # Otherwise, the index is already correct
                        orig_idx = idx.item()
                    _, remaining = wrapper.select_observations(orig_idx)
                else:
                    _, remaining = wrapper.select_observations(idx)

                wrapper.set_data({var_name: remaining})
                idata_idx = wrapper.sample_posterior()

                log_like_idx = wrapper.log_likelihood_i(
                    orig_idx if use_subsample else idx.item(), idata_idx
                ).values.flatten()

                loo_lppd_idx = scale_value * _logsumexp(
                    log_like_idx, b_inv=len(log_like_idx)
                )
                khats[idx] = 0
                loo_i[idx] = loo_lppd_idx

            finally:
                wrapper.set_data({var_name: original_data})

        loo_refitted.elpd_loo = loo_i.values.sum()
        loo_refitted.se = (n_data_points * np.var(loo_i.values)) ** 0.5
        loo_refitted.p_loo = lppd_orig - loo_refitted.elpd_loo / scale_value

        return loo_refitted
    else:
        if verbose:
            _log.info("No problematic observations found")
        return loo_orig
