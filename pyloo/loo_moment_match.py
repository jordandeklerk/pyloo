"""Moment matching for efficient approximate leave-one-out cross-validation (LOO)."""

import inspect
import logging
import warnings
from copy import deepcopy
from typing import Any, Callable, Literal

import numpy as np
import xarray as xr
from arviz.stats.diagnostics import ess

from .base import ISMethod, compute_importance_weights
from .elpd import ELPDData
from .helpers import (
    ParameterConverter,
    ShiftAndCovResult,
    ShiftAndScaleResult,
    ShiftResult,
    UpdateQuantitiesResult,
    extract_log_likelihood_for_observation,
    log_lik_i_upars,
    log_prob_upars,
)
from .split_moment_matching import loo_moment_match_split
from .utils import _logsumexp
from .wrapper.pymc import PyMCWrapper

__all__ = ["loo_moment_match"]

_log = logging.getLogger(__name__)


def loo_moment_match(
    model: PyMCWrapper | Any,
    loo_data: ELPDData,
    post_draws: Callable | None = None,
    log_lik_i: Callable | None = None,
    unconstrain_pars: Callable | None = None,
    log_prob_upars_fn: Callable | None = None,
    log_lik_i_upars_fn: Callable | None = None,
    max_iters: int = 30,
    k_threshold: float | None = None,
    split: bool = True,
    cov: bool = True,
    method: Literal["psis", "sis", "tis"] | ISMethod = "psis",
    verbose: bool = False,
    **kwargs,
) -> ELPDData:
    r"""Moment matching algorithm for updating a loo object when Pareto k estimates are large.

    Parameters
    ----------
    model : PyMCWrapper | Any
        Either a PyMC model wrapper instance or a custom model object that will be passed
        to the user-provided functions
    loo_data : ELPDData
        A loo object to be modified
    post_draws : Callable | None, optional
        A function that takes `model` and returns a matrix of posterior draws of the model parameters.
        Required if not using PyMCWrapper.
    log_lik_i : Callable | None, optional
        A function that takes `model` and `i` and returns a vector of log-likelihood draws
        of the `i`th observation based on the model `model`. Required if not using PyMCWrapper.
    unconstrain_pars : Callable | None, optional
        A function that takes `model` and `pars` and returns posterior draws on the
        unconstrained space based on the posterior draws on the constrained space passed via `pars`.
        Required if not using PyMCWrapper.
    log_prob_upars_fn : Callable | None, optional
        A function that takes `model` and `upars` and returns a vector of log-posterior density
        values of the unconstrained posterior draws passed via `upars`.
        Required if not using PyMCWrapper.
    log_lik_i_upars_fn : Callable | None, optional
        A function that takes `model`, `upars`, and `i` and returns a vector of log-likelihood
        draws of the `i`th observation based on the unconstrained posterior draws passed via `upars`.
        Required if not using PyMCWrapper.
    max_iters : int
        Maximum number of moment matching iterations
    k_threshold : float | None, optional
        Threshold value for Pareto k values above which moment matching is used.
        If None, uses min(1 - 1/log10(n_samples), 0.7)
    split : bool
        Whether to do the split transformation at the end of moment matching
    cov : bool
        Whether to match the covariance matrix of the samples
    method : Literal['psis', 'sis', 'tis'] | ISMethod
        Importance sampling method to use
    verbose : bool
        If True, enables detailed logging output for debugging
    **kwargs : Any
        Additional keyword arguments passed to the custom functions

    Returns
    -------
    ELPDData object (inherits from :class:`pandas.Series`) with the following row/attributes:
    elpd_loo: approximated expected log pointwise predictive density (elpd)
    se: standard error of the elpd
    p_loo: effective number of parameters
    n_samples: number of samples
    n_data_points: number of data points
    warning: bool
        True if using PSIS and the estimated shape parameter of Pareto distribution
        is greater than ``good_k``.
    loo_i: :class:`~xarray.DataArray` with the pointwise predictive accuracy,
            only if pointwise=True
    pareto_k: :class:`~xarray.DataArray` with the Pareto shape parameter k diagnostic values,
            only if pointwise=True and using PSIS method
    scale: scale of the elpd
    looic: leave-one-out information criterion (looic = -2 * elpd_loo)
    looic_se: standard error of the looic

    Notes
    -----
    Moment matching can fail to improve LOO estimates for several reasons such as very high-dimensional
    parameter spaces, multi-modality, weight instability, and insufficient sample size.

    Split moment matching can be used to improve the estimates by transforming only half of the draws
    and using multiple importance sampling to combine them with untransformed draws. This strategy
    provides more stability than transforming all draws, particularly in cases where the transformation
    itself might be imperfect. However, split moment matching is not guaranteed to improve the estimates
    either.

    Examples
    --------
    When we have many Pareto k estimates above the threshold, we can use moment matching to improve the estimates
    and avoid the computational cost of refitting the model :math:`k` times:

    .. code-block:: python

        import pyloo as pl
        import arviz as az

        data = az.load_arviz_data("centered_eight")

        with pm.Model() as model:
            mu = pm.Normal('mu', mu=0, sigma=10)
            sigma = pm.HalfNormal('sigma', sigma=10)
            likelihood = pm.Normal('y', mu=mu, sigma=sigma, observed=data.y)
            idata = pm.sample(1000, tune=1000, idata_kwargs={"log_likelihood": True})

        wrapper = pl.PyMCWrapper(model, idata)
        loo_orig = pl.loo(wrapper, pointwise=True)

        loo_new = pl.loo_moment_match(
            wrapper,
            loo_orig,
            max_iters=30,
            k_threshold=0.7,
            split=True,
            cov=True,
        )

    If you are using a custom model implementation, you need to implement functions for parameter transformation
    and log probability calculations. For example, for a CmdStanPy model, you can do the following:

    .. code-block:: python

        import numpy as np
        import pandas as pd
        import xarray as xr
        import pyloo as pl
        import arviz as az

        from cmdstanpy import CmdStanModel
        from scipy.special import logsumexp

        stan_code =
        data {
          int<lower=1> K;
          int<lower=1> N;
          matrix[N,K] x;
          array[N] int y;
          vector[N] offset;
          real beta_prior_scale;
          real alpha_prior_scale;
        }
        parameters {
          vector[K] beta;
          real intercept;
        }
        model {
          y ~ poisson(exp(x * beta + intercept + offset));
          beta ~ normal(0, beta_prior_scale);
          intercept ~ normal(0, alpha_prior_scale);
        }
        generated quantities {
          vector[N] log_lik;
          for (n in 1:N)
            log_lik[n] = poisson_lpmf(y[n] | exp(x[n] * beta + intercept + offset[n]));
        }

        model = CmdStanModel(stan_code=stan_code)
        fit = model.sample(data=stan_data, chains=4, iter_sampling=1000)

        # Define custom functions required for moment matching
        def post_draws(model_obj, **kwargs):
            fit = model_obj['fit']
            draws_dict = {
                'beta': fit.stan_variables()['beta'].reshape(fit.chains*fit.draws_per_chain, -1),
                'intercept': fit.stan_variables()['intercept'].flatten()
            }
            return draws_dict

        def log_lik_i(model_obj, i, **kwargs):
            fit = model_obj['fit']
            log_lik = fit.stan_variables()['log_lik']
            return log_lik[:, :, i].reshape(fit.chains*fit.draws_per_chain)

        def unconstrain_pars(model_obj, pars, **kwargs):
            K = model_obj['data']['K']
            n_samples = len(pars['intercept'])

            upars = np.zeros((n_samples, K + 1))
            upars[:, 0] = pars['intercept']  # intercept
            upars[:, 1:] = pars['beta']      # beta values
            return upars

        def log_prob_upars(model_obj, upars, **kwargs):
            # This function needs to compute the log posterior density
            return log_prob

        def log_lik_i_upars(model_obj, upars, i, **kwargs):
            # This function needs to compute the log-likelihood for observation i
            return log_lik

        idata = az.from_cmdstanpy(fit)
        loo_orig = pl.loo(idata, pointwise=True)

        loo_mm = pl.loo_moment_match(
            model_obj,
            loo_orig,
            post_draws=post_draws,
            log_lik_i=log_lik_i,
            unconstrain_pars=unconstrain_pars,
            log_prob_upars_fn=log_prob_upars,
            log_lik_i_upars_fn=log_lik_i_upars,
            k_threshold=0.7,
            split=True,
            cov=True
        )

        print("Original ELPD LOO:", loo_orig.elpd_loo)
        print("Improved ELPD LOO:", loo_mm.elpd_loo)

    References
    ----------
    Paananen, T., Piironen, J., Buerkner, P.-C., Vehtari, A. (2020). Implicitly Adaptive Importance
    Sampling. arXiv preprint arXiv:1906.08850.

    See Also
    --------
    loo_subsample : Leave-one-out cross-validation with subsampling
    loo_kfold : K-fold cross-validation
    loo_approximate_posterior : Leave-one-out cross-validation for posterior approximations
    loo_score : Compute LOO score for continuous ranked probability score
    loo_group : Leave-one-group-out cross-validation
    waic : Compute WAIC
    """
    log_level = logging.INFO if verbose else logging.WARNING
    _log.setLevel(log_level)

    if verbose:
        _log.info("Starting loo_moment_match with the following parameters:")
        _log.info(f"  max_iters: {max_iters}")
        _log.info(f"  split: {split}")
        _log.info(f"  cov: {cov}")
        _log.info(f"  method: {method}")

    loo_data = deepcopy(loo_data)

    if hasattr(loo_data, "loo_i") and not hasattr(loo_data, "p_loo_i"):
        loo_data.p_loo_i = xr.DataArray(
            np.zeros_like(loo_data.loo_i.values),
            dims=loo_data.loo_i.dims,
            coords=loo_data.loo_i.coords,
        )

    if isinstance(model, PyMCWrapper):
        if verbose:
            _log.info("Using PyMCWrapper model")

        converter = ParameterConverter(model)
        unconstrained = model.get_unconstrained_parameters()
        upars = converter.dict_to_matrix(unconstrained)
        S = upars.shape[0]

        if k_threshold is None:
            k_threshold = min(1 - 1 / np.log10(S), 0.7)
            if verbose:
                _log.info(f"Using automatically calculated k_threshold: {k_threshold}")

        orig_log_prob = log_prob_upars(model, unconstrained)

    else:
        if verbose:
            _log.info("Using custom model object")

        required_funcs = {
            "post_draws": post_draws,
            "log_lik_i": log_lik_i,
            "unconstrain_pars": unconstrain_pars,
            "log_prob_upars_fn": log_prob_upars_fn,
            "log_lik_i_upars_fn": log_lik_i_upars_fn,
        }

        missing_funcs = [name for name, func in required_funcs.items() if func is None]
        if missing_funcs:
            raise ValueError(
                "When not using PyMCWrapper, you must provide all the following"
                f" functions: {', '.join(required_funcs.keys())}. Missing:"
                f" {', '.join(missing_funcs)}"
            )

        _validate_custom_function(post_draws, ["model"], "post_draws")
        _validate_custom_function(log_lik_i, ["model", "i"], "log_lik_i")
        _validate_custom_function(
            unconstrain_pars, ["model", "pars"], "unconstrain_pars"
        )
        _validate_custom_function(
            log_prob_upars_fn, ["model", "upars"], "log_prob_upars_fn"
        )
        _validate_custom_function(
            log_lik_i_upars_fn, ["model", "upars", "i"], "log_lik_i_upars_fn"
        )

        try:
            pars = post_draws(model, **kwargs)  # type: ignore
            upars = unconstrain_pars(model, pars=pars, **kwargs)  # type: ignore
            upars = _validate_output(upars, "upars", expected_ndim=2)
            if verbose:
                _log.info(
                    f"Obtained unconstrained parameters with shape: {upars.shape}"
                )

        except Exception as e:
            raise ValueError(
                f"Error getting unconstrained parameters: {e}. Make sure your "
                "post_draws and unconstrain_pars functions are implemented correctly."
            ) from e

        S = upars.shape[0]

        if k_threshold is None:
            k_threshold = min(1 - 1 / np.log10(S), 0.7)

        try:
            orig_log_prob = log_prob_upars_fn(model, upars=upars, **kwargs)  # type: ignore
            orig_log_prob = _validate_output(
                orig_log_prob, "orig_log_prob", expected_ndim=1
            )
            if verbose:
                _log.info(
                    "Obtained original log probabilities with shape:"
                    f" {orig_log_prob.shape}"
                )

        except Exception as e:
            raise ValueError(
                f"Error computing log probabilities: {e}. Make sure your "
                "log_prob_upars_fn function is implemented correctly."
            ) from e

    if hasattr(loo_data, "pareto_k"):
        ks = loo_data.pareto_k.values
    else:
        raise ValueError(
            "Moment matching requires pointwise LOO results with Pareto k values. "
            "Please recompute LOO with pointwise=True before using moment_match=True."
        )

    bad_obs = np.where(ks > k_threshold)[0]
    _log.info(f"Found {len(bad_obs)} observations with Pareto k > {k_threshold}")
    kfs = np.zeros_like(ks)

    for i in bad_obs:
        if verbose:
            _log.info(f"\n{'=' * 50}")
            _log.info(f"Processing observation {i} with Pareto k = {ks[i]:.4f}")
            _log.info(f"{'=' * 50}")

        uparsi = upars.copy()
        ki = ks[i]
        kfi = 0

        if isinstance(model, PyMCWrapper):
            try:
                log_lik_result = log_lik_i_upars(model, unconstrained, pointwise=True)
                log_liki = extract_log_likelihood_for_observation(log_lik_result, i)
                if verbose:
                    _log.info(
                        f"Extracted log likelihood for observation {i} with shape:"
                        f" {log_liki.shape}"
                    )

            except Exception as e:
                raise ValueError(
                    f"Error computing log likelihood for observation {i}: {e}. "
                    "Check that your model's log likelihood is correctly configured."
                ) from e

            posterior = model.idata.posterior
            n_chains = len(posterior.chain)
            if n_chains == 1:
                r_eff_i = 1.0
            else:
                ess_i = ess(log_liki, method="mean")
                if isinstance(ess_i, xr.DataArray):
                    ess_i = ess_i.values
                r_eff_i = float(ess_i / len(log_liki))

        else:
            try:
                log_liki = log_lik_i(model, i, **kwargs)  # type: ignore
                if verbose:
                    _log.info(
                        f"Original log_liki type and shape: {type(log_liki)},"
                        f" {np.asarray(log_liki).shape}"
                    )

                log_liki = _validate_output(
                    log_liki, f"log_lik for observation {i}", expected_ndim=1
                )
                if verbose:
                    _log.info(
                        f"Validated log likelihood for observation {i} with shape:"
                        f" {log_liki.shape}"
                    )

            except Exception as e:
                raise ValueError(
                    f"Error computing log likelihood for observation {i}: {e}. "
                    "Make sure your log_lik_i function returns the log likelihood "
                    "for the specified observation as a 1D array."
                ) from e

            log_liki_matrix = np.array(log_liki)
            if len(log_liki_matrix.shape) > 1:
                n_chains = log_liki_matrix.shape[1]
                if n_chains == 1:
                    r_eff_i = 1.0
                else:
                    ess_i = ess(log_liki_matrix, method="mean")
                    if hasattr(ess_i, "values"):
                        ess_i = ess_i.values
                    r_eff_i = float(ess_i / len(log_liki))
            else:
                r_eff_i = 1.0

        is_obj = compute_importance_weights(-log_liki, method=method, reff=r_eff_i)
        lwi, initial_k = is_obj
        _log.info(f"Observation {i}: Initial Pareto k = {initial_k:.4f}")

        total_shift = np.zeros(upars.shape[1])
        total_scaling = np.ones(upars.shape[1])
        total_mapping = np.eye(upars.shape[1])

        iterind = 1

        while iterind <= max_iters and ki > k_threshold:
            if verbose:
                _log.info(f"\nIteration {iterind}/{max_iters} for observation {i}")

            if iterind == max_iters:
                warnings.warn(
                    "Maximum number of moment matching iterations reached. "
                    "Increasing max_iters may improve accuracy.",
                    stacklevel=2,
                )

            improved = False

            # Match means
            if verbose:
                _log.info("Applying mean shift transformation...")

            trans = shift(uparsi, lwi)
            try:
                quantities_i = update_quantities_i(
                    model,
                    trans["upars"],
                    i,
                    orig_log_prob,
                    r_eff_i,
                    converter if isinstance(model, PyMCWrapper) else None,
                    None if isinstance(model, PyMCWrapper) else log_prob_upars_fn,
                    None if isinstance(model, PyMCWrapper) else log_lik_i_upars_fn,
                    method,
                    verbose=verbose,
                    **kwargs,
                )
            except Exception as e:
                if verbose:
                    _log.error(f"Error computing quantities for mean shift: {e}")
                warnings.warn(
                    f"Error during mean shift for observation {i}: {e}. "
                    "Skipping this transformation.",
                    stacklevel=2,
                )
                break

            if quantities_i["ki"] < ki:
                _log.info(
                    f"Observation {i}: Mean shift improved Pareto k from {ki:.4f} to"
                    f" {quantities_i['ki']:.4f}"
                )
                uparsi = trans["upars"]
                total_shift += trans["shift"]
                lwi = quantities_i["lwi"]
                ki = quantities_i["ki"]
                kfi = quantities_i["kfi"]
                log_liki = quantities_i["log_liki"]
                iterind += 1
                improved = True
            else:
                if verbose:
                    _log.info(
                        f"Mean shift did not improve Pareto k: {ki:.4f} vs"
                        f" {quantities_i['ki']:.4f}"
                    )

            # Match means and marginal variances
            if verbose:
                _log.info("Applying mean and scale transformation...")

            trans = shift_and_scale(uparsi, lwi)
            try:
                quantities_i_scale = update_quantities_i(
                    model,
                    trans["upars"],
                    i,
                    orig_log_prob,
                    r_eff_i,
                    converter if isinstance(model, PyMCWrapper) else None,
                    None if isinstance(model, PyMCWrapper) else log_prob_upars_fn,
                    None if isinstance(model, PyMCWrapper) else log_lik_i_upars_fn,
                    method,
                    verbose=verbose,
                    **kwargs,
                )
            except Exception as e:
                if verbose:
                    _log.error(f"Error computing quantities for scale shift: {e}")
                warnings.warn(
                    f"Error during scale shift for observation {i}: {e}. "
                    "Skipping this transformation.",
                    stacklevel=2,
                )
                if improved:
                    continue
                else:
                    break

            if quantities_i_scale["ki"] < ki:
                _log.info(
                    f"Observation {i}: Mean and scale shift improved Pareto k from"
                    f" {ki:.4f} to {quantities_i_scale['ki']:.4f}"
                )
                uparsi = trans["upars"]
                total_shift += trans["shift"]
                total_scaling *= trans["scaling"]
                lwi = quantities_i_scale["lwi"]
                ki = quantities_i_scale["ki"]
                kfi = quantities_i_scale["kfi"]
                log_liki = quantities_i_scale["log_liki"]
                iterind += 1
                improved = True
            else:
                if verbose:
                    _log.info(
                        f"Mean and scale shift did not improve Pareto k: {ki:.4f} vs"
                        f" {quantities_i_scale['ki']:.4f}"
                    )

            # Match means and covariances
            if cov:
                if verbose:
                    _log.info("Applying covariance transformation...")

                trans = shift_and_cov(uparsi, lwi)
                try:
                    quantities_i_cov = update_quantities_i(
                        model,
                        trans["upars"],
                        i,
                        orig_log_prob,
                        r_eff_i,
                        converter if isinstance(model, PyMCWrapper) else None,
                        None if isinstance(model, PyMCWrapper) else log_prob_upars_fn,
                        None if isinstance(model, PyMCWrapper) else log_lik_i_upars_fn,
                        method,
                        verbose=verbose,
                        **kwargs,
                    )
                except Exception as e:
                    if verbose:
                        _log.error(
                            f"Error computing quantities for covariance shift: {e}"
                        )
                    warnings.warn(
                        f"Error during covariance shift for observation {i}: {e}. "
                        "Skipping this transformation.",
                        stacklevel=2,
                    )
                    if improved:
                        continue
                    else:
                        break

                if quantities_i_cov["ki"] < ki:
                    _log.info(
                        f"Observation {i}: Covariance shift improved Pareto k from"
                        f" {ki:.4f} to {quantities_i_cov['ki']:.4f}"
                    )
                    uparsi = trans["upars"]
                    total_shift += trans["shift"]
                    total_mapping = trans["mapping"] @ total_mapping
                    lwi = quantities_i_cov["lwi"]
                    ki = quantities_i_cov["ki"]
                    kfi = quantities_i_cov["kfi"]
                    log_liki = quantities_i_cov["log_liki"]
                    iterind += 1
                    improved = True
                else:
                    if verbose:
                        _log.info(
                            f"Covariance shift did not improve Pareto k: {ki:.4f} vs"
                            f" {quantities_i_cov['ki']:.4f}"
                        )

            # Only break if no transformation improved k
            if not improved:
                _log.info(
                    f"Observation {i}: No further improvement after"
                    f" {iterind - 1} iterations. Final Pareto k = {ki:.4f}"
                )
                break

        if max_iters == 1:
            warnings.warn(
                "Maximum number of moment matching iterations reached with max_iters=1."
                " Increasing max_iters may improve accuracy.",
                stacklevel=2,
            )

        # Split transformation if requested and transformations were successful
        if split and iterind > 1:
            _log.info(f"Performing split transformation for observation {i}")

            try:
                if isinstance(model, PyMCWrapper):
                    split_result = loo_moment_match_split(
                        model,
                        upars,
                        cov,
                        total_shift,
                        total_scaling,
                        total_mapping,
                        i,
                        r_eff_i,
                        method=method,
                        verbose=verbose,
                    )
                else:
                    split_result = loo_moment_match_split(
                        model,
                        upars,
                        cov,
                        total_shift,
                        total_scaling,
                        total_mapping,
                        i,
                        r_eff_i,
                        log_prob_upars_fn=log_prob_upars_fn,
                        log_lik_i_upars_fn=log_lik_i_upars_fn,
                        method=method,
                        verbose=verbose,
                        **kwargs,
                    )

                log_liki = split_result["log_liki"]
                lwi = split_result["lwi"]
                r_eff_i = split_result["r_eff_i"]

                if verbose:
                    _log.info(f"Split transformation completed for observation {i}")

            except Exception as e:
                if verbose:
                    _log.error(f"Error during split transformation: {e}")
                warnings.warn(
                    f"Split transformation failed for observation {i}: {e}. "
                    "Using the last successful transformation instead.",
                    stacklevel=2,
                )

        new_elpd_i = _logsumexp(log_liki + lwi)
        if verbose:
            _log.info(f"New ELPD for observation {i}: {new_elpd_i:.4f}")

        update_loo_data_i(
            loo_data,
            i,
            new_elpd_i,
            ki,
            kfi,
            kfs,
            model if isinstance(model, PyMCWrapper) else None,
            log_liki if not isinstance(model, PyMCWrapper) else None,
            verbose=verbose,
        )

    summary(loo_data, ks, k_threshold, verbose=verbose)

    if np.any(ks > k_threshold):
        warnings.warn(
            "Some Pareto k estimates are still above the threshold. "
            "The model may be misspecified or the data may be highly influential.",
            stacklevel=2,
        )

    if not split and np.any(kfs > k_threshold):
        warnings.warn(
            "The accuracy of self-normalized importance sampling may be bad. "
            "Setting split=True will likely improve accuracy.",
            stacklevel=2,
        )

    return loo_data


def update_quantities_i(
    model: PyMCWrapper | Any,
    upars: np.ndarray,
    i: int,
    orig_log_prob: np.ndarray,
    r_eff_i: float,
    converter: ParameterConverter | None = None,
    log_prob_upars_fn: Callable | None = None,
    log_lik_i_upars_fn: Callable | None = None,
    method: Literal["psis", "sis", "tis"] | ISMethod = "psis",
    verbose: bool = False,
    **kwargs,
) -> UpdateQuantitiesResult:
    """Update the importance weights, Pareto diagnostic and log-likelihood for observation i.

    Parameters
    ----------
    model : PyMCWrapper | Any
        Either a PyMC model wrapper instance OR a custom model object
    upars : np.ndarray
        A matrix representing a sample of vector-valued parameters in the unconstrained space
    i : int
        Observation number
    orig_log_prob : np.ndarray
        Log probability densities of the original draws
    r_eff_i : float
        MCMC effective sample size divided by total sample size
    converter : ParameterConverter | None, optional
        Parameter converter instance for efficient format conversions.
        Required if model is PyMCWrapper.
    log_prob_upars_fn : Callable | None, optional
        Function to compute log probability for unconstrained parameters.
        Required if model is not PyMCWrapper.
    log_lik_i_upars_fn : Callable | None, optional
        Function to compute log likelihood for observation i with unconstrained parameters.
        Required if model is not PyMCWrapper.
    method : Literal['psis', 'sis', 'tis'] | ISMethod
        Importance sampling method to use
    verbose : bool
        If True, enables detailed logging output for debugging
    **kwargs : Any
        Additional keyword arguments passed to custom functions

    Returns
    -------
    dict
        dictionary containing updated quantities:
        - lwi: New log importance weights
        - lwfi: New log importance weights for full distribution
        - ki: New Pareto k value
        - kfi: New Pareto k value for full distribution
        - log_liki: New log likelihood values
    """
    if isinstance(model, PyMCWrapper):
        if converter is None:
            raise ValueError("converter must be provided when using PyMCWrapper")

        upars_dict = converter.matrix_to_dict(upars)
        log_prob_new = log_prob_upars(model, upars_dict)
        log_prob_new = _validate_output(log_prob_new, "log_prob_new", expected_ndim=1)

        try:
            log_lik_result = log_lik_i_upars(model, upars_dict, pointwise=True)
            log_liki_new = extract_log_likelihood_for_observation(log_lik_result, i)
            log_liki_new = _validate_output(
                log_liki_new, f"log_liki_new for obs {i}", expected_ndim=1
            )

            if verbose:
                _log.info(
                    f"New log likelihood for observation {i} computed with shape:"
                    f" {log_liki_new.shape}"
                )

        except Exception as e:
            raise ValueError(
                f"Error computing log likelihood for observation {i}: {e}. "
                "Check that your model's log likelihood is correctly specified."
            ) from e

    else:
        if None in (log_prob_upars_fn, log_lik_i_upars_fn):
            raise ValueError(
                "log_prob_upars_fn and log_lik_i_upars_fn must be provided when not"
                " using PyMCWrapper"
            )

        try:
            log_prob_new = log_prob_upars_fn(model, upars=upars, **kwargs)  # type: ignore
            log_prob_new = _validate_output(
                log_prob_new, "log_prob_new", expected_ndim=1
            )

            if verbose:
                _log.info(
                    f"New log probability computed with shape: {log_prob_new.shape}"
                )

        except Exception as e:
            raise ValueError(
                f"Error computing log probability: {e}. Make sure your"
                " log_prob_upars_fn function returns a 1D array of log probabilities."
            ) from e

        try:
            log_liki_new = log_lik_i_upars_fn(model, upars=upars, i=i, **kwargs)  # type: ignore
            log_liki_new = _validate_output(
                log_liki_new, f"log_liki_new for obs {i}", expected_ndim=1
            )

            if verbose:
                _log.info(
                    f"New log likelihood for observation {i} computed with shape:"
                    f" {log_liki_new.shape}"
                )

        except Exception as e:
            raise ValueError(
                f"Error computing log likelihood for observation {i}: {e}. Make sure"
                " your log_lik_i_upars_fn function returns a 1D array of log"
                " likelihoods."
            ) from e

    log_liki_new = np.array(log_liki_new, dtype=np.float64)
    log_prob_new = np.array(log_prob_new, dtype=np.float64)
    orig_log_prob = np.array(orig_log_prob, dtype=np.float64)

    # Calculate importance ratios
    lr = -log_liki_new + log_prob_new - orig_log_prob
    lr[np.isnan(lr)] = -np.inf

    if verbose:
        _log.info(
            f"Computing importance weights with method: {method}, reff: {r_eff_i:.4f}"
        )

    lwi_new, ki_new = compute_importance_weights(lr, method=method, reff=r_eff_i)

    if verbose:
        _log.info(f"New Pareto k: {ki_new:.4f}")

    # Calculate full importance ratios
    full_lr = log_prob_new - orig_log_prob
    full_lr[np.isnan(full_lr)] = -np.inf
    lwfi_new, kfi_new = compute_importance_weights(full_lr, method=method, reff=r_eff_i)

    if verbose:
        _log.info(f"New full Pareto k: {kfi_new:.4f}")

    return {
        "lwi": lwi_new,
        "lwfi": lwfi_new,
        "ki": ki_new,
        "kfi": kfi_new,
        "log_liki": log_liki_new,
    }


def shift(upars: np.ndarray, lwi: np.ndarray) -> ShiftResult:
    """Shift a matrix of parameters to their weighted mean.

    Parameters
    ----------
    upars : np.ndarray
        A matrix representing a sample of vector-valued parameters in the unconstrained space
    lwi : np.ndarray
        A vector representing the log-weight of each parameter

    Returns
    -------
    dict
        dictionary containing:
        - upars: The transformed parameter matrix
        - shift: The shift that was performed
    """
    mean_original = np.mean(upars, axis=0)
    mean_weighted = np.sum(np.exp(lwi)[:, None] * upars, axis=0)
    shift = mean_weighted - mean_original
    upars_new = upars + shift[None, :]

    return {"upars": upars_new, "shift": shift}


def shift_and_scale(upars: np.ndarray, lwi: np.ndarray) -> ShiftAndScaleResult:
    """Shift a matrix of parameters to their weighted mean and scale the marginal variances.

    Parameters
    ----------
    upars : np.ndarray
        A matrix representing a sample of vector-valued parameters in the unconstrained space
    lwi : np.ndarray
        A vector representing the log-weight of each parameter

    Returns
    -------
    dict
        dictionary containing:
        - upars: The transformed parameter matrix
        - shift: The shift that was performed
        - scaling: The scaling that was performed
    """
    S = upars.shape[0]
    mean_original = np.mean(upars, axis=0)
    mean_weighted = np.sum(np.exp(lwi)[:, None] * upars, axis=0)
    shift = mean_weighted - mean_original
    mii = np.exp(lwi)[:, None] * upars**2
    mii = np.sum(mii, axis=0) - mean_weighted**2
    mii = mii * S / (S - 1)
    scaling = np.sqrt(mii / np.var(upars, axis=0))

    upars_new = upars - mean_original[None, :]
    upars_new = upars_new * scaling[None, :]
    upars_new = upars_new + mean_weighted[None, :]

    return {"upars": upars_new, "shift": shift, "scaling": scaling}


def shift_and_cov(upars: np.ndarray, lwi: np.ndarray) -> ShiftAndCovResult:
    """Shift a matrix of parameters and scale the covariance to match the weighted covariance.

    Parameters
    ----------
    upars : np.ndarray
        A matrix representing a sample of vector-valued parameters in the unconstrained space
    lwi : np.ndarray
        A vector representing the log-weight of each parameter

    Returns
    -------
    dict
        dictionary containing:
        - upars: The transformed parameter matrix
        - shift: The shift that was performed
        - mapping: The mapping matrix that was used
    """
    mean_original = np.mean(upars, axis=0)
    mean_weighted = np.sum(np.exp(lwi)[:, None] * upars, axis=0)
    shift = mean_weighted - mean_original

    covv = np.cov(upars, rowvar=False)
    wcovv = np.cov(upars, rowvar=False, aweights=np.exp(lwi))

    try:
        chol1 = np.linalg.cholesky(wcovv)
        chol2 = np.linalg.cholesky(covv)
        mapping = chol1.T @ np.linalg.inv(chol2.T)
    except np.linalg.LinAlgError as e:
        warnings.warn(
            f"Cholesky decomposition failed during covariance matching: {e}. "
            "Using identity mapping instead.",
            stacklevel=2,
        )
        mapping = np.eye(len(mean_original))

    upars_new = upars - mean_original[None, :]
    upars_new = upars_new @ mapping.T
    upars_new = upars_new + mean_weighted[None, :]

    return {"upars": upars_new, "shift": shift, "mapping": mapping}


def update_loo_data_i(
    loo_data: ELPDData,
    i: int,
    new_elpd_i: float,
    ki: float,
    kfi: float,
    kfs: np.ndarray,
    wrapper: PyMCWrapper | None = None,
    log_liki: np.ndarray | None = None,
    verbose: bool = False,
) -> None:
    """Update LOO data for a specific observation with new ELPD and k values.

    Parameters
    ----------
    loo_data : ELPDData
        The LOO data object to update
    i : int
        Observation index
    new_elpd_i : float
        New ELPD value for the observation
    ki : float
        New Pareto k value
    kfi : float
        New Pareto k value for full distribution
    kfs : np.ndarray
        Array to store kfi values
    wrapper : PyMCWrapper | None, optional
        PyMC model wrapper instance, if using PyMCWrapper
    log_liki : np.ndarray | None, optional
        Log likelihood values for observation i, if using custom functions
    verbose : bool
        If True, enables detailed logging output for debugging
    """
    # Calculate lpd_i for observation i
    if wrapper is not None:
        log_likelihood = wrapper.idata.log_likelihood
        var_name = list(log_likelihood.data_vars)[0]
        log_lik_array = log_likelihood[var_name]

        if hasattr(wrapper, "observed_dims") and var_name in wrapper.observed_dims:
            obs_dims = wrapper.observed_dims[var_name]
            if obs_dims:
                obs_dim = obs_dims[0]
            else:
                obs_dims = [  # type: ignore
                    dim for dim in log_lik_array.dims if dim not in ("chain", "draw")
                ]
                if not obs_dims:
                    raise ValueError(
                        "Could not identify observation dimension in log_likelihood. "
                        "Please check your model's log_likelihood structure."
                    )
                obs_dim = obs_dims[0]
        else:
            obs_dims = [  # type: ignore
                dim for dim in log_lik_array.dims if dim not in ("chain", "draw")
            ]
            if not obs_dims:
                raise ValueError(
                    "Could not identify observation dimension in log_likelihood. "
                    "Please check your model's log_likelihood structure."
                )
            obs_dim = obs_dims[0]

        log_lik_i = log_lik_array.isel({obs_dim: i}).stack(__sample__=("chain", "draw"))
        n_samples = log_lik_i.shape[-1]
        lpd_i = _logsumexp(log_lik_i.values) - np.log(n_samples)
    else:
        if log_liki is None:
            raise ValueError(
                "log_liki must be provided when not using PyMCWrapper. "
                "This is an internal error and should not happen."
            )
        lpd_i = _logsumexp(log_liki) - np.log(len(log_liki))

    p_loo_i = lpd_i - new_elpd_i

    if hasattr(loo_data, "loo_i"):
        # Multi-observation case
        old_elpd_i = loo_data.loo_i.values[i]
        loo_data.loo_i.values[i] = new_elpd_i
        loo_data.p_loo_i.values[i] = p_loo_i

        loo_data["elpd_loo"] = np.sum(loo_data.loo_i.values)
        loo_data["p_loo"] = np.sum(loo_data.p_loo_i.values)

        n_data_points = loo_data.n_data_points
        loo_data["se"] = (n_data_points * np.var(loo_data.loo_i.values)) ** 0.5
        loo_data["p_loo_se"] = (n_data_points * np.var(loo_data.p_loo_i.values)) ** 0.5

        _log.info(
            f"Observation {i}: ELPD changed from {old_elpd_i:.4f} to"
            f" {new_elpd_i:.4f} (diff: {new_elpd_i - old_elpd_i:.4f})"
        )
        if verbose:
            _log.info(f"Observation {i}: p_loo changed to {p_loo_i:.4f}")
    else:
        # Single observation case
        old_elpd_total = loo_data["elpd_loo"]
        loo_data["elpd_loo"] = new_elpd_i
        loo_data["p_loo"] = p_loo_i

        _log.info(
            f"Total ELPD changed from {old_elpd_total:.4f} to {new_elpd_i:.4f} (diff:"
            f" {new_elpd_i - old_elpd_total:.4f})"
        )

    if "looic" in loo_data:
        loo_data["looic"] = -2 * loo_data["elpd_loo"]
        if "se" in loo_data:
            loo_data["looic_se"] = 2 * loo_data["se"]

    # Update Pareto k
    if hasattr(loo_data, "pareto_k"):
        old_k = loo_data.pareto_k.values[i]
        loo_data.pareto_k.values[i] = ki
        _log.info(
            f"Observation {i}: Pareto k changed from {old_k:.4f} to"
            f" {ki:.4f} (improvement: {old_k - ki:.4f})"
        )

    kfs[i] = kfi


def summary(
    loo_data: ELPDData,
    original_ks: np.ndarray,
    k_threshold: float,
    verbose: bool = False,
) -> None:
    """Log a summary of improvements in Pareto k values.

    Parameters
    ----------
    loo_data : ELPDData
        The LOO data object
    original_ks : np.ndarray
        Original Pareto k values
    k_threshold : float
        Threshold for Pareto k values
    verbose : bool
        If True, enables detailed logging output for debugging
    """
    if hasattr(loo_data, "pareto_k"):
        improved_indices = np.where(loo_data.pareto_k.values < original_ks)[0]
        n_improved = len(improved_indices)

        if n_improved > 0:
            avg_improvement = np.mean(
                original_ks[improved_indices]
                - loo_data.pareto_k.values[improved_indices]
            )
            _log.info(
                f"Improved Pareto k for {n_improved} observations. Average improvement:"
                f" {avg_improvement:.4f}"
            )

            if verbose:
                for idx in improved_indices:
                    _log.info(
                        f"  Observation {idx}: {original_ks[idx]:.4f} ->"
                        f" {loo_data.pareto_k.values[idx]:.4f} (improvement:"
                        f" {original_ks[idx] - loo_data.pareto_k.values[idx]:.4f})"
                    )
        else:
            _log.info("No improvements in Pareto k values")

        # Observations that still have high Pareto k values
        high_k_indices = np.where(loo_data.pareto_k.values > k_threshold)[0]
        if len(high_k_indices) > 0:
            _log.info(
                f"{len(high_k_indices)} observations still have Pareto k >"
                f" {k_threshold}"
            )

            if verbose:
                for idx in high_k_indices:
                    _log.info(
                        f"  Observation {idx}: Pareto k ="
                        f" {loo_data.pareto_k.values[idx]:.4f}"
                    )


def _validate_custom_function(func, expected_args, name):
    """Validate that a custom function has the expected signature."""
    sig = inspect.signature(func)
    # Check if the function accepts **kwargs which would cover all expected args
    has_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )

    if not has_kwargs:
        missing = [arg for arg in expected_args if arg not in sig.parameters]
        if missing:
            raise ValueError(
                f"Custom function '{name}' is missing required parameters: {missing}. "
                f"Expected signature should include: {expected_args}"
            )
    return True


def _validate_output(
    array, name, expected_ndim=None, expected_shape=None, allow_none=False
):
    """Validate array outputs from custom functions."""
    if array is None:
        if allow_none:
            return None
        raise ValueError(
            f"Function returned None for {name}. Please check your custom function"
            " implementation."
        )

    try:
        array = np.asarray(array, dtype=np.float64)
    except Exception as e:
        raise ValueError(
            f"Could not convert {name} to numpy array: {e}. "
            "Please ensure your function returns numeric data."
        ) from e

    if np.any(np.isnan(array)):
        raise ValueError(
            f"NaN values detected in {name}. Please check your function for division by"
            " zero, log of negative values, or other numerical issues."
        )

    if expected_ndim is not None and array.ndim != expected_ndim:
        raise ValueError(
            f"Expected {expected_ndim} dimensions for {name}, got {array.ndim}. "
            "Please reshape your output accordingly."
        )

    if expected_shape is not None and array.shape != expected_shape:
        raise ValueError(
            f"Expected shape {expected_shape} for {name}, got {array.shape}. "
            "Please check your function output dimensions."
        )

    return array
