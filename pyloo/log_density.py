"""Compute the log density of a model and its variational approximation."""

import logging

import numpy as np
from pymc.blocking import DictToArrayBijection
from pymc.distributions.dist_math import rho2sigma
from scipy import linalg

__all__ = [
    "get_approximation_params",
    "compute_log_p",
    "compute_log_q",
    "compute_log_weights",
]

_log = logging.getLogger(__name__)

if not logging.root.handlers:
    _log.setLevel(logging.INFO)
    if len(_log.handlers) == 0:
        handler = logging.StreamHandler()
        _log.addHandler(handler)


def get_approximation_params(approx, verbose=False):
    """
    Extract parameters from a variational approximation.

    Parameters
    ----------
    approx : Approximation
        The variational approximation object (MeanField or FullRank)
    verbose : bool, default=False
        Whether to print debug information

    Returns
    -------
    dict
        Dictionary of approximation parameters
    """
    approx_group = approx.groups[0]
    params = {}

    if "rho" in approx_group.params_dict:
        # MeanField approximation
        params["type"] = "meanfield"
        params["mu"] = np.array(approx_group.params_dict["mu"].eval())
        rho = np.array(approx_group.params_dict["rho"].eval())
        params["std"] = np.array(rho2sigma(rho).eval())

        if verbose:
            _log.info("MeanField approximation")
            _log.info(f"Shape of mu: {params['mu'].shape}")
            _log.info(f"Shape of std: {params['std'].shape}")

    elif "L_tril" in approx_group.params_dict:
        # FullRank approximation
        params["type"] = "fullrank"
        params["mu"] = np.array(approx_group.params_dict["mu"].eval())
        params["L"] = np.array(approx_group.L.eval())

        if verbose:
            _log.info("FullRank approximation")
            _log.info(f"Shape of mu: {params['mu'].shape}")
            _log.info(f"Shape of L: {params['L'].shape}")

    else:
        raise TypeError("Approximation must be either MeanField or FullRank ADVI")

    return params


def compute_log_p(model, samples, verbose=False):
    r"""Compute math:`\log p(\theta, y)` for a set of samples.

    Parameters
    ----------
    model : PyMC model
        The probabilistic model
    samples : dict
        Dictionary of samples from a variational approximation
    verbose : bool, default=False
        Whether to print debug information

    Returns
    -------
    np.ndarray
        Array of log p(θ,y) values
    """
    nsample = len(next(iter(samples.values())))
    log_p = np.zeros(nsample)

    with model:
        logp_fn = model.compile_fn(model.logp(sum=True, jacobian=False))

    for i in range(nsample):
        point = {name: records[i] for name, records in samples.items()}

        try:
            log_p[i] = float(logp_fn(point))
        except Exception as e:
            if verbose:
                _log.error(f"Error calculating logp for sample {i}: {e}")
            log_p[i] = np.nan

        if verbose and (i + 1) % 100 == 0:
            _log.info(f"Processed {i + 1}/{nsample} logp calculations")

    return log_p


def compute_log_q(samples, approx_params, verbose=False):
    r"""Compute math:`\log q(\theta)` for a set of samples.

    Parameters
    ----------
    samples : dict
        Dictionary of samples from a variational approximation
    approx_params : dict
        Dictionary of approximation parameters from get_approximation_params
    verbose : bool, default=False
        Whether to print debug information

    Returns
    -------
    np.ndarray
        Array of log q(θ) values
    """
    nsample = len(next(iter(samples.values())))
    log_q = np.zeros(nsample)

    for i in range(nsample):
        point = {name: records[i] for name, records in samples.items()}

        try:
            point_array = DictToArrayBijection.map(point).data

            if approx_params["type"] == "meanfield":
                mu = approx_params["mu"]
                std = approx_params["std"]
                log_probs = (
                    -0.5 * np.log(2 * np.pi)
                    - np.log(std)
                    - 0.5 * ((point_array - mu) / std) ** 2
                )
                log_q[i] = np.sum(log_probs)

            elif approx_params["type"] == "fullrank":
                mu = approx_params["mu"]
                L = approx_params["L"]
                dim = len(mu)
                centered = point_array - mu
                scaled = linalg.solve_triangular(L, centered, lower=True)
                log_det = np.sum(np.log(np.diag(L)))
                log_q[i] = (
                    -0.5 * np.sum(scaled**2) - log_det - 0.5 * dim * np.log(2 * np.pi)
                )

        except Exception as e:
            if verbose:
                _log.error(f"Error calculating logq for sample {i}: {e}")
            log_q[i] = np.nan

        if verbose and (i + 1) % 100 == 0:
            _log.info(f"Processed {i + 1}/{nsample} logq calculations")

    return log_q


def compute_log_weights(approx, nsample=1000, verbose=False):
    r"""Calculate math:`\log w_i(\theta)` for variational approximations.
    Works with both MeanField and FullRank ADVI.

    Parameters
    ----------
    approx : Approximation
        The variational approximation object (MeanField or FullRank)
    nsample : int, default=1000
        Number of samples to use for estimation
    verbose : bool, default=False
        Whether to print debug information

    Returns
    -------
    tuple of np.ndarray
        (log_p, log_q, log_weights) - Arrays of log model probabilities,
        log approximation probabilities, and log importance weights
    """
    model = getattr(approx, "model", approx.model)
    approx_params = get_approximation_params(approx, verbose)

    samples = approx.sample_dict_fn(nsample)

    log_p = compute_log_p(model, samples, verbose)
    log_q = compute_log_q(samples, approx_params, verbose)

    valid_indices = ~(np.isnan(log_p) | np.isnan(log_q))
    if not np.all(valid_indices):
        n_invalid = np.sum(~valid_indices)
        if verbose:
            _log.warning(
                f"Warning: {n_invalid} samples had NaN values and were excluded"
            )
        log_p = log_p[valid_indices]
        log_q = log_q[valid_indices]

    log_weights = log_p - log_q

    return log_p, log_q, log_weights
