"""Tests for the unified importance sampling module."""

import numpy as np
import pytest

from ...base import ISMethod, compute_importance_weights
from ...psis import psislw, vi_psis_sampling
from ...sis import sislw
from ...tis import tislw
from ...utils import get_log_likelihood
from ..helpers import assert_arrays_allclose


@pytest.mark.parametrize("data_fixture", ["centered_eight", "non_centered_eight"])
def test_psis_equivalence(data_fixture, request):
    """Test that unified PSIS gives same results as original implementation."""
    data = request.getfixturevalue(data_fixture)
    log_likelihood = get_log_likelihood(data)
    log_likelihood = log_likelihood.stack(__sample__=("chain", "draw"))

    reff = 0.7
    orig_weights, orig_k = psislw(-log_likelihood, reff=reff)
    new_weights, new_k = compute_importance_weights(
        -log_likelihood, method="psis", reff=reff
    )

    assert_arrays_allclose(orig_weights, new_weights)
    assert_arrays_allclose(orig_k, new_k)
    assert_arrays_allclose(np.exp(new_weights).sum("__sample__"), 1.0, rtol=1e-6)


@pytest.mark.parametrize("data_fixture", ["centered_eight", "non_centered_eight"])
def test_sis_equivalence(data_fixture, request):
    """Test that unified SIS gives same results as original implementation."""
    data = request.getfixturevalue(data_fixture)
    log_likelihood = get_log_likelihood(data)
    log_likelihood = log_likelihood.stack(__sample__=("chain", "draw"))

    orig_weights, orig_ess = sislw(-log_likelihood)
    new_weights, new_ess = compute_importance_weights(-log_likelihood, method="sis")

    assert_arrays_allclose(orig_weights, new_weights)
    assert_arrays_allclose(orig_ess, new_ess)
    assert_arrays_allclose(np.exp(new_weights).sum("__sample__"), 1.0, rtol=1e-6)


@pytest.mark.parametrize("data_fixture", ["centered_eight", "non_centered_eight"])
def test_tis_equivalence(data_fixture, request):
    """Test that unified TIS gives same results as original implementation."""
    data = request.getfixturevalue(data_fixture)
    log_likelihood = get_log_likelihood(data)
    log_likelihood = log_likelihood.stack(__sample__=("chain", "draw"))

    orig_weights, orig_ess = tislw(-log_likelihood)
    new_weights, new_ess = compute_importance_weights(-log_likelihood, method="tis")

    assert_arrays_allclose(orig_weights, new_weights)
    assert_arrays_allclose(orig_ess, new_ess)
    assert_arrays_allclose(np.exp(new_weights).sum("__sample__"), 1.0, rtol=1e-6)


def test_invalid_method(centered_eight):
    """Test that invalid method raises appropriate error."""
    log_likelihood = get_log_likelihood(centered_eight)
    log_likelihood = log_likelihood.stack(__sample__=("chain", "draw"))

    with pytest.raises(ValueError, match="Invalid method"):
        compute_importance_weights(-log_likelihood, method="invalid")


def test_method_case_insensitive(centered_eight):
    """Test that method parameter is case insensitive."""
    log_likelihood = get_log_likelihood(centered_eight)
    log_likelihood = log_likelihood.stack(__sample__=("chain", "draw"))

    standard_methods = [ISMethod.PSIS, ISMethod.SIS, ISMethod.TIS]
    for method in standard_methods:
        upper_weights, upper_diag = compute_importance_weights(
            -log_likelihood, method=method.value.upper()
        )
        lower_weights, lower_diag = compute_importance_weights(
            -log_likelihood, method=method.value.lower()
        )
        assert_arrays_allclose(upper_weights, lower_weights)
        assert_arrays_allclose(upper_diag, lower_diag)


def test_variational_psis():
    """Test variational inference with PSIS method."""
    np.random.seed(42)
    samples = np.random.normal(size=(4, 1000, 10))
    logP = np.random.normal(size=(4, 1000))
    logQ = np.random.normal(size=(4, 1000))
    num_draws = 500

    log_weights, pareto_k = compute_importance_weights(
        method="psis",
        variational=True,
        samples=samples,
        logP=logP,
        logQ=logQ,
        num_draws=num_draws,
        random_seed=42,
    )

    expected = vi_psis_sampling(
        samples=samples,
        logP=logP,
        logQ=logQ,
        num_draws=num_draws,
        method="psis",
        random_seed=42,
    )

    assert_arrays_allclose(log_weights, expected.log_weights)
    assert_arrays_allclose(pareto_k, expected.pareto_k)


def test_variational_psir():
    """Test variational inference with PSIR method."""
    np.random.seed(42)
    samples = np.random.normal(size=(4, 1000, 10))
    logP = np.random.normal(size=(4, 1000))
    logQ = np.random.normal(size=(4, 1000))
    num_draws = 500

    log_weights, pareto_k = compute_importance_weights(
        method="psir",
        variational=True,
        samples=samples,
        logP=logP,
        logQ=logQ,
        num_draws=num_draws,
        random_seed=42,
    )

    expected = vi_psis_sampling(
        samples=samples,
        logP=logP,
        logQ=logQ,
        num_draws=num_draws,
        method="psir",
        random_seed=42,
    )

    assert_arrays_allclose(log_weights, expected.log_weights)
    assert_arrays_allclose(pareto_k, expected.pareto_k)


def test_variational_identity():
    """Test variational inference with IDENTITY method."""
    np.random.seed(42)
    samples = np.random.normal(size=(4, 1000, 10))
    logP = np.random.normal(size=(4, 1000))
    logQ = np.random.normal(size=(4, 1000))
    num_draws = 500

    log_weights, pareto_k = compute_importance_weights(
        method="identity",
        variational=True,
        samples=samples,
        logP=logP,
        logQ=logQ,
        num_draws=num_draws,
        random_seed=42,
    )

    expected = vi_psis_sampling(
        samples=samples,
        logP=logP,
        logQ=logQ,
        num_draws=num_draws,
        method="identity",
        random_seed=42,
    )

    assert_arrays_allclose(log_weights, expected.log_weights)
    assert pareto_k is None


def test_variational_missing_parameters():
    """Test error handling for missing parameters in variational inference."""
    np.random.seed(42)
    samples = np.random.normal(size=(4, 1000, 10))
    logP = np.random.normal(size=(4, 1000))
    logQ = np.random.normal(size=(4, 1000))
    num_draws = 500

    with pytest.raises(
        ValueError, match="samples, logP, logQ, and num_draws must be provided"
    ):
        compute_importance_weights(
            method="psis",
            variational=True,
            samples=None,
            logP=logP,
            logQ=logQ,
            num_draws=num_draws,
        )

    with pytest.raises(
        ValueError, match="samples, logP, logQ, and num_draws must be provided"
    ):
        compute_importance_weights(
            method="psis",
            variational=True,
            samples=samples,
            logP=None,
            logQ=logQ,
            num_draws=num_draws,
        )

    with pytest.raises(
        ValueError, match="samples, logP, logQ, and num_draws must be provided"
    ):
        compute_importance_weights(
            method="psis",
            variational=True,
            samples=samples,
            logP=logP,
            logQ=None,
            num_draws=num_draws,
        )

    with pytest.raises(
        ValueError, match="samples, logP, logQ, and num_draws must be provided"
    ):
        compute_importance_weights(
            method="psis",
            variational=True,
            samples=samples,
            logP=logP,
            logQ=logQ,
            num_draws=None,
        )


def test_variational_unsupported_method():
    """Test error handling for unsupported methods in variational inference."""
    np.random.seed(42)
    samples = np.random.normal(size=(4, 1000, 10))
    logP = np.random.normal(size=(4, 1000))
    logQ = np.random.normal(size=(4, 1000))
    num_draws = 500

    with pytest.raises(ValueError, match="not supported for variational inference"):
        compute_importance_weights(
            method="sis",
            variational=True,
            samples=samples,
            logP=logP,
            logQ=logQ,
            num_draws=num_draws,
        )

    with pytest.raises(ValueError, match="not supported for variational inference"):
        compute_importance_weights(
            method="tis",
            variational=True,
            samples=samples,
            logP=logP,
            logQ=logQ,
            num_draws=num_draws,
        )


def test_standard_missing_log_weights():
    """Test error handling for missing log_weights in standard importance sampling."""
    with pytest.raises(
        ValueError, match="log_weights must be provided when variational=False"
    ):
        compute_importance_weights(
            log_weights=None,
            method="psis",
            variational=False,
        )
