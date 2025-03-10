"""Tests for the unified importance sampling module."""

import numpy as np
import pytest

from ...base import ISMethod, compute_importance_weights
from ...psis import psislw
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

    for method in ISMethod:
        upper_weights, upper_diag = compute_importance_weights(
            -log_likelihood, method=method.value.upper()
        )
        lower_weights, lower_diag = compute_importance_weights(
            -log_likelihood, method=method.value.lower()
        )
        assert_arrays_allclose(upper_weights, lower_weights)
        assert_arrays_allclose(upper_diag, lower_diag)
