"""Tests for the LOO module."""
from copy import deepcopy

import arviz as az
import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose, assert_array_almost_equal

from pyloo.loo import loo
from pyloo.psis import psislw


@pytest.fixture(scope="session")
def centered_eight():
    """Load the centered_eight example dataset from ArviZ."""
    return az.load_arviz_data("centered_eight")


@pytest.fixture(scope="module")
def multidim_model():
    """Create a model with multidimensional log-likelihood."""
    log_like = np.random.randn(4, 100, 10, 2)  # chains, draws, dim1, dim2
    return az.from_dict(
        posterior={"mu": np.random.randn(4, 100, 2)},
        log_likelihood={"obs": log_like},
        observed_data={"obs": np.random.randn(10, 2)},
    )


@pytest.mark.parametrize("scale", ["log", "negative_log", "deviance"])
def test_loo_basic(centered_eight, scale):
    """Test basic LOO computation with different scales."""
    az_result = az.loo(centered_eight, scale=scale)
    pl_result = loo(centered_eight, scale=scale)

    assert_allclose(pl_result["elpd_loo"], az_result["elpd_loo"])
    assert_allclose(pl_result["p_loo"], az_result["p_loo"])
    assert_allclose(pl_result["se"], az_result["se"])
    assert pl_result["scale"] == az_result["scale"]


def test_loo_one_chain(centered_eight):
    """Test LOO computation with single chain."""
    centered_eight_one = deepcopy(centered_eight)
    centered_eight_one.posterior = centered_eight_one.posterior.sel(chain=[0])
    centered_eight_one.log_likelihood = centered_eight_one.log_likelihood.sel(chain=[0])

    result = loo(centered_eight_one)
    assert result is not None
    assert "elpd_loo" in result


def test_loo_pointwise(centered_eight):
    """Test LOO computation with pointwise=True."""
    result = loo(centered_eight, pointwise=True)
    assert result is not None
    assert "loo_i" in result
    assert "pareto_k" in result

    az_result = az.loo(centered_eight, pointwise=True)
    assert_array_almost_equal(result["loo_i"], az_result["loo_i"])
    assert_array_almost_equal(result["pareto_k"], az_result["pareto_k"])


def test_loo_bad_scale(centered_eight):
    """Test LOO computation with invalid scale."""
    with pytest.raises(TypeError, match='Valid scale values are "deviance", "log", "negative_log"'):
        loo(centered_eight, scale="invalid")


def test_loo_missing_loglik():
    """Test LOO computation with missing log_likelihood."""
    data = az.from_dict(posterior={"mu": np.random.randn(4, 100)})
    with pytest.raises(TypeError):
        loo(data)


def test_loo_missing_posterior():
    """Test LOO computation with missing posterior and no reff provided."""
    data = az.from_dict(
        log_likelihood={"obs": np.random.randn(4, 100, 8)},
    )
    with pytest.raises(TypeError, match="Must be able to extract a posterior group from data"):
        loo(data, reff=None)

    assert loo(data, reff=0.7) is not None


def test_loo_warning(centered_eight):
    """Test warning for high Pareto k values."""
    centered_eight = deepcopy(centered_eight)
    # Make one of the observations very influential
    centered_eight.log_likelihood["obs"][:, :, 1] = 10

    with pytest.warns(UserWarning, match="Estimated shape parameter of Pareto"):
        result = loo(centered_eight, pointwise=True)
        assert result is not None
        assert any(k > result["good_k"] for k in result["pareto_k"])


def test_loo_pointwise_warning(centered_eight):
    """Test warning when pointwise LOO equals total LOO."""
    centered_eight = deepcopy(centered_eight)
    # Make all observations have same log likelihood
    centered_eight.log_likelihood["obs"][:] = 1.0

    with pytest.warns(UserWarning) as record:
        result = loo(centered_eight, pointwise=True)
        assert result is not None
        assert any("The point-wise LOO is the same" in str(w.message) for w in record)


def test_psislw_matches(centered_eight):
    """Test that psislw results match between PyLOO and ArviZ."""
    log_likelihood = centered_eight.log_likelihood["obs"].stack(__sample__=("chain", "draw"))
    n_samples = log_likelihood.shape[-1]

    ess_p = az.ess(centered_eight.posterior, method="mean")
    reff = np.hstack([ess_p[v].values.flatten() for v in ess_p.data_vars]).mean() / n_samples

    # Compare PyLOO and ArviZ psislw results
    pl_lw, pl_k = psislw(-log_likelihood, reff)
    az_lw, az_k = az.stats.psislw(-log_likelihood, reff)

    assert_array_almost_equal(pl_lw, az_lw)
    assert_array_almost_equal(pl_k, az_k)


def test_loo_multidim(multidim_model):
    """Test LOO computation with multidimensional log-likelihood."""
    result = loo(multidim_model)
    assert result is not None
    assert "elpd_loo" in result

    az_result = az.loo(multidim_model)
    assert_allclose(result["elpd_loo"], az_result["elpd_loo"])


def test_loo_nan_handling(centered_eight):
    """Test LOO computation with NaN values in log-likelihood."""
    centered_eight = deepcopy(centered_eight)
    log_like = centered_eight.log_likelihood["obs"].values
    log_like[0, 0, 0] = np.nan

    centered_eight.log_likelihood["obs"] = xr.DataArray(
        log_like,
        dims=centered_eight.log_likelihood["obs"].dims,
        coords=centered_eight.log_likelihood["obs"].coords,
    )

    with pytest.warns(UserWarning):
        result = loo(centered_eight)
        assert result is not None
        assert not np.isnan(result["elpd_loo"])


def test_loo_inf_handling(centered_eight):
    """Test LOO computation with infinite values in log-likelihood."""
    centered_eight = deepcopy(centered_eight)
    log_likelihood = centered_eight.log_likelihood["obs"]
    log_like_values = log_likelihood.values.copy()
    log_like_values[0, 0, 0] = np.inf

    centered_eight.log_likelihood["obs"] = xr.DataArray(
        log_like_values, dims=log_likelihood.dims, coords=log_likelihood.coords
    )

    with pytest.warns(UserWarning):
        result = loo(centered_eight)
        assert result is not None
        assert not np.isinf(result["elpd_loo"])


def test_loo_extreme_values(centered_eight):
    """Test LOO computation with extreme values in log-likelihood."""
    centered_eight = deepcopy(centered_eight)
    log_like = centered_eight.log_likelihood["obs"].values

    log_like[0, 0, 0] = 1e10
    log_like[0, 0, 1] = -1e10

    centered_eight.log_likelihood["obs"] = xr.DataArray(
        log_like,
        dims=centered_eight.log_likelihood["obs"].dims,
        coords=centered_eight.log_likelihood["obs"].coords,
    )

    result = loo(centered_eight)
    assert result is not None
    assert np.isfinite(result["elpd_loo"])


def test_loo_constant_values(centered_eight):
    """Test LOO computation with constant values in log-likelihood."""
    centered_eight = deepcopy(centered_eight)
    log_like = centered_eight.log_likelihood["obs"].values
    log_like[:] = 1.0

    centered_eight.log_likelihood["obs"] = xr.DataArray(
        log_like,
        dims=centered_eight.log_likelihood["obs"].dims,
        coords=centered_eight.log_likelihood["obs"].coords,
    )

    with pytest.warns(UserWarning):
        result = loo(centered_eight)
        assert result is not None


def test_loo_multiple_groups(centered_eight):
    """Test LOO computation with multiple log_likelihood groups."""
    centered_eight = deepcopy(centered_eight)
    centered_eight.log_likelihood["obs2"] = centered_eight.log_likelihood["obs"]

    with pytest.raises(TypeError, match="several log likelihood arrays"):
        loo(centered_eight)

    result = loo(centered_eight, var_name="obs")
    assert result is not None
