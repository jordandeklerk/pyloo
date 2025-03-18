"""Tests for the LOO module."""

from copy import deepcopy

import arviz as az
import numpy as np
import pytest
import xarray as xr

from ...loo import loo
from ...psis import psislw
from ...wrapper.pymc import PyMCWrapper
from ..helpers import assert_allclose, assert_array_almost_equal


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
    with pytest.raises(
        TypeError, match='Valid scale values are "deviance", "log", "negative_log"'
    ):
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
    with pytest.raises(
        TypeError, match="Must be able to extract a posterior group from data"
    ):
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
    log_likelihood = centered_eight.log_likelihood["obs"].stack(
        __sample__=("chain", "draw")
    )
    n_samples = log_likelihood.shape[-1]

    ess_p = az.ess(centered_eight.posterior, method="mean")
    reff = (
        np.hstack([ess_p[v].values.flatten() for v in ess_p.data_vars]).mean()
        / n_samples
    )

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


def test_loo_different_methods(centered_eight):
    """Test LOO computation with different IS methods."""
    psis_result = loo(centered_eight, pointwise=True)
    assert "pareto_k" in psis_result
    assert "good_k" in psis_result

    with pytest.warns(UserWarning, match="Using SIS for LOO computation"):
        sis_result = loo(centered_eight, pointwise=True, method="sis")
        assert "ess" in sis_result
        assert "pareto_k" not in sis_result
        assert "good_k" not in sis_result

    with pytest.warns(UserWarning, match="Using TIS for LOO computation"):
        tis_result = loo(centered_eight, pointwise=True, method="tis")
        assert "ess" in tis_result
        assert "pareto_k" not in tis_result
        assert "good_k" not in tis_result


def test_loo_invalid_method(centered_eight):
    """Test LOO computation with invalid method."""
    with pytest.raises(ValueError, match="Invalid method 'invalid'"):
        loo(centered_eight, method="invalid")


def test_loo_sis_tis_low_ess(centered_eight):
    """Test warning for low ESS with SIS/TIS methods."""
    centered_eight = deepcopy(centered_eight)
    centered_eight.log_likelihood["obs"] *= 10

    with pytest.warns(UserWarning, match="Low effective sample size detected"):
        result = loo(centered_eight, method="sis")
        assert result["warning"]

    with pytest.warns(UserWarning, match="Low effective sample size detected"):
        result = loo(centered_eight, method="tis")
        assert result["warning"]


def test_loo_non_pointwise_returns(centered_eight):
    """Test non-pointwise returns for different methods."""
    psis_result = loo(centered_eight, pointwise=False)
    assert "good_k" in psis_result

    sis_result = loo(centered_eight, pointwise=False, method="sis")
    assert "good_k" not in sis_result

    tis_result = loo(centered_eight, pointwise=False, method="tis")
    assert "good_k" not in tis_result


def test_loo_method_results(centered_eight):
    """Test that results from different methods are numerically reasonable."""
    psis_result = loo(centered_eight, pointwise=True)
    sis_result = loo(centered_eight, pointwise=True, method="sis")
    tis_result = loo(centered_eight, pointwise=True, method="tis")

    assert np.all(psis_result["pareto_k"] >= 0)
    assert psis_result["good_k"] > 0
    assert psis_result["good_k"] <= 0.7

    n_samples = sis_result["n_samples"]
    assert np.all(sis_result["ess"] >= 1)
    assert np.all(sis_result["ess"] <= n_samples)
    assert np.all(tis_result["ess"] >= 1)
    assert np.all(tis_result["ess"] <= n_samples)

    elpds = np.array(
        [psis_result["elpd_loo"], sis_result["elpd_loo"], tis_result["elpd_loo"]]
    )
    assert np.all(np.isfinite(elpds))

    ses = np.array([psis_result["se"], sis_result["se"], tis_result["se"]])
    assert np.all(ses > 0)

    max_diff = np.max(np.abs(elpds[:, None] - elpds))
    max_se = 3 * np.max(ses)
    assert max_diff < max_se, (
        f"Maximum difference between ELPDs ({max_diff:.2f}) exceeds "
        f"3 times the maximum standard error ({max_se:.2f})"
    )

    p_loos = np.array([psis_result["p_loo"], sis_result["p_loo"], tis_result["p_loo"]])
    assert np.all(p_loos >= 0)
    assert np.all(p_loos <= n_samples)

    assert np.all(np.isfinite(psis_result["loo_i"]))
    assert np.all(np.isfinite(sis_result["loo_i"]))
    assert np.all(np.isfinite(tis_result["loo_i"]))


def test_loo_moment_matching(problematic_k_model):
    """Test moment matching for LOO computation."""
    model, idata = problematic_k_model
    wrapper = PyMCWrapper(model, idata)

    loo_orig = loo(idata, pointwise=True)
    loo_mm = loo(idata, pointwise=True, moment_match=True, wrapper=wrapper)

    assert np.all(loo_mm.pareto_k.values <= loo_orig.pareto_k.values)


def test_loo_moment_matching_no_pointwise(problematic_k_model):
    """Test moment matching for LOO computation with no pointwise results."""
    model, idata = problematic_k_model
    wrapper = PyMCWrapper(model, idata)

    with pytest.raises(
        ValueError, match="Moment matching requires pointwise LOO results"
    ):
        loo(idata, moment_match=True, wrapper=wrapper)
