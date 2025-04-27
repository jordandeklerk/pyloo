"""Tests for the LOO_i module."""

from copy import deepcopy

import arviz as az
import numpy as np
import pytest
import xarray as xr

from ...loo import loo
from ...loo_i import loo_i
from ..helpers import assert_allclose


@pytest.fixture(scope="session")
def centered_eight():
    return az.load_arviz_data("centered_eight")


@pytest.fixture(scope="module")
def multidim_model():
    log_like = np.random.randn(4, 100, 10, 2)  # chains, draws, dim1, dim2
    return az.from_dict(
        posterior={"mu": np.random.randn(4, 100, 2)},
        log_likelihood={"obs": log_like},
        observed_data={"obs": np.random.randn(10, 2)},
    )


@pytest.mark.parametrize("scale", ["log", "negative_log", "deviance"])
def test_loo_i_basic(centered_eight, scale):
    result = loo_i(0, centered_eight, scale=scale)
    assert result is not None
    assert "elpd_loo" in result
    assert "se" in result
    assert "p_loo" in result
    assert result["scale"] == scale
    assert result["n_data_points"] == 1


def test_loo_i_matches_loo(centered_eight):
    full_loo = loo(centered_eight, pointwise=True)

    for i in range(8):  # centered_eight has 8 observations
        single_loo = loo_i(i, centered_eight, pointwise=True)
        assert_allclose(
            single_loo["loo_i"].values,
            full_loo["loo_i"].values[i],
            rtol=1e-10,
            err_msg=f"Mismatch at observation {i}",
        )
        if "pareto_k" in full_loo:
            assert_allclose(
                single_loo["pareto_k"],
                full_loo["pareto_k"][i],
                rtol=1e-10,
                err_msg=f"Pareto k mismatch at observation {i}",
            )


def test_loo_i_one_chain(centered_eight):
    centered_eight_one = deepcopy(centered_eight)
    centered_eight_one.posterior = centered_eight_one.posterior.sel(chain=[0])
    centered_eight_one.log_likelihood = centered_eight_one.log_likelihood.sel(chain=[0])

    result = loo_i(0, centered_eight_one)
    assert result is not None
    assert "elpd_loo" in result


def test_loo_i_pointwise(centered_eight):
    result = loo_i(0, centered_eight, pointwise=True)
    assert result is not None
    assert "loo_i" in result
    assert "pareto_k" in result


def test_loo_i_bad_scale(centered_eight):
    with pytest.raises(
        TypeError, match='Valid scale values are "deviance", "log", "negative_log"'
    ):
        loo_i(0, centered_eight, scale="invalid")


def test_loo_i_missing_loglik():
    data = az.from_dict(posterior={"mu": np.random.randn(4, 100)})
    with pytest.raises(TypeError):
        loo_i(0, data)


def test_loo_i_missing_posterior():
    data = az.from_dict(
        log_likelihood={"obs": np.random.randn(4, 100, 8)},
    )
    with pytest.raises(
        TypeError, match="Must be able to extract a posterior group from data"
    ):
        loo_i(0, data, reff=None)

    assert loo_i(0, data, reff=0.7) is not None


def test_loo_i_warning(centered_eight):
    centered_eight = deepcopy(centered_eight)
    # Make the first observation very influential
    centered_eight.log_likelihood["obs"][:, :, 0] = 10

    with pytest.warns(UserWarning, match="Estimated shape parameter of Pareto"):
        result = loo_i(0, centered_eight, pointwise=True)
        assert result is not None
        assert result["pareto_k"] > result["good_k"]


def test_loo_i_nan_handling(centered_eight):
    centered_eight = deepcopy(centered_eight)
    log_like = centered_eight.log_likelihood["obs"].values
    log_like[0, 0, 0] = np.nan

    centered_eight.log_likelihood["obs"] = xr.DataArray(
        log_like,
        dims=centered_eight.log_likelihood["obs"].dims,
        coords=centered_eight.log_likelihood["obs"].coords,
    )

    with pytest.warns(UserWarning):
        result = loo_i(0, centered_eight)
        assert result is not None
        assert not np.isnan(result["elpd_loo"])


def test_loo_i_multiple_groups(centered_eight):
    centered_eight = deepcopy(centered_eight)
    centered_eight.log_likelihood["obs2"] = centered_eight.log_likelihood["obs"]

    with pytest.raises(TypeError, match="several log likelihood arrays"):
        loo_i(0, centered_eight)

    result = loo_i(0, centered_eight, var_name="obs")
    assert result is not None


def test_loo_i_different_methods(centered_eight):
    psis_result = loo_i(0, centered_eight, pointwise=True)
    assert "pareto_k" in psis_result
    assert "good_k" in psis_result

    with pytest.warns(UserWarning, match="Using SIS for LOO computation"):
        sis_result = loo_i(0, centered_eight, pointwise=True, method="sis")
        assert "ess" in sis_result
        assert "pareto_k" not in sis_result
        assert "good_k" not in sis_result

    with pytest.warns(UserWarning, match="Using TIS for LOO computation"):
        tis_result = loo_i(0, centered_eight, pointwise=True, method="tis")
        assert "ess" in tis_result
        assert "pareto_k" not in tis_result
        assert "good_k" not in tis_result


def test_loo_i_invalid_method(centered_eight):
    with pytest.raises(ValueError, match="Invalid method 'invalid'"):
        loo_i(0, centered_eight, method="invalid")


def test_loo_i_sis_tis_low_ess(centered_eight):
    centered_eight = deepcopy(centered_eight)
    centered_eight.log_likelihood["obs"] *= 10

    with pytest.warns(UserWarning, match="Low effective sample size detected"):
        result = loo_i(0, centered_eight, method="sis")
        assert result["warning"]

    with pytest.warns(UserWarning, match="Low effective sample size detected"):
        result = loo_i(0, centered_eight, method="tis")
        assert result["warning"]


def test_loo_i_method_results(centered_eight):
    psis_result = loo_i(0, centered_eight, pointwise=True)
    sis_result = loo_i(0, centered_eight, pointwise=True, method="sis")
    tis_result = loo_i(0, centered_eight, pointwise=True, method="tis")

    assert psis_result["pareto_k"] >= 0
    assert psis_result["good_k"] > 0
    assert psis_result["good_k"] <= 0.7

    n_samples = sis_result["n_samples"]
    assert sis_result["ess"] >= 1
    assert sis_result["ess"] <= n_samples
    assert tis_result["ess"] >= 1
    assert tis_result["ess"] <= n_samples

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


def test_loo_i_invalid_index(centered_eight):
    with pytest.raises(IndexError):
        loo_i(100, centered_eight)

    with pytest.raises(TypeError):
        loo_i("invalid", centered_eight)

    with pytest.raises(ValueError):
        loo_i([0, 1], centered_eight)


def test_loo_i_multidim(multidim_model):
    result = loo_i(0, multidim_model)
    assert result is not None
    assert "elpd_loo" in result

    result = loo_i(10, multidim_model)
    assert result is not None
    assert "elpd_loo" in result
