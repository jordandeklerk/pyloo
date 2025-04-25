"""Tests for the LOO module."""

from copy import deepcopy

import arviz as az
import numpy as np
import pytest
import xarray as xr

from ...loo import loo
from ...loo_subsample import loo_subsample
from ...psis import psislw
from ...wrapper.pymc.pymc import PyMCWrapper
from ..helpers import assert_allclose, assert_array_almost_equal


@pytest.fixture(scope="session")
def centered_eight():
    return az.load_arviz_data("centered_eight")


@pytest.fixture(scope="module")
def multidim_model():
    log_like = np.random.randn(4, 100, 10, 2) 
    return az.from_dict(
        posterior={"mu": np.random.randn(4, 100, 2)},
        log_likelihood={"obs": log_like},
        observed_data={"obs": np.random.randn(10, 2)},
    )


@pytest.mark.parametrize("scale", ["log", "negative_log", "deviance"])
def test_loo_basic(centered_eight, scale):
    az_result = az.loo(centered_eight, scale=scale)
    pl_result = loo(centered_eight, scale=scale)

    assert_allclose(pl_result["elpd_loo"], az_result["elpd_loo"])
    assert_allclose(pl_result["p_loo"], az_result["p_loo"])
    assert_allclose(pl_result["se"], az_result["se"])
    assert pl_result["scale"] == az_result["scale"]


def test_loo_one_chain(centered_eight):
    centered_eight_one = deepcopy(centered_eight)
    centered_eight_one.posterior = centered_eight_one.posterior.sel(chain=[0])
    centered_eight_one.log_likelihood = centered_eight_one.log_likelihood.sel(chain=[0])

    result = loo(centered_eight_one)
    assert result is not None
    assert "elpd_loo" in result


def test_loo_pointwise(centered_eight):
    result = loo(centered_eight, pointwise=True)
    assert result is not None
    assert "loo_i" in result
    assert "pareto_k" in result

    az_result = az.loo(centered_eight, pointwise=True)
    assert_array_almost_equal(result["loo_i"], az_result["loo_i"])
    assert_array_almost_equal(result["pareto_k"], az_result["pareto_k"])


def test_loo_bad_scale(centered_eight):
    with pytest.raises(
        TypeError, match='Valid scale values are "deviance", "log", "negative_log"'
    ):
        loo(centered_eight, scale="invalid")


def test_loo_missing_loglik():
    data = az.from_dict(posterior={"mu": np.random.randn(4, 100)})
    with pytest.raises(TypeError):
        loo(data)


def test_loo_missing_posterior():
    data = az.from_dict(
        log_likelihood={"obs": np.random.randn(4, 100, 8)},
    )
    with pytest.raises(
        TypeError, match="Must be able to extract a posterior group from data"
    ):
        loo(data, reff=None)

    assert loo(data, reff=0.7) is not None


def test_loo_warning(centered_eight):
    centered_eight = deepcopy(centered_eight)
    # Make one of the observations very influential
    centered_eight.log_likelihood["obs"][:, :, 1] = 10

    with pytest.warns(UserWarning, match="Estimated shape parameter of Pareto"):
        result = loo(centered_eight, pointwise=True)
        assert result is not None
        assert any(k > result["good_k"] for k in result["pareto_k"])


def test_loo_pointwise_warning(centered_eight):
    centered_eight = deepcopy(centered_eight)
    # Make all observations have same log likelihood
    centered_eight.log_likelihood["obs"][:] = 1.0

    with pytest.warns(UserWarning) as record:
        result = loo(centered_eight, pointwise=True)
        assert result is not None
        assert any("The point-wise LOO is the same" in str(w.message) for w in record)


def test_psislw_matches(centered_eight):
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
    result = loo(multidim_model)
    assert result is not None
    assert "elpd_loo" in result

    az_result = az.loo(multidim_model)
    assert_allclose(result["elpd_loo"], az_result["elpd_loo"])


def test_loo_nan_handling(centered_eight):
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
    centered_eight = deepcopy(centered_eight)
    centered_eight.log_likelihood["obs2"] = centered_eight.log_likelihood["obs"]

    with pytest.raises(TypeError, match="several log likelihood arrays"):
        loo(centered_eight)

    result = loo(centered_eight, var_name="obs")
    assert result is not None


def test_loo_different_methods(centered_eight):
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
    with pytest.raises(ValueError, match="Invalid method 'invalid'"):
        loo(centered_eight, method="invalid")


def test_loo_sis_tis_low_ess(centered_eight):
    centered_eight = deepcopy(centered_eight)
    centered_eight.log_likelihood["obs"] *= 10

    with pytest.warns(UserWarning, match="Low effective sample size detected"):
        result = loo(centered_eight, method="sis")
        assert result["warning"]

    with pytest.warns(UserWarning, match="Low effective sample size detected"):
        result = loo(centered_eight, method="tis")
        assert result["warning"]


def test_loo_non_pointwise_returns(centered_eight):
    psis_result = loo(centered_eight, pointwise=False)
    assert "good_k" in psis_result

    sis_result = loo(centered_eight, pointwise=False, method="sis")
    assert "good_k" not in sis_result

    tis_result = loo(centered_eight, pointwise=False, method="tis")
    assert "good_k" not in tis_result


def test_loo_method_results(centered_eight):
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
    model, idata = problematic_k_model
    wrapper = PyMCWrapper(model, idata)

    loo_orig = loo(idata, pointwise=True)
    loo_mm = loo(idata, pointwise=True, moment_match=True, wrapper=wrapper)

    assert np.all(loo_mm.pareto_k.values <= loo_orig.pareto_k.values)


def test_loo_moment_matching_no_pointwise(problematic_k_model):
    model, idata = problematic_k_model
    wrapper = PyMCWrapper(model, idata)

    with pytest.raises(
        ValueError, match="Moment matching requires pointwise LOO results"
    ):
        loo(idata, moment_match=True, wrapper=wrapper)


def test_loo_jacobian_adjustment(centered_eight):
    centered_eight = deepcopy(centered_eight)

    y = centered_eight.observed_data.obs.values
    result_no_adj = loo(centered_eight, pointwise=True)
    jacobian_adj = np.log(np.abs(2 * y))
    result_with_adj = loo(centered_eight, pointwise=True, jacobian=jacobian_adj)

    expected_adjusted_loo_i = result_no_adj.loo_i.values + jacobian_adj
    assert_array_almost_equal(result_with_adj.loo_i.values, expected_adjusted_loo_i)

    assert result_with_adj["elpd_loo"] == np.sum(result_with_adj.loo_i.values)
    assert result_with_adj["elpd_loo"] != result_no_adj["elpd_loo"]

    assert_array_almost_equal(result_with_adj.pareto_k, result_no_adj.pareto_k)

    with pytest.raises(
        ValueError,
        match=(
            "Jacobian adjustment requires pointwise LOO results. Please set"
            " pointwise=True when using jacobian_adjustment."
        ),
    ):
        loo(centered_eight, pointwise=False, jacobian=jacobian_adj)

    wrong_shape_adj = np.ones(len(y) + 1)
    with pytest.raises(
        ValueError, match="Jacobian adjustment shape .* does not match loo_i shape"
    ):
        loo(centered_eight, pointwise=True, jacobian=wrong_shape_adj)


def test_loo_wells(wells_model):
    _, idata = wells_model
    result = loo(idata, pointwise=True)
    result_subsample = loo_subsample(
        idata,
        observations=100,
        estimator="diff_srs",
        loo_approximation="plpd",
        pointwise=True,
    )

    assert result is not None
    assert "elpd_loo" in result
    assert "p_loo" in result
    assert "looic" in result
    assert "looic_se" in result


def test_loo_roaches(roaches_model):
    _, idata = roaches_model
    result = loo(idata, pointwise=True)

    assert result is not None
    assert "elpd_loo" in result
    assert "p_loo" in result
    assert "looic" in result
    assert "looic_se" in result


def test_loo_mixture(problematic_k_model):
    _, idata = problematic_k_model

    mix_result = loo(idata, pointwise=True, mixture=True)
    reg_result = loo(idata, pointwise=True)

    assert mix_result is not None
    assert reg_result is not None

    assert "elpd_loo" in mix_result
    assert "se" in mix_result
    assert "loo_i" in mix_result

    assert "p_loo" not in mix_result
    assert "looic" not in mix_result

    assert "p_loo" in reg_result
    assert "looic" in reg_result

    assert mix_result["elpd_loo"] != reg_result["elpd_loo"]

    assert np.isfinite(mix_result["elpd_loo"])
    assert np.isfinite(reg_result["elpd_loo"])

    assert np.all(np.isfinite(mix_result.loo_i.values))
    assert np.all(np.isfinite(reg_result.loo_i.values))


def test_loo_moment_matching_custom_funcs():
    mock_model = {"data": {"N": 8, "K": 2}, "fit_params": {"S": 400}}

    n_chains = 4
    n_draws = 100
    n_obs = 8
    n_params = 2
    S = n_chains * n_draws

    log_like = np.random.randn(n_chains, n_draws, n_obs) * 0.5
    log_like[:, :, 0] = np.random.randn(n_chains, n_draws) * 5
    idata = az.from_dict(
        posterior={"beta": np.random.randn(n_chains, n_draws, n_params - 1)},
        log_likelihood={"obs": log_like},
        observed_data={"obs": np.random.randn(n_obs)},
    )

    def mock_post_draws(model, **kwargs):
        K = model["data"]["K"]
        return {
            "beta": np.random.randn(S, K),
            "intercept": np.random.randn(S),
        }

    def mock_log_lik_i(model, i, **kwargs):
        S_fit = model["fit_params"]["S"]
        return np.random.randn(S_fit) * (5 if i == 0 else 0.5)

    def mock_unconstrain_pars(model, pars, **kwargs):
        S_fit = model["fit_params"]["S"]
        K = model["data"]["K"]
        upars = np.zeros((S_fit, K + 1))
        upars[:, 0] = pars["intercept"]
        upars[:, 1:] = pars["beta"]
        return upars

    def mock_log_prob_upars_fn(model, upars, **kwargs):
        S_fit = model["fit_params"]["S"]
        return np.random.randn(S_fit) - 0.5 * np.sum(upars**2, axis=1)

    def mock_log_lik_i_upars_fn(model, upars, i, **kwargs):
        S_fit = model["fit_params"]["S"]
        base_lik = -0.5 * np.sum((upars - i) ** 2, axis=1)
        noise = np.random.randn(S_fit) * (5 if i == 0 else 0.5)
        return base_lik + noise

    loo_mm_custom = loo(
        idata,
        pointwise=True,
        moment_match=True,
        model_obj=mock_model,
        post_draws=mock_post_draws,
        log_lik_i=mock_log_lik_i,
        unconstrain_pars=mock_unconstrain_pars,
        log_prob_upars_fn=mock_log_prob_upars_fn,
        log_lik_i_upars_fn=mock_log_lik_i_upars_fn,
        verbose=True,
    )

    assert loo_mm_custom is not None
    assert "elpd_loo" in loo_mm_custom
    assert "pareto_k" in loo_mm_custom
    assert "loo_i" in loo_mm_custom
    assert np.all(np.isfinite(loo_mm_custom.pareto_k))

    loo_orig = loo(idata, pointwise=True)
    assert loo_mm_custom.pareto_k[0] <= loo_orig.pareto_k[0] + 1e-6

    with pytest.raises(
        ValueError, match="following functions must be passed via kwargs"
    ):
        loo(
            idata,
            pointwise=True,
            moment_match=True,
            model_obj=mock_model,
            post_draws=mock_post_draws,
            log_lik_i=mock_log_lik_i,
            unconstrain_pars=mock_unconstrain_pars,
            log_prob_upars_fn=mock_log_prob_upars_fn,
            # log_lik_i_upars_fn is missing
        )
