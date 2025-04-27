"""Tests for non-factorized multivariate models."""


from unittest.mock import patch

import numpy as np
import pymc as pm
import pytest
import xarray as xr
from arviz import InferenceData

from ...elpd import ELPDData
from ...loo_nonfactor import _validate_model_structure, loo_nonfactor


def test_loo_nonfactor_basic(mvn_inference_data):
    loo_results = loo_nonfactor(mvn_inference_data, var_name="y", pointwise=True)

    assert isinstance(loo_results, ELPDData)
    assert "elpd_loo" in loo_results
    assert "p_loo" in loo_results
    assert "loo_i" in loo_results
    assert "pareto_k" in loo_results
    assert loo_results.loo_i.shape == (mvn_inference_data.observed_data.y.shape[0],)
    assert loo_results.pareto_k.shape == (mvn_inference_data.observed_data.y.shape[0],)
    assert not np.isnan(loo_results.elpd_loo)
    assert not np.isnan(loo_results.p_loo)


def test_loo_nonfactor_student_t_basic(mvt_inference_data):
    loo_results = loo_nonfactor(
        mvt_inference_data, var_name="y", pointwise=True, model_type="student_t"
    )

    assert isinstance(loo_results, ELPDData)
    assert "elpd_loo" in loo_results
    assert "p_loo" in loo_results
    assert "loo_i" in loo_results
    assert "pareto_k" in loo_results
    assert loo_results.loo_i.shape == (mvt_inference_data.observed_data.y.shape[0],)
    assert loo_results.pareto_k.shape == (mvt_inference_data.observed_data.y.shape[0],)
    assert not np.isnan(loo_results.elpd_loo)
    assert not np.isnan(loo_results.p_loo)


def test_loo_nonfactor_precision_input(mvn_precision_data):
    loo_results = loo_nonfactor(
        mvn_precision_data, var_name="y", prec_var_name="prec", pointwise=True
    )

    assert isinstance(loo_results, ELPDData)
    assert "elpd_loo" in loo_results
    assert "p_loo" in loo_results
    assert "loo_i" in loo_results
    assert "pareto_k" in loo_results
    assert loo_results.loo_i.shape == (mvn_precision_data.observed_data.y.shape[0],)
    assert loo_results.pareto_k.shape == (mvn_precision_data.observed_data.y.shape[0],)
    assert not np.isnan(loo_results.elpd_loo)
    assert not np.isnan(loo_results.p_loo)


def test_loo_nonfactor_student_t_precision_input(mvt_precision_data):
    loo_results = loo_nonfactor(
        mvt_precision_data,
        var_name="y",
        prec_var_name="prec",
        pointwise=True,
        model_type="student_t",
    )

    assert isinstance(loo_results, ELPDData)
    assert "elpd_loo" in loo_results
    assert "p_loo" in loo_results
    assert "loo_i" in loo_results
    assert "pareto_k" in loo_results
    assert loo_results.loo_i.shape == (mvt_precision_data.observed_data.y.shape[0],)
    assert loo_results.pareto_k.shape == (mvt_precision_data.observed_data.y.shape[0],)
    assert not np.isnan(loo_results.elpd_loo)
    assert not np.isnan(loo_results.p_loo)


def test_verify_mvn_structure(mvn_validation_data):

    valid_idata = mvn_validation_data["valid"]
    no_mu_idata = mvn_validation_data["no_mu"]
    no_cov_prec_idata = mvn_validation_data["no_cov_prec"]
    no_posterior_idata = mvn_validation_data["no_posterior"]

    assert _validate_model_structure(valid_idata, "mu", None, None) is True

    with pytest.warns(UserWarning, match="Mean vector .* not found"):
        assert (
            _validate_model_structure(no_mu_idata, "wrong_mu_name", None, None) is False
        )

    with pytest.warns(
        UserWarning, match="Neither covariance nor precision matrix found"
    ):
        assert (
            _validate_model_structure(
                no_cov_prec_idata, "mu", "wrong_cov", "wrong_prec"
            )
            is False
        )

    assert _validate_model_structure(no_posterior_idata, "mu", None, None) is False


def test_verify_student_t_structure(mvt_validation_data):

    valid_idata = mvt_validation_data["valid"]
    missing_df_idata = mvt_validation_data["missing_df"]

    assert _validate_model_structure(valid_idata, "mu", None, None, "student_t") is True

    with pytest.warns(UserWarning, match="Degrees of freedom.*not found"):
        assert (
            _validate_model_structure(missing_df_idata, "mu", None, None, "student_t")
            is False
        )


def test_loo_nonfactor_warnings(missing_cov_data):
    with pytest.warns(
        UserWarning,
        match="loo_nonfactor\\(\\) with model_type='normal' requires the correct model",
    ):
        with pytest.warns(
            UserWarning, match="Neither covariance nor precision matrix found"
        ):
            with pytest.raises(
                ValueError, match="Could not find posterior samples for covariance"
            ):
                loo_nonfactor(missing_cov_data, var_name="y")


def test_loo_nonfactor_both_cov_prec(both_cov_prec_data):
    loo_results = loo_nonfactor(both_cov_prec_data, var_name="y")
    assert isinstance(loo_results, ELPDData)

    loo_results_prec = loo_nonfactor(
        both_cov_prec_data, var_name="y", prec_var_name="prec", cov_var_name=None
    )
    assert isinstance(loo_results_prec, ELPDData)


def test_loo_nonfactor_custom_names(mvn_custom_names_data):
    loo_results = loo_nonfactor(
        mvn_custom_names_data,
        var_name="observations",
        mu_var_name="mean_vector",
        cov_var_name="covariance_matrix",
    )

    assert isinstance(loo_results, ELPDData)
    assert not np.isnan(loo_results.elpd_loo)
    assert not np.isnan(loo_results.p_loo)

    with pytest.warns(UserWarning, match="Mean vector 'wrong_mu' not found"):
        with pytest.raises(ValueError, match="Posterior variable 'wrong_mu' not found"):
            loo_nonfactor(
                mvn_custom_names_data,
                var_name="observations",
                mu_var_name="wrong_mu",
                cov_var_name="covariance_matrix",
            )


def test_loo_nonfactor_student_t_custom_names(mvt_custom_names_data):
    loo_results = loo_nonfactor(
        mvt_custom_names_data,
        var_name="observations",
        mu_var_name="location",
        cov_var_name="scale_matrix",
        model_type="student_t",
        df_var_name="nu",
    )

    assert isinstance(loo_results, ELPDData)
    assert not np.isnan(loo_results.elpd_loo)
    assert not np.isnan(loo_results.p_loo)

    with pytest.warns(UserWarning, match="Degrees of freedom.*not found"):
        with pytest.raises(ValueError, match="Degrees of freedom variable.*not found"):
            loo_nonfactor(
                mvt_custom_names_data,
                var_name="observations",
                mu_var_name="location",
                cov_var_name="scale_matrix",
                model_type="student_t",
                df_var_name="wrong_df",
            )


def test_loo_nonfactor_student_t_negative_df(mvt_negative_df_data):
    with pytest.warns(UserWarning, match="Non-positive degrees of freedom"):
        loo_results = loo_nonfactor(
            mvt_negative_df_data, var_name="y", model_type="student_t", pointwise=True
        )
        assert isinstance(loo_results, ELPDData)
        assert not np.isnan(loo_results.elpd_loo)
        assert not np.isnan(loo_results.p_loo)


def test_loo_nonfactor_singular_matrices(singular_matrix_data):
    with pytest.warns(UserWarning, match="Invalid values detected in log-likelihood"):
        loo_results = loo_nonfactor(singular_matrix_data, var_name="y")
        assert isinstance(loo_results, ELPDData)


@pytest.mark.skipif(pm is None, reason="PyMC not installed")
def test_loo_nonfactor_pymc_model(mvn_spatial_model):
    model, idata, _, _, _ = mvn_spatial_model
    n_obs = idata.observed_data.y_obs.shape[0]

    loo_results = loo_nonfactor(
        idata,
        var_name="y_obs",
        mu_var_name="mu",
        cov_var_name="cov",
        pointwise=True,
    )

    assert isinstance(loo_results, ELPDData)

    assert "elpd_loo" in loo_results
    assert "p_loo" in loo_results
    assert "loo_i" in loo_results
    assert "pareto_k" in loo_results
    assert loo_results.loo_i.shape == (n_obs,)
    assert loo_results.pareto_k.shape == (n_obs,)

    assert not np.isnan(loo_results.elpd_loo)
    assert not np.isnan(loo_results.p_loo)
    assert not np.any(np.isnan(loo_results.loo_i))
    assert not np.any(np.isnan(loo_results.pareto_k))


@pytest.mark.skipif(pm is None, reason="PyMC not installed")
def test_loo_nonfactor_student_t_pymc_model(mvt_spatial_model):
    model, idata, _, _, _ = mvt_spatial_model
    n_obs = idata.observed_data.y_obs.shape[0]

    loo_results = loo_nonfactor(
        idata,
        var_name="y_obs",
        mu_var_name="mu",
        cov_var_name="cov",
        model_type="student_t",
        df_var_name="df_det",
        pointwise=True,
    )

    assert isinstance(loo_results, ELPDData)

    assert "elpd_loo" in loo_results
    assert "p_loo" in loo_results
    assert "loo_i" in loo_results
    assert "pareto_k" in loo_results
    assert loo_results.loo_i.shape == (n_obs,)
    assert loo_results.pareto_k.shape == (n_obs,)

    assert not np.isnan(loo_results.elpd_loo)
    assert not np.isnan(loo_results.p_loo)
    assert not np.any(np.isnan(loo_results.loo_i))
    assert not np.any(np.isnan(loo_results.pareto_k))


@pytest.mark.parametrize("method", ["psis", "sis", "tis"])
def test_loo_nonfactor_methods(mvn_inference_data, method):
    if method != "psis":
        with pytest.warns(UserWarning, match=f"Using {method.upper()} for LOO"):
            loo_results = loo_nonfactor(
                mvn_inference_data, var_name="y", method=method, pointwise=True
            )
    else:
        loo_results = loo_nonfactor(
            mvn_inference_data, var_name="y", method=method, pointwise=True
        )

    assert isinstance(loo_results, ELPDData)
    assert "elpd_loo" in loo_results
    assert "p_loo" in loo_results
    assert "loo_i" in loo_results
    diag_name = "pareto_k" if method == "psis" else "ess"
    assert diag_name in loo_results
    assert loo_results[diag_name].shape == (
        mvn_inference_data.observed_data.y.shape[0],
    )
    assert not np.isnan(loo_results.elpd_loo)


@pytest.mark.parametrize("scale", ["log", "negative_log", "deviance"])
def test_loo_nonfactor_scales(mvn_inference_data, scale):
    loo_results = loo_nonfactor(
        mvn_inference_data, var_name="y", scale=scale, pointwise=True
    )
    assert isinstance(loo_results, ELPDData)
    assert loo_results.scale == scale
    assert "elpd_loo" in loo_results
    assert "looic" in loo_results

    expected_looic = -2 * loo_results.elpd_loo

    assert np.isclose(loo_results.looic, expected_looic)


def test_loo_nonfactor_no_pointwise(mvn_inference_data):
    loo_results = loo_nonfactor(mvn_inference_data, var_name="y", pointwise=False)
    assert isinstance(loo_results, ELPDData)
    assert "elpd_loo" in loo_results
    assert "p_loo" in loo_results
    assert "loo_i" not in loo_results
    assert "pareto_k" not in loo_results
    assert "ess" not in loo_results


def test_loo_nonfactor_invalid_scale(mvn_inference_data):
    with pytest.raises(TypeError, match='Valid scale values are "deviance", "log"'):
        loo_nonfactor(mvn_inference_data, var_name="y", scale="invalid_scale")


def test_loo_nonfactor_invalid_method(mvn_inference_data):
    with pytest.raises(ValueError, match="Invalid method 'invalid_method'"):
        loo_nonfactor(mvn_inference_data, var_name="y", method="invalid_method")


def test_loo_nonfactor_missing_var_name_ambiguous(mvn_inference_data):
    idata_copy = mvn_inference_data.copy()
    idata_copy.observed_data["y2"] = xr.DataArray(
        np.random.randn(idata_copy.observed_data.y.shape[0]),
        dims=idata_copy.observed_data.y.dims,
        coords=idata_copy.observed_data.y.coords,
    )

    with pytest.raises(ValueError, match="Multiple variables found in observed_data"):
        loo_nonfactor(idata_copy)


def test_loo_nonfactor_missing_var_name_success(mvn_inference_data):
    idata_single_obs = mvn_inference_data.copy()
    obs_vars = list(idata_single_obs.observed_data.data_vars)
    if len(obs_vars) > 1:
        for var in obs_vars[1:]:
            del idata_single_obs.observed_data[var]

    loo_results = loo_nonfactor(idata_single_obs)
    assert isinstance(loo_results, ELPDData)
    assert "elpd_loo" in loo_results


def test_loo_nonfactor_var_name_not_found(mvn_inference_data):
    with pytest.raises(ValueError, match="Variable 'wrong_name' not found"):
        loo_nonfactor(mvn_inference_data, var_name="wrong_name")


def test_loo_nonfactor_y_ndim_error(mvn_inference_data):
    idata_copy = mvn_inference_data.copy()
    y_orig = idata_copy.observed_data["y"]
    idata_copy.observed_data["y"] = xr.DataArray(
        y_orig.values[:, np.newaxis],
        dims=(y_orig.dims[0], "dummy_dim"),
        coords={y_orig.dims[0]: y_orig.coords[y_orig.dims[0]]},
    )
    with pytest.raises(ValueError, match="Observed data 'y' must be 1-dimensional"):
        loo_nonfactor(idata_copy, var_name="y")


def test_loo_nonfactor_mu_shape_error(mvn_inference_data):
    idata_copy = mvn_inference_data.copy()
    mu_orig = idata_copy.posterior["mu"]
    idata_copy.posterior["mu"] = xr.DataArray(
        mu_orig.values[:, :, :-1],
        dims=mu_orig.dims,
        coords={
            "chain": mu_orig.coords["chain"],
            "draw": mu_orig.coords["draw"],
            mu_orig.dims[-1]: mu_orig.coords[mu_orig.dims[-1]][:-1],
        },
    )
    loo_results = loo_nonfactor(idata_copy, var_name="y")
    assert isinstance(loo_results, ELPDData)


def test_loo_nonfactor_cov_shape_error(mvn_inference_data):
    idata_copy = mvn_inference_data.copy()
    cov_orig = idata_copy.posterior["cov"]
    new_cov_vals = cov_orig.values[:, :, :-1, :-1]
    idata_copy.posterior["cov"] = xr.DataArray(
        new_cov_vals,
        dims=cov_orig.dims,
        coords={
            "chain": cov_orig.coords["chain"],
            "draw": cov_orig.coords["draw"],
            cov_orig.dims[-2]: cov_orig.coords[cov_orig.dims[-2]][:-1],
            cov_orig.dims[-2]: cov_orig.coords[cov_orig.dims[-2]][:-1],
            cov_orig.dims[-1]: cov_orig.coords[cov_orig.dims[-1]][:-1],
        },
    )
    loo_results = loo_nonfactor(idata_copy, var_name="y")
    assert isinstance(loo_results, ELPDData)


def test_loo_nonfactor_prec_shape_error(mvn_precision_data):
    idata_copy = mvn_precision_data.copy()
    prec_orig = idata_copy.posterior["prec"]
    new_prec_vals = prec_orig.values[:, :, :-1, :-1]
    idata_copy.posterior["prec"] = xr.DataArray(
        new_prec_vals,
        dims=prec_orig.dims,
        coords={
            "chain": prec_orig.coords["chain"],
            "draw": prec_orig.coords["draw"],
            prec_orig.dims[-2]: prec_orig.coords[prec_orig.dims[-2]][:-1],
            prec_orig.dims[-2]: prec_orig.coords[prec_orig.dims[-2]][:-1],
            prec_orig.dims[-1]: prec_orig.coords[prec_orig.dims[-1]][:-1],
        },
    )
    loo_results = loo_nonfactor(idata_copy, var_name="y", prec_var_name="prec")
    assert isinstance(loo_results, ELPDData)


def test_loo_nonfactor_student_t_missing_df(mvn_inference_data):
    with pytest.warns(UserWarning, match="Degrees of freedom variable 'df' not found"):
        with pytest.raises(
            ValueError, match="Degrees of freedom variable 'df' not found"
        ):
            loo_nonfactor(mvn_inference_data, var_name="y", model_type="student_t")


def test_loo_nonfactor_student_t_df_shape_error(mvt_inference_data):
    idata_copy = mvt_inference_data.copy()
    df_orig = idata_copy.posterior["df"]
    idata_copy.posterior["df"] = xr.DataArray(
        df_orig.values[:, :, np.newaxis],
        dims=df_orig.dims + ("extra_dim",),
        coords={
            "chain": df_orig.coords["chain"],
            "draw": df_orig.coords["draw"],
            "extra_dim": [0],
        },
    )
    idata_copy = mvt_inference_data.copy()
    df_orig = idata_copy.posterior["df"]
    idata_copy.posterior["df"] = xr.DataArray(
        np.random.rand(df_orig.shape[0], df_orig.shape[1], 5),
        dims=("chain", "draw", "df_dim"),
        coords={
            "chain": df_orig.coords["chain"],
            "draw": df_orig.coords["draw"],
            "df_dim": np.arange(5),
        },
    )
    with pytest.raises(TypeError):
        loo_nonfactor(idata_copy, var_name="y", model_type="student_t")


def test_loo_nonfactor_no_observed_data():
    idata = InferenceData(
        posterior=xr.Dataset({"mu": (("chain", "draw"), np.random.randn(2, 10))})
    )
    with pytest.raises(
        TypeError, match="Must be able to extract an observed_data group"
    ):
        loo_nonfactor(idata)


def test_loo_nonfactor_no_posterior_data():
    idata = InferenceData(
        observed_data=xr.Dataset({"y": (("obs",), np.random.randn(5))})
    )
    with pytest.raises(TypeError, match="Must be able to extract a posterior group"):
        loo_nonfactor(idata)


def test_loo_nonfactor_manual_reff(mvn_inference_data):
    manual_reff = 0.5
    loo_results = loo_nonfactor(
        mvn_inference_data, var_name="y", reff=manual_reff, pointwise=True
    )
    assert isinstance(loo_results, ELPDData)
    assert "elpd_loo" in loo_results


@patch("pyloo.loo_nonfactor.ess")
def test_loo_nonfactor_low_ess_warning(mock_ess, mvn_inference_data):
    n_samples = (
        mvn_inference_data.posterior.dims["draw"]
        * mvn_inference_data.posterior.dims["chain"]
    )
    mock_ess.return_value = xr.Dataset({"mu": (("param_dim",), [n_samples * 0.05])})

    with pytest.warns(UserWarning, match="Low effective sample size detected"):
        loo_results = loo_nonfactor(
            mvn_inference_data, var_name="y", method="sis", pointwise=True
        )
    assert isinstance(loo_results, ELPDData)
    assert loo_results.warning is True
    assert "ess" in loo_results

    with pytest.warns(UserWarning, match="Low effective sample size detected"):
        loo_results_tis = loo_nonfactor(
            mvn_inference_data, var_name="y", method="tis", pointwise=True
        )
    assert isinstance(loo_results_tis, ELPDData)
    assert loo_results_tis.warning is True
    assert "ess" in loo_results_tis


@patch("pyloo.loo_nonfactor.compute_importance_weights")
def test_loo_nonfactor_high_pareto_k_warning(mock_compute_weights, mvn_inference_data):
    obs_var = "y"
    obs_dim = mvn_inference_data.observed_data[obs_var].dims[0]
    n_obs = mvn_inference_data.observed_data[obs_var].shape[0]
    n_samples = (
        mvn_inference_data.posterior.sizes["draw"]
        * mvn_inference_data.posterior.sizes["chain"]
    )

    high_k_values = np.full(n_obs, 0.8)
    dummy_log_weights = np.random.randn(n_obs, n_samples)
    mock_compute_weights.return_value = (
        xr.DataArray(dummy_log_weights, dims=(obs_dim, "__sample__")),
        xr.DataArray(high_k_values, dims=obs_dim, name="pareto_k"),
    )

    with pytest.warns(
        UserWarning, match="Estimated shape parameter of Pareto distribution is greater"
    ):
        loo_results = loo_nonfactor(
            mvn_inference_data, var_name=obs_var, method="psis", pointwise=True
        )

    assert isinstance(loo_results, ELPDData)
    assert loo_results.warning is True
    assert "pareto_k" in loo_results
    assert np.all(loo_results.pareto_k > 0.7)


def test_loo_nonfactor_nan_inf_loglik_warning(mvn_inference_data):
    idata_copy = mvn_inference_data.copy()
    cov_vals = idata_copy.posterior["cov"].values
    cov_vals[0, 0, :, :] = np.zeros_like(cov_vals[0, 0, :, :])
    idata_copy.posterior["cov"] = xr.DataArray(
        cov_vals,
        dims=idata_copy.posterior["cov"].dims,
        coords=idata_copy.posterior["cov"].coords,
    )

    with pytest.warns(
        UserWarning, match="Invalid values detected in log-likelihood calculation"
    ):
        loo_results = loo_nonfactor(idata_copy, var_name="y", pointwise=True)

    assert isinstance(loo_results, ELPDData)
    assert not np.isnan(loo_results.elpd_loo)
    assert "loo_i" in loo_results
    assert np.any(np.isnan(loo_results.loo_i.values))


def test_loo_nonfactor_student_t_beta_nan_warning(mvt_inference_data):
    idata_copy = mvt_inference_data.copy()
    prec_vals = np.linalg.inv(idata_copy.posterior["cov"].values)
    prec_vals[0, 0, :, :] = 0
    idata_copy.posterior["prec"] = xr.DataArray(
        prec_vals,
        dims=idata_copy.posterior["cov"].dims,
        coords=idata_copy.posterior["cov"].coords,
    )
    del idata_copy.posterior["cov"]

    with pytest.warns(
        UserWarning, match="Invalid values detected in log-likelihood calculation"
    ):
        loo_results = loo_nonfactor(
            idata_copy,
            var_name="y",
            model_type="student_t",
            prec_var_name="prec",
            pointwise=True,
        )

    assert isinstance(loo_results, ELPDData)
    assert not np.isnan(loo_results.elpd_loo)
    assert np.any(np.isnan(loo_results.loo_i.values))
