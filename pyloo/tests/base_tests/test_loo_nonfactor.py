"""Tests for non-factorized multivariate models."""

import logging

import numpy as np
import pymc as pm
import pytest

from pyloo import loo_nonfactor
from pyloo.elpd import ELPDData


def test_loo_nonfactor_basic(mvn_inference_data):
    """Test basic functionality of loo_nonfactor."""
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
    """Test basic functionality of loo_nonfactor with Student-t model."""
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
    """Test loo_nonfactor using precision matrix input."""
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
    """Test loo_nonfactor with Student-t model using precision matrix input."""
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
    """Test the model structure verification helper function."""
    from pyloo.loo_nonfactor import _validate_mvn_structure

    valid_idata = mvn_validation_data["valid"]
    no_mu_idata = mvn_validation_data["no_mu"]
    no_cov_prec_idata = mvn_validation_data["no_cov_prec"]
    no_posterior_idata = mvn_validation_data["no_posterior"]

    assert _validate_mvn_structure(valid_idata, "mu", None, None) is True

    with pytest.warns(UserWarning, match="Mean vector .* not found"):
        assert (
            _validate_mvn_structure(no_mu_idata, "wrong_mu_name", None, None) is False
        )

    with pytest.warns(
        UserWarning, match="Neither covariance nor precision matrix found"
    ):
        assert (
            _validate_mvn_structure(no_cov_prec_idata, "mu", "wrong_cov", "wrong_prec")
            is False
        )

    assert _validate_mvn_structure(no_posterior_idata, "mu", None, None) is False


def test_verify_student_t_structure(mvt_validation_data):
    """Test validation of Student-t model structure."""
    from pyloo.loo_nonfactor import _validate_mvn_structure

    valid_idata = mvt_validation_data["valid"]
    missing_df_idata = mvt_validation_data["missing_df"]

    assert _validate_mvn_structure(valid_idata, "mu", None, None, "student_t") is True

    with pytest.warns(UserWarning, match="Degrees of freedom.*not found"):
        assert (
            _validate_mvn_structure(missing_df_idata, "mu", None, None, "student_t")
            is False
        )


def test_loo_nonfactor_warnings(missing_cov_data):
    """Test that appropriate warnings are raised for incorrect model structures."""
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
    """Test that loo_nonfactor works when both covariance and precision matrices are provided."""
    loo_results = loo_nonfactor(both_cov_prec_data, var_name="y")
    assert isinstance(loo_results, ELPDData)

    loo_results_prec = loo_nonfactor(
        both_cov_prec_data, var_name="y", prec_var_name="prec", cov_var_name=None
    )
    assert isinstance(loo_results_prec, ELPDData)


def test_loo_nonfactor_custom_names(mvn_custom_names_data):
    """Test loo_nonfactor with custom variable names."""
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
    """Test loo_nonfactor with Student-t model using custom variable names."""
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
    """Test loo_nonfactor with Student-t model including negative degrees of freedom."""
    with pytest.warns(UserWarning, match="Non-positive degrees of freedom"):
        loo_results = loo_nonfactor(
            mvt_negative_df_data, var_name="y", model_type="student_t", pointwise=True
        )
        assert isinstance(loo_results, ELPDData)
        assert not np.isnan(loo_results.elpd_loo)
        assert not np.isnan(loo_results.p_loo)


def test_loo_nonfactor_singular_matrices(singular_matrix_data):
    """Test loo_nonfactor handling of singular matrices."""
    with pytest.warns(UserWarning, match="Invalid values detected in log-likelihood"):
        loo_results = loo_nonfactor(singular_matrix_data, var_name="y")
        assert isinstance(loo_results, ELPDData)


@pytest.mark.skipif(pm is None, reason="PyMC not installed")
def test_loo_nonfactor_pymc_model(mvn_spatial_model):
    """Test loo_nonfactor with a realistic spatial model using a joint MVN likelihood."""
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

    logging.info(f"loo_nonfactor successful: elpd_loo={loo_results.elpd_loo}")
    logging.info(f"p_loo={loo_results.p_loo}")
    logging.info(f"p_loo_se={loo_results.p_loo_se}")

    logging.info(f"{loo_results}")


@pytest.mark.skipif(pm is None, reason="PyMC not installed")
def test_loo_nonfactor_student_t_pymc_model(mvt_spatial_model):
    """Test loo_nonfactor with a realistic spatial model using a multivariate Student-t likelihood."""
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

    logging.info(f"Student-t loo_nonfactor successful: elpd_loo={loo_results.elpd_loo}")
    logging.info(f"p_loo={loo_results.p_loo}")
    logging.info(f"p_loo_se={loo_results.p_loo_se}")

    logging.info(loo_results)
