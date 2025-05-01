"""Tests for exact refitting in LOO-CV."""

import numpy as np
import pymc as pm
import pytest

from ...elpd import ELPDData
from ...loo import loo
from ...reloo import reloo
from ...wrapper.pymc.pymc import PyMCWrapper
from ..helpers import assert_arrays_allclose


def test_reloo_validates_required_methods():
    class NonPyMCModel:
        pass

    class WrapperWithInvalidModel:
        def __init__(self):
            self.model = NonPyMCModel()

        def check_implemented_methods(self, methods):
            return []

    wrapper = WrapperWithInvalidModel()
    with pytest.raises(TypeError, match=".*PyMC model.*"):
        reloo(wrapper)

    class IncompleteWrapper:
        def __init__(self):
            self.model = pm.Model()

        def check_implemented_methods(self, methods):
            return ["select_observations", "log_likelihood_i"]

    wrapper = IncompleteWrapper()
    with pytest.raises(TypeError, match=".*select_observations.*log_likelihood_i.*"):
        reloo(wrapper)


def test_reloo_no_problematic_observations(simple_model):
    model, idata = simple_model
    wrapper = PyMCWrapper(model, idata)

    loo_result = loo(idata, pointwise=True)
    loo_result.pareto_k = np.full_like(loo_result.pareto_k, 0.1)

    result = reloo(wrapper, loo_result, k_thresh=0.7)
    assert result is loo_result


def test_reloo_with_problematic_observations(simple_model):
    model, idata = simple_model
    wrapper = PyMCWrapper(model, idata)

    loo_result = loo(idata, pointwise=True)
    loo_result.pareto_k = np.array([0.1, 0.8, 0.3])

    result = reloo(wrapper, loo_result, k_thresh=0.7)

    assert isinstance(result, ELPDData)
    assert result is not loo_result
    assert result.pareto_k[1] == 0


@pytest.mark.parametrize("scale", ["log", "negative_log", "deviance"])
def test_reloo_different_scales(simple_model, scale):
    model, idata = simple_model
    wrapper = PyMCWrapper(model, idata)

    loo_result = loo(idata, pointwise=True, scale=scale)
    loo_result.pareto_k = np.array([0.1, 0.8, 0.3])

    result = reloo(wrapper, loo_result, k_thresh=0.7)
    assert result.scale == scale


def test_reloo_verbose_logging(simple_model, caplog):
    model, idata = simple_model
    wrapper = PyMCWrapper(model, idata)

    loo_result = loo(idata, pointwise=True)
    loo_result.pareto_k = np.array([0.1, 0.8, 0.3])

    reloo(wrapper, loo_result, k_thresh=0.7, verbose=True)


def test_reloo_without_original_loo(simple_model):
    model, idata = simple_model
    wrapper = PyMCWrapper(model, idata)

    result = reloo(wrapper)
    assert isinstance(result, ELPDData)


@pytest.mark.parametrize("k_thresh", [0.5, 0.7, 0.9])
def test_reloo_different_thresholds(simple_model, k_thresh):
    model, idata = simple_model
    wrapper = PyMCWrapper(model, idata)

    loo_result = loo(idata, pointwise=True)
    loo_result.pareto_k = np.array([0.6, 0.7, 0.8])

    result = reloo(wrapper, loo_result, k_thresh=k_thresh)
    n_refits = sum(k > k_thresh for k in [0.6, 0.7, 0.8])
    assert sum(result.pareto_k == 0) == n_refits


def test_reloo_data_restoration(simple_model):
    model, idata = simple_model
    wrapper = PyMCWrapper(model, idata)

    original_data = wrapper.get_observed_data()
    original_shape = original_data.shape

    loo_result = loo(idata, pointwise=True)
    loo_result.pareto_k = np.array([0.1, 0.8, 0.3])

    reloo(wrapper, loo_result, k_thresh=0.7)

    current_data = wrapper.get_observed_data()
    assert (
        current_data.shape == original_shape
    ), f"Data shape changed from {original_shape} to {current_data.shape}"
    assert_arrays_allclose(current_data, original_data)


def test_reloo_with_missing_values(simple_model, caplog):
    model, idata = simple_model
    wrapper = PyMCWrapper(model, idata)

    data = wrapper.get_observed_data()
    data[1] = np.nan
    wrapper.set_data({wrapper.get_observed_name(): data})

    loo_result = loo(idata, pointwise=True)
    loo_result.pareto_k = np.array([0.1, 0.8, 0.3])

    # Test that it runs without error with missing values
    result = reloo(wrapper, loo_result, k_thresh=0.7)
    assert isinstance(result, ELPDData)


def test_reloo_sequential_refits(simple_model):
    model, idata = simple_model
    wrapper = PyMCWrapper(model, idata)

    loo_result = loo(idata, pointwise=True)
    loo_result.pareto_k = np.array([0.8, 0.9, 0.3])

    result = reloo(wrapper, loo_result, k_thresh=0.7)

    assert isinstance(result, ELPDData)
    assert all(k == 0 for k in result.pareto_k[:2])
    assert result.pareto_k[2] == 0.3


def test_reloo_sequential_refits_hierarchical(hierarchical_model):
    model, idata = hierarchical_model
    wrapper = PyMCWrapper(model, idata)

    loo_result = loo(idata, pointwise=True)
    loo_result.pareto_k = np.full_like(loo_result.pareto_k, 0.3)
    loo_result.pareto_k = np.array([0.8, 0.3, 0.3, 0.9, 0.3, 0.3, 0.85, 0.3])

    result = reloo(wrapper, loo_result, k_thresh=0.7)

    assert isinstance(result, ELPDData)
    assert result.pareto_k[0] == 0
    assert result.pareto_k[3] == 0
    assert result.pareto_k[6] == 0
    assert all(result.pareto_k[i] == 0.3 for i in [1, 2, 4, 5, 7])


def test_reloo_sequential_refits_poisson(poisson_model):
    model, idata = poisson_model
    wrapper = PyMCWrapper(model, idata)

    loo_result = loo(idata, pointwise=True)
    loo_result.pareto_k = np.full_like(loo_result.pareto_k, 0.3)
    loo_result.pareto_k[0] = 0.8
    loo_result.pareto_k[10] = 0.85
    loo_result.pareto_k[20] = 0.9

    result = reloo(wrapper, loo_result, k_thresh=0.7)

    assert isinstance(result, ELPDData)
    assert result.pareto_k[0] == 0
    assert result.pareto_k[10] == 0
    assert result.pareto_k[20] == 0
    assert all(k == 0.3 for k in result.pareto_k[1:10])
    assert all(k == 0.3 for k in result.pareto_k[11:20])
    assert all(k == 0.3 for k in result.pareto_k[21:])


def test_reloo_sequential_refits_multi_observed(multi_observed_model):
    model, idata = multi_observed_model
    wrapper = PyMCWrapper(model, idata)

    loo_result_y1 = loo(idata, pointwise=True, var_name="y1")
    loo_result_y1.pareto_k = np.full_like(loo_result_y1.pareto_k, 0.3)
    loo_result_y1.pareto_k[0] = 0.8
    loo_result_y1.pareto_k[10] = 0.85

    result_y1 = reloo(wrapper, loo_result_y1, k_thresh=0.7)

    assert isinstance(result_y1, ELPDData)
    assert result_y1.pareto_k[0] == 0
    assert result_y1.pareto_k[10] == 0
    assert all(k == 0.3 for k in result_y1.pareto_k[1:10])
    assert all(k == 0.3 for k in result_y1.pareto_k[11:])

    loo_result_y2 = loo(idata, pointwise=True, var_name="y2")
    loo_result_y2.pareto_k = np.full_like(loo_result_y2.pareto_k, 0.3)
    loo_result_y2.pareto_k[5] = 0.8
    loo_result_y2.pareto_k[15] = 0.85

    result_y2 = reloo(wrapper, loo_result_y2, k_thresh=0.7)

    assert isinstance(result_y2, ELPDData)
    assert result_y2.pareto_k[5] == 0
    assert result_y2.pareto_k[15] == 0
    assert all(k == 0.3 for k in result_y2.pareto_k[:5])
    assert all(k == 0.3 for k in result_y2.pareto_k[6:15])
    assert all(k == 0.3 for k in result_y2.pareto_k[16:])


@pytest.mark.integration
def test_reloo_hierarchical_model(hierarchical_model):
    model, idata = hierarchical_model
    wrapper = PyMCWrapper(model, idata)

    loo_result = loo(idata, pointwise=True)
    result = reloo(wrapper, loo_result, k_thresh=0.7)

    assert isinstance(result, ELPDData)
    assert hasattr(result, "elpd_loo")
    assert hasattr(result, "pareto_k")


@pytest.mark.integration
def test_reloo_poisson_model(poisson_model):
    model, idata = poisson_model
    wrapper = PyMCWrapper(model, idata)

    loo_result = loo(idata, pointwise=True)
    result = reloo(wrapper, loo_result, k_thresh=0.7)

    assert isinstance(result, ELPDData)
    assert hasattr(result, "elpd_loo")
    assert hasattr(result, "pareto_k")


@pytest.mark.integration
def test_reloo_multi_observed_model(multi_observed_model):
    model, idata = multi_observed_model
    wrapper = PyMCWrapper(model, idata)

    loo_result = loo(idata, pointwise=True, var_name="y1")
    result = reloo(wrapper, loo_result, k_thresh=0.7)

    assert isinstance(result, ELPDData)
    assert hasattr(result, "elpd_loo")
    assert hasattr(result, "pareto_k")


def test_reloo_with_subsampling(simple_model):
    model, idata = simple_model
    wrapper = PyMCWrapper(model, idata)

    result = reloo(
        wrapper,
        k_thresh=0.7,
        use_subsample=True,
        subsample_observations=2,
    )

    assert isinstance(result, ELPDData)
    assert hasattr(result, "elpd_loo")
    assert hasattr(result, "pareto_k")
    assert len(result.loo_i) == len(wrapper.get_observed_data())


@pytest.mark.parametrize(
    "approximation,estimator",
    [
        ("plpd", "diff_srs"),
        ("lpd", "srs"),
        ("tis", "hh_pps"),
        ("sis", "diff_srs"),
    ],
)
def test_reloo_subsample_parameters(simple_model, approximation, estimator):
    model, idata = simple_model
    wrapper = PyMCWrapper(model, idata)

    result = reloo(
        wrapper,
        k_thresh=0.7,
        use_subsample=True,
        subsample_observations=2,
        subsample_approximation=approximation,
        subsample_estimator=estimator,
    )

    assert isinstance(result, ELPDData)
    assert hasattr(result, "elpd_loo")


def test_reloo_subsample_specific_indices(simple_model):
    model, idata = simple_model
    wrapper = PyMCWrapper(model, idata)

    indices = np.array([0, 2])
    loo_result = loo(idata, pointwise=True)
    loo_result.pareto_k = np.array([0.8, 0.3, 0.3])

    result = reloo(
        wrapper,
        loo_orig=loo_result,
        k_thresh=0.7,
        use_subsample=True,
        subsample_observations=indices,
    )

    assert isinstance(result, ELPDData)
    assert result.pareto_k[0] == 0
    assert len(result.loo_i) == len(wrapper.get_observed_data())


@pytest.mark.integration
def test_reloo_subsample_hierarchical(hierarchical_model):
    model, idata = hierarchical_model
    wrapper = PyMCWrapper(model, idata)

    for observations in [4, np.array([0, 2, 4, 6])]:
        result = reloo(
            wrapper,
            k_thresh=0.7,
            use_subsample=True,
            subsample_observations=observations,
        )

        assert isinstance(result, ELPDData)
        assert hasattr(result, "elpd_loo")
        assert hasattr(result, "pareto_k")
        assert len(result.loo_i) == len(wrapper.get_observed_data())


def test_reloo_with_problematic_k(problematic_k_model):
    model, idata = problematic_k_model
    wrapper = PyMCWrapper(model, idata)

    loo_result = loo(idata, pointwise=True)

    result = reloo(wrapper, loo_result, k_thresh=0.7)

    assert np.all(result.pareto_k <= loo_result.pareto_k)
    assert np.all(result.elpd_loo > loo_result.elpd_loo)
