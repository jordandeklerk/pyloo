"""Tests for moment matching in LOO-CV."""

from copy import deepcopy

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from ...loo import loo
from ...loo_moment_match import (
    loo_moment_match,
    loo_moment_match_split,
    shift,
    shift_and_cov,
    shift_and_scale,
    update_quantities_i,
)
from ...moment_match_helpers import (
    ParameterConverter,
    _initialize_array,
    log_lik_i_upars,
    log_prob_upars,
)
from ...wrapper.pymc_wrapper import PyMCWrapper


@pytest.fixture
def problematic_model(problematic_k_model):
    """Create a model with problematic Pareto k values."""
    model, idata = problematic_k_model
    wrapper = PyMCWrapper(model, idata)
    return wrapper


def test_loo_moment_match_basic(problematic_model):
    """Test basic functionality of loo_moment_match."""
    param_names = list(problematic_model.get_unconstrained_parameters().keys())
    loo_orig = loo(problematic_model.idata, pointwise=True)
    original_k_values = loo_orig.pareto_k.values.copy()

    high_k_indices = np.where(original_k_values > 0.7)[0]
    assert len(high_k_indices) > 0, "Test requires observations with high Pareto k"

    unconstrained = problematic_model.get_unconstrained_parameters()
    param_arrays = []
    for name in param_names:
        param = unconstrained[name].values.flatten()
        param_arrays.append(param)

    loo_orig_copy = deepcopy(loo_orig)

    loo_mm = loo_moment_match(
        problematic_model,
        loo_orig_copy,
        max_iters=30,
        k_threshold=0.7,
        split=True,
        cov=True,
    )

    improvements = []
    for idx in high_k_indices:
        orig_k = original_k_values[idx]
        mm_k = loo_mm.pareto_k[idx]
        improvement = orig_k - mm_k
        improvements.append(improvement)

    assert np.any(np.array(improvements) >= 0), "No Pareto k values improved"


def test_loo_moment_match_split(problematic_model):
    """Test split moment matching."""
    loo_orig = loo(problematic_model.idata, pointwise=True)
    original_elpd = loo_orig.elpd_loo

    import copy

    loo_orig_copy1 = copy.deepcopy(loo_orig)
    loo_orig_copy2 = copy.deepcopy(loo_orig)

    loo_mm_split = loo_moment_match(
        problematic_model,
        loo_orig_copy1,
        max_iters=10,
        k_threshold=0.7,
        split=True,
        cov=True,
    )

    loo_mm_regular = loo_moment_match(
        problematic_model,
        loo_orig_copy2,
        max_iters=10,
        k_threshold=0.7,
        split=False,
        cov=True,
    )

    assert loo_mm_split.elpd_loo >= original_elpd - 1e-10
    assert loo_mm_regular.elpd_loo >= original_elpd - 1e-10

    if loo_mm_split.elpd_loo != loo_mm_regular.elpd_loo:
        rel_diff = abs(loo_mm_split.elpd_loo - loo_mm_regular.elpd_loo) / abs(
            loo_mm_regular.elpd_loo
        )
        assert rel_diff < 0.1


def test_loo_moment_match_different_methods(problematic_model):
    """Test moment matching with different importance sampling methods."""
    loo_orig = loo(problematic_model.idata, pointwise=True)
    original_elpd = loo_orig.elpd_loo

    methods = ["psis", "sis", "tis"]
    results = {}

    import copy

    for method in methods:
        loo_orig_copy = copy.deepcopy(loo_orig)

        results[method] = loo_moment_match(
            problematic_model,
            loo_orig_copy,
            max_iters=10,
            k_threshold=0.7,
            split=True,
            cov=True,
            method=method,
        )

        assert results[method].elpd_loo >= original_elpd - 1e-10

    for m1 in methods:
        for m2 in methods:
            if m1 != m2 and results[m1].elpd_loo != results[m2].elpd_loo:
                rel_diff = abs(results[m1].elpd_loo - results[m2].elpd_loo) / abs(
                    results[m1].elpd_loo
                )
                assert rel_diff < 0.2


def test_loo_moment_match_iterations(problematic_model):
    """Test moment matching with different iteration counts."""
    loo_orig = loo(problematic_model.idata, pointwise=True)

    iters = [1, 5, 15]
    results = {}

    for iter_count in iters:
        results[iter_count] = loo_moment_match(
            problematic_model,
            loo_orig,
            max_iters=iter_count,
            k_threshold=0.7,
            split=False,
            cov=True,
        )

        assert results[iter_count].elpd_loo >= loo_orig.elpd_loo - 1e-10

    assert results[5].elpd_loo >= results[1].elpd_loo - 1e-10
    assert results[15].elpd_loo >= results[5].elpd_loo - 1e-10


def test_shift_transformation():
    """Test the shift transformation."""
    rng = np.random.default_rng(42)
    upars = rng.normal(size=(100, 5))
    lwi = rng.normal(size=100)
    lwi = lwi - np.max(lwi)

    result = shift(upars, lwi)

    assert "upars" in result
    assert "shift" in result
    assert result["upars"].shape == upars.shape
    assert result["shift"].shape == (5,)

    weighted_mean = np.sum(np.exp(lwi)[:, None] * upars, axis=0)
    original_mean = np.mean(upars, axis=0)
    expected_shift = weighted_mean - original_mean

    assert_allclose(result["shift"], expected_shift)
    assert_allclose(np.mean(result["upars"], axis=0), weighted_mean)


def test_shift_and_scale_transformation():
    """Test the shift and scale transformation."""
    rng = np.random.default_rng(42)
    upars = rng.normal(size=(100, 5))
    lwi = rng.normal(size=100)
    lwi = lwi - np.max(lwi)

    result = shift_and_scale(upars, lwi)

    assert "upars" in result
    assert "shift" in result
    assert "scaling" in result
    assert result["upars"].shape == upars.shape
    assert result["shift"].shape == (5,)
    assert result["scaling"].shape == (5,)

    weighted_mean = np.sum(np.exp(lwi)[:, None] * upars, axis=0)
    assert_allclose(np.mean(result["upars"], axis=0), weighted_mean)

    S = upars.shape[0]
    weighted_var = np.sum(np.exp(lwi)[:, None] * upars**2, axis=0) - weighted_mean**2
    weighted_var = weighted_var * S / (S - 1)
    original_var = np.var(upars, axis=0)
    expected_scaling = np.sqrt(weighted_var / original_var)

    assert_allclose(result["scaling"], expected_scaling)


def test_shift_and_cov_transformation():
    """Test the shift and covariance transformation."""
    rng = np.random.default_rng(42)
    upars = rng.normal(size=(100, 5))
    lwi = rng.normal(size=100)
    lwi = lwi - np.max(lwi)

    result = shift_and_cov(upars, lwi)

    assert "upars" in result
    assert "shift" in result
    assert "mapping" in result
    assert result["upars"].shape == upars.shape
    assert result["shift"].shape == (5,)
    assert result["mapping"].shape == (5, 5)

    weighted_mean = np.sum(np.exp(lwi)[:, None] * upars, axis=0)
    assert_allclose(np.mean(result["upars"], axis=0), weighted_mean)

    original_cov = np.cov(upars, rowvar=False)
    weighted_cov = np.cov(upars, rowvar=False, aweights=np.exp(lwi))
    transformed_cov = np.cov(result["upars"], rowvar=False)

    diff_orig = np.linalg.norm(original_cov - weighted_cov)
    diff_trans = np.linalg.norm(transformed_cov - weighted_cov)
    assert diff_trans < diff_orig


def test_update_quantities_i(problematic_model):
    """Test the update_quantities_i function."""
    unconstrained = problematic_model.get_unconstrained_parameters()
    param_names = list(unconstrained.keys())
    param_converter = ParameterConverter(problematic_model)
    param_arrays = []
    for name in param_names:
        param = unconstrained[name].values.flatten()
        param_arrays.append(param)

    min_size = min(len(arr) for arr in param_arrays)
    upars = np.column_stack([arr[:min_size] for arr in param_arrays])

    orig_log_prob = np.zeros(upars.shape[0])
    for name, param in unconstrained.items():
        var = problematic_model.get_variable(name)
        if var is not None and hasattr(var, "logp"):
            if isinstance(param, xr.DataArray):
                param = param.values
            log_prob_part = var.logp(param).eval()
            orig_log_prob += log_prob_part

    i = 0
    r_eff_i = 0.9

    result = update_quantities_i(
        problematic_model, upars, i, orig_log_prob, r_eff_i, converter=param_converter
    )

    assert "lwi" in result
    assert "lwfi" in result
    assert "ki" in result
    assert "kfi" in result
    assert "log_liki" in result

    assert result["lwi"].shape == (upars.shape[0],)
    assert result["lwfi"].shape == (upars.shape[0],)
    assert np.isscalar(result["ki"]) or isinstance(result["ki"], np.ndarray)
    assert isinstance(result["kfi"], (int, float, np.ndarray))
    assert result["log_liki"].shape == (upars.shape[0],)

    assert_allclose(np.exp(result["lwi"]).sum(), 1.0, rtol=1e-6)


def test_loo_moment_match_split_function(problematic_model):
    """Test the loo_moment_match_split function directly."""
    unconstrained = problematic_model.get_unconstrained_parameters()
    param_names = list(unconstrained.keys())

    param_arrays = []
    for name in param_names:
        param = unconstrained[name].values.flatten()
        param_arrays.append(param)

    min_size = min(len(arr) for arr in param_arrays)
    upars = np.column_stack([arr[:min_size] for arr in param_arrays])

    i = 0
    r_eff_i = 0.9

    dim = upars.shape[1]
    total_shift = np.random.normal(size=dim) * 0.1
    total_scaling = np.ones(dim) + np.random.normal(size=dim) * 0.05
    total_mapping = np.eye(dim) + np.random.normal(size=(dim, dim)) * 0.01

    result = loo_moment_match_split(
        problematic_model,
        upars,
        True,
        total_shift,
        total_scaling,
        total_mapping,
        i,
        r_eff_i,
    )

    assert "lwi" in result
    assert "lwfi" in result
    assert "log_liki" in result
    assert "r_eff_i" in result

    assert result["lwi"].shape == (upars.shape[0],)
    assert result["lwfi"].shape == (upars.shape[0],)
    assert result["log_liki"].shape == (upars.shape[0],)
    assert isinstance(result["r_eff_i"], float)

    assert_allclose(np.exp(result["lwi"]).sum(), 1.0, rtol=1e-6)


def test_initialize_array():
    """Test the _initialize_array function."""
    arr = np.ones(5)
    dim = 5
    result = _initialize_array(arr, np.zeros, dim)
    assert_allclose(result, arr)

    arr = np.ones(3)
    dim = 5
    result = _initialize_array(arr, np.zeros, dim)
    assert_allclose(result, np.zeros(dim))

    arr = np.eye(3)
    dim = 5
    result = _initialize_array(arr, np.eye, dim)
    assert_allclose(result, np.eye(dim))


def test_loo_moment_match_with_custom_threshold(problematic_model):
    """Test moment matching with custom k threshold."""
    loo_orig = loo(problematic_model.idata, pointwise=True)

    thresholds = [0.5, 0.7, 0.9]
    results = {}

    for threshold in thresholds:
        results[threshold] = loo_moment_match(
            problematic_model,
            loo_orig,
            max_iters=10,
            k_threshold=threshold,
            split=False,
            cov=True,
        )

    n_improved_05 = np.sum(results[0.5].pareto_k < loo_orig.pareto_k)
    n_improved_07 = np.sum(results[0.7].pareto_k < loo_orig.pareto_k)
    n_improved_09 = np.sum(results[0.9].pareto_k < loo_orig.pareto_k)

    assert n_improved_05 >= n_improved_07 >= n_improved_09


def test_loo_moment_match_roaches_model(roaches_model):
    """Test moment matching with the roaches model."""
    model, idata = roaches_model
    wrapper = PyMCWrapper(model, idata)

    loo_orig = loo(idata, pointwise=True)

    k_threshold = 0.7
    high_k = np.where(loo_orig.pareto_k > k_threshold)[0]

    loo_mm = loo_moment_match(
        wrapper,
        loo_orig,
        max_iters=30,
        k_threshold=k_threshold,
        split=True,
        cov=True,
    )

    print(loo_orig)
    print(loo_mm)

    improvements = loo_orig.pareto_k[high_k] - loo_mm.pareto_k[high_k]
    assert np.any(improvements > 0), "No Pareto k values improved"


def test_parameter_converter_initialization(mmm_model):
    """Test initialization of ParameterConverter."""
    model, idata = mmm_model
    wrapper = PyMCWrapper(model, idata)
    converter = ParameterConverter(wrapper)

    assert len(converter.param_names) > 0
    assert converter.total_size > 0

    unconstrained = wrapper.get_unconstrained_parameters()
    for name in unconstrained:
        info = converter.get_param_info(name)
        assert info.name == name
        assert info.flattened_size > 0
        assert info.end_idx > info.start_idx


def test_dict_to_matrix_conversion(mmm_model):
    """Test conversion from dictionary to matrix format."""
    model, idata = mmm_model
    wrapper = PyMCWrapper(model, idata)
    converter = ParameterConverter(wrapper)

    unconstrained = wrapper.get_unconstrained_parameters()

    matrix = converter.dict_to_matrix(unconstrained)

    n_samples = (
        unconstrained[converter.param_names[0]].shape[0]
        * unconstrained[converter.param_names[0]].shape[1]
    )
    assert matrix.shape == (n_samples, converter.total_size)

    for name in converter.param_names:
        info = converter.get_param_info(name)
        param_values = unconstrained[name].values
        if param_values.ndim > 2:
            param_values = param_values.reshape(
                param_values.shape[0], param_values.shape[1], -1
            )
        param_values = param_values.reshape(-1, info.flattened_size)
        np.testing.assert_array_equal(
            matrix[:, info.start_idx : info.end_idx], param_values
        )


def test_matrix_to_dict_conversion(mmm_model):
    """Test conversion from matrix to dictionary format."""
    model, idata = mmm_model
    wrapper = PyMCWrapper(model, idata)
    converter = ParameterConverter(wrapper)

    unconstrained = wrapper.get_unconstrained_parameters()
    matrix = converter.dict_to_matrix(unconstrained)

    result = converter.matrix_to_dict(matrix)

    assert set(result.keys()) == set(unconstrained.keys())

    for name in unconstrained:
        assert result[name].shape == unconstrained[name].shape
        np.testing.assert_array_equal(result[name].values, unconstrained[name].values)


def test_multidimensional_parameters(mmm_model):
    """Test handling of multi-dimensional parameters."""
    model, idata = mmm_model
    wrapper = PyMCWrapper(model, idata)
    converter = ParameterConverter(wrapper)

    unconstrained = wrapper.get_unconstrained_parameters()

    multidim_params = {
        name: param for name, param in unconstrained.items() if param.values.ndim > 2
    }

    if not multidim_params:
        pytest.skip("No multi-dimensional parameters found in test model")

    for name, param in multidim_params.items():
        single_param = {name: param}

        matrix = converter.dict_to_matrix(single_param)
        result = converter.matrix_to_dict(matrix)

        assert result[name].shape == param.shape
        np.testing.assert_array_equal(result[name].values, param.values)


def test_parameter_info_consistency(mmm_model):
    """Test consistency of parameter info with actual data."""
    model, idata = mmm_model
    wrapper = PyMCWrapper(model, idata)
    converter = ParameterConverter(wrapper)

    unconstrained = wrapper.get_unconstrained_parameters()

    total_size = 0
    for name in converter.param_names:
        info = converter.get_param_info(name)
        param = unconstrained[name]

        expected_size = np.prod(
            [size for dim, size in param.sizes.items() if dim not in ("chain", "draw")]
        ).astype(int)
        assert info.flattened_size == expected_size

        assert info.start_idx == total_size
        assert info.end_idx == total_size + info.flattened_size
        total_size += info.flattened_size

    assert converter.total_size == total_size


def test_roundtrip_conversion(mmm_model):
    """Test that converting to matrix and back preserves all information."""
    model, idata = mmm_model
    wrapper = PyMCWrapper(model, idata)
    converter = ParameterConverter(wrapper)

    original = wrapper.get_unconstrained_parameters()

    matrix = converter.dict_to_matrix(original)
    result = converter.matrix_to_dict(matrix)

    for name in original:
        xr.testing.assert_equal(result[name], original[name])


def test_error_handling(mmm_model):
    """Test error handling in ParameterConverter."""
    model, idata = mmm_model
    wrapper = PyMCWrapper(model, idata)
    converter = ParameterConverter(wrapper)

    with pytest.raises(KeyError):
        converter.get_param_info("nonexistent_parameter")

    invalid_matrix = np.random.randn(100, converter.total_size + 1)
    with pytest.raises(ValueError):
        converter.matrix_to_dict(invalid_matrix)

    unconstrained = wrapper.get_unconstrained_parameters()
    invalid_dict = {
        name: unconstrained[name] for name in list(unconstrained.keys())[:-1]
    }
    matrix = converter.dict_to_matrix(invalid_dict)
    assert matrix.shape[1] == converter.total_size


def test_parameter_ordering(mmm_model):
    """Test that parameter ordering is consistent."""
    model, idata = mmm_model
    wrapper = PyMCWrapper(model, idata)
    converter = ParameterConverter(wrapper)

    unconstrained = wrapper.get_unconstrained_parameters()

    matrix = converter.dict_to_matrix(unconstrained)

    for name in converter.param_names:
        info = converter.get_param_info(name)
        param_values = unconstrained[name].values
        if param_values.ndim > 2:
            param_values = param_values.reshape(
                param_values.shape[0], param_values.shape[1], -1
            )
        param_values = param_values.reshape(-1, info.flattened_size)

        np.testing.assert_array_equal(
            matrix[:, info.start_idx : info.end_idx], param_values
        )


def test_converter_with_log_prob_upars(mmm_model):
    """Test that ParameterConverter works correctly with log_prob_upars."""
    model, idata = mmm_model
    wrapper = PyMCWrapper(model, idata)
    converter = ParameterConverter(wrapper)

    unconstrained = wrapper.get_unconstrained_parameters()
    original_log_prob = log_prob_upars(wrapper, unconstrained)

    matrix = converter.dict_to_matrix(unconstrained)
    converted = converter.matrix_to_dict(matrix)

    converted_log_prob = log_prob_upars(wrapper, converted)

    np.testing.assert_allclose(original_log_prob, converted_log_prob)


def test_converter_with_log_lik_i_upars(mmm_model):
    """Test that ParameterConverter works correctly with log_lik_i_upars."""
    model, idata = mmm_model
    wrapper = PyMCWrapper(model, idata)
    converter = ParameterConverter(wrapper)

    unconstrained = wrapper.get_unconstrained_parameters()
    original_log_lik = log_lik_i_upars(wrapper, unconstrained, pointwise=True)

    matrix = converter.dict_to_matrix(unconstrained)
    converted = converter.matrix_to_dict(matrix)

    converted_log_lik = log_lik_i_upars(wrapper, converted, pointwise=True)

    xr.testing.assert_allclose(original_log_lik, converted_log_lik)


def test_converter_with_multidim_log_lik(mmm_model):
    """Test ParameterConverter with multi-dimensional log likelihood."""
    model, idata = mmm_model
    wrapper = PyMCWrapper(model, idata)
    converter = ParameterConverter(wrapper)

    unconstrained = wrapper.get_unconstrained_parameters()
    original_log_lik = log_lik_i_upars(wrapper, unconstrained, pointwise=False)

    matrix = converter.dict_to_matrix(unconstrained)
    converted = converter.matrix_to_dict(matrix)

    converted_log_lik = log_lik_i_upars(wrapper, converted, pointwise=False)

    assert set(original_log_lik.log_likelihood.dims) == set(
        converted_log_lik.log_likelihood.dims
    )

    for group in original_log_lik.log_likelihood.data_vars:
        orig_ll = original_log_lik.log_likelihood[group]
        conv_ll = converted_log_lik.log_likelihood[group]

        assert set(orig_ll.dims) == set(conv_ll.dims)

        xr.testing.assert_allclose(orig_ll, conv_ll)
