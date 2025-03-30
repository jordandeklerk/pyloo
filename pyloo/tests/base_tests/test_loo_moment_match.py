"""Tests for moment matching in LOO-CV."""

import logging
from copy import deepcopy

import numpy as np
import pytest
import xarray as xr

from ...elpd import ELPDData
from ...helpers import (
    ParameterConverter,
    _initialize_array,
    extract_log_likelihood_for_observation,
    log_lik_i_upars,
    log_prob_upars,
)
from ...loo import loo
from ...loo_moment_match import (
    _validate_custom_function,
    _validate_output,
    loo_moment_match,
    shift,
    shift_and_cov,
    shift_and_scale,
    update_quantities_i,
)
from ...split_moment_matching import loo_moment_match_split
from ...wrapper.pymc import PyMCWrapper
from ..helpers import assert_allclose

logger = logging.getLogger(__name__)


class MockCmdStanModel:
    """A mock CmdStanPy model for testing CmdStanPy integration."""

    def __init__(self, n_samples=1000, n_obs=20, n_predictors=3, seed=42):
        """Initialize with simulated Poisson regression data."""
        self.rng = np.random.RandomState(seed)
        self.n_samples = n_samples
        self.n_obs = n_obs
        self.n_predictors = n_predictors

        self.x = self.rng.normal(0, 1, (n_obs, n_predictors))
        self.offset = np.zeros(n_obs)

        self.beta = self.rng.normal(0, 0.5, n_predictors)
        self.intercept = 0.5

        linear_pred = self.x @ self.beta + self.intercept + self.offset
        self.y = self.rng.poisson(np.exp(linear_pred))

        self.draws = {
            "beta": self.rng.normal(self.beta, 0.1, (n_samples, n_predictors)),
            "intercept": self.rng.normal(self.intercept, 0.1, n_samples),
        }

        self.log_lik = np.zeros((n_samples, n_obs))
        for s in range(n_samples):
            beta_s = self.draws["beta"][s]
            intercept_s = self.draws["intercept"][s]

            for i in range(n_obs):
                mu_i = np.exp(np.dot(self.x[i], beta_s) + intercept_s + self.offset[i])
                self.log_lik[s, i] = (
                    self.y[i] * np.log(mu_i)
                    - mu_i
                    - np.log(np.math.factorial(self.y[i]))
                )

    def stan_variables(self):
        """Return a dictionary of Stan variables."""
        return {
            "beta": self.draws["beta"].reshape(1, self.n_samples, self.n_predictors),
            "intercept": self.draws["intercept"].reshape(1, self.n_samples),
            "log_lik": self.log_lik.reshape(1, self.n_samples, self.n_obs),
        }


@pytest.fixture
def problematic_model(problematic_k_model):
    """Create a model with problematic Pareto k values."""
    model, idata = problematic_k_model
    wrapper = PyMCWrapper(model, idata)
    return wrapper


class CustomModel:
    """A simple custom model for testing."""

    def __init__(self, n_samples=1000, n_obs=20, seed=42):
        """Initialize the custom model with simulated data."""
        self.rng = np.random.RandomState(seed)
        self.n_samples = n_samples
        self.n_obs = n_obs

        self.alpha = 0.5
        self.beta = 1.2
        self.sigma = 0.5

        self.x = self.rng.normal(0, 1, n_obs)
        self.y = self.alpha + self.beta * self.x + self.rng.normal(0, self.sigma, n_obs)

        self.draws = {
            "alpha": self.rng.normal(0.5, 0.1, n_samples),
            "beta": self.rng.normal(1.2, 0.2, n_samples),
            "sigma": np.abs(self.rng.normal(0.5, 0.1, n_samples)),
        }

        self._compute_log_likelihood()

    def _compute_log_likelihood(self):
        """Compute log-likelihood values for all observations and draws."""
        n_samples = self.n_samples
        n_obs = self.n_obs

        self.log_likelihood = np.zeros((n_samples, n_obs))

        for s in range(n_samples):
            alpha = self.draws["alpha"][s]
            beta = self.draws["beta"][s]
            sigma = self.draws["sigma"][s]

            for i in range(n_obs):
                mean = alpha + beta * self.x[i]
                self.log_likelihood[s, i] = (
                    -0.5 * np.log(2 * np.pi * sigma**2)
                    - 0.5 * ((self.y[i] - mean) / sigma) ** 2
                )


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
        verbose=True,
    )

    logger.info(loo_orig)
    logger.info(loo_mm)

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
        verbose=True,
    )

    logger.info(loo_mm)
    logger.info(loo_orig)

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
        assert_allclose(result[name], original[name])


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

    assert_allclose(original_log_prob, converted_log_prob)


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

    assert_allclose(original_log_lik, converted_log_lik)


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

        assert_allclose(orig_ll, conv_ll)


def post_draws_custom(model, **kwargs):
    """Get posterior draws from custom model."""
    draws = np.column_stack(
        [model.draws["alpha"], model.draws["beta"], model.draws["sigma"]]
    )
    return draws


def log_lik_i_custom(model, i, **kwargs):
    """Get log-likelihood for observation i from custom model."""
    return model.log_likelihood[:, i]


def unconstrain_pars_custom(model, pars, **kwargs):
    """Convert parameters to unconstrained space for custom model."""
    upars = pars.copy()
    upars[:, 2] = np.log(pars[:, 2])
    return upars


def log_prob_upars_custom(model, upars, **kwargs):
    """Compute log probability for unconstrained parameters for custom model."""
    log_prob = np.zeros(len(upars))
    log_prob += -0.5 * upars[:, 0] ** 2 - 0.5 * np.log(2 * np.pi)
    log_prob += -0.5 * upars[:, 1] ** 2 - 0.5 * np.log(2 * np.pi)
    log_prob += -0.5 * upars[:, 2] ** 2 - 0.5 * np.log(2 * np.pi)
    log_prob += upars[:, 2]

    return log_prob


def log_lik_i_upars_custom(model, upars, i, **kwargs):
    """Compute log-likelihood for observation i with unconstrained parameters."""
    n_samples = len(upars)
    log_lik = np.zeros(n_samples)

    for s in range(n_samples):
        alpha = upars[s, 0]
        beta = upars[s, 1]
        sigma = np.exp(upars[s, 2])

        mean = alpha + beta * model.x[i]
        log_lik[s] = (
            -0.5 * np.log(2 * np.pi * sigma**2)
            - 0.5 * ((model.y[i] - mean) / sigma) ** 2
        )

    return log_lik


def create_mock_loo_data(model, k_values=None):
    """Create a mock LOO data object for testing."""
    n_obs = model.n_obs

    if k_values is None:
        k_values = np.random.uniform(0.1, 0.5, n_obs)
        k_values[0] = 0.8
        k_values[1] = 0.9

    elpd_values = -2.0 - k_values

    loo_i = xr.DataArray(
        elpd_values,
        dims=["observation_id"],
        coords={"observation_id": np.arange(n_obs)},
    )

    pareto_k = xr.DataArray(
        k_values, dims=["observation_id"], coords={"observation_id": np.arange(n_obs)}
    )

    elpd_loo_value = elpd_values.sum()
    se_value = np.sqrt(n_obs * np.var(elpd_values))
    p_loo_value = 3.0

    loo_data = ELPDData({
        "elpd_loo": elpd_loo_value,
        "p_loo": p_loo_value,
        "loo_i": loo_i,
        "pareto_k": pareto_k,
        "n_samples": model.n_samples,
        "n_data_points": n_obs,
        "scale": "log",
        "se": se_value,
        "warning": np.any(k_values > 0.7),
        "looic": -2 * elpd_loo_value,
        "looic_se": 2 * se_value,
        "p_loo_se": 0.5,
    })

    return loo_data


@pytest.fixture
def custom_model():
    """Create a custom model for testing."""
    return CustomModel(n_samples=1000, n_obs=20, seed=42)


def test_loo_moment_match_custom_implementation(custom_model):
    """Test that loo_moment_match works with custom implementations."""
    loo_data = create_mock_loo_data(custom_model)
    original_k_values = loo_data.pareto_k.values.copy()

    k_threshold = 0.7
    high_k_indices = np.where(original_k_values > k_threshold)[0]
    assert len(high_k_indices) > 0, "Test requires observations with high Pareto k"

    loo_data_copy = deepcopy(loo_data)

    loo_mm = loo_moment_match(
        custom_model,
        loo_data_copy,
        post_draws=post_draws_custom,
        log_lik_i=log_lik_i_custom,
        unconstrain_pars=unconstrain_pars_custom,
        log_prob_upars_fn=log_prob_upars_custom,
        log_lik_i_upars_fn=log_lik_i_upars_custom,
        max_iters=10,
        k_threshold=k_threshold,
        split=True,
        cov=True,
    )

    improvements = []
    for idx in high_k_indices:
        orig_k = original_k_values[idx]
        mm_k = loo_mm.pareto_k[idx].values
        improvement = orig_k - mm_k
        improvements.append(improvement)

    assert np.any(np.array(improvements) > 0), "No Pareto k values improved"
    assert loo_mm.elpd_loo >= loo_data.elpd_loo - 1e-10


def test_loo_moment_match_custom_vs_split(custom_model):
    """Test split moment matching with custom implementation."""
    loo_data = create_mock_loo_data(custom_model)
    original_elpd = loo_data.elpd_loo

    loo_data_copy1 = deepcopy(loo_data)
    loo_data_copy2 = deepcopy(loo_data)

    loo_mm_split = loo_moment_match(
        custom_model,
        loo_data_copy1,
        post_draws=post_draws_custom,
        log_lik_i=log_lik_i_custom,
        unconstrain_pars=unconstrain_pars_custom,
        log_prob_upars_fn=log_prob_upars_custom,
        log_lik_i_upars_fn=log_lik_i_upars_custom,
        max_iters=10,
        k_threshold=0.7,
        split=True,
        cov=True,
    )

    loo_mm_regular = loo_moment_match(
        custom_model,
        loo_data_copy2,
        post_draws=post_draws_custom,
        log_lik_i=log_lik_i_custom,
        unconstrain_pars=unconstrain_pars_custom,
        log_prob_upars_fn=log_prob_upars_custom,
        log_lik_i_upars_fn=log_lik_i_upars_custom,
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


def test_loo_moment_match_custom_missing_functions(custom_model):
    """Test error handling when required functions are missing."""
    loo_data = create_mock_loo_data(custom_model)

    with pytest.raises(ValueError, match="must provide all the following functions"):
        loo_moment_match(
            custom_model,
            loo_data,
            max_iters=10,
            k_threshold=0.7,
        )

    with pytest.raises(ValueError, match="must provide all the following functions"):
        loo_moment_match(
            custom_model,
            loo_data,
            post_draws=post_draws_custom,
            log_lik_i=log_lik_i_custom,
            max_iters=10,
            k_threshold=0.7,
        )


def test_loo_moment_match_custom_different_methods(custom_model):
    """Test moment matching with different importance sampling methods for custom model."""
    loo_data = create_mock_loo_data(custom_model)
    original_elpd = loo_data.elpd_loo

    methods = ["psis", "sis", "tis"]
    results = {}

    for method in methods:
        loo_data_copy = deepcopy(loo_data)

        results[method] = loo_moment_match(
            custom_model,
            loo_data_copy,
            post_draws=post_draws_custom,
            log_lik_i=log_lik_i_custom,
            unconstrain_pars=unconstrain_pars_custom,
            log_prob_upars_fn=log_prob_upars_custom,
            log_lik_i_upars_fn=log_lik_i_upars_custom,
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
                assert rel_diff < 0.2, f"Methods {m1} and {m2} differ too much"


def test_loo_moment_match_pymc_vs_custom(problematic_model):
    """Test that PyMCWrapper and custom function implementations yield same results."""
    wrapper = problematic_model
    loo_orig = loo(wrapper.idata, pointwise=True)

    loo_data_pymc = deepcopy(loo_orig)
    loo_mm_pymc = loo_moment_match(
        wrapper,
        loo_data_pymc,
        max_iters=10,
        k_threshold=0.7,
        split=True,
        cov=True,
    )

    converter = ParameterConverter(wrapper)
    unconstrained_draws_dict = wrapper.get_unconstrained_parameters()

    def post_draws_from_wrapper(model, **kwargs):
        """Get posterior draws from wrapper."""
        posterior = model.idata.posterior
        stacked_posterior = posterior.stack(__sample__=("chain", "draw"))
        draw_list = [
            stacked_posterior[var].values for var in model.get_parameter_names()
        ]
        return np.column_stack(draw_list)

    def log_lik_i_from_wrapper(model, i, **kwargs):
        """Get log-likelihood for observation i from wrapper."""
        log_likelihood = model.idata.log_likelihood
        var_name = list(log_likelihood.data_vars)[0]
        stacked_log_lik = log_likelihood[var_name].stack(__sample__=("chain", "draw"))
        obs_dim = [dim for dim in stacked_log_lik.dims if dim != "__sample__"][0]
        return stacked_log_lik.isel({obs_dim: i}).values

    def unconstrain_pars_from_wrapper(model, pars, **kwargs):
        """Convert parameters to unconstrained space for wrapper."""
        return converter.dict_to_matrix(unconstrained_draws_dict)

    def log_prob_upars_from_wrapper(model, upars, **kwargs):
        """Compute log probability for unconstrained parameters for wrapper."""
        upars_dict = converter.matrix_to_dict(upars)
        return log_prob_upars(model, upars_dict)

    def log_lik_i_upars_from_wrapper(model, upars, i, **kwargs):
        """Compute log-likelihood for observation i with unconstrained parameters for wrapper."""
        upars_dict = converter.matrix_to_dict(upars)
        log_lik_result = log_lik_i_upars(model, upars_dict, pointwise=True)
        return extract_log_likelihood_for_observation(log_lik_result, i)

    loo_data_custom = deepcopy(loo_orig)
    loo_mm_custom = loo_moment_match(
        wrapper,
        loo_data_custom,
        post_draws=post_draws_from_wrapper,
        log_lik_i=log_lik_i_from_wrapper,
        unconstrain_pars=unconstrain_pars_from_wrapper,
        log_prob_upars_fn=log_prob_upars_from_wrapper,
        log_lik_i_upars_fn=log_lik_i_upars_from_wrapper,
        max_iters=10,
        k_threshold=0.7,
        split=True,
        cov=True,
    )

    assert_allclose(loo_mm_pymc.elpd_loo, loo_mm_custom.elpd_loo, rtol=1e-6)
    assert_allclose(loo_mm_pymc.p_loo, loo_mm_custom.p_loo, rtol=1e-6)
    assert_allclose(loo_mm_pymc.se, loo_mm_custom.se, rtol=1e-6)
    assert_allclose(loo_mm_pymc.loo_i, loo_mm_custom.loo_i, rtol=1e-6)
    assert_allclose(loo_mm_pymc.pareto_k, loo_mm_custom.pareto_k, rtol=1e-6)

    logger.info(loo_mm_pymc)
    logger.info(loo_mm_custom)


def test_loo_moment_match_cmdstan_example():
    """Test the CmdStanPy example from the docstring."""
    mock_cmdstan = MockCmdStanModel(n_samples=1000, n_obs=20, n_predictors=3)

    model_obj = {
        "fit": mock_cmdstan,
        "data": {"K": mock_cmdstan.n_predictors, "N": mock_cmdstan.n_obs},
    }

    def post_draws_cmdstan(model_obj, **kwargs):
        fit = model_obj["fit"]
        draws_dict = {
            "beta": fit.stan_variables()["beta"].reshape(fit.n_samples, -1),
            "intercept": fit.stan_variables()["intercept"].flatten(),
        }
        beta_array = draws_dict["beta"]
        intercept_array = draws_dict["intercept"].reshape(-1, 1)
        return np.hstack([intercept_array, beta_array])

    def log_lik_i_cmdstan(model_obj, i, **kwargs):
        fit = model_obj["fit"]
        log_lik = fit.stan_variables()["log_lik"]
        return log_lik.reshape(-1, fit.n_obs)[:, i]

    def unconstrain_pars_cmdstan(model_obj, pars, **kwargs):
        # Parameters are already unconstrained
        return pars

    def log_prob_upars_cmdstan(model_obj, upars, **kwargs):
        K = model_obj["data"]["K"]
        n_samples = upars.shape[0]

        log_prob = np.zeros(n_samples)
        alpha_prior_scale = 10.0
        log_prob += (
            -0.5 * (upars[:, 0] / alpha_prior_scale) ** 2
            - np.log(alpha_prior_scale)
            - 0.5 * np.log(2 * np.pi)
        )

        beta_prior_scale = 10.0
        for k in range(1, K + 1):
            log_prob += (
                -0.5 * (upars[:, k] / beta_prior_scale) ** 2
                - np.log(beta_prior_scale)
                - 0.5 * np.log(2 * np.pi)
            )

        return log_prob

    def log_lik_i_upars_cmdstan(model_obj, upars, i, **kwargs):
        fit = model_obj["fit"]
        K = model_obj["data"]["K"]

        n_samples = upars.shape[0]
        log_lik = np.zeros(n_samples)

        for s in range(n_samples):
            intercept = upars[s, 0]
            beta = upars[s, 1 : K + 1]

            x_i = fit.x[i]
            y_i = fit.y[i]
            offset_i = fit.offset[i]

            mu_i = np.exp(np.dot(x_i, beta) + intercept + offset_i)

            log_lik[s] = y_i * np.log(mu_i) - mu_i - np.log(np.math.factorial(y_i))

        return log_lik

    k_values = np.random.uniform(0.1, 0.5, mock_cmdstan.n_obs)
    k_values[0] = 0.8
    k_values[1] = 0.9
    loo_data = create_mock_loo_data(mock_cmdstan, k_values)

    loo_data_copy = deepcopy(loo_data)

    loo_mm = loo_moment_match(
        model_obj,
        loo_data_copy,
        post_draws=post_draws_cmdstan,
        log_lik_i=log_lik_i_cmdstan,
        unconstrain_pars=unconstrain_pars_cmdstan,
        log_prob_upars_fn=log_prob_upars_cmdstan,
        log_lik_i_upars_fn=log_lik_i_upars_cmdstan,
        max_iters=10,
        k_threshold=0.7,
        split=True,
        cov=True,
        verbose=True,
    )

    high_k_indices = np.where(k_values > 0.7)[0]
    assert len(high_k_indices) > 0, "Test requires observations with high Pareto k"

    improvements = []
    for idx in high_k_indices:
        orig_k = k_values[idx]
        mm_k = loo_mm.pareto_k[idx].values
        improvement = orig_k - mm_k
        improvements.append(improvement)

    assert np.any(np.array(improvements) > 0), "No Pareto k values improved"
    assert loo_mm.elpd_loo >= loo_data.elpd_loo - 1e-10

    logger.info(f"Original ELPD LOO: {loo_data.elpd_loo}")
    logger.info(f"Improved ELPD LOO: {loo_mm.elpd_loo}")
    logger.info(f"Original k values: {k_values}")
    logger.info(f"Improved k values: {loo_mm.pareto_k.values}")

    logger.info(loo_mm)


def test_validate_custom_function():
    """Test the _validate_custom_function utility."""

    def good_func(model, i, extra=None):
        return i

    def missing_param_func(model):
        return 0

    def kwargs_func(model, **kwargs):
        return kwargs.get("i", 0)

    def extra_params_func(model, i, extra1=None, extra2=None):
        return i

    assert _validate_custom_function(good_func, ["model", "i"], "good_func") is True
    assert _validate_custom_function(kwargs_func, ["model", "i"], "kwargs_func") is True
    assert (
        _validate_custom_function(
            extra_params_func, ["model", "i"], "extra_params_func"
        )
        is True
    )
    with pytest.raises(ValueError, match="missing required parameters"):
        _validate_custom_function(
            missing_param_func, ["model", "i"], "missing_param_func"
        )


def test_validate_output():
    """Test the _validate_output utility."""
    good_array = np.array([1.0, 2.0, 3.0])
    validated = _validate_output(good_array, "good_array", expected_ndim=1)
    assert_allclose(validated, good_array)

    bad_ndim_array = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="dimensions for bad_ndim_array"):
        _validate_output(bad_ndim_array, "bad_ndim_array", expected_ndim=1)

    nan_array = np.array([1.0, np.nan, 3.0])
    with pytest.raises(ValueError, match="NaN values detected"):
        _validate_output(nan_array, "nan_array", expected_ndim=1)

    with pytest.raises(ValueError, match="returned None for none_value"):
        _validate_output(None, "none_value", expected_ndim=1, allow_none=False)

    assert (
        _validate_output(None, "none_value", expected_ndim=1, allow_none=True) is None
    )

    shape_array = np.array([[1.0, 2.0], [3.0, 4.0]])
    validated = _validate_output(shape_array, "shape_array", expected_shape=(2, 2))
    assert_allclose(validated, shape_array)

    with pytest.raises(ValueError, match="Expected shape"):
        _validate_output(shape_array, "shape_array", expected_shape=(3, 2))

    list_input = [1.0, 2.0, 3.0]
    validated = _validate_output(list_input, "list_input", expected_ndim=1)
    assert isinstance(validated, np.ndarray)
    assert_allclose(validated, np.array(list_input))
