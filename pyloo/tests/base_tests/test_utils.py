import arviz as az
import numpy as np
import pytest
import xarray as xr
from arviz import InferenceData

from ...utils import (
    compute_estimates,
    compute_log_mean_exp,
    extract_log_likelihood,
    is_constant,
    reshape_draws,
    to_inference_data,
    validate_data,
)
from ..helpers import (
    assert_arrays_allclose,
    assert_arrays_equal,
    assert_finite,
    assert_shape_equal,
)


def test_to_inference_data_real(centered_eight, non_centered_eight):
    assert isinstance(to_inference_data(centered_eight), az.InferenceData)
    assert isinstance(to_inference_data(non_centered_eight), az.InferenceData)

    idata = to_inference_data(centered_eight)
    assert_arrays_equal(idata.log_likelihood.obs, centered_eight.log_likelihood.obs)


def test_to_inference_data_invalid():
    with pytest.raises(ValueError):
        to_inference_data([1, 2, 3])
    with pytest.raises(ValueError):
        to_inference_data({"a": 1})


def test_extract_log_likelihood_real(centered_eight, non_centered_eight):
    log_lik_c, chain_ids_c = extract_log_likelihood(centered_eight)
    assert_finite(log_lik_c)
    assert log_lik_c.shape[1] == len(centered_eight.observed_data.obs)

    log_lik_nc, chain_ids_nc = extract_log_likelihood(non_centered_eight)
    assert_finite(log_lik_nc)
    assert log_lik_nc.shape[1] == len(non_centered_eight.observed_data.obs)

    n_chains_c = centered_eight.posterior.chain.size
    n_draws_c = centered_eight.posterior.draw.size

    n_chains_nc = non_centered_eight.posterior.chain.size
    n_draws_nc = non_centered_eight.posterior.draw.size

    expected_ids_c = np.repeat(np.arange(1, n_chains_c + 1), n_draws_c)
    expected_ids_nc = np.repeat(np.arange(1, n_chains_nc + 1), n_draws_nc)

    assert_arrays_equal(chain_ids_c, expected_ids_c)
    assert_arrays_equal(chain_ids_nc, expected_ids_nc)


def test_extract_log_likelihood_synthetic(rng):
    draws, chains, obs = 100, 4, 8
    log_like = rng.normal(size=(chains, draws, obs))

    dataset = xr.Dataset(
        data_vars={"obs": (["chain", "draw", "observation"], log_like)},
        coords={"chain": np.arange(chains), "draw": np.arange(draws), "observation": np.arange(obs)},
    )

    idata = az.InferenceData(log_likelihood=dataset)

    log_lik, chain_ids = extract_log_likelihood(idata, var_name="obs")
    assert_shape_equal(log_lik, rng.normal(size=(chains * draws, obs)))
    assert len(chain_ids) == chains * draws


def test_compute_log_mean_exp(numpy_arrays):
    x = numpy_arrays["random_weights"]

    result = compute_log_mean_exp(x)
    expected = np.log(np.mean(np.exp(x)))
    assert_arrays_allclose(result, expected)

    result_axis = compute_log_mean_exp(x, axis=0)
    expected_axis = np.log(np.mean(np.exp(x), axis=0))
    assert_arrays_allclose(result_axis, expected_axis)


def test_compute_estimates(log_likelihood_data):
    x = log_likelihood_data.values
    result = compute_estimates(x)

    assert "estimate" in result
    assert "se" in result

    assert_arrays_allclose(result["estimate"], np.sum(x, axis=0))
    assert_arrays_allclose(result["se"], np.sqrt(x.shape[0] * np.var(x, axis=0, ddof=1)))


def test_validate_data_numpy(numpy_arrays, extreme_data):
    """Test validation of numpy arrays."""
    x = numpy_arrays["normal"]
    assert_arrays_equal(validate_data(x), x)

    with pytest.raises(ValueError, match="Array has incorrect shape"):
        validate_data(x, check_shape=(10, 10))
    with pytest.raises(ValueError, match="Input contains NaN values"):
        validate_data(np.array([1.0, np.nan, 3.0]))
    with pytest.raises(ValueError, match="Input contains infinite values"):
        validate_data(np.array([1.0, np.inf, 3.0]))

    inf_array = np.array([1.0, np.inf, 3.0])
    assert_arrays_equal(validate_data(inf_array, allow_inf=True), inf_array)

    large_array = np.array([1e40, 2e40])
    with pytest.raises(ValueError, match="numerical instability"):
        validate_data(large_array)

    validate_data(extreme_data)


def test_validate_data_inference(centered_eight):
    """Test validation of InferenceData objects."""
    validated = validate_data(centered_eight)
    assert isinstance(validated, InferenceData)
    assert hasattr(validated, "log_likelihood")

    with pytest.raises(TypeError, match="Failed to validate or convert input: Variable 'nonexistent' not found"):
        validate_data(centered_eight, var_name="nonexistent")

    idata_no_loglik = az.InferenceData(posterior=centered_eight.posterior)
    with pytest.raises(
        TypeError, match="Failed to validate or convert input: InferenceData object must have a log_likelihood group"
    ):
        validate_data(idata_no_loglik)


def test_validate_data_conversion():
    """Test validation of objects that need conversion."""
    with pytest.raises(TypeError, match="Failed to validate or convert input: Lists and tuples cannot be converted"):
        validate_data([1, 2, 3])
    with pytest.raises(TypeError, match="Failed to validate or convert input: Dictionary values must be array-like"):
        validate_data({"a": 1, "b": "string"})

    data = np.array([[[1, 2, 3]]])
    dataset = xr.Dataset(
        data_vars={"obs": (["chain", "draw", "observation"], data)},
        coords={"chain": [0], "draw": [0], "observation": [0, 1, 2]},
    )
    idata = az.InferenceData(log_likelihood=dataset)
    validated = validate_data(idata)
    assert isinstance(validated, InferenceData)
    assert hasattr(validated, "log_likelihood")
    assert hasattr(validated, "log_likelihood")


def test_reshape_draws(multidim_data):
    llm = multidim_data["llm"]
    ll1 = multidim_data["ll1"]

    x_2d = llm.reshape(-1, np.prod(llm.shape[2:]))
    chain_ids = np.repeat(np.arange(1, llm.shape[0] + 1), llm.shape[1])
    reshaped, ids = reshape_draws(x_2d, chain_ids)
    assert_shape_equal(reshaped, llm.reshape(llm.shape[1], llm.shape[0], -1))

    flat_ll1, _ = reshape_draws(ll1)
    assert flat_ll1.shape == (ll1.shape[0] * ll1.shape[1], ll1.shape[2])


def test_is_constant(rng):
    x = np.full(10, 1.5)
    assert is_constant(x)

    x = rng.normal(size=100)
    assert not is_constant(x)

    x = np.ones(10) + rng.normal(0, 1e-10, 10)
    assert is_constant(x, tol=1e-9)
    assert not is_constant(x, tol=1e-11)
