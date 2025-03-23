import arviz as az
import numpy as np
import pytest

from ...utils import get_log_likelihood, reshape_draws, to_inference_data
from ..helpers import assert_arrays_equal, assert_shape_equal


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


def test_reshape_draws(multidim_data):
    llm = multidim_data["llm"]
    ll1 = multidim_data["ll1"]

    x_2d = llm.reshape(-1, np.prod(llm.shape[2:]))
    chain_ids = np.repeat(np.arange(1, llm.shape[0] + 1), llm.shape[1])
    reshaped, ids = reshape_draws(x_2d, chain_ids)
    assert_shape_equal(reshaped, llm.reshape(llm.shape[1], llm.shape[0], -1))

    flat_ll1, _ = reshape_draws(ll1)
    assert flat_ll1.shape == (ll1.shape[0] * ll1.shape[1], ll1.shape[2])


def test_get_log_likelihood(centered_eight):
    ll = get_log_likelihood(centered_eight)
    assert ll is not None

    ll_obs = get_log_likelihood(centered_eight, var_name="obs")
    assert ll_obs is not None

    idata_no_loglik = az.InferenceData(posterior=centered_eight.posterior)
    with pytest.raises(TypeError, match="log likelihood not found"):
        get_log_likelihood(idata_no_loglik)

    with pytest.raises(TypeError, match="No log likelihood data named"):
        get_log_likelihood(centered_eight, var_name="nonexistent")
