import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ...utils import (
    ELPDData,
    autocorr,
    autocov,
    compute_estimates,
    compute_log_mean_exp,
    extract_log_likelihood,
    get_log_likelihood,
    is_constant,
    reshape_draws,
    smooth_data,
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
    x = numpy_arrays["normal"]
    assert_arrays_equal(validate_data(x, min_chains=1, min_draws=1), x)

    with pytest.raises(ValueError, match="Array has incorrect shape"):
        validate_data(x, check_shape=(10, 10), min_chains=1, min_draws=1)
    with pytest.raises(ValueError, match="Input contains NaN values"):
        validate_data(np.array([1.0, np.nan, 3.0]), min_chains=1, min_draws=1)
    with pytest.raises(ValueError, match="Input contains infinite values"):
        validate_data(np.array([1.0, np.inf, 3.0]), min_chains=1, min_draws=1)

    inf_array = np.array([1.0, np.inf, 3.0])
    assert_arrays_equal(validate_data(inf_array, allow_inf=True, min_chains=1, min_draws=1), inf_array)

    large_array = np.array([1e40, 2e40])
    with pytest.raises(ValueError, match="numerical instability"):
        validate_data(large_array, min_chains=1, min_draws=1)

    validate_data(extreme_data, min_chains=1, min_draws=1)


def test_validate_data_inference(centered_eight):
    validated = validate_data(centered_eight)
    assert isinstance(validated, az.InferenceData)
    assert hasattr(validated, "log_likelihood")

    with pytest.raises(TypeError, match="Failed to validate or convert input: Variable 'nonexistent' not found"):
        validate_data(centered_eight, var_name="nonexistent")

    idata_no_loglik = az.InferenceData(posterior=centered_eight.posterior)
    with pytest.raises(
        TypeError, match="Failed to validate or convert input: InferenceData object must have a log_likelihood group"
    ):
        validate_data(idata_no_loglik)


def test_validate_data_conversion():
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
    validated = validate_data(idata, min_chains=1, min_draws=1)
    assert isinstance(validated, az.InferenceData)
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


def test_autocov(rng):
    n = 5000
    phi = 0.7
    x = np.zeros(n)
    x[0] = rng.normal()
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal()

    cov = autocov(x)

    assert cov.shape == (n,)

    theoretical_var = 1 / (1 - phi**2)
    assert np.abs(cov[0] - theoretical_var) < 0.5

    for k in range(1, 5):
        theoretical_cov = theoretical_var * (phi**k)
        assert np.abs(cov[k] - theoretical_cov) < 0.2

    x_2d = rng.normal(size=(10, n))
    cov_2d = autocov(x_2d, axis=1)
    assert cov_2d.shape == (10, n)


def test_autocorr(rng):
    n = 1000
    phi = 0.7
    x = np.zeros(n)
    x[0] = rng.normal()
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal()

    corr = autocorr(x)

    assert corr.shape == (n,)
    assert np.abs(corr[0] - 1.0) < 1e-10
    assert np.all(np.abs(corr) <= 1.0 + 1e-10)

    for k in range(1, 5):
        assert np.abs(corr[k] - phi**k) < 0.2


def test_smooth_data(rng):
    n_obs = 50
    t = np.linspace(0, 1, n_obs)
    signal = np.sin(2 * np.pi * t)
    noise = 0.1 * rng.normal(size=n_obs)
    obs = signal + noise

    n_samples = 100
    pp = np.array([signal + 0.1 * rng.normal(size=n_obs) for _ in range(n_samples)])

    obs_smooth, pp_smooth = smooth_data(obs, pp)

    assert obs_smooth.shape == obs.shape
    assert pp_smooth.shape == pp.shape

    assert np.corrcoef(obs_smooth, signal)[0, 1] > 0.95
    assert all(np.corrcoef(pp_smooth[i], signal)[0, 1] > 0.95 for i in range(n_samples))

    obs_diff = np.abs(np.diff(obs))
    obs_smooth_diff = np.abs(np.diff(obs_smooth))
    assert np.mean(obs_smooth_diff) < np.mean(obs_diff)

    pp_diff = np.abs(np.diff(pp, axis=1))
    pp_smooth_diff = np.abs(np.diff(pp_smooth, axis=1))
    assert np.mean(np.mean(pp_smooth_diff, axis=1)) < np.mean(np.mean(pp_diff, axis=1))


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


def test_elpd_data():
    data = {
        "elpd_loo": -5.0,
        "p_loo": 2.5,
        "scale": "log",
        "n_samples": 1000,
        "n_data_points": 100,
        "warning": False,
    }
    elpd = ELPDData(data)

    assert elpd.kind == "loo"
    assert elpd.scale_str == "elpd"
    assert not elpd.has_warnings()
    assert not elpd.has_pareto_k()

    str_repr = str(elpd)
    assert "Computed from 1000 posterior samples" in str_repr
    assert "100 observations" in str_repr
    assert "elpd_loo" in str_repr
    assert "p_loo" in str_repr

    elpd.validate()

    invalid_data = data.copy()
    invalid_data["scale"] = "invalid"
    with pytest.raises(ValueError, match="Invalid scale"):
        ELPDData(invalid_data).validate()

    missing_data = {k: v for k, v in data.items() if k != "n_samples"}
    with pytest.raises(ValueError, match="Missing required fields"):
        ELPDData(missing_data).validate()

    warn_data = data.copy()
    warn_data["warning"] = True
    warn_elpd = ELPDData(warn_data)
    assert warn_elpd.has_warnings()
    assert "Warning" in str(warn_elpd)

    k_data = data.copy()
    k_values = np.array([0.1, 0.5, 0.8, 1.2])
    k_data["pareto_k"] = pd.Series(k_values)
    k_data["good_k"] = 0.6
    k_elpd = ELPDData(k_data)

    assert k_elpd.has_pareto_k()

    summary = k_elpd.get_pareto_k_summary()
    assert summary["k_threshold"] == 0.6
    assert len(summary["counts"]) == 3

    assert summary["counts"][0] == 2
    assert summary["counts"][1] == 1
    assert summary["counts"][2] == 1
    assert np.allclose(summary["percentages"], [50, 25, 25])

    k_str = str(k_elpd)
    assert "Pareto k diagnostic values" in k_str
    assert "good" in k_str
    assert "bad" in k_str
    assert "very bad" in k_str

    copied = k_elpd.copy(deep=True)
    assert isinstance(copied, ELPDData)
    assert copied.has_pareto_k()
    assert np.array_equal(copied.pareto_k, k_elpd.pareto_k)
    copied.pareto_k[0] = 999
    assert not np.array_equal(copied.pareto_k, k_elpd.pareto_k)

    shallow = k_elpd.copy(deep=False)
    assert isinstance(shallow, ELPDData)
    shallow.pareto_k[0] = 999
    assert np.array_equal(shallow.pareto_k, k_elpd.pareto_k)


def test_validate_data_enhanced(numpy_arrays):
    x = numpy_arrays["normal"]

    x_2d = x.reshape(2, -1)

    with pytest.raises(ValueError, match="Number of chains .* is less than min_chains"):
        validate_data(x_2d[:1], min_chains=2)

    with pytest.raises(ValueError, match="Number of draws .* is less than min_draws"):
        validate_data(x_2d[:, :3], min_draws=4)

    with pytest.warns(UserWarning, match="Number of chains"):
        validate_data(x_2d[:1], min_chains=2, raise_on_failure=False)

    x_with_nan = x_2d.copy()
    x_with_nan[0, 0] = np.nan

    with pytest.raises(ValueError):
        validate_data(x_with_nan, nan_axis=0)

    validate_data(x_with_nan, nan_policy="all", min_chains=1, min_draws=1)
