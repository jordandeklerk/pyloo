import arviz as az
import numpy as np
import pytest
from numpy.testing import assert_allclose

from ...waic import waic


@pytest.fixture(scope="session")
def centered_eight():
    return az.load_arviz_data("centered_eight")


def test_waic_basic(centered_eight):
    result = waic(centered_eight)

    az_result = az.waic(centered_eight)

    assert_allclose(result["elpd_waic"], az_result["elpd_waic"], rtol=1e-7)
    assert_allclose(result["p_waic"], az_result["p_waic"], rtol=1e-7)
    assert_allclose(result["se"], az_result["se"], rtol=1e-7)


def test_waic_pointwise(centered_eight):
    result = waic(centered_eight, pointwise=True)

    az_result = az.waic(centered_eight, pointwise=True)

    assert_allclose(result["elpd_waic"], az_result["elpd_waic"], rtol=1e-7)
    assert_allclose(result["p_waic"], az_result["p_waic"], rtol=1e-7)
    assert_allclose(result["se"], az_result["se"], rtol=1e-7)
    assert_allclose(result["waic_i"].values, az_result["waic_i"].values, rtol=1e-7)


@pytest.mark.parametrize("scale", ["log", "negative_log", "deviance"])
def test_waic_scales(centered_eight, scale):
    result = waic(centered_eight, scale=scale)
    az_result = az.waic(centered_eight, scale=scale)

    assert_allclose(result["elpd_waic"], az_result["elpd_waic"], rtol=1e-7)
    assert result["scale"] == scale


def test_waic_invalid_scale(centered_eight):
    with pytest.raises(TypeError, match="Valid scale values are"):
        waic(centered_eight, scale="invalid")


def test_waic_nan_inf(centered_eight):
    data = centered_eight.copy()
    data.log_likelihood["obs"][:, :, 0] = np.nan
    data.log_likelihood["obs"][:, :, 1] = np.inf

    with pytest.warns(UserWarning, match="NaN values detected"):
        with pytest.warns(UserWarning, match="Infinite values detected"):
            result = waic(data)
            assert result is not None


def test_waic_multiple_vars(centered_eight):
    data = centered_eight.copy()
    data.log_likelihood["obs2"] = data.log_likelihood["obs"]

    with pytest.raises(TypeError, match="Found several log likelihood arrays"):
        waic(data)

    result = waic(data, var_name="obs")
    assert result is not None


def test_waic_missing_loglik():
    data = az.from_dict({"posterior": {"mu": np.random.normal(0, 1, 1000)}})
    with pytest.raises(TypeError, match="log likelihood not found"):
        waic(data)


def test_waic_constant_loglik(centered_eight):
    data = centered_eight.copy()
    data.log_likelihood["obs"][:] = 1.0

    with pytest.warns(UserWarning, match="The point-wise WAIC is the same"):
        result = waic(data, pointwise=True)
        assert np.allclose(result["waic_i"].values, result["waic_i"].values[0])
