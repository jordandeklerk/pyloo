"""Tests for ELPD base class."""

import pandas as pd
import pytest

from ...elpd import ELPDData


def test_elpd_data_error():
    with pytest.raises(IndexError):
        repr(ELPDData(data=[0, 1, 2], index=["not IC", "se", "p"]))


def test_elpd_data_initialization():
    data = [1.0, 0.5, 0.1, 100, False, 1000]
    index = ["elpd_loo", "se", "p_loo", "n_samples", "warning", "n_data_points"]
    elpd = ELPDData(data=data, index=index)

    assert isinstance(elpd, ELPDData)
    assert isinstance(elpd, pd.Series)
    assert len(elpd) == 6
    assert elpd["elpd_loo"] == 1.0
    assert elpd["se"] == 0.5
    assert elpd["p_loo"] == 0.1
    assert elpd.n_samples == 100
    assert elpd.warning is False
    assert elpd.n_data_points == 1000


def test_elpd_data_copy():
    data = [1.0, 0.5, 0.1, 100, False, 1000]
    index = ["elpd_loo", "se", "p_loo", "n_samples", "warning", "n_data_points"]
    elpd = ELPDData(data=data, index=index)

    shallow_copy = elpd.copy(deep=False)
    assert isinstance(shallow_copy, ELPDData)
    assert all(shallow_copy == elpd)

    deep_copy = elpd.copy(deep=True)
    assert isinstance(deep_copy, ELPDData)
    assert all(deep_copy == elpd)
    assert deep_copy is not elpd


def test_elpd_data_properties():
    data = [1.0, 0.5, 0.1, 100, False, 1000]
    index = ["elpd_loo", "se", "p_loo", "n_samples", "warning", "n_data_points"]
    elpd = ELPDData(data=data, index=index)

    assert elpd.n_samples == 100
    assert elpd.n_data_points == 1000
    assert elpd.warning is False
    assert elpd.method == "psis"  # default method

    elpd.method = "tis"
    assert elpd.method == "tis"

    estimates = {"mean": 1.0, "std": 0.5}
    elpd.estimates = estimates
    assert elpd.estimates == estimates


def test_elpd_str_subsampled_format():
    data = {
        "elpd_loo": 1.0,
        "se": 0.5,
        "p_loo": 0.1,
        "n_samples": 1000,
        "warning": False,
        "n_data_points": 100,
        "subsampling_SE": 0.3,
        "subsample_size": 50,
    }
    elpd = ELPDData(data=pd.Series(data))

    str_output = str(elpd)
    assert "Computed from 1000 by 50 subsampled" in str_output
    assert "subsampling SE" in str_output


def test_elpd_invalid_kind():
    data = [1.0, 0.5, 0.1, 100, False, 1000]
    index = ["invalid_kind", "se", "p_loo", "n_samples", "warning", "n_data_points"]
    elpd = ELPDData(data=data, index=index)

    with pytest.raises(ValueError, match="Invalid ELPDData object"):
        str(elpd)


def test_elpd_with_warning():
    data = {
        "elpd_loo": 1.0,
        "se": 0.5,
        "p_loo": 0.1,
        "p_loo_se": 0.2,
        "looic": 100,
        "looic_se": 10,
        "n_samples": 1000,
        "warning": True,
        "n_data_points": 100,
    }
    elpd = ELPDData(data=pd.Series(data))

    str_output = str(elpd)
    assert "There has been a warning during the calculation" in str_output
