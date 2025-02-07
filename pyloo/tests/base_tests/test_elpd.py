"""Tests for ELPD base class."""

import pytest

from ...elpd import ELPDData


def test_elpd_data_error():
    with pytest.raises(IndexError):
        repr(ELPDData(data=[0, 1, 2], index=["not IC", "se", "p"]))
