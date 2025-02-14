"""Tests for rcparams.py module."""

import pytest

from ...rcparams import (
    RcParams,
    _validate_boolean,
    _validate_scale,
    defaultParams,
    rcParams,
)


def test_validate_boolean():
    assert _validate_boolean(True) is True
    assert _validate_boolean(False) is False
    with pytest.raises(ValueError, match="Value must be True or False, not 1"):
        _validate_boolean(1)
    with pytest.raises(ValueError, match="Value must be True or False, not true"):
        _validate_boolean("true")


def test_validate_scale():
    valid_scales = ["deviance", "log", "negative_log"]
    for scale in valid_scales:
        assert _validate_scale(scale) == scale
        assert _validate_scale(scale.upper()) == scale

    with pytest.raises(ValueError, match="Scale must be one of"):
        _validate_scale("invalid_scale")
    with pytest.raises(ValueError, match="Scale must be one of"):
        _validate_scale(123)


def test_rcparams_init():
    rc = RcParams()
    assert rc["stats.ic_pointwise"] is False
    assert rc["stats.ic_scale"] == "log"

    rc = RcParams({"stats.ic_pointwise": True})
    assert rc["stats.ic_pointwise"] is True
    assert rc["stats.ic_scale"] == "log"


def test_rcparams_setitem():
    rc = RcParams()

    rc["stats.ic_pointwise"] = True
    assert rc["stats.ic_pointwise"] is True

    rc["stats.ic_scale"] = "deviance"
    assert rc["stats.ic_scale"] == "deviance"

    with pytest.raises(ValueError, match="Value must be True or False"):
        rc["stats.ic_pointwise"] = "invalid"

    with pytest.raises(ValueError, match="Scale must be one of"):
        rc["stats.ic_scale"] = "invalid"

    with pytest.raises(KeyError, match="is not a valid rc parameter"):
        rc["invalid.key"] = True


def test_rcparams_deletion_prevention():
    rc = RcParams()

    with pytest.raises(TypeError, match="RcParams keys cannot be deleted"):
        del rc["stats.ic_pointwise"]

    with pytest.raises(TypeError, match="RcParams keys cannot be deleted"):
        rc.clear()

    with pytest.raises(TypeError, match="RcParams keys cannot be deleted"):
        rc.pop("stats.ic_pointwise")

    with pytest.raises(TypeError, match="RcParams keys cannot be deleted"):
        rc.popitem()


def test_rcparams_setdefault():
    rc = RcParams()
    with pytest.raises(
        TypeError, match="Defaults in RcParams are handled on object initialization"
    ):
        rc.setdefault("stats.ic_pointwise", True)


def test_rcparams_string_representation():
    rc = RcParams()

    assert repr(rc).startswith("RcParams({")
    assert "'stats.ic_pointwise': False" in repr(rc)
    assert "'stats.ic_scale': 'log'" in repr(rc)

    str_repr = str(rc)
    assert "stats.ic_pointwise    : False" in str_repr
    assert "stats.ic_scale        : log" in str_repr


def test_rcparams_iteration():
    rc = RcParams()

    keys = list(rc)
    assert len(keys) == len(defaultParams)
    assert all(key in defaultParams for key in keys)
    assert keys == sorted(keys)

    assert len(rc) == len(defaultParams)


def test_rcparams_copy():
    rc = RcParams()
    rc_copy = rc.copy()

    assert isinstance(rc_copy, dict)
    assert rc_copy == rc._underlying_storage
    assert rc_copy is not rc._underlying_storage


def test_global_rcparams():
    assert isinstance(rcParams, RcParams)
    assert rcParams["stats.ic_pointwise"] is False
    assert rcParams["stats.ic_scale"] == "log"
