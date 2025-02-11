"""pyloo rcparams based on matplotlib's implementation."""

from collections.abc import MutableMapping
from typing import Any, Dict, Set


def _validate_boolean(value: Any) -> bool:
    """Validate boolean values."""
    if isinstance(value, bool):
        return value
    raise ValueError(f"Value must be True or False, not {value}")


def _validate_scale(value: Any) -> str:
    """Validate scale parameter."""
    valid_scales: Set[str] = {"deviance", "log", "negative_log"}
    if isinstance(value, str) and value.lower() in valid_scales:
        return value.lower()
    raise ValueError(f"Scale must be one of {valid_scales}, not {value}")


defaultParams = {
    "stats.ic_pointwise": (False, _validate_boolean),
    "stats.ic_scale": ("log", _validate_scale),
}


class RcParams(MutableMapping):
    """Class to contain pyloo default parameters."""

    validate = {key: validate_fun for key, (_, validate_fun) in defaultParams.items()}

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize RcParams with default values and optional overrides."""
        self._underlying_storage: Dict[str, Any] = {}
        super().__init__()
        for key, (value, _) in defaultParams.items():
            self._underlying_storage[key] = value
        self.update(*args, **kwargs)
        self.update(*args, **kwargs)

    def __setitem__(self, key: str, val: Any) -> None:
        """Add validation to __setitem__ function."""
        try:
            try:
                cval = self.validate[key](val)
            except ValueError as verr:
                raise ValueError(f"Key {key}: {str(verr)}") from verr
            self._underlying_storage[key] = cval
        except KeyError as err:
            raise KeyError(
                f"{key} is not a valid rc parameter " f"(see rcParams.keys() for a list of valid parameters)"
            ) from err

    def __getitem__(self, key: str) -> Any:
        """Use underlying dict's getitem method."""
        return self._underlying_storage[key]

    def __delitem__(self, key: str) -> None:
        """Raise TypeError if someone tries to delete a key from RcParams."""
        raise TypeError("RcParams keys cannot be deleted")

    def clear(self) -> None:
        """Raise TypeError if someone tries to delete all keys from RcParams."""
        raise TypeError("RcParams keys cannot be deleted")

    def pop(self, key: str, default: Any = None) -> None:
        """Raise TypeError if someone tries to delete a key from RcParams."""
        raise TypeError("RcParams keys cannot be deleted. Use .get(key) or RcParams[key] to check values")

    def popitem(self) -> tuple[Any, Any]:
        """Raise TypeError if someone tries to delete a key from RcParams."""
        raise TypeError("RcParams keys cannot be deleted. Use .get(key) or RcParams[key] to check values")

    def setdefault(self, key: str, default: Any = None) -> None:
        """Raise error when using setdefault."""
        raise TypeError(
            "Defaults in RcParams are handled on object initialization. " "Use pyloo configuration file instead."
        )

    def __repr__(self) -> str:
        """Customize repr of RcParams objects."""
        class_name = self.__class__.__name__
        return f"{class_name}({self._underlying_storage})"

    def __str__(self) -> str:
        """Customize str/print of RcParams objects."""
        return "\n".join(f"{key:<22}: {value}" for key, value in sorted(self._underlying_storage.items()))

    def __iter__(self):
        """Yield sorted list of keys."""
        yield from sorted(self._underlying_storage.keys())

    def __len__(self) -> int:
        """Use underlying dict's len method."""
        return len(self._underlying_storage)

    def copy(self) -> Dict[str, Any]:
        """Get a copy of the RcParams object."""
        return dict(self._underlying_storage)


rcParams = RcParams()
