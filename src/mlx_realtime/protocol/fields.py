"""Validation helpers for client events. Every problem names its dotted ``param`` path.

Errors use the two codes of the GA material: ``invalid_event`` for an event that cannot be
processed as a whole (bad JSON, missing or unsupported type, wrong state) and ``invalid_value``
for a field that is missing, mistyped, out of range, unsupported or unknown.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Literal, cast


class ProtocolError(Exception):
    def __init__(
        self,
        code: Literal["invalid_event", "invalid_value"],
        message: str,
        param: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code: Literal["invalid_event", "invalid_value"] = code
        self.message = message
        self.param = param


def child(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def element(path: str, index: int) -> str:
    return f"{path}[{index}]"


def json_type(value: object) -> str:
    match value:
        case None:
            return "null"
        case bool():
            return "a boolean"
        case int():
            return "an integer"
        case float():
            return "a number"
        case str():
            return "a string"
        case list():
            return "an array"
        case dict():
            return "an object"
        case _:
            return type(value).__name__


def _type_error(param: str, expected: str, value: object) -> ProtocolError:
    message = (
        f"Invalid type for '{param}': expected {expected}, but got {json_type(value)} instead."
    )
    return ProtocolError("invalid_value", message, param)


def value_error(param: str, message: str) -> ProtocolError:
    return ProtocolError("invalid_value", f"Invalid value for '{param}': {message}", param)


def expect_object(value: object, param: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise _type_error(param, "an object", value)
    return cast(dict[str, object], value)


def expect_list(value: object, param: str) -> list[object]:
    if not isinstance(value, list):
        raise _type_error(param, "an array", value)
    return cast(list[object], value)


def expect_str(value: object, param: str) -> str:
    if not isinstance(value, str):
        raise _type_error(param, "a string", value)
    return value


def expect_bool(value: object, param: str) -> bool:
    if not isinstance(value, bool):
        raise _type_error(param, "a boolean", value)
    return value


def expect_int(
    value: object, param: str, *, minimum: int | None = None, maximum: int | None = None
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _type_error(param, "an integer", value)
    _check_range(value, param, minimum, maximum)
    return value


def expect_number(
    value: object, param: str, *, minimum: float | None = None, maximum: float | None = None
) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _type_error(param, "a number", value)
    _check_range(value, param, minimum, maximum)
    return float(value)


def expect_choice[T: str](value: object, param: str, choices: Collection[T]) -> T:
    """A string among ``choices``."""
    text = expect_str(value, param)
    for choice in choices:
        if text == choice:
            return choice
    supported = ", ".join(f"'{choice}'" for choice in choices)
    raise ProtocolError(
        "invalid_value", f"Invalid value: '{text}'. Supported values are: {supported}.", param
    )


def require(obj: Mapping[str, object], key: str, path: str) -> object:
    if key not in obj:
        param = child(path, key)
        raise ProtocolError("invalid_value", f"Missing required parameter: '{param}'.", param)
    return obj[key]


def check_keys(
    obj: Mapping[str, object],
    path: str,
    allowed: Collection[str],
    *,
    unsupported: Mapping[str, str] | None = None,
    beta: Collection[str] = (),
) -> None:
    """Reject any key outside ``allowed``; ``unsupported`` explains GA fields this server lacks."""
    for key in obj:
        if key in allowed:
            continue
        param = child(path, key)
        if unsupported and key in unsupported:
            raise ProtocolError(
                "invalid_value", f"'{param}' is not supported: {unsupported[key]}", param
            )
        if key in beta:
            message = (
                f"Unknown parameter: '{param}'. It belongs to the beta API; send the GA shape."
            )
            raise ProtocolError("invalid_value", message, param)
        raise ProtocolError("invalid_value", f"Unknown parameter: '{param}'.", param)


def _check_range(value: float, param: str, minimum: float | None, maximum: float | None) -> None:
    if minimum is not None and value < minimum:
        raise value_error(param, f"expected a value >= {minimum}, but got {value}.")
    if maximum is not None and value > maximum:
        raise value_error(param, f"expected a value <= {maximum}, but got {value}.")
