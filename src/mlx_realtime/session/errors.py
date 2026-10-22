"""Errors as the client sees them, and the conditions that end a session (ARCHITECTURE §8)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast


@dataclass(frozen=True, slots=True)
class ErrorInfo:
    """An ``error`` event's payload; ``event_id`` names the client event that caused it."""

    type: Literal["invalid_request_error", "server_error"]
    code: str | None
    message: str
    param: str | None = None
    event_id: str | None = None


def first_leaf[E: Exception](group: ExceptionGroup[E]) -> E:
    """The first exception in ``group``, looking through nested groups (TaskGroups nest them)."""
    first = group.exceptions[0]
    while isinstance(first, ExceptionGroup):
        first = cast(ExceptionGroup[E], first).exceptions[0]
    return first


class SessionFatal(Exception):
    """Ends the session: the client gets an ``error`` with ``code``, then a close frame."""

    def __init__(self, code: str, message: str, *, close_code: int = 1011) -> None:
        super().__init__(message)
        self.code = code
        self.close_code = close_code
