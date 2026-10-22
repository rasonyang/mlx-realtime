"""JSON-lines logging on stdout, with the ids of the current session, turn, response and item.

The ids live in context variables. Tasks inherit the context they were created in, so a task
started while a response id is bound logs that id without passing it around.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime

session_id: ContextVar[str | None] = ContextVar("session_id", default=None)
turn_id: ContextVar[str | None] = ContextVar("turn_id", default=None)
response_id: ContextVar[str | None] = ContextVar("response_id", default=None)
item_id: ContextVar[str | None] = ContextVar("item_id", default=None)

_CONTEXT = (
    ("session_id", session_id),
    ("turn_id", turn_id),
    ("response_id", response_id),
    ("item_id", item_id),
)

# Attributes every LogRecord has; anything else on a record came from `extra=`.
_STANDARD_ATTRS = frozenset(
    vars(logging.LogRecord("", logging.INFO, "", 0, "", None, None)).keys() | {"message", "asctime"}
)


class JsonFormatter(logging.Formatter):
    """One JSON object per record: time, level, logger, message, bound ids and extras."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for name, var in _CONTEXT:
            value = var.get()
            if value is not None:
                payload[name] = value
        payload.update(
            (key, value)
            for key, value in vars(record).items()
            if key not in _STANDARD_ATTRS and not key.startswith("_")
        )
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str) -> None:
    """Send every logger, uvicorn's included, through one JSON handler on stdout."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True


@contextmanager
def bound(
    *,
    session: str | None = None,
    turn: str | None = None,
    response: str | None = None,
    item: str | None = None,
) -> Generator[None]:
    """Bind the given ids for the duration of the block; ``None`` leaves an id unchanged."""
    tokens = [
        var.set(value)
        for var, value in (
            (session_id, session),
            (turn_id, turn),
            (response_id, response),
            (item_id, item),
        )
        if value is not None
    ]
    try:
        yield
    finally:
        for token in reversed(tokens):
            token.var.reset(token)
