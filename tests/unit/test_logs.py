from __future__ import annotations

import asyncio
import json
import logging
import sys

from mlx_realtime import logs


def render(message: str, **extra: object) -> dict[str, object]:
    record = logging.LogRecord(
        "mlx_realtime.test", logging.WARNING, __file__, 1, message, None, None
    )
    record.__dict__.update(extra)
    return json.loads(logs.JsonFormatter().format(record))


def test_records_are_json_lines_with_extras() -> None:
    line = render("transcript ready", item="item_1", sent_ms=120)
    assert line["level"] == "warning"
    assert line["logger"] == "mlx_realtime.test"
    assert line["msg"] == "transcript ready"
    assert (line["item"], line["sent_ms"]) == ("item_1", 120)
    assert str(line["ts"]).endswith("+00:00")
    assert "session_id" not in line


def test_bound_ids_appear_and_are_restored() -> None:
    with logs.bound(session="sess_1", turn="turn_1"):
        with logs.bound(response="resp_1", item="item_1"):
            inner = render("x")
        outer = render("y")
    after = render("z")
    assert {key: inner[key] for key in ("session_id", "turn_id", "response_id", "item_id")} == {
        "session_id": "sess_1",
        "turn_id": "turn_1",
        "response_id": "resp_1",
        "item_id": "item_1",
    }
    assert (outer["session_id"], "response_id" in outer) == ("sess_1", False)
    assert "session_id" not in after


async def test_tasks_inherit_the_ids_bound_when_they_start() -> None:
    async def child() -> dict[str, object]:
        await asyncio.sleep(0)
        return render("from a task")

    with logs.bound(session="sess_2", response="resp_2"):
        task = asyncio.create_task(child())
    line = await task
    assert (line["session_id"], line["response_id"]) == ("sess_2", "resp_2")


def test_exceptions_are_included() -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        record = logging.LogRecord("t", logging.ERROR, __file__, 1, "failed", None, None)
        record.exc_info = sys.exc_info()
    line = json.loads(logs.JsonFormatter().format(record))
    assert "ValueError: boom" in line["exc"]
