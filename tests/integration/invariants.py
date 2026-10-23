"""Causal invariants of a server event stream (ARCHITECTURE §11)."""

from __future__ import annotations

import base64
from collections.abc import Sequence

from tests.support import Event

_DELTAS = {
    "response.output_audio.delta": "response.output_audio.done",
    "response.output_audio_transcript.delta": "response.output_audio_transcript.done",
    "response.output_text.delta": "response.output_text.done",
}
_CLOSING_ORDER = [
    "response.output_audio.done",
    "response.output_audio_transcript.done",
    "response.output_text.done",
    "response.content_part.done",
    "response.output_item.done",
]


def assert_causal(events: Sequence[Event], *, ended: bool = True) -> None:
    """Check ordering and completeness of every response, item and stream in ``events``.

    - ``response.created`` precedes everything that names the response; nothing names it after
      its ``response.done``.
    - Deltas follow their content part's ``added`` and precede their stream's ``done``; each
      stream is closed exactly once, and its deltas add up to the final text.
    - An item's closing events come in the GA order, before ``response.done``.
    - Items are added before they are done; audio is whole samples of the response's output
      format (16-bit for ``audio/pcm``; any byte count for G.711).

    ``ended=False``: the session was cut off (a disconnect), so responses may stay open.
    """
    created: set[str | None] = set()
    finished: set[str | None] = set()
    parts_open: set[str | None] = set()
    closed: dict[tuple[str | None, str], int] = {}
    deltas: dict[tuple[str | None, str], str] = {}
    items_added: set[str | None] = set()
    closing_seen: dict[str | None, list[str]] = {}
    widths: dict[str | None, int] = {}  # bytes per sample of each response's audio
    for index, event in enumerate(events):
        kind: str = event["type"]
        where = f"event {index} ({kind})"
        response_id: str | None = event.get("response_id") or (
            event["response"]["id"] if kind in ("response.created", "response.done") else None
        )
        if kind == "response.created":
            created.add(response_id)
            audio_format = event["response"]["audio"]["output"]["format"]["type"]
            widths[response_id] = 2 if audio_format == "audio/pcm" else 1
        elif response_id is not None:
            assert response_id in created, f"{where}: response not created yet"
            assert response_id not in finished, f"{where}: response already done"
        item = event.get("item")
        item_id: str | None = event.get("item_id") or (
            item["id"] if isinstance(item, dict) else None
        )
        if kind == "response.content_part.added":
            parts_open.add(item_id)
        if kind in _DELTAS:
            assert item_id in parts_open, f"{where}: delta before its content part"
            assert (item_id, _DELTAS[kind]) not in closed, f"{where}: delta after done"
            if kind == "response.output_audio.delta":
                width = widths.get(response_id, 2)
                assert len(base64.b64decode(event["delta"])) % width == 0, f"{where}: half a sample"
            else:
                deltas[(item_id, kind)] = deltas.get((item_id, kind), "") + event["delta"]
        if kind in _CLOSING_ORDER:
            closed[(item_id, kind)] = closed.get((item_id, kind), 0) + 1
            assert closed[(item_id, kind)] == 1, f"{where}: closed twice"
            seen = closing_seen.setdefault(item_id, [])
            assert all(
                _CLOSING_ORDER.index(previous) < _CLOSING_ORDER.index(kind) for previous in seen
            ), f"{where}: closing order {[*seen, kind]}"
            seen.append(kind)
        if kind == "response.output_audio_transcript.done":
            spoken = deltas.get((item_id, "response.output_audio_transcript.delta"), "")
            assert event["transcript"] == spoken, f"{where}: transcript differs from its deltas"
        if kind == "response.output_text.done":
            written = deltas.get((item_id, "response.output_text.delta"), "")
            assert event["text"] == written, f"{where}: text differs from its deltas"
        if kind == "conversation.item.added":
            items_added.add(item_id)
        if kind == "conversation.item.done":
            assert item_id in items_added, f"{where}: done before added"
        if kind == "response.done":
            finished.add(response_id)
            for output in event["response"]["output"]:
                if output["type"] == "message":
                    assert (output["id"], "response.output_item.done") in closed, (
                        f"{where}: output item {output['id']} never closed"
                    )
    if not ended:
        return
    assert created == finished, f"responses never finished: {created - finished}"
    for (item_id, kind), text in deltas.items():
        closing = _DELTAS[kind]
        assert (item_id, closing) in closed, f"{kind} stream of {item_id} never closed ({text!r})"
