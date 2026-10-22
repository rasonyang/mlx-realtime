"""What the writer actually handed to the socket, per assistant item (ARCHITECTURE §4.2, §6.5).

Audio and transcript deltas of an audio response are queued with a ``Delivery``: the response
generation and a receipt. When a response is cancelled the actor marks its generation stale,
the writer discards the deltas still queued for it, and the ledger tells the actor exactly
which audio and transcript reached the client.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mlx_realtime.audio.pcm import WIRE_RATE, bytes_to_ms


@dataclass(frozen=True, slots=True)
class Delivery:
    generation: int
    item_id: str
    segment_index: int
    kind: Literal["audio", "transcript"]
    audio_bytes: int = 0


class OutputLedger:
    """Shared by the actor (reads, cancels generations) and the writer (records)."""

    def __init__(self) -> None:
        self._stale: set[int] = set()
        self._audio_bytes: dict[str, int] = {}
        self._transcripts: dict[str, set[int]] = {}

    def discard(self, generation: int) -> None:
        """Deltas of ``generation`` still queued must not be written."""
        self._stale.add(generation)

    def is_stale(self, delivery: Delivery) -> bool:
        return delivery.generation in self._stale

    def record(self, delivery: Delivery) -> None:
        """The writer hands ``delivery``'s event to the socket."""
        if delivery.kind == "audio":
            total = self._audio_bytes.get(delivery.item_id, 0)
            self._audio_bytes[delivery.item_id] = total + delivery.audio_bytes
        else:
            self._transcripts.setdefault(delivery.item_id, set()).add(delivery.segment_index)

    def sent_ms(self, item_id: str) -> int:
        """Milliseconds of the item's audio handed to the socket."""
        return bytes_to_ms(self._audio_bytes.get(item_id, 0), WIRE_RATE)

    def transcript_sent(self, item_id: str, segment_index: int) -> bool:
        return segment_index in self._transcripts.get(item_id, set())
