"""What the writer actually handed to the socket, per assistant item (ARCHITECTURE §4.2, §6.5).

Audio and transcript deltas of an audio response are queued with a ``Delivery``: the response
generation and a receipt. When a response is cancelled the actor marks its generation stale,
the writer discards the deltas still queued for it, and the ledger tells the actor exactly
which audio and transcript reached the client, and when its first audio did (D5). A listener
(the session's echo reference, D17) hears of every audio delta sent or discarded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from mlx_realtime.audio.pcm import WIRE_RATE, bytes_to_ms
from mlx_realtime.session.clock import Clock


@dataclass(frozen=True, slots=True)
class Delivery:
    generation: int
    item_id: str
    segment_index: int
    kind: Literal["audio", "transcript"]
    audio_bytes: int = 0
    pcm: bytes = b""  # the audio itself (audio deliveries)


class OutputListener(Protocol):
    def audio_sent(self, item_id: str, generation: int, pcm: bytes) -> None: ...
    def audio_dropped(self, generation: int) -> None: ...


class OutputLedger:
    """Shared by the actor (reads, cancels generations) and the writer (records)."""

    def __init__(self, clock: Clock | None = None) -> None:
        self._clock = clock
        self._stale: set[int] = set()
        self._audio_bytes: dict[str, int] = {}
        self._first_audio: dict[str, float] = {}
        self._transcripts: dict[str, set[int]] = {}
        self.listener: OutputListener | None = None

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
            if self._clock is not None:
                self._first_audio.setdefault(delivery.item_id, self._clock.now())
            if self.listener is not None:
                self.listener.audio_sent(delivery.item_id, delivery.generation, delivery.pcm)
        else:
            self._transcripts.setdefault(delivery.item_id, set()).add(delivery.segment_index)

    def dropped(self, delivery: Delivery) -> None:
        """The writer discards ``delivery`` (its response was cancelled)."""
        if delivery.kind == "audio" and self.listener is not None:
            self.listener.audio_dropped(delivery.generation)

    def first_audio_at(self, item_id: str) -> float | None:
        """When the item's first audio delta was handed to the socket (the clock's time)."""
        return self._first_audio.get(item_id)

    def sent_ms(self, item_id: str) -> int:
        """Milliseconds of the item's audio handed to the socket."""
        return bytes_to_ms(self._audio_bytes.get(item_id, 0), WIRE_RATE)

    def transcript_sent(self, item_id: str, segment_index: int) -> bool:
        return segment_index in self._transcripts.get(item_id, set())
