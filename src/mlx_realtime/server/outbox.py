"""The queue in front of a session's single writer (ARCHITECTURE §4.4).

A backlog of more than ``limit`` events means the client is not reading: the session ends.
Producers of bulk output (the TTS stage) wait for room below half the limit before adding more,
so a fast pipeline never trips the limit on its own.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass

from mlx_realtime.session.errors import SessionFatal
from mlx_realtime.session.output import Delivery


@dataclass(frozen=True, slots=True)
class Outbound:
    payload: str  # one serialized server event
    delivery: Delivery | None  # audio-response deltas: discarded if their response was cancelled


class Outbox:
    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._low_water = limit // 2
        self._queue: deque[Outbound] = deque()
        self._ready = asyncio.Event()
        self._room = asyncio.Event()
        self._room.set()

    def put(self, entry: Outbound) -> None:
        if len(self._queue) >= self._limit:
            raise SessionFatal(
                "slow_client", "The client is not reading its events.", close_code=1008
            )
        self._queue.append(entry)
        self._ready.set()
        if len(self._queue) >= self._low_water:
            self._room.clear()

    async def get(self) -> Outbound:
        while not self._queue:
            self._ready.clear()
            await self._ready.wait()
        entry = self._queue.popleft()
        if len(self._queue) < self._low_water:
            self._room.set()
        return entry

    async def wait_for_room(self) -> None:
        """Yield once, so the actor and the writer can catch up, then wait while the backlog is
        at or above half the limit."""
        await asyncio.sleep(0)
        await self._room.wait()

    def drain(self) -> list[Outbound]:
        """Take everything still queued (the closing flush)."""
        entries = list(self._queue)
        self._queue.clear()
        self._room.set()
        return entries
