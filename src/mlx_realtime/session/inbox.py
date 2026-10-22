"""The actor's FIFO inbox: client commands and task signals, in arrival order (ARCHITECTURE §4.4).

Client input is bounded by ``max_audio_ms`` of queued audio plus ``max_commands`` other
commands. Exceeding either bound is fatal for the session: audio is never dropped silently.
Signals from session-owned tasks are not bounded here; their producers bound themselves.
"""

from __future__ import annotations

import asyncio
from collections import deque

from mlx_realtime.audio.pcm import WIRE_RATE, ms_to_bytes
from mlx_realtime.session.commands import AppendAudio, Command
from mlx_realtime.session.errors import SessionFatal
from mlx_realtime.session.signals import Signal


class Inbox:
    def __init__(self, *, max_audio_ms: int, max_commands: int) -> None:
        self._max_audio_bytes = ms_to_bytes(max_audio_ms, WIRE_RATE)
        self._max_commands = max_commands
        # Each entry keeps its charge against the client bounds: (message, audio bytes, command).
        self._queue: deque[tuple[Command | Signal, int, bool]] = deque()
        self._audio_bytes = 0
        self._commands = 0
        self._ready = asyncio.Event()

    def put_command(self, command: Command) -> None:
        if isinstance(command, AppendAudio):
            if self._audio_bytes + len(command.pcm) > self._max_audio_bytes:
                raise SessionFatal("input_queue_overflow", "Too much input audio is queued.")
            self._audio_bytes += len(command.pcm)
            self._push((command, len(command.pcm), False))
            return
        if self._commands >= self._max_commands:
            raise SessionFatal("input_queue_overflow", "Too many client events are queued.")
        self._commands += 1
        self._push((command, 0, True))

    def put_signal(self, signal: Signal) -> None:
        self._push((signal, 0, False))

    async def get(self) -> Command | Signal:
        while not self._queue:
            self._ready.clear()
            await self._ready.wait()
        message, audio_bytes, is_command = self._queue.popleft()
        self._audio_bytes -= audio_bytes
        self._commands -= is_command
        return message

    def _push(self, entry: tuple[Command | Signal, int, bool]) -> None:
        self._queue.append(entry)
        self._ready.set()
