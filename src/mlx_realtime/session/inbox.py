"""The actor's FIFO inbox: client commands and task signals, in arrival order (ARCHITECTURE §4.4).

Client input is bounded by ``max_audio_ms`` of queued audio plus ``max_commands`` other
commands. Exceeding either bound is fatal for the session: audio is never dropped silently.
Signals from session-owned tasks are not bounded here; their producers bound themselves.

Queued audio is charged by its duration in the input format it will be decoded in: the format
of the last queued ``session.update`` that sets one, until the actor has handled every such
update (then the session's effective format, which a rejected update leaves unchanged).
"""

from __future__ import annotations

import asyncio
from collections import deque

from mlx_realtime.audio.formats import PCM, AudioFormat
from mlx_realtime.session.commands import AppendAudio, Command, UpdateSession
from mlx_realtime.session.errors import SessionFatal
from mlx_realtime.session.settings import UNSET
from mlx_realtime.session.signals import Signal


class Inbox:
    def __init__(
        self, *, max_audio_ms: int, max_commands: int, input_format: AudioFormat = PCM
    ) -> None:
        self._max_audio_us = max_audio_ms * 1000
        self._max_commands = max_commands
        # Each entry keeps its charge against the client bounds: (message, audio µs, command).
        self._queue: deque[tuple[Command | Signal, int, bool]] = deque()
        self._audio_us = 0
        self._commands = 0
        self._format = input_format  # what queued appends will be decoded as
        self._format_updates = 0  # queued session.updates that set the input format
        self._ready = asyncio.Event()

    def put_command(self, command: Command) -> None:
        if isinstance(command, AppendAudio):
            charge = len(command.audio) * 1_000_000 // self._format.bytes_per_second
            if self._audio_us + charge > self._max_audio_us:
                raise SessionFatal("input_queue_overflow", "Too much input audio is queued.")
            self._audio_us += charge
            self._push((command, charge, False))
            return
        if self._commands >= self._max_commands:
            raise SessionFatal("input_queue_overflow", "Too many client events are queued.")
        if isinstance(command, UpdateSession) and command.patch.input_format is not UNSET:
            self._format = command.patch.input_format
            self._format_updates += 1
        self._commands += 1
        self._push((command, 0, True))

    def input_format_settled(self, effective: AudioFormat) -> None:
        """The actor handled a ``session.update`` that set the input format (applied or
        rejected); ``effective`` is the session's input format now."""
        self._format_updates = max(self._format_updates - 1, 0)
        if self._format_updates == 0:
            self._format = effective

    def put_signal(self, signal: Signal) -> None:
        self._push((signal, 0, False))

    async def get(self) -> Command | Signal:
        while not self._queue:
            self._ready.clear()
            await self._ready.wait()
        message, audio_us, is_command = self._queue.popleft()
        self._audio_us -= audio_us
        self._commands -= is_command
        return message

    def _push(self, entry: tuple[Command | Signal, int, bool]) -> None:
        self._queue.append(entry)
        self._ready.set()
