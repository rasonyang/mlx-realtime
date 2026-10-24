"""One Realtime session over one WebSocket (ARCHITECTURE §4.2, §6.1, §8).

The session scope is a TaskGroup of four tasks: the reader, the single writer, the actor, and a
lifetime watch (shutdown, maximum duration). Every way a session ends is an exception raised in
that group; ``run`` maps it to the close frame, after a bounded flush of what is still queued.
An exception in one session ends only that session.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass

from starlette.websockets import WebSocket, WebSocketDisconnect

from mlx_realtime import logs
from mlx_realtime.protocol.decode import decode_text_frame, reject_binary_frame
from mlx_realtime.protocol.encode import EventEncoder
from mlx_realtime.server.gateway import Gateway
from mlx_realtime.server.outbox import Outbound, Outbox
from mlx_realtime.session.actor import SessionActor
from mlx_realtime.session.errors import ErrorInfo, SessionFatal, first_leaf
from mlx_realtime.session.facts import Fact, OutputAudioDelta, OutputTranscriptDelta
from mlx_realtime.session.inbox import Inbox
from mlx_realtime.session.output import OutputLedger
from mlx_realtime.session.settings import SessionType, settings_for

logger = logging.getLogger(__name__)


class _ClientGone(Exception):
    pass


class _Stop(Exception):
    def __init__(self, code: int, reason: str) -> None:
        super().__init__(reason)
        self.code = code
        self.reason = reason


@dataclass(frozen=True, slots=True)
class _Close:
    code: int
    reason: str
    error: ErrorInfo | None = None
    flush: bool = True  # write what is still queued before the close frame
    client_gone: bool = False


class Connection:
    def __init__(
        self,
        websocket: WebSocket,
        gateway: Gateway,
        *,
        model: str | None,
        session_type: SessionType = "realtime",
    ) -> None:
        config = gateway.config
        self._ws = websocket
        self._session_id = gateway.ids.session()
        self._write_timeout_s = config.limits.client_write_timeout_ms / 1000
        self._max_duration_s = config.gateway.max_session_s
        settings = settings_for(
            gateway.defaults, session_type, gateway.defaults.model if model is None else model
        )
        self._session_type = session_type
        self._inbox = Inbox(
            max_audio_ms=config.limits.inbox_max_ms,
            max_commands=config.limits.inbox_max_commands,
            input_format=settings.input_format,
        )
        self._outbox = Outbox(config.limits.outbound_queue_events)
        deps = gateway.session_deps()
        self._ledger = OutputLedger(deps.clock)
        self._encoder = EventEncoder()
        self._stopping = asyncio.Event()
        self._stop = _Stop(1001, "server shutting down")
        self._actor = SessionActor(
            session_id=self._session_id,
            conversation_id=gateway.ids.conversation(),
            settings=settings,
            defaults=gateway.defaults,
            deps=deps,
            inbox=self._inbox,
            ledger=self._ledger,
            emit=self._emit,
            wait_for_room=self._outbox.wait_for_room,
        )

    def stop(self, code: int, reason: str) -> None:
        """Ask the session to close (server shutdown)."""
        self._stop = _Stop(code, reason)
        self._stopping.set()

    async def run(self) -> None:
        with logs.bound(session=self._session_id):
            logger.info("session started", extra={"type": self._session_type})
            close = await self._serve()
            await self._close(close)
            logger.info("session closed", extra={"code": close.code, "reason": close.reason})

    async def _serve(self) -> _Close:
        close = _Close(1011, "internal_error")
        try:
            async with asyncio.TaskGroup() as tasks:
                tasks.create_task(self._read())
                tasks.create_task(self._write())
                tasks.create_task(self._actor.run())
                tasks.create_task(self._lifetime())
        except* _ClientGone:
            close = _Close(1000, "client closed", client_gone=True)
        except* _Stop as group:
            stop = first_leaf(group)
            close = _Close(stop.code, stop.reason)
        except* SessionFatal as group:
            fatal = first_leaf(group)
            logger.warning("session failed", extra={"code": fatal.code, "error": str(fatal)})
            close = _Close(
                fatal.close_code,
                fatal.code,
                ErrorInfo("server_error", fatal.code, str(fatal)),
                flush=fatal.code != "slow_client",
            )
        except* Exception as group:
            logger.error("session crashed", exc_info=group.exceptions[0])
            error = ErrorInfo("server_error", "internal_error", "The session failed unexpectedly.")
            close = _Close(1011, "internal_error", error)
        return close

    async def _read(self) -> None:
        while True:
            message = await self._ws.receive()
            if message["type"] == "websocket.disconnect":
                raise _ClientGone
            text: object = message.get("text")
            command = decode_text_frame(text) if isinstance(text, str) else reject_binary_frame()
            self._inbox.put_command(command)

    async def _write(self) -> None:
        while True:
            await self._send(await self._outbox.get())

    async def _lifetime(self) -> None:
        try:
            async with asyncio.timeout(self._max_duration_s):
                await self._stopping.wait()
        except TimeoutError:
            raise _Stop(1000, "maximum session duration reached") from None
        raise self._stop

    async def _send(self, entry: Outbound) -> None:
        delivery = entry.delivery
        if delivery is not None:
            if self._ledger.is_stale(delivery):
                self._ledger.dropped(delivery)
                return  # audio of a cancelled response: flushed, never sent (§6.5 step 4)
            self._ledger.record(delivery)
        try:
            async with asyncio.timeout(self._write_timeout_s):
                await self._ws.send_text(entry.payload)
        except TimeoutError:
            message = "A write to the client stalled."
            raise SessionFatal("slow_client", message, close_code=1008) from None
        except WebSocketDisconnect:
            raise _ClientGone from None

    def _emit(self, fact: Fact) -> None:
        delivery = (
            fact.delivery if isinstance(fact, OutputAudioDelta | OutputTranscriptDelta) else None
        )
        for event in self._encoder.encode(fact):
            self._outbox.put(Outbound(_serialize(event), delivery))

    async def _close(self, close: _Close) -> None:
        """Bounded: flush what is queued, report a fatal error, send the close frame."""
        if close.client_gone:
            return
        try:
            async with asyncio.timeout(self._write_timeout_s):
                if close.flush:
                    for entry in self._outbox.drain():
                        await self._send(entry)
                if close.error is not None:
                    await self._ws.send_text(_serialize(self._encoder.error(close.error)))
                await self._ws.close(close.code, close.reason)
        except (TimeoutError, WebSocketDisconnect, SessionFatal, _ClientGone):
            logger.info("close handshake skipped: the client is gone or not reading")


def _serialize(event: dict[str, object]) -> str:
    return json.dumps(event, ensure_ascii=False, separators=(",", ":"))
