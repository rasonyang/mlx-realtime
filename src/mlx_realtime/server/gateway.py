"""Process-wide gateway state shared by every connection: configuration, providers, the speech
and end-of-turn detectors, the session defaults, admission and readiness.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from mlx_realtime.config import Config, ConfigError, TurnConfig
from mlx_realtime.protocol.fields import ProtocolError
from mlx_realtime.protocol.ids import IdFactory
from mlx_realtime.protocol.session import decode_session_defaults
from mlx_realtime.providers.base import Backend, Readiness
from mlx_realtime.providers.registry import Providers, build_providers
from mlx_realtime.server.readiness import ReadinessMonitor
from mlx_realtime.session.actor import SessionDeps
from mlx_realtime.session.settings import SessionSettings
from mlx_realtime.vad.base import SpeechDetector, TurnEndDetector
from mlx_realtime.vad.executors import VadRuntime
from mlx_realtime.vad.scripted import ScriptedSpeech, ScriptedTurnEnd

logger = logging.getLogger(__name__)


class Stoppable(Protocol):
    def stop(self, code: int, reason: str) -> None: ...


class Admission:
    """At most ``max_sessions`` sessions at once (ARCHITECTURE §8)."""

    def __init__(self, max_sessions: int) -> None:
        self._max_sessions = max_sessions
        self._active: set[Stoppable] = set()
        self._idle = asyncio.Event()
        self._idle.set()

    @property
    def active(self) -> int:
        return len(self._active)

    @property
    def full(self) -> bool:
        return len(self._active) >= self._max_sessions

    def enter(self, session: Stoppable) -> None:
        """Register a session; check ``full`` first, with no await in between."""
        self._active.add(session)
        self._idle.clear()

    def leave(self, session: Stoppable) -> None:
        self._active.discard(session)
        if not self._active:
            self._idle.set()

    async def stop_all(self, code: int, reason: str, timeout_s: float) -> None:
        """Ask every session to close, and wait (bounded) until they have."""
        for session in list(self._active):
            session.stop(code, reason)
        try:
            async with asyncio.timeout(timeout_s):
                await self._idle.wait()
        except TimeoutError:
            logger.warning(
                "sessions still open after shutdown request", extra={"count": self.active}
            )


class Gateway:
    """``providers`` and the detector factories default to what ``config`` selects; tests pass
    their own."""

    def __init__(
        self,
        config: Config,
        providers: Providers | None = None,
        *,
        speech_detector: Callable[[], SpeechDetector] | None = None,
        turn_end_detector: Callable[[], TurnEndDetector | None] | None = None,
    ) -> None:
        try:
            self.defaults: SessionSettings = decode_session_defaults(config.session_defaults)
        except ProtocolError as error:
            raise ConfigError(f"invalid configuration:\n  {error.param}: {error.message}") from None
        self.config = config
        self.providers = providers or build_providers(
            config.providers, warmup_instructions=self.defaults.instructions
        )
        self.ids = IdFactory()
        self.admission = Admission(config.gateway.max_sessions)
        backends = self.providers.backends()
        if speech_detector is None and config.turn.vad.type == "silero":
            runtime = vad_runtime(config.turn)
            backends["vad"] = _VadBackend(runtime)
            speech_detector = runtime.speech_detector
            if turn_end_detector is None:
                turn_end_detector = runtime.turn_end_detector
        if turn_end_detector is None:
            smart_turn = config.turn.smart_turn.enabled
            turn_end_detector = ScriptedTurnEnd if smart_turn else _no_turn_end_detector
        self._speech_detector = speech_detector or ScriptedSpeech
        self._turn_end_detector = turn_end_detector
        self._backends = backends
        self.readiness = ReadinessMonitor(backends)

    def authorized(self, authorization: str | None) -> bool:
        """Bearer auth with ``gateway.api_key``; an empty key disables it."""
        key = self.config.gateway.api_key
        if not key:
            return True
        expected = f"Bearer {key}".encode()
        return authorization is not None and hmac.compare_digest(authorization.encode(), expected)

    def session_deps(self) -> SessionDeps:
        return SessionDeps(
            llm=self.providers.llm,
            asr=self.providers.asr,
            tts=self.providers.tts,
            speech=self._speech_detector(),
            turn_end=self._turn_end_detector(),
            ids=self.ids,
            limits=self.config.limits,
            turn=self.config.turn,
            segmenter=self.config.text.segmenter,
            max_delta_ms=self.config.output.max_delta_ms,
        )

    async def aclose(self) -> None:
        for backend in self._backends.values():
            await backend.aclose()


def vad_runtime(turn: TurnConfig) -> VadRuntime:
    """Silero, and Smart Turn when enabled, on their CPU executors (the model files must exist)."""
    smart_turn = turn.smart_turn
    return VadRuntime(
        silero_model=Path(turn.vad.model),
        reset_interval_s=turn.vad.model_reset_interval_s,
        smart_turn_model=Path(smart_turn.model) if smart_turn.enabled else None,
        workers=smart_turn.workers,
        intra_op_threads=smart_turn.intra_op_threads,
    )


class _VadBackend(Backend):
    """The CPU detectors as a readiness backend: warmed with one inference each."""

    def __init__(self, runtime: VadRuntime) -> None:
        self._runtime = runtime

    async def warmup(self) -> None:
        await self._runtime.warmup()

    async def ready(self) -> Readiness:
        return Readiness(ready=True, detail="cpu")

    async def aclose(self) -> None:
        await self._runtime.aclose()


def _no_turn_end_detector() -> TurnEndDetector | None:
    return None
