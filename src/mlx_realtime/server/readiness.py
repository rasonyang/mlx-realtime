"""Backend warmup and readiness (ARCHITECTURE §2: startup, warmup, readiness; §8: backend health).

Each backend is warmed once it answers, all of them at once; one that is not up yet (still
starting, or loading its model) is retried with exponential backoff. The gateway admits
sessions from the first moment all of them were warm; later failures surface per response and
on ``/readyz``.

After that the monitor keeps them warm. Every ``watch_s`` it probes each warm backend; one that
answers "not ready" (unreachable, or its model no longer loaded) has gone down or been restarted
by its supervisor, and is warmed again, with the same retries, until it is ready: a restarted
``mlx_audio.server`` loads a model only when a request asks for it, and the first caller turn
must not be the one that waits for it. A probe that times out is a busy backend, not a lost one,
and changes nothing.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping

from mlx_realtime.providers.base import Backend, ProviderError, Readiness

logger = logging.getLogger(__name__)

PROBE_TIMEOUT_S = 1.0
WATCH_S = 5.0


class ReadinessMonitor:
    def __init__(
        self,
        backends: Mapping[str, Backend],
        *,
        retry_s: float = 0.5,
        max_retry_s: float = 5.0,
        watch_s: float | None = WATCH_S,
        probe_timeout_s: float = PROBE_TIMEOUT_S,
    ) -> None:
        self._backends = dict(backends)
        self._retry_s = retry_s
        self._max_retry_s = max_retry_s
        self._watch_s = watch_s
        self._probe_timeout_s = probe_timeout_s
        self._warm: set[str] = set()
        self.ever_ready = False

    async def run(self) -> None:
        """Warm every backend, then keep them warm (the module's docstring) until cancelled;
        with ``watch_s`` None, only warm them."""
        await self.warm_up()
        if self._watch_s is None:
            return
        async with asyncio.TaskGroup() as tasks:
            while True:
                await asyncio.sleep(self._watch_s)
                for name, backend in self._backends.items():
                    if name not in self._warm:
                        continue  # being warmed again
                    try:
                        state = await self._answer(backend)
                    except Exception:
                        logger.exception("backend probe failed", extra={"backend": name})
                        continue
                    if state is None or state.ready:
                        continue
                    self._warm.discard(name)
                    logger.warning(
                        "backend lost; warming it again once it answers",
                        extra={"backend": name, "detail": state.detail},
                    )
                    tasks.create_task(self._warm_until_ready(name, backend))

    async def warm_up(self) -> None:
        """Warm every backend, retrying each until it succeeds."""
        async with asyncio.TaskGroup() as tasks:
            for name, backend in self._backends.items():
                tasks.create_task(self._warm_until_ready(name, backend))
        self.ever_ready = True
        logger.info("all backends warm", extra={"backends": sorted(self._backends)})

    async def report(self) -> tuple[bool, dict[str, dict[str, object]]]:
        """Overall readiness and per-backend detail, probed now. Ready: every backend answers
        ready and is warm (a backend being warmed again after a restart is not)."""
        details: dict[str, dict[str, object]] = {}
        for name, backend in self._backends.items():
            state = await self._answer(backend)
            if state is None:
                state = Readiness(ready=False, detail="TimeoutError")
            details[name] = {
                "ready": state.ready,
                "warm": name in self._warm,
                "detail": state.detail,
            }
        ready = self.ever_ready and all(
            detail["ready"] and detail["warm"] for detail in details.values()
        )
        return ready, details

    async def _warm_until_ready(self, name: str, backend: Backend) -> None:
        delay = self._retry_s
        while not await _warm_one(name, backend):
            await asyncio.sleep(delay)
            delay = min(delay * 2, self._max_retry_s)
        self._warm.add(name)
        logger.info("backend warm", extra={"backend": name})

    async def _answer(self, backend: Backend) -> Readiness | None:
        """The backend's readiness, or None when it did not answer within the probe timeout."""
        try:
            async with asyncio.timeout(self._probe_timeout_s):
                return await backend.ready()
        except ProviderError as error:
            return Readiness(ready=False, detail=str(error) or type(error).__name__)
        except TimeoutError:
            return None


async def _warm_one(name: str, backend: Backend) -> bool:
    try:
        await backend.warmup()
        state = await backend.ready()
    except ProviderError as error:
        logger.warning("backend warmup failed", extra={"backend": name, "error": str(error)})
        return False
    except Exception:
        logger.exception("backend warmup failed", extra={"backend": name})
        return False
    if not state.ready:
        logger.warning("backend not ready", extra={"backend": name, "detail": state.detail})
    return state.ready
