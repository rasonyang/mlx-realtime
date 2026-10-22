"""Backend warmup and readiness (ARCHITECTURE §2: startup, warmup, readiness).

Each backend is warmed once it answers, all of them at once; one that is not up yet (still
starting, or loading its model) is retried with exponential backoff. The gateway admits
sessions from the first moment all of them were warm; later failures surface per response and
on ``/readyz``.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping

from mlx_realtime.providers.base import Backend, ProviderError, Readiness

logger = logging.getLogger(__name__)

_PROBE_TIMEOUT_S = 1.0


class ReadinessMonitor:
    def __init__(
        self,
        backends: Mapping[str, Backend],
        *,
        retry_s: float = 0.5,
        max_retry_s: float = 5.0,
    ) -> None:
        self._backends = dict(backends)
        self._retry_s = retry_s
        self._max_retry_s = max_retry_s
        self._warm: set[str] = set()
        self.ever_ready = False

    async def warm_up(self) -> None:
        """Warm every backend, retrying each until it succeeds."""
        async with asyncio.TaskGroup() as tasks:
            for name, backend in self._backends.items():
                tasks.create_task(self._warm_until_ready(name, backend))
        self.ever_ready = True
        logger.info("all backends warm", extra={"backends": sorted(self._backends)})

    async def report(self) -> tuple[bool, dict[str, dict[str, object]]]:
        """Overall readiness and per-backend detail, probed now."""
        details: dict[str, dict[str, object]] = {}
        for name, backend in self._backends.items():
            state = await _probe(backend)
            details[name] = {
                "ready": state.ready,
                "warm": name in self._warm,
                "detail": state.detail,
            }
        ready = self.ever_ready and all(detail["ready"] for detail in details.values())
        return ready, details

    async def _warm_until_ready(self, name: str, backend: Backend) -> None:
        delay = self._retry_s
        while not await _warm_one(name, backend):
            await asyncio.sleep(delay)
            delay = min(delay * 2, self._max_retry_s)
        self._warm.add(name)
        logger.info("backend warm", extra={"backend": name})


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


async def _probe(backend: Backend) -> Readiness:
    try:
        async with asyncio.timeout(_PROBE_TIMEOUT_S):
            return await backend.ready()
    except (ProviderError, TimeoutError) as error:
        return Readiness(ready=False, detail=str(error) or type(error).__name__)
