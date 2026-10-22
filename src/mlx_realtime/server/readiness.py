"""Backend warmup and readiness (ARCHITECTURE §2: startup, warmup, readiness).

Each backend is warmed once it answers. The gateway admits sessions after the first moment all
of them were ready; later failures surface per response and on ``/readyz``.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping

from mlx_realtime.providers.base import Backend, ProviderError, Readiness

logger = logging.getLogger(__name__)

_PROBE_TIMEOUT_S = 1.0


class ReadinessMonitor:
    def __init__(self, backends: Mapping[str, Backend], *, retry_s: float = 1.0) -> None:
        self._backends = dict(backends)
        self._retry_s = retry_s
        self._warm: set[str] = set()
        self.ever_ready = False

    async def warm_up(self) -> None:
        """Warm every backend, retrying the ones that are not ready yet."""
        while True:
            for name, backend in self._backends.items():
                if name not in self._warm and await self._warm_one(name, backend):
                    self._warm.add(name)
            if len(self._warm) == len(self._backends):
                break
            await asyncio.sleep(self._retry_s)
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

    @staticmethod
    async def _warm_one(name: str, backend: Backend) -> bool:
        try:
            await backend.warmup()
            state = await backend.ready()
        except ProviderError as error:
            logger.warning("backend warmup failed", extra={"backend": name, "error": str(error)})
            return False
        return state.ready


async def _probe(backend: Backend) -> Readiness:
    try:
        async with asyncio.timeout(_PROBE_TIMEOUT_S):
            return await backend.ready()
    except (ProviderError, TimeoutError) as error:
        return Readiness(ready=False, detail=str(error) or type(error).__name__)
