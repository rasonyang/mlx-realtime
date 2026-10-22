"""The session's clock: the event loop's in service, a fake one in tests (ARCHITECTURE §11)."""

from __future__ import annotations

import asyncio
from typing import Protocol


class Clock(Protocol):
    def now(self) -> float:
        """Seconds on a monotonic clock."""
        ...

    async def sleep(self, seconds: float) -> None: ...


class LoopClock:
    """The running event loop's clock."""

    def now(self) -> float:
        return asyncio.get_running_loop().time()

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)
