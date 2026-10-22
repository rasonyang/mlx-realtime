"""HTTP plumbing shared by the backend providers (ARCHITECTURE §4.4, §5.2).

- ``RequestClock`` keeps one request's deadlines: the first item (first token or transcript)
  and the total are absolute; the idle deadline restarts at every item. Expiry raises
  ``ProviderTimeout``. Connecting is bounded by the client's own connect timeout.
- ``Slots`` bounds the requests in flight per backend; waiting longer than the queue timeout
  raises ``ProviderError("overloaded")``.
- Transport failures and HTTP statuses map to ``ProviderError`` kinds.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx2

from mlx_realtime.config import HttpTimeouts
from mlx_realtime.providers.base import ProviderError, ProviderTimeout, TimeoutStage

_DETAIL_CHARS = 300


@dataclass(frozen=True, slots=True)
class Deadlines:
    connect_s: float
    first_item_s: float
    idle_s: float
    total_s: float

    @classmethod
    def of(cls, timeouts: HttpTimeouts) -> Deadlines:
        return cls(
            timeouts.connect_ms / 1000,
            timeouts.first_item_ms / 1000,
            timeouts.idle_ms / 1000,
            timeouts.total_ms / 1000,
        )

    @classmethod
    def warmup(cls, timeouts: HttpTimeouts) -> Deadlines:
        """A warmup may wait for the backend to load its model; only connecting stays short."""
        patient = timeouts.warmup_ms / 1000
        return cls(timeouts.connect_ms / 1000, patient, patient, patient)


class RequestClock:
    def __init__(self, deadlines: Deadlines) -> None:
        now = asyncio.get_running_loop().time()
        self._deadlines = deadlines
        self._first_item_at = now + deadlines.first_item_s
        self._total_at = now + deadlines.total_s
        self._got_item = False

    def item(self) -> None:
        """An item arrived: from now on each wait is bounded by the idle deadline."""
        self._got_item = True

    @asynccontextmanager
    async def waiting(self) -> AsyncGenerator[None]:
        """Bound one wait for the backend: until the first item, then until the next one."""
        stage: TimeoutStage = "idle" if self._got_item else "first_item"
        stage_at = (
            asyncio.get_running_loop().time() + self._deadlines.idle_s
            if self._got_item
            else self._first_item_at
        )
        if self._total_at < stage_at:
            stage, stage_at = "total", self._total_at
        timeout = asyncio.timeout_at(stage_at)
        try:
            async with timeout:
                yield
        except TimeoutError:
            if timeout.expired():
                raise ProviderTimeout(stage) from None
            raise


class Slots:
    """At most ``limit`` requests in flight to one backend."""

    def __init__(self, limit: int, queue_timeout_s: float) -> None:
        self._semaphore = asyncio.Semaphore(limit)
        self._queue_timeout_s = queue_timeout_s

    @asynccontextmanager
    async def slot(self) -> AsyncGenerator[None]:
        try:
            async with asyncio.timeout(self._queue_timeout_s):
                await self._semaphore.acquire()
        except TimeoutError:
            raise ProviderError(
                "overloaded", "too many requests in flight", retryable=True
            ) from None
        try:
            yield
        finally:
            self._semaphore.release()


def new_client(
    base_url: str, connect_s: float, transport: httpx2.AsyncBaseTransport | None = None
) -> httpx2.AsyncClient:
    """A client for one internal backend: no proxies from the environment, and no timeouts but
    the connect timeout (the request clock bounds everything else)."""
    return httpx2.AsyncClient(
        base_url=base_url,
        timeout=httpx2.Timeout(None, connect=connect_s),
        transport=transport,
        trust_env=False,
    )


def transport_error(error: httpx2.HTTPError) -> ProviderError:
    if isinstance(error, httpx2.ConnectTimeout):
        return ProviderTimeout("connect")
    if isinstance(error, httpx2.ConnectError):
        return ProviderError("unavailable", f"cannot connect: {error}", retryable=True)
    if isinstance(error, httpx2.TransportError):
        return ProviderError("unavailable", f"connection failed: {error}", retryable=True)
    return ProviderError("internal", f"HTTP client error: {error}")


def status_error(status: int, body: bytes) -> ProviderError:
    detail = body[:_DETAIL_CHARS].decode("utf-8", errors="replace")
    message = f"HTTP {status}: {detail}"
    if status == 429:
        return ProviderError("overloaded", message, retryable=True)
    if status == 503:
        return ProviderError("unavailable", message, retryable=True)
    if status >= 500:
        return ProviderError("internal", message)
    return ProviderError("bad_response", message)
