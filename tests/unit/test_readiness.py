"""The readiness monitor (``server/readiness.py``): warmup at startup, and warming a backend again
after it went down or was restarted (a restarted ``mlx_audio.server`` has no model loaded)."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncGenerator, Callable

import pytest

from mlx_realtime.providers.base import ProviderError, Readiness
from mlx_realtime.server.readiness import ReadinessMonitor


class FakeServer:
    """A backend process: ``up`` (it answers), ``loaded`` (its model is in memory, which only a
    warmup does), ``slow_s`` (how long a probe takes to answer)."""

    def __init__(self, *, ready_unloaded: bool = False) -> None:
        self.up = True
        self.loaded = False
        self.slow_s = 0.0
        self.warmups = 0
        self.ready_unloaded = ready_unloaded  # like mlx_lm.server's /health
        self.hold = asyncio.Event()
        self.hold.set()

    async def warmup(self) -> None:
        if not self.up:
            raise ProviderError("unavailable", "cannot connect")
        await self.hold.wait()
        self.warmups += 1
        self.loaded = True

    async def ready(self) -> Readiness:
        await asyncio.sleep(self.slow_s)
        if not self.up:
            return Readiness(ready=False, detail="cannot connect")
        if self.ready_unloaded:
            return Readiness(ready=True, detail="ok")
        return Readiness(ready=self.loaded, detail="ok" if self.loaded else "model not loaded")

    async def aclose(self) -> None:
        pass

    def crash(self) -> None:
        self.up, self.loaded = False, False


@contextlib.asynccontextmanager
async def monitoring(
    monitor: ReadinessMonitor,
) -> AsyncGenerator[asyncio.Task[None]]:
    task = asyncio.create_task(monitor.run())
    try:
        async with asyncio.timeout(2):
            while not monitor.ever_ready:  # noqa: ASYNC110 (a flag, not an event)
                await asyncio.sleep(0.005)
        yield task
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def until(check: Callable[[], bool], timeout_s: float = 2.0) -> None:
    async with asyncio.timeout(timeout_s):
        while not check():  # noqa: ASYNC110 (the fakes expose flags, not events)
            await asyncio.sleep(0.005)


async def until_ready(monitor: ReadinessMonitor, timeout_s: float = 2.0) -> None:
    async with asyncio.timeout(timeout_s):
        while not (await monitor.report())[0]:  # noqa: ASYNC110 (polled, as /readyz is)
            await asyncio.sleep(0.005)


def monitor_of(**backends: FakeServer) -> ReadinessMonitor:
    return ReadinessMonitor(backends, retry_s=0.01, max_retry_s=0.02, watch_s=0.02,
                            probe_timeout_s=0.05)  # fmt: skip


async def test_a_restarted_backend_is_warmed_again() -> None:
    asr, llm = FakeServer(), FakeServer()
    monitor = monitor_of(asr=asr, llm=llm)
    async with monitoring(monitor):
        assert (asr.warmups, llm.warmups) == (1, 1)
        assert (await monitor.report())[0]
        asr.crash()
        await until(lambda: not asr.up)
        await asyncio.sleep(0.1)  # down for a few probes: retried, not yet warm
        ready, details = await monitor.report()
        assert not ready
        assert details["asr"] == {"ready": False, "warm": False, "detail": "cannot connect"}
        assert details["llm"]["ready"]
        asr.up = True  # restarted by its supervisor, its model not loaded
        await until_ready(monitor)
        details = (await monitor.report())[1]
        assert details["asr"] == {"ready": True, "warm": True, "detail": "ok"}
        assert (asr.warmups, llm.warmups) == (2, 1)


async def test_a_backend_is_not_ready_until_it_is_warm_again() -> None:
    """mlx_lm.server answers /health as soon as it is up; the gateway's warmup has not run."""
    llm = FakeServer(ready_unloaded=True)
    monitor = monitor_of(llm=llm)
    async with monitoring(monitor):
        llm.hold.clear()
        llm.crash()
        await until(lambda: llm.up is False)
        await asyncio.sleep(0.05)  # it answered "cannot connect": warming it again
        llm.up = True  # back, its warmup held
        await asyncio.sleep(0.1)
        ready, details = await monitor.report()
        assert (ready, details["llm"]) == (False, {"ready": True, "warm": False, "detail": "ok"})
        llm.hold.set()
        await until_ready(monitor)
        assert llm.warmups == 2


async def test_a_backend_that_lost_its_model_is_warmed_again() -> None:
    tts = FakeServer()
    monitor = monitor_of(tts=tts)
    async with monitoring(monitor):
        tts.loaded = False  # a restart quicker than one probe interval
        await until(lambda: tts.warmups == 2)
        await until_ready(monitor)


async def test_a_slow_probe_is_a_busy_backend_not_a_lost_one() -> None:
    llm = FakeServer()
    monitor = monitor_of(llm=llm)
    async with monitoring(monitor):
        llm.slow_s = 0.1  # past the probe timeout
        await asyncio.sleep(0.3)
        assert llm.warmups == 1
        ready, details = await monitor.report()
        assert (ready, details["llm"]["detail"], details["llm"]["warm"]) == (
            False,
            "TimeoutError",
            True,
        )


async def test_without_a_watch_the_monitor_only_warms() -> None:
    asr = FakeServer()
    monitor = ReadinessMonitor({"asr": asr}, retry_s=0.01, watch_s=None)
    await asyncio.wait_for(monitor.run(), 1)  # returns once warm
    assert monitor.ever_ready
    asr.crash()
    asr.up = True
    await asyncio.sleep(0.05)
    assert asr.warmups == 1


async def test_a_probe_that_raises_does_not_stop_the_watch(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class Broken(FakeServer):
        broken = False

        async def ready(self) -> Readiness:
            if self.broken:
                raise RuntimeError("bug")
            return await super().ready()

    asr, llm = Broken(), FakeServer()
    monitor = monitor_of(asr=asr, llm=llm)
    async with monitoring(monitor) as task:
        asr.broken = True
        await asyncio.sleep(0.1)
        assert not task.done()
        llm.crash()
        llm.up = True
        await until(lambda: llm.warmups == 2)
    assert "backend probe failed" in caplog.text
