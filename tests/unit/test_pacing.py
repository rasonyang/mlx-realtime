"""Output pacing (D6) on a fake clock: as soon as available, at most ``max_lead_ms`` ahead of real
time from the response's first delta, and a steady flow at the cap."""

from __future__ import annotations

import itertools

from mlx_realtime.session.pacing import Pacer


class FakeClock:
    """Virtual time: ``sleep`` advances it at once and records how long it was asked for."""

    def __init__(self, start: float = 50.0) -> None:
        self.t = start
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.t

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds


async def send_all(pacer: Pacer, clock: FakeClock, deltas: int, delta_ms: float) -> list[float]:
    """Admit ``deltas`` deltas that are all ready at once; return each one's send time (ms)."""
    sent: list[float] = []
    for _ in range(deltas):
        await pacer.admit(delta_ms)
        sent.append(clock.now() * 1000)
    return sent


async def test_audio_under_the_cap_goes_out_at_once() -> None:
    clock = FakeClock()
    pacer = Pacer(10_000, clock)
    sent = await send_all(pacer, clock, 100, 100)  # 10 s of audio
    assert clock.sleeps == []
    assert sent == [50_000.0] * 100
    assert pacer.lead_ms() == 10_000


async def test_at_the_cap_deltas_flow_at_the_playback_rate() -> None:
    clock = FakeClock()
    pacer = Pacer(10_000, clock)
    sent = await send_all(pacer, clock, 300, 100)  # 30 s of audio, all synthesized at once
    start = sent[0]
    assert sent[:100] == [start] * 100  # the first 10 s at once
    gaps = [later - earlier for earlier, later in itertools.pairwise(sent)]
    assert all(abs(gap - 100) < 1e-6 for gap in gaps[100:])  # then one delta per 100 ms
    assert max(gaps) < 2000  # never near AICC's 2 s watchdog
    # The lead never exceeds the cap: audio sent minus elapsed time, at every send.
    for index, at in enumerate(sent):
        assert (index + 1) * 100 - (at - start) <= 10_000 + 1e-6


async def test_real_time_starts_at_the_first_delta() -> None:
    clock = FakeClock()
    pacer = Pacer(1000, clock)
    clock.t += 5.0  # the response took 5 s to produce its first audio
    sent = await send_all(pacer, clock, 20, 100)
    assert sent[9] == sent[0]  # 1 s of lead, from the first delta, not from the start
    assert sent[10] - sent[0] == 100


async def test_audio_slower_than_real_time_is_never_held() -> None:
    clock = FakeClock()
    pacer = Pacer(1000, clock)
    for _ in range(50):
        await pacer.admit(100)
        clock.t += 0.15  # synthesis delivers 100 ms of audio every 150 ms
    assert clock.sleeps == []
    assert pacer.lead_ms() < 100


async def test_a_new_pacer_per_response() -> None:
    clock = FakeClock()
    first = Pacer(1000, clock)
    await send_all(first, clock, 30, 100)
    second = Pacer(1000, clock)
    assert second.lead_ms() == 0
    before = len(clock.sleeps)
    await send_all(second, clock, 10, 100)
    assert len(clock.sleeps) == before  # a fresh response starts with no lead
