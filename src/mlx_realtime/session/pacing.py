"""Output pacing (D6, ARCHITECTURE §7).

Audio deltas go out as soon as they are synthesized, as GA does over WebSocket, but never more
than ``max_lead_ms`` ahead of real time, where real time starts at the response's first delta:

- the lead bounds the client's queue (AICC queues at most 30 s and drops the rest);
- audio not yet sent stays in the gateway, where a barge-in discards it;
- at the cap, each delta waits exactly as long as the audio before it plays, so deltas keep
  flowing at the playback rate, never with a gap anywhere near AICC's 2 s watchdog.
"""

from __future__ import annotations

from mlx_realtime.session.clock import Clock


class Pacer:
    def __init__(self, max_lead_ms: int, clock: Clock) -> None:
        self._max_lead_ms = max_lead_ms
        self._clock = clock
        self._start: float | None = None
        self._sent_ms = 0.0

    def lead_ms(self) -> float:
        """How far the audio admitted so far runs ahead of real time."""
        if self._start is None:
            return 0.0
        return self._sent_ms - (self._clock.now() - self._start) * 1000

    async def admit(self, audio_ms: float) -> None:
        """Wait until ``audio_ms`` more audio can go out without passing the lead cap."""
        now = self._clock.now()
        if self._start is None:
            self._start = now
        excess_ms = self._sent_ms + audio_ms - (now - self._start) * 1000 - self._max_lead_ms
        if excess_ms > 0:
            await self._clock.sleep(excess_ms / 1000)
        self._sent_ms += audio_ms
