"""A simulated AICC call for the echo and barge-in tests (M4, D17): AICC's media path and a
telephone line with echo, in real time, as M0's S11 simulation built it (spikes/aec/s11_chain.py).

Downlink, per our ``response.output_audio.delta``:

- AICC's 24 → 8 kHz downsampler (media/resample.go: a 33-tap Hamming windowed sinc, ÷3), and
  its framer (20 ms frames; the last one of a response padded with silence);
- AICC's playout (voice/rtp.go ``Run``) on a 20 ms ticker of its own: playback starts at 3
  queued frames, or 60 ms after the queue was last empty; more than 2 empty ticks re-buffer;
- G.711 μ-law, then the echo path: a seeded, exponentially decaying random impulse response of
  64-128 taps at 8 kHz, scaled to an echo return loss, after a bulk delay. (No ITU G.168
  coefficients: those are © ITU.)

Uplink, one frame per tick: the echo, the caller's speech (injected), line noise → G.711 μ-law
→ AICC's per-frame linear 8 → 24 kHz upsampler → one 20 ms ``input_audio_buffer.append``.

AICC's barge-in is modelled too: on ``speech_started`` once its 800 ms guard has passed, or on a
cancelled ``response.done``, it flushes its queue and truncates the item to the frames it played.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from mlx_realtime.audio.pcm import CANONICAL_RATE
from mlx_realtime.audio.resample import Resampler
from tests.support import Client, Event

TICK_S = 0.020
FRAME_8K = 160  # one 20 ms G.711 frame
PREBUFFER_FRAMES = 3
UNDERRUN_TICKS = 2
AICC_BARGE_GUARD_S = 0.8

type Floats = npt.NDArray[np.float64]


def _down_taps(factor: int = 3, taps: int = 33) -> Floats:
    """media.NewDownsampler(3, 32): an odd 33-tap windowed sinc, cutoff 0.5/factor, Hamming."""
    index = np.arange(taps, dtype=np.float64)
    x = index - (taps - 1) / 2
    cutoff = 0.5 / factor
    with np.errstate(invalid="ignore", divide="ignore"):
        h = np.where(x == 0, 2 * cutoff, np.sin(2 * np.pi * cutoff * x) / (np.pi * x))
    h *= 0.54 - 0.46 * np.cos(2 * np.pi * index / (taps - 1))
    return h / h.sum()


_DOWN = _down_taps()


def ulaw_roundtrip(samples: Floats) -> Floats:
    """G.711 μ-law encode and decode (the continuous companding law, 8-bit quantized)."""
    x = np.clip(samples / 32768.0, -1.0, 1.0)
    mu = 255.0
    encoded = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    quantized = np.round(encoded * 127.0) / 127.0
    return np.sign(quantized) * ((1 + mu) ** np.abs(quantized) - 1) / mu * 32768.0


def upsample_frame(frame8: Floats) -> Floats:
    """AICC's uplink: linear interpolation ×3 within one 20 ms frame, the last sample held."""
    following = np.concatenate([frame8[1:], frame8[-1:]])
    step = (following - frame8) / 3
    return (frame8[:, None] + step[:, None] * np.arange(3)[None, :]).reshape(-1)


def echo_path(seed: int, *, taps: int = 96, erl_db: float = 10.0) -> Floats:
    """A decaying random impulse response at 8 kHz with an echo return loss of ``erl_db``."""
    rng = np.random.default_rng(seed)
    response = rng.standard_normal(taps) * np.exp(-np.arange(taps) / (taps / 5))
    response /= np.sqrt(np.sum(response**2))
    return response * 10 ** (-erl_db / 20)


def to_pcm(samples: Floats) -> bytes:
    return np.clip(np.rint(samples), -32768, 32767).astype("<i2").tobytes()


def from_pcm(pcm: bytes) -> Floats:
    return np.frombuffer(pcm, dtype="<i2").astype(np.float64)


@dataclass
class _Frame:
    item_id: str
    samples: Floats  # 160 at 8 kHz


@dataclass
class CallLog:
    speech_started: list[float] = field(default_factory=list[float])  # loop times
    playback_starts: list[float] = field(default_factory=list[float])
    truncates: list[tuple[str, int]] = field(default_factory=list[tuple[str, int]])
    played_frames: dict[str, int] = field(default_factory=dict[str, int])


class SimulatedCall:
    """AICC plus a phone line with echo, driving ``client`` in real time (see the module notes).

    ``echo_db``: the echo return loss (None: no echo). ``delay_s``: the echo path's bulk delay.
    ``phase_s``: AICC's ticker phase within 20 ms.
    """

    def __init__(
        self,
        client: Client,
        *,
        echo_db: float | None = 10.0,
        delay_s: float = 0.1,
        noise_db: float = -65.0,
        seed: int = 7,
        phase_s: float = 0.007,
    ) -> None:
        self.client = client
        self.log = CallLog()
        self.events: list[tuple[float, Event]] = []
        self._echo = None if echo_db is None else echo_path(seed, erl_db=echo_db)
        self._delay = round(delay_s * 8000)
        self._noise_rms = 32768.0 * 10 ** ((noise_db - 6.2) / 20)
        self._rng = np.random.default_rng(seed + 1)
        self._phase_s = phase_s
        # Downlink state.
        self._down_history = np.zeros(len(_DOWN) - 1)
        self._down_phase = 0
        self._pending8 = np.zeros(0)  # downsampled, not yet a whole frame
        self._item: str | None = None  # the item whose audio is arriving
        self._queue: deque[_Frame] = deque()
        self._playing = False
        self._underruns = 0
        self._deadline: float | None = None
        self._audible_since: float | None = None
        taps = 0 if self._echo is None else len(self._echo)
        self._overlap = np.zeros(max(taps - 1, 0))  # the echo response's tail, overlap-add
        self._line = np.zeros(self._delay)  # echo on its way back (the bulk delay)
        self._caller: deque[float] = deque()  # the caller's speech at 8 kHz
        self._flushed: set[str] = set()
        self._task: asyncio.Task[None] | None = None
        self._reader: asyncio.Task[None] | None = None
        self.done = asyncio.Event()  # set on every response.done
        self._arrived = asyncio.Event()

    async def __aenter__(self) -> SimulatedCall:
        self._task = asyncio.create_task(self._ticker())
        self._reader = asyncio.create_task(self._read())
        return self

    async def __aexit__(self, *_: object) -> None:
        for task in (self._task, self._reader):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    def say(self, pcm16k: bytes, *, gain_db: float = 0.0) -> float:
        """The caller speaks ``pcm16k`` (canonical PCM) from the next tick; returns its length."""
        resampler = Resampler(CANONICAL_RATE, 8000)
        voice = from_pcm(resampler.process(pcm16k) + resampler.flush()) * 10 ** (gain_db / 20)
        self._caller.extend(voice.tolist())
        return len(voice) / 8000

    def types(self) -> list[str]:
        return [event["type"] for _, event in self.events]

    async def until(self, event_type: str, *, after: int = 0, timeout_s: float = 10.0) -> int:
        """The index in ``events`` of the first ``event_type`` at or after ``after``."""
        async with asyncio.timeout(timeout_s):
            while True:
                for index in range(after, len(self.events)):
                    if self.events[index][1]["type"] == event_type:
                        return index
                self._arrived.clear()
                await self._arrived.wait()

    # The server's events

    async def _read(self) -> None:
        while True:
            event = await self.client.recv(wait_s=3600)
            now = asyncio.get_running_loop().time()
            self.events.append((now, event))
            self._arrived.set()
            match event["type"]:
                case "response.output_audio.delta":
                    self._downlink(event["item_id"], base64.b64decode(event["delta"]))
                case "response.output_audio.done":
                    self._end_of_audio()
                case "input_audio_buffer.speech_started":
                    self.log.speech_started.append(now)
                    if self._audible_since is not None and (
                        now - self._audible_since >= AICC_BARGE_GUARD_S
                    ):
                        await self._barge_in()
                case "response.done":
                    if event["response"]["status"] == "cancelled":
                        await self._barge_in()
                    self.done.set()
                case _:
                    pass

    def _downlink(self, item_id: str, pcm24: bytes) -> None:
        if item_id != self._item:
            self._end_of_audio()
            self._item = item_id
        if item_id in self._flushed:
            return
        samples = from_pcm(pcm24)
        stream = np.concatenate([self._down_history, samples])
        filtered = np.convolve(stream, _DOWN, mode="valid")  # one output per input sample
        self._down_history = stream[len(stream) - (len(_DOWN) - 1) :]
        picked = filtered[(-self._down_phase) % 3 :: 3]
        self._down_phase = (self._down_phase + len(samples)) % 3
        self._pending8 = np.concatenate([self._pending8, picked])
        while len(self._pending8) >= FRAME_8K:
            self._queue.append(_Frame(item_id, self._pending8[:FRAME_8K]))
            self._pending8 = self._pending8[FRAME_8K:]

    def _end_of_audio(self) -> None:
        """AICC's framer pads the last frame of a response with silence."""
        if self._item is not None and len(self._pending8) and self._item not in self._flushed:
            padded = np.concatenate([self._pending8, np.zeros(FRAME_8K - len(self._pending8))])
            self._queue.append(_Frame(self._item, padded))
        self._pending8 = np.zeros(0)

    async def _barge_in(self) -> None:
        """AICC flushes what it has not played and truncates the item to what it played."""
        item = self._item
        if item is None or item in self._flushed:
            return
        self._flushed.add(item)
        self._queue = deque(frame for frame in self._queue if frame.item_id != item)
        self._pending8 = np.zeros(0)
        self._audible_since = None
        played = self.log.played_frames.get(item, 0)
        self.log.truncates.append((item, played * 20))
        await self.client.send(
            {
                "type": "conversation.item.truncate",
                "item_id": item,
                "content_index": 0,
                "audio_end_ms": played * 20,
            }
        )

    # The 20 ms ticker: playout, echo, uplink

    async def _ticker(self) -> None:
        loop = asyncio.get_running_loop()
        start = loop.time() + self._phase_s
        tick = 0
        while True:
            due = start + tick * TICK_S
            await asyncio.sleep(max(due - loop.time(), 0.0))
            now = loop.time()
            played = self._playout(now)
            await self._uplink(played)
            tick += 1

    def _playout(self, now: float) -> Floats:
        queued = len(self._queue)
        frame: _Frame | None = None
        if self._playing:
            if queued:
                frame = self._queue.popleft()
                self._underruns = 0
            else:
                self._underruns += 1
                if self._underruns > UNDERRUN_TICKS:
                    self._playing, self._underruns = False, 0
                    self._audible_since = None
        elif queued >= PREBUFFER_FRAMES or (
            queued and self._deadline is not None and now >= self._deadline
        ):
            self._playing = True
            self.log.playback_starts.append(now)
            if self._audible_since is None:
                self._audible_since = now
            frame = self._queue.popleft()
        elif not queued:
            self._deadline = now + PREBUFFER_FRAMES * TICK_S
        if frame is None:
            return np.zeros(FRAME_8K)
        self.log.played_frames[frame.item_id] = self.log.played_frames.get(frame.item_id, 0) + 1
        return frame.samples

    async def _uplink(self, played: Floats) -> None:
        echo = np.zeros(FRAME_8K)
        if self._echo is not None:
            # What played, G.711-coded, through the echo path, after the bulk delay.
            full = np.convolve(ulaw_roundtrip(played), self._echo)
            full[: len(self._overlap)] += self._overlap
            produced, self._overlap = full[:FRAME_8K], full[FRAME_8K:]
            line = np.concatenate([self._line, produced])
            echo, self._line = line[:FRAME_8K], line[FRAME_8K:]
        caller = np.zeros(FRAME_8K)
        for index in range(min(FRAME_8K, len(self._caller))):
            caller[index] = self._caller.popleft()
        noise = self._rng.standard_normal(FRAME_8K) * self._noise_rms
        uplink = ulaw_roundtrip(echo + caller + noise)
        audio = base64.b64encode(to_pcm(upsample_frame(uplink))).decode()
        await self.client.send({"type": "input_audio_buffer.append", "audio": audio})
