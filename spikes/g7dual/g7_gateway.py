"""G7 dual path: one simulated call through the gateway's own echo path. Throwaway spike code.

The gateway's classes, as the session actor and the VAD feeder drive them (src/mlx_realtime/
session/echo.py, feeder.py, actor.py), on a manual clock:

- the client's 24 kHz uplink arrives in 20 ms appends, 3 ms after each frame's end, and goes
  through the InputBuffer's decoder (soxr 24 -> 16 kHz) and ``EchoReference.capture_arrived``;
- the canonical audio goes through ``EchoStage.feed`` (AEC3 for detection, the ASR path's
  canceller per ``aec.asr_input``, the GCC-PHAT estimator, the hold-back); held frames are
  released as the clock advances and after every event, as the feeder does;
- our output: ``response_started``, then each delta ``audio_queued`` + ``audio_sent`` at its send
  time, ``response_finished`` after the last; a barge-in ``flush`` + ``response_finished`` at the
  cut, and AICC's truncate (``truncated``, its really played frames) two transits (6 ms) later.

Returns the three tracks the InputBuffer holds (raw, AEC3's, the ASR path's) over the call, the
clean chain through the same decoder, the echo stats, and the CPU of the stage's executor work.
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import g7_chain as gc
import s11_chain as ch
from mlx_realtime.audio.aec import make_engine
from mlx_realtime.audio.codec import InputDecoder
from mlx_realtime.audio.formats import PCM
from mlx_realtime.config import AecConfig, BargeInConfig
from mlx_realtime.session.echo import (
    EchoParams,
    EchoReference,
    EchoRuntime,
    EchoStage,
    Processed,
)

T0 = 1000.0  # loop time of the call's start
UPLINK_TRANSIT_S = 0.003
TRUNCATE_AFTER_S = 0.006  # AICC flushes on speech_started and truncates at once: 2 transits
APPEND = 480  # 20 ms at 24 kHz


class ManualClock:
    def __init__(self, t: float) -> None:
        self.t = t

    def now(self) -> float:
        return self.t

    async def sleep(self, seconds: float) -> None:
        self.t += seconds


def decode(up24: np.ndarray) -> np.ndarray:
    """The InputBuffer's decoder over 20 ms appends: canonical 16 kHz int16."""
    decoder = InputDecoder(PCM, "HQ")
    pcm = up24.astype("<i2").tobytes()
    out = b"".join(decoder.decode(pcm[i:i + 2 * APPEND]) for i in range(0, len(pcm), 2 * APPEND))
    return np.frombuffer(out + decoder.flush(), dtype="<i2").astype(np.int16)


async def run_call(call: gc.Call, aec: AecConfig, executor: ThreadPoolExecutor) -> dict:
    clock = ManualClock(T0)
    reference = EchoReference(EchoParams.of(aec, BargeInConfig()), clock, "HQ")
    runtime = EchoRuntime(aec)
    stage = EchoStage(make_engine(aec.engine), reference, executor, runtime.asr_path())
    runtime.executor.shutdown()
    run_ns = [0]
    inner = stage._run  # noqa: SLF001 - the executor work, timed whole (engines + estimator)

    def timed(*args):  # type: ignore[no-untyped-def]
        began = time.thread_time_ns()
        try:
            return inner(*args)
        finally:
            run_ns[0] += time.thread_time_ns() - began

    stage._run = timed  # type: ignore[method-assign]  # noqa: SLF001

    decoder = InputDecoder(PCM, "HQ")
    n16 = len(call.up24) * 2 // 3 + 4096
    raw = np.zeros(n16, dtype=np.int16)
    vad = np.zeros(n16, dtype=np.int16)
    asr = np.zeros(n16, dtype=np.int16)
    written = [0]

    def put(chunks: list[Processed]) -> None:
        for chunk in chunks:
            a = chunk.start
            pcm = np.frombuffer(chunk.pcm, dtype="<i2")
            b = min(a + len(pcm), n16)
            vad[a:b] = pcm[: b - a]
            if chunk.asr is not None:
                asr[a:b] = np.frombuffer(chunk.asr, dtype="<i2")[: b - a]

    # ---- the events, in time order: (t, order, kind, payload)
    events: list[tuple[float, int, str, object]] = []
    first: dict[int, float] = {}
    last: dict[int, float] = {}
    for d in call.deltas:
        first.setdefault(d.response, d.t)
        last[d.response] = d.t
        events.append((d.t, 1, "delta", d))
    for r, t in first.items():
        events.append((t, 0, "start", r))
    # A response cut while still sending is cancelled (flush, then done); one whose audio had
    # all been sent was done already, and the barge-in only brings AICC's truncate.
    cancelled = {cut.response for cut in call.cuts if not cut.complete}
    for r, t in last.items():
        if r not in cancelled:
            events.append((t, 2, "finish", r))
    for cut in call.cuts:
        events.append((cut.t, 3, "cut", cut))
        events.append((cut.t + TRUNCATE_AFTER_S, 3, "truncate", cut))
    pcm24 = call.up24.astype("<i2").tobytes()
    n_appends = len(call.up24) // APPEND
    for j in range(n_appends):
        events.append(((j + 1) * 0.02 + UPLINK_TRANSIT_S, 4, "append", j))
    events.sort(key=lambda e: (e[0], e[1]))

    cut_done: set[int] = set()
    for t, _, kind, payload in events:
        clock.t = T0 + t
        if kind == "start":
            reference.response_started()
        elif kind == "delta":
            d = payload
            if d.response in cut_done:
                continue
            x24 = ch.far_material(call.responses[d.response])["x24"]
            reference.audio_queued(d.response + 1)
            reference.audio_sent(f"item_{d.response}", d.response + 1,
                                 x24[d.a24:d.b24].astype("<i2").tobytes(), PCM)
        elif kind == "finish":
            reference.response_finished(payload + 1)
        elif kind == "cut":
            cut = payload
            if not cut.complete and cut.response not in cut_done:
                cut_done.add(cut.response)
                reference.flush(cut.response + 1)
                reference.response_finished(cut.response + 1)
        elif kind == "truncate":
            cut = payload
            reference.truncated(f"item_{cut.response}", cut.played_frames * 20)
        elif kind == "append":
            j = payload
            canonical = decoder.decode(pcm24[j * 2 * APPEND:(j + 1) * 2 * APPEND])
            start = written[0]
            count = len(canonical) // 2
            b = min(start + count, n16)
            raw[start:b] = np.frombuffer(canonical, dtype="<i2")[: b - start]
            written[0] += count
            reference.capture_arrived(start + count)
            if count:
                put(await stage.feed(start, canonical))
        put(await stage.release())
    put(await stage.release(force=True))
    n = written[0]
    stats = reference.stats
    audio_s = n / 16000
    return {
        "raw": raw[:n], "aec3": vad[:n], "asr": asr[:n] if aec.asr_input == "speex" else None,
        "summary": {
            "audio_s": round(audio_s, 3),
            "erle_db": stats.erle_db(), "speex_erle_db": stats.speex_erle.median_db(),
            "speex_aligned_at_s": None if stats.speex_aligned_at is None else round(stats.speex_aligned_at / 16000, 3),
            "speex_shift_ms": reference.speex_shift_ms, "speex_realignments": stats.speex_realignments,
            "speex_resets": stats.speex_resets, "realignments": stats.realignments, "anchors": stats.anchors,
            "bypasses": stats.bypasses, "echo_absent": reference.echo_absent, "shift_ms": reference.shift_ms,
            "tick_phase_ms": round(reference.tick_phase_ms, 2), "estimates": stats.estimates,
            "cpu_ms_per_audio_s": {
                "aec3": round(stats.engine_ns / 1e6 / audio_s, 3),
                "speex_path": round(stats.speex_ns / 1e6 / audio_s, 3),
                "executor_total": round(run_ns[0] / 1e6 / audio_s, 3),
            },
        },
    }


def run_call_sync(call: gc.Call, aec: AecConfig) -> dict:
    with ThreadPoolExecutor(1) as executor:
        return asyncio.run(run_call(call, aec, executor))
