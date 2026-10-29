"""Building blocks of the hold measurement (bursts.py): the simulated AICC call with residual-echo
bursts and injected sounds, two gateways (baseline and fix) over the real ASR and TTS, trials, and
the reading of the gateway's capture events. Run in the gateway's environment (root project).

The burst model (an assumption, not a measured phone AEC): the audio the client really played is
delayed by ``delay`` (316 or 506 ms), passed through a short random impulse response and a
300-2400 Hz band-pass (a phone AEC's suppressor leaves the mid band), soft-clipped (tanh at 0.7 of
the local rms), normalised per 20 ms frame and scaled to ``level`` dB under the played audio's
smoothed level, with a frame-wise gain fluctuation (AR(1), rho 0.5, sigma 5 dB) and dropouts (10 %
of the frames at -25 dB). Lengths 1.5-2.5 s. AICC's G.711 and 8 -> 24 kHz uplink follow as for
any line signal (tests/phone.py).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import os
import subprocess
import sys
import wave
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO = Path(__file__).parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import tests.support as support  # noqa: E402
from mlx_realtime.audio.resample import Resampler  # noqa: E402
from mlx_realtime.config import validate_config  # noqa: E402
from mlx_realtime.providers.mock import EchoLLM  # noqa: E402
from mlx_realtime.providers.registry import Providers, build_asr, build_tts  # noqa: E402
from mlx_realtime.providers.voices import VoiceBook  # noqa: E402
from mlx_realtime.server.gateway import Gateway  # noqa: E402
from tests.integration.test_echo import AICC_SESSION  # noqa: E402
from tests.phone import FRAME_8K, TICK_S, SimulatedCall, echo_path  # noqa: E402
from tests.support import Client  # noqa: E402

HOLD = Path(__file__).parent / "captures" / os.environ.get("HOLD_DIR", "hold")
SAY = "Say exactly this, word for word, and add nothing else:\n"
RING = 8000 * 30  # a multiple of FRAME_8K

# The burst model's parameters (documented in the module notes and in results/hold.md).
BAND_HZ = (300.0, 2400.0)
CLIP_REL = 0.7
GAIN_SIGMA_DB = 5.0
GAIN_RHO = 0.5
DROPOUT_P = 0.10
DROPOUT_DB = -25.0
IR_TAPS = 48
PROFILES: dict[str, dict[str, Any]] = {
    # The main model (module notes above): a strongly processed leak.
    "default": {"band_hz": BAND_HZ, "clip_rel": CLIP_REL, "gain_sigma_db": GAIN_SIGMA_DB,
                "dropout_p": DROPOUT_P, "dropout_db": DROPOUT_DB},
    # An optimistic bracket: a cleaner, wideband, nearly linear leak (the ASR understands more of it).
    "mild": {"band_hz": (200.0, 3400.0), "clip_rel": 1.5, "gain_sigma_db": 2.5,
             "dropout_p": 0.03, "dropout_db": -15.0},
}


def data_root() -> Path:
    """spikes/data (git-ignored): this worktree's, else the main checkout's."""
    if env := os.environ.get("MLXRT_SPIKES_DATA"):
        return Path(env)
    if (REPO / "spikes" / "data").is_dir():
        return REPO / "spikes" / "data"
    common = subprocess.run(
        ["git", "rev-parse", "--git-common-dir"], cwd=REPO, capture_output=True, text=True
    ).stdout.strip()
    return (REPO / common).resolve().parent / "spikes" / "data"


def read_wav(path: Path) -> bytes:
    with wave.open(str(path)) as file:
        assert (file.getframerate(), file.getnchannels(), file.getsampwidth()) == (16000, 1, 2)
        return file.readframes(file.getnframes())


def write_wav(path: Path, pcm: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(16000)
        file.writeframes(pcm)


def onset_s(pcm16k: bytes, *, below_db: float = 18.0) -> float:
    """The first voiced instant of a clip: the first 20 ms frame (three in a row) whose rms is within
    ``below_db`` of the clip's loud frames (90th percentile)."""
    x = np.frombuffer(pcm16k, dtype="<i2").astype(np.float64)
    frames = x[: len(x) // 320 * 320].reshape(-1, 320)
    rms = np.sqrt(np.mean(frames**2, axis=1)) + 1e-9
    level = np.percentile(rms, 90)
    hot = rms >= level * 10 ** (-below_db / 20)
    for index in range(len(hot) - 2):
        if hot[index] and hot[index + 1] and hot[index + 2]:
            return index * 0.02
    return 0.0


# The burst model


def _bandpass(band: tuple[float, float], taps: int = 63) -> np.ndarray:
    index = np.arange(taps) - (taps - 1) / 2

    def lowpass(cut_hz: float) -> np.ndarray:
        fc = cut_hz / 8000
        with np.errstate(invalid="ignore", divide="ignore"):
            h = np.where(index == 0, 2 * fc, np.sin(2 * np.pi * fc * index) / (np.pi * index))
        return h * np.hamming(taps)

    return lowpass(band[1]) - lowpass(band[0])


@dataclass
class Burst:
    kind: str  # A1 | A2 | A3
    first_frame: int
    frames: int
    delay_s: float
    level_db: float
    seed: int
    profile: str = "default"
    done: bool = False
    src_energy: float = 0.0
    out_energy: float = 0.0
    playback_pos_s: float | None = None  # seconds of audio played before the burst began
    to_end_s: float | None = None  # seconds of audio left to play when it began (A1: how late)
    _rng: Any = field(default=None, repr=False)
    _h: Any = field(default=None, repr=False)
    _hist: Any = field(default=None, repr=False)
    _ema_z: float = 0.0
    _ema_src: float = 0.0
    _gain_db: float = 0.0

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        ir = echo_path(self.seed, taps=IR_TAPS, erl_db=0.0)
        self._p = PROFILES[self.profile]
        self._h = np.convolve(ir, _bandpass(self._p["band_hz"]))
        self._hist = np.zeros(len(self._h) - 1)

    @property
    def delay_samples(self) -> int:
        return round(self.delay_s * 8000)

    @property
    def end_frame(self) -> int:
        return self.first_frame + self.frames

    def realised_db(self) -> float | None:
        if self.src_energy <= 0 or self.out_energy <= 0:
            return None
        return 10 * math.log10(self.out_energy / self.src_energy)

    def frame(self, source: np.ndarray) -> np.ndarray:
        """One 20 ms frame of the leak, from the frame the client played ``delay`` ago."""
        stream = np.concatenate([self._hist, source])
        z = np.convolve(stream, self._h, mode="valid")[: len(source)]
        self._hist = stream[len(stream) - (len(self._h) - 1) :]
        rms_z = float(np.sqrt(np.mean(z**2)))
        self._ema_z = rms_z if self._ema_z == 0 else 0.8 * self._ema_z + 0.2 * rms_z
        c = max(self._p["clip_rel"] * self._ema_z, 1.0)
        y = np.tanh(z / c) * c
        rms_y = float(np.sqrt(np.mean(y**2)))
        rms_s = float(np.sqrt(np.mean(source**2)))
        self._ema_src = 0.5 * self._ema_src + 0.5 * rms_s
        self._gain_db = GAIN_RHO * self._gain_db + math.sqrt(1 - GAIN_RHO**2) * float(
            self._rng.normal(0.0, self._p["gain_sigma_db"])
        )
        gain = self._p["dropout_db"] if self._rng.random() < self._p["dropout_p"] else self._gain_db
        self.src_energy += float(np.sum(source**2))
        if rms_y < 1e-6 or rms_s < 5.0:
            return np.zeros(len(source))
        out = y / rms_y * self._ema_src * 10 ** ((self.level_db + gain) / 20)
        self.out_energy += float(np.sum(out**2))
        return out


@dataclass
class Plan:
    kind: str
    delay_s: float
    level_db: float
    length_s: float
    seed: int
    offset_s: float = 0.0  # A3: where the leak starts, from the playback start
    past_guard_s: float = 0.6  # A2: how far past the 2 s onset guard the leak ends
    profile: str = "default"


class _SpeechLog(list):  # type: ignore[type-arg]
    """The call's speech_started times, each with whether AICC had audio queued at that moment."""

    def __init__(self, call: Any) -> None:
        super().__init__()
        self.call = call

    def append(self, value: float) -> None:
        super().append(value)
        call = self.call
        call.ss_snap.append(
            {"t": value, "audio": call.active_audio, "queued_frames": len(call._queue),  # noqa: SLF001
             "audible_s": None if call._audible_since is None else value - call._audible_since}  # noqa: SLF001
        )


class HoldCall(SimulatedCall):
    """The simulated AICC call plus residual-echo bursts of what really played, injected sounds,
    and the logs the trials need (tick times, playback starts, effective flushes)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.n = 0  # the uplink frame being built (input ms = 20 n)
        self.tick_t: list[float] = []
        self.ring = np.zeros(RING)
        self.bursts: list[Burst] = []
        self.start_n: list[int] = []
        self.flushes: list[dict[str, Any]] = []
        self.cleaning = False
        self.ss_snap: list[dict[str, Any]] = []
        self.log.speech_started = _SpeechLog(self)  # type: ignore[assignment]
        self._plan_start: Plan | None = None
        self._plan_end: Plan | None = None

    @property
    def active_audio(self) -> bool:
        return self._playing or bool(self._queue)

    # Plans

    def plan_at_start(self, plan: Plan) -> None:
        """A2 / A3: the burst is placed from the next playback start."""
        self._plan_start = plan

    def plan_at_end(self, plan: Plan) -> None:
        """A1: placed when the current item's audio is complete, from its modelled end."""
        self._plan_end = plan

    def _add(self, plan: Plan, first_frame: int, frames: int) -> None:
        burst = Burst(plan.kind, first_frame, frames, plan.delay_s, plan.level_db, plan.seed, plan.profile)
        item = self._item
        played = self.log.played_frames.get(item, 0) if item else 0
        source_frame = first_frame - burst.delay_samples // FRAME_8K  # where its source starts
        burst.playback_pos_s = (played + source_frame - self.n) * TICK_S
        burst.to_end_s = (len(self._queue) - (source_frame - self.n)) * TICK_S
        self.bursts.append(burst)

    def _playout(self, now: float) -> Any:
        before = len(self.log.playback_starts)
        out = super()._playout(now)
        if len(self.log.playback_starts) > before:
            self.start_n.append(self.n)
            plan, self._plan_start = self._plan_start, None
            if plan is not None:
                self._place_from_start(plan, self.n)
        return out

    def _place_from_start(self, plan: Plan, n0: int) -> None:
        min_first = n0 + math.ceil(plan.delay_s * 8000 / FRAME_8K)
        frames = round(plan.length_s / TICK_S)
        if plan.kind == "A2":
            end_frame = n0 + round((2.0 + plan.past_guard_s) / TICK_S)
            first = max(min_first, end_frame - frames)
            frames = end_frame - first
        else:  # A3
            first = max(min_first, n0 + round(plan.offset_s / TICK_S))
        self._add(plan, first, frames)

    def _end_of_audio(self) -> None:
        plan = self._plan_end
        super()._end_of_audio()
        if plan is None or self._item is None:
            return
        self._plan_end = None
        left = len(self._queue)
        frames = round(plan.length_s / TICK_S)
        end_frame = self.n + left
        first = max(self.n + 1, end_frame - frames + math.ceil(plan.delay_s * 8000 / FRAME_8K))
        self._add(plan, first, frames)

    async def _barge_in(self) -> None:
        item = self._item
        before = item in self._flushed if item else True
        dropped = sum(1 for frame in self._queue if frame.item_id == item)
        at = asyncio.get_running_loop().time()
        n = self.n
        await super()._barge_in()
        if item is not None and not before and item in self._flushed:
            self.flushes.append(
                {"t": at, "n": n, "item": item, "frames_dropped": dropped, "cleanup": self.cleaning}
            )

    # The line

    def _injected(self, played: np.ndarray) -> np.ndarray:
        n = self.n
        base = (n * FRAME_8K) % RING
        self.ring[base : base + FRAME_8K] = played
        out = np.zeros(FRAME_8K)
        for burst in self.bursts:
            if burst.first_frame <= n < burst.end_frame:
                start = n * FRAME_8K - burst.delay_samples
                if start >= 0:
                    index = (np.arange(FRAME_8K) + start) % RING
                    out += burst.frame(self.ring[index])
            elif n == burst.end_frame:
                burst.done = True
        return out

    async def _uplink(self, played: np.ndarray) -> None:
        self.tick_t.append(asyncio.get_running_loop().time())
        await super()._uplink(played)
        self.n += 1

    def inject(self, pcm16k: bytes) -> int:
        """The caller (or a sound) enters the line from the next free tick; its first frame."""
        first = self.n + len(self._caller) // FRAME_8K
        self.say(pcm16k)
        return first

    def tick_time(self, n: int) -> float:
        return self.tick_t[n]


# Sounds


def _db(x: float) -> float:
    return 10 ** (x / 20)


def synth_noise(kind: str, seed: int, peak_dbfs: float) -> bytes:
    """Non-speech sounds at 16 kHz: ``cup`` (set down: damped resonances), ``keys`` (typing
    bursts), ``clicks`` (a click train)."""
    rng = np.random.default_rng(seed)
    rate = 16000
    out = np.zeros(int(rate * 2.2))
    lead = int(0.3 * rate)

    def put(offset: int, sig: np.ndarray, gain: float = 1.0) -> None:
        end = min(len(out), offset + len(sig))
        out[offset:end] += gain * sig[: end - offset]

    def ring(freqs: list[float], taus: list[float], seconds: float) -> np.ndarray:
        t = np.arange(int(seconds * rate)) / rate
        sig = sum(
            np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28)) * np.exp(-t / tau)
            for f, tau in zip(freqs, taus, strict=True)
        )
        return sig / (np.max(np.abs(sig)) + 1e-9)

    if kind == "cup":
        freqs = [float(rng.uniform(700, 1200)), float(rng.uniform(1800, 2800)), float(rng.uniform(3200, 5200))]
        sig = ring(freqs, [0.09, 0.05, 0.03], 0.5)
        thump = rng.standard_normal(int(0.006 * rate)) * np.exp(-np.arange(int(0.006 * rate)) / 20)
        put(lead, sig)
        put(lead, thump / np.max(np.abs(thump)), 0.8)
        put(lead + int(rng.uniform(0.09, 0.2) * rate), sig, 0.35)  # a small bounce
    elif kind == "keys":
        t = lead
        for _ in range(int(rng.integers(6, 13))):
            click = ring([float(rng.uniform(1500, 3500))], [0.004], 0.03)
            thump = ring([float(rng.uniform(120, 200))], [0.012], 0.05)
            put(t, click, float(rng.uniform(0.6, 1.0)))
            put(t, thump, 0.5)
            t += int(rng.uniform(0.07, 0.19) * rate)
            if t > len(out) - int(0.1 * rate):
                break
    else:  # clicks
        t = lead
        for _ in range(10):
            put(t, ring([float(rng.uniform(900, 1300))], [0.003], 0.02))
            t += int(0.125 * rate)
            if t > len(out) - int(0.05 * rate):
                break
    out *= _db(peak_dbfs) * 32767 / (np.max(np.abs(out)) + 1e-9)
    return np.clip(np.rint(out), -32768, 32767).astype("<i2").tobytes()


def normalise_active(pcm16k: bytes, target_dbov: float = -26.0) -> bytes:
    """Scale to an active level of ``target_dbov`` (the datasets' level), peak-limited."""
    x = np.frombuffer(pcm16k, dtype="<i2").astype(np.float64)
    frames = x[: len(x) // 320 * 320].reshape(-1, 320)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    active = rms > np.percentile(rms, 95) * 10 ** (-25 / 20)
    level = float(np.sqrt(np.mean(frames[active] ** 2))) if active.any() else 1.0
    gain = 32768 * _db(target_dbov) / level
    gain = min(gain, 30000 / (np.max(np.abs(x)) + 1e-9))
    return np.clip(np.rint(x * gain), -32768, 32767).astype("<i2").tobytes()


def trim_silence(pcm16k: bytes, keep_ms: int = 60) -> bytes:
    x = np.frombuffer(pcm16k, dtype="<i2").astype(np.float64)
    frames = x[: len(x) // 320 * 320].reshape(-1, 320)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    hot = np.nonzero(rms > np.max(rms) * 10 ** (-35 / 20))[0]
    if not len(hot):
        return pcm16k
    keep = keep_ms // 20
    lo, hi = max(hot[0] - keep, 0), min(hot[-1] + 1 + keep, len(rms))
    return frames[lo:hi].astype("<i2").tobytes()


# The rig: two gateways (baseline and fix) over one pair of real backends


class Rig:
    def __init__(self, capture_root: Path = HOLD) -> None:
        self.root = capture_root
        self.addresses: dict[str, str] = {}
        self.config: dict[str, Any] = {}
        self.providers: Providers | None = None
        self.settings: dict[str, Any] = {}
        self.stack = contextlib.AsyncExitStack()

    def _raw(self, confirm: bool, capture: Path) -> dict[str, Any]:
        raw = yaml.safe_load((REPO / "config.yaml").read_text())
        raw["barge_in"]["confirm_while_playing"] = confirm
        raw["aec"]["capture_dir"] = str(capture)
        raw["aec"]["capture_max_s"] = 1800
        raw["limits"]["inbox_max_ms"] = 10_000
        raw["observability"]["log_level"] = "warning"
        return raw

    async def __aenter__(self) -> Rig:
        support.TIMEOUT_S = 900.0
        base = validate_config(self._raw(False, self.root / "baseline"))
        self.providers = Providers(
            llm=EchoLLM(),
            asr=build_asr(base.providers.asr),
            tts=build_tts(base.providers.tts, VoiceBook(base.voices)),
        )
        for name, confirm in (("baseline", False), ("fix", True)):
            config = validate_config(self._raw(confirm, self.root / name))
            self.config[name] = config
            gateway = Gateway(config, self.providers)
            self.addresses[name] = await self.stack.enter_async_context(support.running(gateway))
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stack.aclose()

    async def synth(self, text: str, voice: str, lang: str) -> bytes:
        """``text`` spoken by the gateway's TTS in ``voice``, as 16 kHz PCM, silence trimmed."""
        assert self.providers is not None
        raw = b""
        async for chunk in self.providers.tts.stream(text, voice, lang):  # type: ignore[arg-type]
            raw += chunk.pcm
        resampler = Resampler(24000, 16000)
        pcm = resampler.process(raw) + resampler.flush()
        return normalise_active(trim_silence(pcm))

    @asynccontextmanager
    async def session(
        self, condition: str, *, echo_db: float | None = None, delay_s: float = 0.1, seed: int = 7,
        warmup_text: str | None = None,
    ) -> AsyncIterator[tuple[Client, HoldCall, str]]:
        """A session on the AICC profile (create_response off: the harness drives the answers),
        its call running; the warm-up answer lets the canceller settle (bypass on a line with no
        echo) before the first trial."""
        async with Client.connect(self.addresses[condition]) as client:
            created = await client.until("conversation.created")
            await client.send(
                {
                    "type": "session.update",
                    "session": {
                        "type": "realtime",
                        **AICC_SESSION,
                        "audio": {
                            **AICC_SESSION["audio"],  # type: ignore[dict-item]
                            "input": {
                                **AICC_SESSION["audio"]["input"],  # type: ignore[index]
                                "turn_detection": {
                                    "type": "server_vad",
                                    "silence_duration_ms": 500,
                                    "create_response": False,
                                    "interrupt_response": True,
                                },
                                "transcription": {"model": "qwen3-asr"},
                            },
                        },
                    },
                }
            )
            updated = await client.until("session.updated")
            session_id = next(
                e["session"]["id"]
                for e in [*created, *updated, *client.events]
                if e["type"] in ("session.created", "session.updated")
            )
            async with HoldCall(client, echo_db=echo_db, delay_s=delay_s, seed=seed) as call:
                await asyncio.sleep(0.5)
                if warmup_text:
                    await play_answer(call, client, warmup_text)
                    await drain(call, 40.0)
                    await asyncio.sleep(1.5)
                yield client, call, session_id


# Trial helpers


async def play_answer(call: HoldCall, client: Client, text: str) -> tuple[int, float]:
    """Ask for ``text`` as a verbatim response; returns (event index before it, playback start)."""
    call.done.clear()
    index, starts = len(call.events), len(call.log.playback_starts)
    await client.send({"type": "response.create", "response": {"instructions": SAY + text}})
    async with asyncio.timeout(30.0):
        while len(call.log.playback_starts) == starts:  # noqa: ASYNC110
            await asyncio.sleep(0.01)
    return index, call.log.playback_starts[-1]


async def drain(call: HoldCall, timeout_s: float) -> None:
    """Until the response is done and the client has played everything it was sent."""
    async with asyncio.timeout(timeout_s):
        await call.done.wait()
        while call.active_audio:  # noqa: ASYNC110
            await asyncio.sleep(0.05)


async def cleanup(call: HoldCall, client: Client) -> None:
    """End the trial's answer: cancel it if still being produced, and clear the client's queue
    (a hang-up of the audio; logged as cleanup, not as a flush by the gateway's events)."""
    call.cleaning = True
    try:
        if not call.done.is_set():
            await client.send({"type": "response.cancel"})
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(5.0):
                    await call.done.wait()
        if call.active_audio:
            await call._barge_in()  # noqa: SLF001
        await asyncio.sleep(0.4)
    finally:
        call.cleaning = False


def client_events(call: HoldCall, since: int, kinds: set[str] | None = None) -> list[dict[str, Any]]:
    out = []
    for t, event in call.events[since:]:
        if kinds is None or event["type"] in kinds:
            out.append({"t": t, **{k: v for k, v in event.items() if k not in ("delta", "response")},
                        **({"status": event["response"]["status"]} if event["type"] == "response.done" else {})})
    return out


async def wait_event(
    call: HoldCall, kind: str, since: int, timeout_s: float, *, item: str | None = None
) -> tuple[float, dict[str, Any]] | None:
    deadline = asyncio.get_running_loop().time() + timeout_s
    seen = since
    while True:
        while seen < len(call.events):
            t, event = call.events[seen]
            seen += 1
            if event["type"] == kind and (item is None or event.get("item_id") == item):
                return t, event
        left = deadline - asyncio.get_running_loop().time()
        if left <= 0:
            return None
        call._arrived.clear()  # noqa: SLF001
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(left):
                await call._arrived.wait()  # noqa: SLF001


# Reading the capture


def load_events(directory: Path) -> list[dict[str, Any]]:
    lines = (directory / "events.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def session_dir(root: Path, session_id: str) -> Path | None:
    found = sorted(root.glob(f"*_{session_id}"))
    return found[-1] if found else None


ITEM_KINDS = {
    "hold", "hold_check", "hold_end", "hold_release", "hold_drop", "hold_cancel",
    "speech_started", "speech_stopped", "transcript", "barge_in_outcome", "echo_turn",
}


def capture_items(events: list[dict[str, Any]], lo_ms: float, hi_ms: float) -> dict[str, Any]:
    """The turns whose gate fired (a hold, or a speech_started) in [lo, hi] input ms, with every
    event of theirs, and the canceller's bypass state at ``lo``."""
    by_item: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        if event["kind"] in ITEM_KINDS and event.get("item"):
            by_item.setdefault(event["item"], []).append(event)
    bypass = None
    for event in events:
        if event["kind"] == "bypass" and event.get("t") is not None and event["t"] <= lo_ms:
            bypass = event.get("on")
    items = []
    for item, evs in by_item.items():
        gate = [e for e in evs if e["kind"] in ("hold", "speech_started") and lo_ms <= e["t"] <= hi_ms]
        if not gate:
            continue
        first = min(gate, key=lambda e: e["t"])
        row: dict[str, Any] = {"item": item, "gate_t": first["t"], "onset_ms": first.get("onset_ms"),
                               "audio_start_ms": first.get("audio_start_ms")}
        row["held"] = any(e["kind"] == "hold" for e in evs)
        row["checks"] = [
            {k: e.get(k) for k in ("t", "text", "units", "unmatched", "verdict", "final", "check", "psr", "psr_why",
                                   "played_text", "audio_start_ms", "audio_end_ms")}
            for e in evs if e["kind"] == "hold_check"
        ]
        for kind in ("hold_release", "hold_drop", "hold_cancel", "hold_end", "speech_started",
                     "speech_stopped", "transcript", "barge_in_outcome", "echo_turn"):
            match = [e for e in evs if e["kind"] == kind]
            if match:
                row[kind] = {k: v for k, v in match[0].items() if k not in ("kind", "item", "played_spans")}
        items.append(row)
    items.sort(key=lambda r: r["gate_t"])
    return {"bypass": bypass, "items": items}
