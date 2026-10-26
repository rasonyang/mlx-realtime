"""Echo captures from the simulated AICC call, with known ground truth (the validation of analyze.py).

Run in the gateway's environment (it imports tests/phone.py and the gateway):

    uv run --locked python spikes/live_echo/simulate.py [--out spikes/live_echo/captures/simulated] [name ...]

Each scenario runs the real gateway in real time with ``aec.capture_dir`` on: the simulated call
(tests/phone.py: AICC's 24 -> 8 kHz downsampler and playout, G.711 mu-law both ways, a synthetic
or ITU-T G.168 echo path after a bulk delay, AICC's 8 -> 24 kHz uplink) hears three of our
answers in silence, then a caller who talks over a fourth. Beside the capture it writes
``ground_truth.json``: the path's bulk delay, ERL and dominant tap, and the input-timeline time
at which the simulated client really started playing each run (what the playout model estimates).
The G.168 coefficients (git-ignored, (c) ITU) are used only if present locally.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).parents[2]
sys.path.insert(0, str(REPO))

from tests.integration.test_echo import AICC_SESSION, CALLER, LINE, SAY, echo_session  # noqa: E402
from tests.phone import FRAME_8K, SimulatedCall, echo_path  # noqa: E402
from tests.speech import EN, clip  # noqa: E402

G168 = REPO / "spikes" / "data" / "echo" / "g168_paths.json"
AICC_DOWNSAMPLER_DELAY_MS = 16 / 24  # the 33-tap symmetric FIR at 24 kHz
FRAME_MS = 20.0

SCENARIOS: dict[str, dict[str, object]] = {
    "erl10_d100": {"erl_db": 10.0, "delay_ms": 100.0},
    "erl15_d180": {"erl_db": 15.0, "delay_ms": 180.0, "seed": 11},
    "g168_m4": {"erl_db": 12.0, "delay_ms": 120.0, "g168": "m4"},
    "no_echo": {"erl_db": None, "delay_ms": 100.0},
}


class TruthCall(SimulatedCall):
    """The simulated call, logging the uplink frame at which each playback run really starts."""

    def __init__(self, *args: object, path: np.ndarray | None = None, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        if path is not None:
            self._echo = path
            self._overlap = np.zeros(len(path) - 1)
        self.uplinks = 0
        self.start_frames: list[int] = []

    @property
    def playing(self) -> bool:
        return self._playing or bool(self._queue)

    def _playout(self, now: float):  # type: ignore[no-untyped-def]
        before = len(self.log.playback_starts)
        out = super()._playout(now)
        if len(self.log.playback_starts) > before:
            self.start_frames.append(self.uplinks)  # this tick's frame is input frame `uplinks`
        return out

    async def _uplink(self, played: np.ndarray) -> None:
        await super()._uplink(played)
        self.uplinks += 1


def path_of(spec: dict[str, object]) -> tuple[np.ndarray | None, dict[str, object]]:
    """The echo path at 8 kHz and its description."""
    erl = spec["erl_db"]
    if erl is None:
        return None, {"erl_db": None, "peak_tap": None}
    if "g168" in spec:
        models = {m["name"]: m for m in json.loads(G168.read_text())["models"]}
        model = models[str(spec["g168"])]
        floor = float(model["min_erl_css_db"])
        erl = max(float(erl), floor)  # type: ignore[arg-type]
        taps = 10 ** (-erl / 20) * float(model["K_css"]) * np.asarray(model["coefficients"], float)
        return taps, {"erl_db": erl, "peak_tap": int(model["peak_index"]), "g168": model["name"]}
    taps = echo_path(int(spec.get("seed", 7)), erl_db=float(erl))  # type: ignore[arg-type]
    return taps, {"erl_db": float(erl), "peak_tap": int(np.argmax(np.abs(taps)))}  # type: ignore[arg-type]


async def run(name: str, spec: dict[str, object], out: Path) -> None:
    path, described = path_of(spec)
    delay_ms = float(spec["delay_ms"])  # type: ignore[arg-type]
    capture_dir = out / name
    for old in capture_dir.glob("*") if capture_dir.exists() else []:  # only our own scenario dir
        for file in old.iterdir():
            file.unlink()
        old.rmdir()
    config = {"aec": {"capture_dir": str(capture_dir)}}
    erl = spec["erl_db"]
    async with (
        echo_session(config, AICC_SESSION) as client,
        TruthCall(
            client,
            echo_db=None if erl is None else float(erl),  # type: ignore[arg-type]
            delay_s=delay_ms / 1000,
            seed=int(spec.get("seed", 7)),  # type: ignore[arg-type]
            path=path,
        ) as call,
    ):
        await asyncio.sleep(0.5)
        for _ in range(3):  # three answers heard in silence
            call.done.clear()
            await client.send(
                {"type": "response.create", "response": {"instructions": SAY + LINE}}
            )
            async with asyncio.timeout(60.0):
                await call.done.wait()
                while call.playing:  # noqa: ASYNC110 (the server is up to 2 s ahead of the client)
                    await asyncio.sleep(0.05)
            await asyncio.sleep(1.5)
        call.done.clear()
        await client.send({"type": "response.create", "response": {"instructions": SAY + LINE}})
        before = len(call.log.playback_starts)
        async with asyncio.timeout(10.0):
            while len(call.log.playback_starts) == before:  # noqa: ASYNC110
                await asyncio.sleep(0.01)
        await asyncio.sleep(6.0)
        caller_at = call.uplinks * FRAME_MS  # the caller's first frame is the next one
        call.say(clip(EN))
        async with asyncio.timeout(10.0):
            await call.done.wait()
        await asyncio.sleep(4.0)  # the caller's turn is transcribed and answered
        starts = [frame * FRAME_MS for frame in call.start_frames]
    await asyncio.sleep(1.5)  # the capture's writer thread finishes the files
    peak = described["peak_tap"]
    truth = {
        "scenario": name,
        "erl_db": described["erl_db"],
        "bulk_delay_ms": delay_ms,
        "peak_tap": peak,
        # Echo of audio the client played at input time t shows at t + this (the dominant tap,
        # after the downsampler's group delay):
        "dominant_delay_ms": None
        if peak is None
        else delay_ms + peak * 1000 / 8000 + AICC_DOWNSAMPLER_DELAY_MS,
        "client_play_start_ms": starts,
        "caller_onset_ms": caller_at,
        "frame_8k": FRAME_8K,
        "g168": described.get("g168"),
        "path": [] if path is None else [float(v) for v in path],  # in captures/ only: not committed
        "caller_line": CALLER,
    }
    found = sorted(capture_dir.iterdir())
    if not found:
        raise SystemExit(f"{name}: the gateway wrote no capture")
    (found[-1] / "ground_truth.json").write_text(json.dumps(truth, indent=2) + "\n")
    print(f"{name}: {found[-1]}")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*", help=f"scenarios (default: all of {list(SCENARIOS)})")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "captures" / "simulated")
    args = parser.parse_args()
    if args.out.name == "live" or "live" in args.out.parts:
        raise SystemExit("captures/live holds real call captures: simulate elsewhere")
    for name in args.names or list(SCENARIOS):
        spec = SCENARIOS[name]
        if "g168" in spec and not G168.exists():
            print(f"{name}: skipped, the G.168 coefficients are not present locally")
            continue
        await run(name, spec, args.out)


if __name__ == "__main__":
    asyncio.run(main())
