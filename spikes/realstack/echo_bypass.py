"""How often the GCC-PHAT delay estimator confirms an echo that is not there, and how often the
gateway's EchoReference leaves its no-echo bypass on it (session/echo.py).

    uv run --locked --project spikes/realstack python spikes/realstack/echo_bypass.py OUT.json

Reference: our output, stood in for by 40 FLEURS zh utterances back to back (16 kHz). Capture:
(a) a caller talking over it with no echo (40 FLEURS en utterances, at 0 and -15 dB re the far
end), (b) line noise only (-65 dBov), (c) AICC's harness room tone (a 182 Hz sine at amplitude
30), and with a real echo (a 120 ms delay at ERL 6 or 15 dB, plus the caller). For each: the
windows evaluated, the valid ones (PSR >= 8), the estimator's confirmations (two valid windows
agreeing within 2 ms) and those at a wrong lag; then the estimates through an EchoReference
whose bypass is engaged, counting its resumes (the bypass engaged again after each) and the
window of the first one. Run it on the working tree, and on the previous rule with
``git stash push src/mlx_realtime/session/echo.py``, to compare.
"""

from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

import numpy as np

from mlx_realtime.audio.delay import DelayEstimate, DelayEstimator
from mlx_realtime.config import AecConfig, BargeInConfig
from mlx_realtime.session.echo import EchoParams, EchoReference

DATA = Path(__file__).resolve().parents[2] / "spikes" / "data" / "asr" / "16k"
DELAY_MS = 120


class Clock:
    t = 100.0

    def now(self) -> float:
        return self.t

    async def sleep(self, seconds: float) -> None:
        self.t += seconds


def load(prefix: str, count: int) -> np.ndarray:
    parts = []
    for path in sorted(DATA.glob(f"{prefix}_*.wav"))[:count]:
        with wave.open(str(path)) as file:
            parts.append(np.frombuffer(file.readframes(file.getnframes()), dtype="<i2"))
    return np.concatenate(parts).astype(np.float64)


def estimates(reference: np.ndarray, capture: np.ndarray) -> list[DelayEstimate]:
    estimator = DelayEstimator()
    found: list[DelayEstimate] = []
    for i in range(0, len(reference) - 160, 160):
        frame = np.clip(capture[i : i + 160], -32768, 32767).astype(np.int16)
        found += estimator.push(frame, reference[i : i + 160].astype(np.int16))
    return found


def resumes(sequence: list[DelayEstimate]) -> tuple[int, int | None]:
    params = EchoParams.of(AecConfig.model_validate({"bypass_without_echo_ms": 1500}), BargeInConfig())
    echo = EchoReference(params, Clock())
    silent = DelayEstimate(0, valid=False, psr=4.0)
    for _ in range(3):
        echo.delay_estimate(0, silent)
    assert echo.echo_absent
    count, first = 0, None
    for index, estimate in enumerate(sequence):
        echo.delay_estimate(0, estimate)
        if not echo.echo_absent:
            count += 1
            first = index if first is None else first
            for _ in range(3):  # bypass it again: count the next resume
                echo.delay_estimate(0, silent)
    return count, first


def main() -> None:
    rng = np.random.default_rng(2)
    reference, caller = load("zh", 40), load("en", 40)
    n = min(len(reference), len(caller))
    reference, caller = reference[:n], caller[:n]
    noise = rng.normal(0, 32768 * 10 ** (-65 / 20), n)
    tone = 30 * np.sin(np.arange(n) * 0.5 / 7.0)  # AICC's 30*sin(i/7) at 8 kHz, at 16 kHz
    delay = DELAY_MS * 16
    late = np.concatenate([np.zeros(delay), reference[:-delay]])
    cases: dict[str, tuple[np.ndarray, int | None]] = {
        "caller 0 dB, no echo": (caller + noise, None),
        "caller -15 dB, no echo": (caller * 10 ** (-15 / 20) + noise, None),
        "line noise, no echo": (noise, None),
        "room tone, no echo": (tone, None),
        "echo ERL 6 dB": (late * 10 ** (-6 / 20) + noise, DELAY_MS),
        "echo ERL 6 dB + caller 0 dB": (late * 10 ** (-6 / 20) + caller + noise, DELAY_MS),
        "echo ERL 15 dB + caller 0 dB": (late * 10 ** (-15 / 20) + caller + noise, DELAY_MS),
        "echo ERL 15 dB + caller -15 dB": (
            late * 10 ** (-15 / 20) + caller * 10 ** (-15 / 20) + noise,
            DELAY_MS,
        ),
    }
    results: dict[str, object] = {"minutes": round(n / 16000 / 60, 2)}
    for name, (capture, true_ms) in cases.items():
        found = estimates(reference, capture)
        confirmed = [e for e in found if e.confirmed]
        wrong = [e for e in confirmed if true_ms is None or abs(e.lag / 16 - true_ms) > 2]
        count, first = resumes(found)
        results[name] = {
            "windows": len(found),
            "valid": sum(e.valid for e in found),
            "confirmed": len(confirmed),
            "confirmed_wrong_lag": len(wrong),
            "wrong_lags_ms": sorted({round(e.lag / 16) for e in wrong})[:12],
            "bypass_resumes": count,
            "first_resume_window": first,
        }
        print(name, json.dumps(results[name]))
    Path(sys.argv[1]).write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")


main()
