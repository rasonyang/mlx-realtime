"""A3 install check for the S11 echo-canceller candidates.

For each candidate: import it, list its public API, and push one second of synthetic 16 kHz mono
int16 audio through it in 10 ms frames, to prove that frames are processed. Nothing is measured:
the only per-candidate outputs are "frames processed", "output differs from input", and the
errors raised by deliberately wrong inputs (to record API constraints).

Run: uv run python install_check.py
Writes: ../results/a3_aec_install_check.json

Candidates that cannot stay in the lockfile were checked on 2026-10-22/23 and then removed with
`uv remove`; a re-run without them reports them as skipped, and the committed JSON is from the run
in which they were present:
- speexdsp 0.1.1 (xiongyihui) and speexaec 0.1.3 build only against Homebrew's libspeexdsp
  (speexdsp: CFLAGS/CXXFLAGS=-I/opt/homebrew/include LDFLAGS=-L/opt/homebrew/lib; speexaec:
  SPEEXDSP_PREFIX=/opt/homebrew), so `uv sync --locked` alone cannot rebuild them.
- pyaec 1.0.1 installs from a wheel, but never sets SPEEX_ECHO_SET_SAMPLING_RATE (8 kHz constants
  at any rate) and does not check buffer lengths, so it is not a faithful SpeexDSP vehicle.
speex_ctypes.py (the S11 part 2 SpeexDSP vehicle) needs only the Homebrew library at run time.
"""

from __future__ import annotations

import importlib.metadata as md
import json
import pathlib
import platform
import sys
import traceback
from typing import Any

import numpy as np

SR = 16000
FRAME = 160  # 10 ms
FILTER = 2048  # 128 ms at 16 kHz, for the engines that take a tail length
SECONDS = 1

rng = np.random.default_rng(0)
FAR = (rng.standard_normal(SR * SECONDS) * 3000).clip(-32768, 32767).astype(np.int16)
_delay = int(0.040 * SR)  # synthetic "echo": far end delayed 40 ms, -10 dB, plus low noise
_echo = np.concatenate([np.zeros(_delay), FAR[:-_delay] * 10 ** (-10 / 20)])
NEAR = (_echo + rng.standard_normal(SR * SECONDS) * 30).clip(-32768, 32767).astype(np.int16)
N_FRAMES = SR * SECONDS // FRAME


def frames():
    for i in range(N_FRAMES):
        s = slice(i * FRAME, (i + 1) * FRAME)
        yield NEAR[s], FAR[s]


def public_api(obj: Any) -> list[str]:
    return sorted(n for n in dir(obj) if not n.startswith("_"))


def version(dist: str) -> str | None:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


def check_livekit() -> dict[str, Any]:
    from livekit import rtc

    res: dict[str, Any] = {"dist": "livekit", "version": version("livekit")}
    res["api"] = public_api(rtc.AudioProcessingModule)
    apm = rtc.AudioProcessingModule(
        echo_cancellation=True, noise_suppression=False, high_pass_filter=False, auto_gain_control=False
    )
    apm.set_stream_delay_ms(0)
    n, changed = 0, False
    for near, far in frames():
        r = rtc.AudioFrame(far.tobytes(), SR, 1, FRAME)
        c = rtc.AudioFrame(near.tobytes(), SR, 1, FRAME)
        apm.process_reverse_stream(r)  # render (reference) first, then capture: lock-step
        apm.process_stream(c)  # in place
        out = np.frombuffer(c.data, dtype=np.int16)
        changed |= not np.array_equal(out, near)
        n += 1
    res["frames_processed"] = n
    res["output_differs_from_input"] = bool(changed)

    rates: dict[str, str] = {}
    for sr in (8000, 16000, 24000, 32000, 44100, 48000):
        try:
            a = rtc.AudioProcessingModule(echo_cancellation=True)
            k = sr // 100
            a.process_reverse_stream(rtc.AudioFrame(bytes(2 * k), sr, 1, k))
            a.process_stream(rtc.AudioFrame(bytes(2 * k), sr, 1, k))
            rates[str(sr)] = "accepted (10 ms frame)"
        except Exception as e:  # noqa: BLE001 - recording the constraint
            rates[str(sr)] = f"rejected: {e}"
    res["rates_10ms_frames"] = rates
    try:
        a = rtc.AudioProcessingModule(echo_cancellation=True)
        a.process_stream(rtc.AudioFrame(bytes(2 * 320), SR, 1, 320))
        res["frame_20ms_at_16k"] = "accepted"
    except Exception as e:  # noqa: BLE001
        res["frame_20ms_at_16k"] = f"rejected: {e}"
    # Which half of a 20 ms frame is touched? The high-pass filter changes any processed sample.
    a = rtc.AudioProcessingModule(high_pass_filter=True)
    x = NEAR[:320].copy()
    c = rtc.AudioFrame(x.tobytes(), SR, 1, 320)
    a.process_stream(c)
    y = np.frombuffer(c.data, dtype=np.int16)
    res["frame_20ms_at_16k_hpf_only"] = {
        "first_10ms_changed": bool(not np.array_equal(y[:160], x[:160])),
        "second_10ms_changed": bool(not np.array_equal(y[160:], x[160:])),
    }
    # A frame that is not a whole number of 10 ms: run in a child process, because it may abort.
    import subprocess

    code = (
        "from livekit import rtc\n"
        "a = rtc.AudioProcessingModule(echo_cancellation=True)\n"
        "a.process_stream(rtc.AudioFrame(bytes(480), 16000, 1, 240))\n"
        "print('returned normally')\n"
    )
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    panic = [ln for ln in p.stderr.splitlines() if "panicked" in ln or "multiple of 10ms" in ln]
    res["frame_15ms_at_16k_child_process"] = {"returncode": p.returncode, "stderr_panic_lines": panic}
    return res


def check_pywebrtc_audio() -> dict[str, Any]:
    if version("pywebrtc-audio") is None:
        return {"dist": "pywebrtc-audio", "skipped": "not installed"}
    import subprocess

    import pywebrtc_audio

    res: dict[str, Any] = {"dist": "pywebrtc-audio", "version": version("pywebrtc-audio")}
    res["api"] = public_api(pywebrtc_audio.EchoCanceller)
    ec = pywebrtc_audio.EchoCanceller(sample_rate=SR, num_channels=1, stream_delay_ms=0)
    n, changed = 0, False
    for near, far in frames():
        out = np.asarray(ec.process(near, far), dtype=np.int16)
        changed |= not np.array_equal(out, near)
        n += 1
    res["frames_processed"] = n
    res["output_differs_from_input"] = bool(changed)
    ec2 = pywebrtc_audio.EchoCanceller(sample_rate=SR)
    whole = np.asarray(ec2.process(NEAR, FAR))
    res["one_call_1s_arrays"] = {"out_len": int(whole.shape[0]), "in_len": int(NEAR.shape[0])}
    code = (
        "import numpy as np, pywebrtc_audio as p\n"
        "e = p.EchoCanceller(sample_rate=16000)\n"
        "o = e.process(np.zeros(240, np.int16), np.zeros(240, np.int16))\n"
        "print('returned', len(o))\n"
    )
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    res["frame_15ms_at_16k_child_process"] = {
        "returncode": p.returncode,
        "stdout": p.stdout.strip()[-200:],
        "stderr_tail": p.stderr.strip()[-300:],
    }
    return res


def check_speexaec() -> dict[str, Any]:
    if version("speexaec") is None:
        return {"dist": "speexaec", "skipped": "not installed (removed after the A3 check; see module docstring)"}
    import speexaec

    res: dict[str, Any] = {"dist": "speexaec", "version": version("speexaec")}
    res["api"] = public_api(speexaec.EchoCanceller)
    ec = speexaec.EchoCanceller(FRAME, FILTER, SR)
    n, changed = 0, False
    for near, far in frames():
        out = np.asarray(ec.process(near, far), dtype=np.int16)
        changed |= not np.array_equal(out, near)
        n += 1
    res["frames_processed"] = n
    res["output_differs_from_input"] = bool(changed)
    try:
        ec.process(NEAR[:100], FAR[:100])
        res["short_frame"] = "accepted"
    except Exception as e:  # noqa: BLE001 - recording the constraint
        res["short_frame"] = f"rejected: {e!r}"
    return res


def check_speex_ctypes() -> dict[str, Any]:
    from speex_ctypes import _LIB_PATH, SpeexAec

    res: dict[str, Any] = {"dist": "speex_ctypes.py over Homebrew libspeexdsp", "lib": _LIB_PATH}
    res["api"] = public_api(SpeexAec)
    for preprocess in (False, True):
        ec = SpeexAec(FRAME, FILTER, SR, preprocess=preprocess, denoise=False)
        n, changed = 0, False
        for near, far in frames():
            out = ec.process(near, far)
            changed |= not np.array_equal(out, near)
            n += 1
        res[f"preprocess={preprocess}"] = {"frames_processed": n, "output_differs_from_input": bool(changed)}
    return res


def check_pyaec() -> dict[str, Any]:
    if version("pyaec") is None:
        return {"dist": "pyaec", "skipped": "not installed (removed after the A3 check; see module docstring)"}
    import pyaec

    res: dict[str, Any] = {"dist": "pyaec", "version": version("pyaec")}
    res["api"] = public_api(pyaec.Aec)
    res["native_lib"] = pathlib.Path(pyaec.__file__).with_name("libaec.dylib").name
    for preprocess in (False, True):
        aec = pyaec.Aec(FRAME, FILTER, SR, preprocess)
        n, changed = 0, False
        for near, far in frames():
            out = np.asarray(aec.cancel_echo(near.tolist(), far.tolist()), dtype=np.int16)
            changed |= not np.array_equal(out, near)
            n += 1
        res[f"enable_preprocess={preprocess}"] = {
            "frames_processed": n,
            "output_differs_from_input": bool(changed),
        }
    return res


def check_speexdsp() -> dict[str, Any]:
    if version("speexdsp") is None:
        return {"dist": "speexdsp", "skipped": "not installed (removed after the A3 check; see module docstring)"}
    from speexdsp import EchoCanceller

    res: dict[str, Any] = {"dist": "speexdsp", "version": version("speexdsp")}
    res["api"] = public_api(EchoCanceller)
    ec = EchoCanceller.create(FRAME, FILTER, SR)
    n, changed = 0, False
    for near, far in frames():
        out = np.frombuffer(ec.process(near.tobytes(), far.tobytes()), dtype=np.int16)
        changed |= not np.array_equal(out, near)
        n += 1
    res["frames_processed"] = n
    res["output_differs_from_input"] = bool(changed)
    return res


def check_baseline() -> dict[str, Any]:
    from baseline import PbfdafGeigel

    res: dict[str, Any] = {"dist": "numpy (self-written baseline.py)", "version": version("numpy")}
    res["api"] = public_api(PbfdafGeigel)
    ec = PbfdafGeigel(sample_rate=SR, block=FRAME, tail_ms=128.0)
    res["tail_samples"] = ec.tail_samples
    n, changed, finite = 0, False, True
    for near, far in frames():
        out = ec.process(near, far)
        finite &= bool(np.all(np.isfinite(out)))
        changed |= not np.array_equal(np.round(out).astype(np.int16), near)
        n += 1
    res["frames_processed"] = n
    res["output_differs_from_input"] = bool(changed)
    res["output_finite"] = finite
    return res


def main() -> None:
    report: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "signal": {"sample_rate": SR, "frame": FRAME, "seconds": SECONDS, "kind": "seeded white noise; capture = far end delayed 40 ms at -10 dB + noise"},
        "candidates": {},
    }
    for name, fn in (
        ("livekit_apm", check_livekit),
        ("pywebrtc_audio", check_pywebrtc_audio),
        ("speex_ctypes", check_speex_ctypes),
        ("speexaec", check_speexaec),
        ("pyaec", check_pyaec),
        ("speexdsp_xiongyihui", check_speexdsp),
        ("numpy_baseline", check_baseline),
    ):
        try:
            report["candidates"][name] = {"ok": True, **fn()}
        except Exception as e:  # noqa: BLE001 - an install check records failures
            report["candidates"][name] = {"ok": False, "error": repr(e), "trace": traceback.format_exc()}
    out = pathlib.Path(__file__).resolve().parent.parent / "results" / "a3_aec_install_check.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
