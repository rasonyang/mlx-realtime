"""A1 audio smoke check (M0): mlx_audio.server 0.6.1 with Qwen3-TTS, Qwen3-ASR and Kokoro.

Correctness only. Timings are single observations (the first request per model includes its lazy
load), not benchmarks. The script starts the server on 127.0.0.1:8103, sends a few requests, and stops
the server. Steps whose model is not in the local HF cache yet are recorded as skipped.

Run from the repository root:

    TMPDIR=<scratch dir> uv run --project spikes/audio python spikes/audio/smoke_audio.py

TMPDIR keeps jieba's dictionary cache and the server's --log-dir out of the system temp dir. The server
runs with HF_HUB_OFFLINE=1: every model is pre-downloaded, and the HF route here is slow.

Writes to spikes/results/: a1_audio_smoke.json, a1_tts_smoke.pcm (the Qwen3-TTS zh output, raw s16le
mono 24 kHz), a1_tts_smoke_16k.wav (that PCM resampled with soxr for the ASR request), and
a1_kokoro_{zh,en}.pcm.
"""

from __future__ import annotations

import datetime as dt
import http.client
import json
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
import uuid
import wave
from pathlib import Path

import numpy as np
import soxr
from huggingface_hub import constants

HOST, PORT = "127.0.0.1", 8103
SPIKE_DIR = Path(__file__).resolve().parent
RESULTS = SPIKE_DIR.parent / "results"
CACHE = Path(constants.HF_HUB_CACHE)

TTS = "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit"
ASR = "mlx-community/Qwen3-ASR-1.7B-8bit"
KOKORO = "mlx-community/Kokoro-82M-bf16"

TTS_TEXT_ZH = "您好，欢迎致电客服中心，请问有什么可以帮您？"
KOKORO_TEXT_ZH = "您好，欢迎致电客服中心，请问有什么可以帮您？"
KOKORO_TEXT_EN = "Hello, thank you for calling. How can I help you today?"


def snapshot(repo: str) -> Path | None:
    """Local snapshot dir if the cache holds a complete download (no partial blobs)."""
    d = CACHE / ("models--" + repo.replace("/", "--"))
    ref, blobs = d / "refs" / "main", d / "blobs"
    if not ref.is_file() or not blobs.is_dir():
        return None
    if any(p.name.endswith(".incomplete") for p in blobs.iterdir()):
        return None
    snap = d / "snapshots" / ref.read_text().strip()
    return snap if (snap / "config.json").is_file() else None


# --- HTTP helpers (stdlib, so chunk arrival is visible) -----------------------------------------
def post(path: str, body: bytes, content_type: str, timeout: float = 600) -> dict:
    t0 = time.perf_counter()
    conn = http.client.HTTPConnection(HOST, PORT, timeout=timeout)
    conn.request("POST", path, body=body, headers={"Content-Type": content_type})
    resp = conn.getresponse()
    ms_headers = (time.perf_counter() - t0) * 1000
    reads, data = [], bytearray()
    while chunk := resp.read1(1 << 20):  # at most one HTTP chunk per read on chunked bodies
        reads.append([round((time.perf_counter() - t0) * 1000, 1), len(chunk)])
        data += chunk
    conn.close()
    # http.client splits one HTTP chunk into several reads (8 KiB buffer); reads that arrive within
    # 2 ms of each other are merged into one arrival, which matches the server's chunks here.
    arrivals: list[list[float]] = []
    for t, n in reads:
        if arrivals and t - arrivals[-1][0] < 2.0:
            arrivals[-1][1] += n
        else:
            arrivals.append([t, n])
    return {
        "status": resp.status,
        "arrivals": arrivals,
        "content_type": resp.getheader("content-type"),
        "transfer_encoding": resp.getheader("transfer-encoding"),
        "ms_to_headers": round(ms_headers, 1),
        "ms_to_first_read": reads[0][0] if reads else None,
        "ms_total": round((time.perf_counter() - t0) * 1000, 1),
        "reads": reads,
        "bytes": len(data),
        "body": bytes(data),
    }


def speech(payload: dict) -> dict:
    r = post("/v1/audio/speech", json.dumps(payload, ensure_ascii=False).encode(), "application/json")
    r["request"] = payload
    return r


def transcribe(wav: bytes, fields: dict[str, str]) -> dict:
    boundary = "a1smoke" + uuid.uuid4().hex
    parts = [
        f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
        for k, v in fields.items()
    ]
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="clip.wav"\r\n'
        "Content-Type: audio/wav\r\n\r\n".encode()
        + wav
        + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    r = post("/v1/audio/transcriptions", b"".join(parts), f"multipart/form-data; boundary={boundary}")
    r["request"] = fields | {"file": f"clip.wav ({len(wav)} bytes)"}
    try:
        r["json"] = json.loads(r["body"])
    except ValueError:
        r["text_body"] = r["body"].decode("utf-8", "replace")[:2000]
    return r


# --- audio helpers ------------------------------------------------------------------------------
def pcm_stats(pcm: bytes, rate: int) -> dict:
    x = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    return {
        "bytes": len(pcm),
        "even_byte_count": len(pcm) % 2 == 0,
        "samples": int(x.size),
        "assumed_rate_hz": rate,
        "seconds": round(x.size / rate, 3),
        "peak_dbfs": round(20 * np.log10(max(float(np.abs(x).max(initial=0)), 1e-9)), 1),
        "rms_dbfs": round(10 * np.log10(max(float(np.mean(x**2)) if x.size else 0.0, 1e-18)), 1),
    }


def wav16k(pcm: bytes, rate: int) -> bytes:
    x = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    y = soxr.resample(x, rate, 16000)
    y16 = np.clip(np.round(y * 32767), -32768, 32767).astype("<i2")
    buf = tempfile.SpooledTemporaryFile()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(y16.tobytes())
    buf.seek(0)
    return buf.read()


def strip(r: dict) -> dict:
    return {k: v for k, v in r.items() if k != "body"}


# --- server lifecycle ---------------------------------------------------------------------------
def port_open() -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex((HOST, PORT)) == 0


def server_pid() -> int | None:
    out = subprocess.run(["lsof", "-t", f"-iTCP:{PORT}", "-sTCP:LISTEN"], capture_output=True, text=True)
    pids = [int(p) for p in out.stdout.split()]
    return pids[0] if pids else None


def footprint_bytes(pid: int | None) -> int | None:
    if pid is None:
        return None
    out = subprocess.run(["footprint", "-p", str(pid), "--noCategories", "-f", "bytes"],
                         capture_output=True, text=True)
    m = re.search(r"Footprint:\s*(\d+)\s*B", out.stdout)
    return int(m.group(1)) if m else None


def start_server(log_dir: Path) -> tuple[subprocess.Popen, Path, float]:
    if port_open():
        raise SystemExit(f"port {PORT} is already in use")
    log_path = log_dir / "server.out"
    env = dict(os.environ, HF_HUB_OFFLINE="1")
    cmd = ["uv", "run", "--project", str(SPIKE_DIR), "python", "-m", "mlx_audio.server",
           "--host", HOST, "--port", str(PORT), "--log-dir", str(log_dir / "logs")]
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=log_path.open("w"), stderr=subprocess.STDOUT, env=env,
                            start_new_session=True)
    while time.perf_counter() - t0 < 180:
        if proc.poll() is not None:
            raise SystemExit(f"server exited early, see {log_path}")
        try:
            conn = http.client.HTTPConnection(HOST, PORT, timeout=2)
            conn.request("GET", "/v1/models")
            if conn.getresponse().status == 200:
                return proc, log_path, time.perf_counter() - t0
        except OSError:
            time.sleep(0.5)
    raise SystemExit("server did not become ready in 180 s")


def stop_server(proc: subprocess.Popen) -> dict:
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=20)
        how = "SIGTERM"
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)
        how = "SIGKILL"
    for _ in range(20):
        if not port_open():
            break
        time.sleep(0.25)
    return {"stopped_with": how, "returncode": proc.returncode, "port_free_after": not port_open()}


# --- the smoke steps ----------------------------------------------------------------------------
def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    snaps = {repo: snapshot(repo) for repo in (TTS, ASR, KOKORO)}
    result: dict = {
        "check": "A1 audio smoke: mlx_audio.server on 127.0.0.1:8103 (correctness only, not a benchmark)",
        "started_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "models_local": {r: (s.name if s else None) for r, s in snaps.items()},
        "steps": {},
    }
    steps = result["steps"]

    # Facts read from the downloaded configs, not guessed.
    tts_voice = tts_lang = None
    if snaps[TTS]:
        cfg = json.loads((snaps[TTS] / "config.json").read_text())["talker_config"]
        result["tts_speakers"] = sorted(cfg["spk_id"])
        result["tts_dialect_speakers"] = {k: v for k, v in cfg["spk_is_dialect"].items() if v}
        result["tts_codec_languages"] = sorted(cfg["codec_language_id"])
        tts_voice = "vivian" if "vivian" in cfg["spk_id"] else sorted(cfg["spk_id"])[0]
        tts_lang = "chinese"
        assert tts_lang in cfg["codec_language_id"]
    asr_lang = None
    if snaps[ASR]:
        cfg = json.loads((snaps[ASR] / "config.json").read_text())
        langs = cfg.get("support_languages") or cfg.get("thinker_config", {}).get("support_languages")
        result["asr_support_languages"] = langs
        asr_lang = next((x for x in (langs or []) if x.lower() == "chinese"), None)
    kokoro_voices: list[str] = []
    if snaps[KOKORO]:
        kokoro_voices = sorted(p.stem for p in (snaps[KOKORO] / "voices").glob("*.safetensors"))
        result["kokoro_voices_zh"] = [v for v in kokoro_voices if v.startswith("z")]
        sr = json.loads((snaps[KOKORO] / "config.json").read_text()).get("sample_rate")
        result["kokoro_sample_rate"] = {
            "hz": sr or 24000,
            "source": "repo config.json" if sr else
            "repo config.json has none; mlx_audio/tts/models/kokoro/kokoro.py ModelConfig.sample_rate = 24000",
        }

    def run_step(name: str, fn) -> dict | None:
        """Run one request; record its result, or the exception, and carry on."""
        try:
            out = fn()
            steps[name] = out
            return out
        except Exception as exc:  # noqa: BLE001 - a smoke check records failures
            steps[name] = {"failed": f"{type(exc).__name__}: {exc}"}
            return None

    tmp = Path(tempfile.mkdtemp(prefix="a1_mlx_audio_"))  # kept: holds the server log
    proc, log_path, ready_s = start_server(tmp)
    result["server"] = {
        "cmd": f"uv run --project spikes/audio python -m mlx_audio.server --host {HOST} --port {PORT} "
        "--log-dir <tmp>/logs",
        "env": {"HF_HUB_OFFLINE": "1"},
        "seconds_to_ready": round(ready_s, 1),
        "log_file": str(log_path),
    }
    try:
        tts_pcm = None
        if snaps[TTS]:
            base = {"model": TTS, "input": TTS_TEXT_ZH, "voice": tts_voice, "lang_code": tts_lang,
                    "response_format": "pcm", "stream": True}

            def tts_cold() -> dict:  # default streaming_interval (2.0 s); cold: includes the lazy load
                r = speech(base)
                (RESULTS / "a1_tts_smoke.pcm").write_bytes(r["body"])
                return strip(r) | {"pcm": pcm_stats(r["body"], 24000), "_pcm": r["body"]}

            def tts_warm() -> dict:  # expect chunks of 0.32 s x 12.5 Hz = 4 codec frames
                r = speech(base | {"streaming_interval": 0.32})
                return strip(r) | {
                    "pcm": pcm_stats(r["body"], 24000),
                    "arrival_seconds_of_audio": [round(n / 2 / 24000, 3) for _, n in r["arrivals"]],
                }

            cold = run_step("tts_zh_stream_default_interval", tts_cold)
            if cold:
                tts_pcm = cold.pop("_pcm")
                (RESULTS / "a1_tts_smoke_16k.wav").write_bytes(wav16k(tts_pcm, 24000))
            run_step("tts_zh_stream_interval_0.32", tts_warm)
        else:
            steps["tts"] = {"skipped": f"{TTS} not downloaded yet"}

        if snaps[ASR] and tts_pcm:
            wav = (RESULTS / "a1_tts_smoke_16k.wav").read_bytes()
            run_step("asr_on_tts_auto_language",
                     lambda: strip(transcribe(wav, {"model": ASR, "response_format": "verbose_json"})))
            if asr_lang:
                run_step("asr_on_tts_language_name", lambda: strip(transcribe(
                    wav, {"model": ASR, "response_format": "verbose_json", "language": asr_lang})))
        else:
            steps["asr"] = {"skipped": "ASR model not downloaded yet, or no TTS audio"}

        if snaps[KOKORO]:
            zh_voice = next(v for v in kokoro_voices if v.startswith("zf_"))
            # mlx-audio 0.6.1 resolves a voice *name* in Model.REPO_ID = "prince-canuma/Kokoro-82M"
            # (the loader never passes repo_id), not in the Kokoro repo that was loaded. With the cache
            # holding only mlx-community/Kokoro-82M-bf16 and HF_HUB_OFFLINE=1 this is expected to fail;
            # a voice given as a path to a .safetensors file is loaded directly.
            run_step("kokoro_zh_voice_by_name", lambda: strip(speech(
                {"model": KOKORO, "input": KOKORO_TEXT_ZH, "voice": zh_voice, "lang_code": "z",
                 "response_format": "pcm", "stream": True})))
            for tag, text, lang, voice in (("zh", KOKORO_TEXT_ZH, "z", zh_voice),
                                           ("en", KOKORO_TEXT_EN, "a", "af_heart")):
                assert voice in kokoro_voices, voice
                voice_path = str(snaps[KOKORO] / "voices" / f"{voice}.safetensors")

                def kokoro(text=text, lang=lang, voice_path=voice_path, tag=tag) -> dict:
                    r = speech({"model": KOKORO, "input": text, "voice": voice_path, "lang_code": lang,
                                "response_format": "pcm", "stream": True})
                    (RESULTS / f"a1_kokoro_{tag}.pcm").write_bytes(r["body"])
                    return strip(r) | {"pcm": pcm_stats(r["body"], 24000), "_pcm": r["body"]}

                out = run_step(f"kokoro_{tag}_voice_path", kokoro)
                pcm = out.pop("_pcm") if out else b""
                if snaps[ASR] and pcm:
                    run_step(f"asr_on_kokoro_{tag}", lambda pcm=pcm: strip(transcribe(
                        wav16k(pcm, 24000), {"model": ASR, "response_format": "verbose_json"})))
        else:
            steps["kokoro"] = {"skipped": f"{KOKORO} not downloaded yet"}

        pid = server_pid()
        result["server"]["pid"] = pid
        result["server"]["footprint_bytes_after_requests"] = footprint_bytes(pid)
        try:
            conn = http.client.HTTPConnection(HOST, PORT, timeout=5)
            conn.request("GET", "/v1/models")
            result["server"]["v1_models_after"] = json.loads(conn.getresponse().read())
        except (OSError, ValueError) as exc:
            result["server"]["v1_models_after"] = repr(exc)
    finally:
        result["server"]["stop"] = stop_server(proc)
        lines = log_path.read_text(errors="replace").splitlines()
        result["server"]["log_line_count"] = len(lines)
        result["server"]["log_warnings_and_errors"] = [
            ln for ln in lines
            if re.search(r"warn|error|exception|traceback", ln, re.I) and "token" not in ln.lower()
        ][:60]
        # Keep every traceback block in full (up to 80 lines each).
        blocks, cur = [], None
        for ln in lines:
            if ln.startswith("Traceback") or "Exception in ASGI application" in ln:
                cur = [ln]
                blocks.append(cur)
            elif cur is not None and len(cur) < 80:
                cur.append(ln)
                if ln and not ln.startswith((" ", "\t", "Traceback", "During", "The above")) and len(cur) > 2:
                    cur = None
        result["server"]["log_tracebacks"] = blocks[:5]
        result["server"]["log_tail"] = [ln for ln in lines[-40:] if "token" not in ln.lower()]

    result["finished_at"] = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    out = RESULTS / "a1_audio_smoke.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    for name, step in steps.items():
        summary = step.get("skipped") or step.get("failed") or {
            k: step.get(k) for k in ("status", "ms_to_headers", "ms_to_first_read", "ms_total", "bytes")
        }
        print(name, summary)
        if "pcm" in step:
            print("   pcm:", step["pcm"])
        if "json" in step:
            print("   json:", json.dumps(step["json"], ensure_ascii=False)[:300])
    print("server:", {k: v for k, v in result["server"].items() if k not in ("log_tail", "log_warnings_and_errors")})
    print(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
