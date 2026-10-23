"""Shared helpers for M0 work package B2: spikes S3 (TTS), S4 (ASR) and S9 (normalization probe).

Throwaway spike code. Every script in this package imports this module; nothing here is production code.

- Servers: `mlx_audio.server` 0.6.1, started with HF_HUB_OFFLINE=1 and PYTHONUNBUFFERED=1. Its output is
  read by a tee thread that prefixes every line with time.perf_counter_ns() (mach_absolute_time on macOS,
  so it is comparable with the client's timestamps and with the traced server's own trace lines).
- HTTP client: a raw-socket HTTP/1.1 client that parses the chunked body itself, so every chunk the
  server sends is seen with its exact size and arrival time. Reads block (no busy-spin).
- Scoring: the A2 `text_norm` rule (copied from spikes/datasets/a2_data.py `norm_text`), and
  CER / WER / MER with substitution, deletion and insertion counts.
"""

from __future__ import annotations

import datetime as dt
import importlib.metadata as md
import io
import json
import os
import platform
import re
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
import uuid
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Every model is pre-downloaded; nothing may be fetched behind our back (orchestrator rule). Set before
# any huggingface_hub / mlx_audio import in the scripts that import this module.
os.environ["HF_HUB_OFFLINE"] = "1"

SPIKE_DIR = Path(__file__).resolve().parent
REPO = SPIKE_DIR.parents[1]
RESULTS = REPO / "spikes" / "results"
DATA = REPO / "spikes" / "data"
# Server logs, the jieba cache (TMPDIR) and other scratch files; set B2_SCRATCH to choose the place.
SCRATCH = Path(os.environ.get("B2_SCRATCH") or Path(tempfile.gettempdir()) / "mlxrt_b2")
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"

HOST = "127.0.0.1"
ASR_PORT = 8102
TTS_PORT = 8103

TTS = "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit"
TTS_4BIT = "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit"
TTS_17B = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
ASR = "mlx-community/Qwen3-ASR-1.7B-8bit"
ASR_06B = "mlx-community/Qwen3-ASR-0.6B-8bit"
KOKORO = "mlx-community/Kokoro-82M-bf16"

TTS_RATE = 24000  # Qwen3-TTS and Kokoro both emit 24 kHz (A1 smoke check)
ASR_RATE = 16000

# lang_code values the gateway would send (ARCHITECTURE §5.3: chinese / english / auto; mixed text is
# tagged zh). Kokoro uses its own codes: z = Mandarin (misaki[zh]), a = American English.
QWEN_LANG = {"zh": "chinese", "en": "english", "mix": "chinese"}
KOKORO_LANG = {"zh": "z", "en": "a", "mix": "z"}
ASR_LANG = {"zh": "Chinese", "en": "English", "mix": "Chinese"}


# ------------------------------------------------------------------------------------------------
# Environment and models
# ------------------------------------------------------------------------------------------------
def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def loadavg1() -> float:
    return round(os.getloadavg()[0], 2)


def models_json() -> dict:
    return json.loads((RESULTS / "models.json").read_text())


def model_entry(repo: str) -> dict | None:
    return next((m for m in models_json()["models"] if m["repo"] == repo), None)


def model_ok(repo: str) -> bool:
    """A comparison model is used only once models.json lists it with status ok (orchestrator rule)."""
    m = model_entry(repo)
    return bool(m and m.get("status") == "ok")


def snapshot_dir(repo: str) -> Path:
    d = HF_CACHE / ("models--" + repo.replace("/", "--"))
    return d / "snapshots" / (d / "refs" / "main").read_text().strip()


def model_revision(repo: str) -> dict:
    m = model_entry(repo) or {}
    snap = None
    try:
        snap = snapshot_dir(repo).name
    except OSError:
        pass
    return {"repo": repo, "sha_models_json": m.get("sha"), "status": m.get("status"), "sha_local_ref": snap}


def kokoro_voice_path(name: str) -> str:
    """mlx-audio 0.6.1 resolves a Kokoro voice *name* in prince-canuma/Kokoro-82M; offline we pass the
    .safetensors path inside our own mlx-community/Kokoro-82M-bf16 snapshot instead."""
    p = snapshot_dir(KOKORO) / "voices" / f"{name}.safetensors"
    assert p.is_file(), p
    return str(p)


def environment() -> dict:
    """Versions, device info and settings, as SPIKE.md §1 asks every run to record."""
    pkgs = ["mlx", "mlx-metal", "mlx-audio", "transformers", "numpy", "soxr", "misaki", "fastapi",
            "uvicorn", "starlette", "httptools", "huggingface-hub"]
    versions = {}
    for p in pkgs:
        try:
            versions[p] = md.version(p)
        except md.PackageNotFoundError:
            versions[p] = None
    try:
        import mlx.core as mx

        dev = mx.device_info()
    except Exception as exc:  # noqa: BLE001
        dev = repr(exc)

    def sysctl(name: str) -> str | None:
        r = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True)
        return r.stdout.strip() or None

    return {
        "recorded_at": now_iso(),
        "python": sys.version.split()[0],
        "macos": platform.mac_ver()[0],
        "versions": versions,
        "mx_device_info": dev,
        "sysctl": {k: sysctl(k) for k in ("iogpu.wired_limit_mb", "hw.memsize", "machdep.cpu.brand_string",
                                          "vm.swapusage")},
        "uv": subprocess.run(["uv", "--version"], capture_output=True, text=True).stdout.strip(),
    }


# ------------------------------------------------------------------------------------------------
# Statistics
# ------------------------------------------------------------------------------------------------
def stats(xs) -> dict:
    """n, p50, p95, max (numpy linear-interpolated percentiles), plus min and mean."""
    a = np.asarray([x for x in xs if x is not None], dtype=float)
    if a.size == 0:
        return {"n": 0}
    p50, p95 = np.percentile(a, [50, 95])
    return {"n": int(a.size), "p50": round(float(p50), 2), "p95": round(float(p95), 2),
            "max": round(float(a.max()), 2), "min": round(float(a.min()), 2),
            "mean": round(float(a.mean()), 2)}


# ------------------------------------------------------------------------------------------------
# Raw-socket HTTP/1.1 client
# ------------------------------------------------------------------------------------------------
class _Reader:
    def __init__(self, sock: socket.socket, t0: int):
        self.sock, self.t0, self.buf = sock, t0, bytearray()
        self.first_recv_ns: int | None = None

    def _fill(self) -> bool:
        data = self.sock.recv(1 << 18)  # blocking read
        if not data:
            return False
        if self.first_recv_ns is None:
            self.first_recv_ns = time.perf_counter_ns()
        self.buf += data
        return True

    def readline(self) -> bytes:
        while True:
            i = self.buf.find(b"\r\n")
            if i >= 0:
                line = bytes(self.buf[:i])
                del self.buf[: i + 2]
                return line
            if not self._fill():
                raise ConnectionError("connection closed while reading a line")

    def read_exact(self, n: int, on_first=None) -> bytes:
        """Read n bytes; on_first() is called once the first of them is in the buffer."""
        first_called = False
        while len(self.buf) < n:
            if self.buf and on_first and not first_called:
                on_first()
                first_called = True
            if not self._fill():
                raise ConnectionError(f"connection closed with {len(self.buf)}/{n} bytes")
        if on_first and not first_called:
            on_first()
        out = bytes(self.buf[:n])
        del self.buf[:n]
        return out

    def read_all(self) -> bytes:
        while self._fill():
            pass
        out = bytes(self.buf)
        self.buf.clear()
        return out


def ms(t_ns: int, t0_ns: int) -> float:
    return round((t_ns - t0_ns) / 1e6, 2)


@dataclass
class HttpResult:
    status: int | None = None
    headers: dict = field(default_factory=dict)
    t0_ns: int = 0  # perf_counter_ns just before connect()
    connect_ms: float | None = None
    headers_ms: float | None = None  # status line + headers fully received
    first_byte_ms: float | None = None  # first body byte (for TTS: first audio bytes)
    chunks: list = field(default_factory=list)  # [ms_first_byte, ms_complete, nbytes] per HTTP chunk
    end_ms: float | None = None
    body: bytes = b""
    closed_early: bool = False
    close_ns: int | None = None
    error: str | None = None

    def summary(self) -> dict:
        d = {k: getattr(self, k) for k in ("status", "connect_ms", "headers_ms", "first_byte_ms", "end_ms",
                                           "closed_early", "error")}
        d["bytes"] = len(self.body)
        d["n_chunks"] = len(self.chunks)
        d["content_type"] = self.headers.get("content-type")
        d["transfer_encoding"] = self.headers.get("transfer-encoding")
        return d


def http_request(port: int, method: str, path: str, body: bytes = b"", content_type: str | None = None,
                 *, close_after_chunks: int | None = None, timeout: float = 600.0) -> HttpResult:
    r = HttpResult()
    r.t0_ns = t0 = time.perf_counter_ns()
    sock = None
    try:
        sock = socket.create_connection((HOST, port), timeout=timeout)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        r.connect_ms = ms(time.perf_counter_ns(), t0)
        head = [f"{method} {path} HTTP/1.1", f"Host: {HOST}:{port}", "Connection: close",
                "Accept: */*", f"Content-Length: {len(body)}"]
        if content_type:
            head.append(f"Content-Type: {content_type}")
        sock.sendall(("\r\n".join(head) + "\r\n\r\n").encode() + body)
        rd = _Reader(sock, t0)
        status_line = rd.readline().decode("latin-1")
        r.status = int(status_line.split()[1])
        while True:
            line = rd.readline().decode("latin-1")
            if not line:
                break
            k, _, v = line.partition(":")
            r.headers[k.strip().lower()] = v.strip()
        r.headers_ms = ms(time.perf_counter_ns(), t0)
        out = bytearray()
        if r.headers.get("transfer-encoding", "").lower() == "chunked":
            while True:
                size = int(rd.readline().split(b";")[0], 16)
                if size == 0:
                    while rd.readline():  # trailers, then the final CRLF
                        pass
                    break
                first = []
                data = rd.read_exact(size, on_first=lambda first=first: first.append(time.perf_counter_ns()))
                t_done = time.perf_counter_ns()
                rd.read_exact(2)  # CRLF after the chunk data
                if r.first_byte_ms is None:
                    r.first_byte_ms = ms(first[0], t0)
                r.chunks.append([ms(first[0], t0), ms(t_done, t0), size])
                out += data
                if close_after_chunks is not None and len(r.chunks) >= close_after_chunks:
                    r.close_ns = time.perf_counter_ns()
                    r.closed_early = True
                    break
        else:
            n = r.headers.get("content-length")
            if n is not None:
                n = int(n)
                if n > 0:
                    first = []
                    out += rd.read_exact(n, on_first=lambda first=first: first.append(time.perf_counter_ns()))
                    r.first_byte_ms = ms(first[0], t0)
            else:
                out += rd.read_all()
                if out:
                    r.first_byte_ms = ms(rd.first_recv_ns, t0)
            if out:
                r.chunks.append([r.first_byte_ms, ms(time.perf_counter_ns(), t0), len(out)])
        r.end_ms = ms(time.perf_counter_ns(), t0)
        r.body = bytes(out)
    except Exception as exc:  # noqa: BLE001 - a spike records failures and carries on
        r.error = f"{type(exc).__name__}: {exc}"
    finally:
        if sock is not None:
            try:
                if r.closed_early:
                    sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()
    return r


def tts_request(payload: dict, port: int = TTS_PORT, **kw) -> HttpResult:
    return http_request(port, "POST", "/v1/audio/speech",
                        json.dumps(payload, ensure_ascii=False).encode(), "application/json", **kw)


def asr_request(wav: bytes, fields: dict[str, str], port: int = ASR_PORT) -> tuple[HttpResult, dict | None]:
    boundary = "b2" + uuid.uuid4().hex
    parts = [f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
             for k, v in fields.items() if v is not None]
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="clip.wav"\r\n'
                 "Content-Type: audio/wav\r\n\r\n".encode() + wav + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    r = http_request(port, "POST", "/v1/audio/transcriptions", b"".join(parts),
                     f"multipart/form-data; boundary={boundary}")
    js = None
    if r.body and (r.headers.get("content-type") or "").startswith("application/json"):
        try:
            js = json.loads(r.body)
        except ValueError:
            js = None
    return r, js


def get_json(port: int, path: str) -> dict | str:
    r = http_request(port, "GET", path, timeout=10)
    try:
        return json.loads(r.body)
    except ValueError:
        return r.error or r.body.decode("utf-8", "replace")


# ------------------------------------------------------------------------------------------------
# Server lifecycle
# ------------------------------------------------------------------------------------------------
def port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex((HOST, port)) == 0


def listener_pid(port: int) -> int | None:
    out = subprocess.run(["lsof", "-t", f"-iTCP:{port}", "-sTCP:LISTEN"], capture_output=True, text=True)
    pids = [int(p) for p in out.stdout.split()]
    return pids[0] if pids else None


def footprint_bytes(pid: int | None) -> int | None:
    if pid is None:
        return None
    out = subprocess.run(["footprint", "-p", str(pid), "--noCategories", "-f", "bytes"],
                         capture_output=True, text=True)
    m = re.search(r"Footprint:\s*(\d+)\s*B", out.stdout)
    return int(m.group(1)) if m else None


STARTED: list["Server"] = []


def stop_all() -> None:
    """Stop every server this process started that is still running (call from a finally block)."""
    for s in STARTED:
        if s.proc is not None and s.proc.poll() is None:
            print(f"stopping {s.name}: {s.stop()}")


class Server:
    """One mlx_audio.server process (stock, or the trace launcher s3_traced_server.py)."""

    def __init__(self, name: str, port: int, *, traced: bool = False, extra_env: dict | None = None):
        self.name, self.port, self.traced = name, port, traced
        self.dir = SCRATCH / "servers" / f"{name}_{dt.datetime.now():%H%M%S}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.dir / "server.log"
        self.lines: list[tuple[int, str]] = []
        self.proc: subprocess.Popen | None = None
        self.extra_env = extra_env or {}
        self.started_ns = 0
        self.ready_s = None

    @property
    def cmd(self) -> list[str]:
        base = ["uv", "run", "--locked", "--project", str(SPIKE_DIR), "python"]
        if self.traced:
            base += [str(SPIKE_DIR / "s3_traced_server.py")]
        else:
            base += ["-m", "mlx_audio.server"]
        return base + ["--host", HOST, "--port", str(self.port), "--log-dir", str(self.dir / "logs")]

    def describe(self) -> dict:
        cmd = " ".join(self.cmd).replace(str(SPIKE_DIR), "spikes/audio").replace(str(self.dir), "<log dir>")
        return {"name": self.name, "port": self.port, "traced": self.traced, "cmd": cmd,
                "env": {"HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1", "TMPDIR": "<scratch>"} | self.extra_env,
                "log": str(self.log_path), "seconds_to_ready": self.ready_s}

    def _tee(self) -> None:
        assert self.proc and self.proc.stdout
        with self.log_path.open("w") as f:
            pending = b""
            while True:
                data = self.proc.stdout.read1(65536)
                if not data:
                    break
                pending += data
                parts = re.split(rb"[\r\n]", pending)
                pending = parts.pop()
                t = time.perf_counter_ns()
                for p in parts:
                    if p.strip():
                        s = p.decode("utf-8", "replace")
                        self.lines.append((t, s))
                        f.write(f"{t} {s}\n")
                f.flush()

    def start(self, timeout: float = 240) -> "Server":
        if port_open(self.port):
            raise SystemExit(f"port {self.port} is already in use")
        tmp = SCRATCH / "tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1", TMPDIR=str(tmp), **self.extra_env)
        self.started_ns = time.perf_counter_ns()
        self.proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                                     cwd=str(REPO), start_new_session=True)
        STARTED.append(self)
        threading.Thread(target=self._tee, daemon=True).start()
        while (time.perf_counter_ns() - self.started_ns) / 1e9 < timeout:
            if self.proc.poll() is not None:
                raise SystemExit(f"{self.name} exited early, see {self.log_path}")
            if port_open(self.port):
                r = http_request(self.port, "GET", "/v1/models", timeout=5)
                if r.status == 200:
                    self.ready_s = round((time.perf_counter_ns() - self.started_ns) / 1e9, 2)
                    return self
            time.sleep(0.25)
        raise SystemExit(f"{self.name} not ready in {timeout} s")

    def pid(self) -> int | None:
        return listener_pid(self.port)

    def footprint(self) -> int | None:
        return footprint_bytes(self.pid())

    def log_since(self, t_ns: int) -> list[tuple[int, str]]:
        return [(t, s) for t, s in self.lines if t >= t_ns]

    def stop(self) -> dict:
        if not self.proc:
            return {}
        if self.proc.poll() is not None:
            return {"already_exited": True, "returncode": self.proc.returncode}
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            self.proc.wait(timeout=30)
            how = "SIGTERM"
        except subprocess.TimeoutExpired:
            os.killpg(self.proc.pid, signal.SIGKILL)
            self.proc.wait(timeout=10)
            how = "SIGKILL"
        for _ in range(40):
            if not port_open(self.port):
                break
            time.sleep(0.25)
        time.sleep(0.3)  # let the tee thread drain
        return {"stopped_with": how, "returncode": self.proc.returncode, "port_free_after": not port_open(self.port)}

    def __enter__(self) -> "Server":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


LOG_NOISE = re.compile(r"pkg_resources|import pkg_resources|Setuptools<81")


def interesting_log(lines: list[tuple[int, str]], t0_ns: int = 0) -> list[str]:
    """Server log lines other than uvicorn access lines and known noise, with ms relative to t0."""
    out = []
    for t, s in lines:
        if '"POST /v1/audio' in s or '"GET /v1/models' in s or LOG_NOISE.search(s):
            continue
        out.append(f"{(t - t0_ns) / 1e6:.1f} ms: {s}" if t0_ns else s)
    return out


# ------------------------------------------------------------------------------------------------
# Audio helpers
# ------------------------------------------------------------------------------------------------
def pcm_to_float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0


def float_to_pcm(x: np.ndarray) -> bytes:
    return np.clip(np.round(x * 32767), -32768, 32767).astype("<i2").tobytes()


def wav_bytes(pcm: bytes, rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def write_wav(path: Path, pcm: bytes, rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(wav_bytes(pcm, rate))


def read_wav(path: Path) -> tuple[bytes, int]:
    with wave.open(str(path), "rb") as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2, path
        return w.readframes(w.getnframes()), w.getframerate()


def tts_pcm_to_asr_wav(pcm: bytes, rate: int = TTS_RATE) -> bytes:
    """24 kHz s16le TTS output -> 16 kHz WAV for the ASR (soxr HQ, as A1 did)."""
    import soxr

    y = soxr.resample(pcm_to_float(pcm), rate, ASR_RATE)
    return wav_bytes(float_to_pcm(y), ASR_RATE)


def voiced_span(pcm: bytes, rate: int = TTS_RATE, frame_ms: int = 10, rel_db: float = 35.0,
                floor_db: float = -55.0) -> dict:
    """Leading/trailing silence by an energy threshold: a 10 ms frame is voiced when its RMS is within
    rel_db of the loudest frame and above floor_db (dBFS). Returns the span between the first and last
    voiced frame (internal pauses stay in)."""
    x = pcm_to_float(pcm)
    n = rate * frame_ms // 1000
    if x.size < n:
        return {"total_s": x.size / rate, "voiced_s": 0.0, "lead_s": None, "trail_s": None}
    frames = x[: x.size // n * n].reshape(-1, n)
    db = 10 * np.log10(np.mean(frames**2, axis=1) + 1e-12)
    thr = max(db.max() - rel_db, floor_db)
    idx = np.flatnonzero(db > thr)
    total = x.size / rate
    if idx.size == 0:
        return {"total_s": total, "voiced_s": 0.0, "lead_s": None, "trail_s": None, "threshold_db": thr}
    first, last = int(idx[0]), int(idx[-1])
    return {"total_s": round(total, 3), "voiced_s": round((last - first + 1) * frame_ms / 1000, 3),
            "lead_s": round(first * frame_ms / 1000, 3), "trail_s": round(total - (last + 1) * frame_ms / 1000, 3),
            "threshold_db": round(float(thr), 1)}


# ------------------------------------------------------------------------------------------------
# Text normalization and error rates
# ------------------------------------------------------------------------------------------------
CJK = "㐀-䶿一-鿿豈-﫿"
_SPACE_NEXT_TO_CJK = re.compile(rf"(?<=[{CJK}])\s+|\s+(?=[{CJK}])")
_CJK_CHAR = re.compile(rf"[{CJK}]")
_TOKEN = re.compile(rf"[{CJK}]|[^\s{CJK}]+")


def norm_text(text: str) -> str:
    """Same rule as spikes/datasets/a2_data.py norm_text (the manifest's text_norm):
    NFKC, curly apostrophes -> ', lower-case, every P*/S* char -> space (except an apostrophe between
    letters), collapse whitespace, drop whitespace next to CJK. Numbers are not verbalized."""
    s = unicodedata.normalize("NFKC", text).replace("’", "'").replace("‘", "'")
    s = s.lower()
    out = []
    for i, ch in enumerate(s):
        cat = unicodedata.category(ch)
        if ch == "'" and 0 < i < len(s) - 1 and s[i - 1].isalpha() and s[i + 1].isalpha():
            out.append(ch)
        elif cat[0] in "PS":
            out.append(" ")
        else:
            out.append(ch)
    s = re.sub(r"\s+", " ", "".join(out)).strip()
    return _SPACE_NEXT_TO_CJK.sub("", s)


def tokens(text_norm: str, kind: str) -> list[str]:
    """cer: characters without spaces; wer: whitespace words; mer: one token per CJK character and
    one per Latin/digit word (A2's suggested tokenization for code-switched speech)."""
    if kind == "cer":
        return [c for c in text_norm if not c.isspace()]
    if kind == "wer":
        return text_norm.split()
    if kind == "mer":
        return _TOKEN.findall(text_norm)
    raise ValueError(kind)


def units(text: str) -> int:
    """Length as the segmenter counts it (ARCHITECTURE §7): CJK characters plus Latin words."""
    return len(tokens(norm_text(text), "mer"))


def edit_ops(ref: list[str], hyp: list[str]) -> dict:
    """Levenshtein alignment with S/D/I counts."""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        ri = ref[i - 1]
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (ri != hyp[j - 1]))
    i, j, s, dl, ins = n, m, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            s += ref[i - 1] != hyp[j - 1]
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            dl += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return {"ref": n, "edits": int(d[n, m]), "sub": int(s), "del": int(dl), "ins": int(ins)}


def error_rate(ref_text: str, hyp_text: str, kind: str, *, ref_is_norm: bool = False) -> dict:
    ref = ref_text if ref_is_norm else norm_text(ref_text)
    hyp = norm_text(hyp_text or "")
    e = edit_ops(tokens(ref, kind), tokens(hyp, kind))
    e["rate"] = round(e["edits"] / e["ref"], 4) if e["ref"] else None
    return e


def pooled(errs: list[dict]) -> dict:
    """Corpus-level rate: total edits / total reference tokens."""
    ref = sum(e["ref"] for e in errs)
    tot = {k: sum(e[k] for e in errs) for k in ("edits", "sub", "del", "ins")}
    return {"n_utts": len(errs), "ref_tokens": ref, **tot, "rate": round(tot["edits"] / ref, 4) if ref else None}


KIND_FOR_SET = {"zh": "cer", "en": "wer", "mix": "mer"}


def save_json(name: str, obj: dict) -> Path:
    p = RESULTS / name
    p.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n")
    return p
