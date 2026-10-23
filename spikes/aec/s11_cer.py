"""S11 part 2: near-end ASR error during double talk, per canceller (dCER). Throwaway spike code.

    cd spikes/aec
    uv run python s11_cer.py            # waits for nothing: refuses to start unless B2 has finished
    uv run python s11_cer.py --force    # start anyway (only when the GPU is otherwise idle)

Input: the double-talk calls of the `dt` part (s11_run.py gen/run --part dt): every near-end utterance
+-300 ms, cut from (a) the clean chain (near end + noise through the same telephone uplink, no echo, no
canceller: the reference condition), (b) the raw capture (variant none:engine) and (c) each canceller's
output. 40 zh + 40 en FLEURS utterances x ERL {6, 15} dB x near end {0, -15} dB re the far end.

ASR: Qwen3-ASR-1.7B-8bit on mlx_audio.server 0.6.1 from spikes/audio (the S4 setup), started here on
127.0.0.1:8112 with HF_HUB_OFFLINE=1 and stopped at the end; one request at a time; multipart WAV
16 kHz, `language` given (Chinese / English), response_format verbose_json; 3 warm-up requests discarded.

Scoring: the manifest's text_norm rule (A2; the same as B2's norm_text): CER for zh, WER for en,
micro-averaged (total edits / total reference tokens). dCER = error(variant) - error(clean), paired by
utterance, with a 95 % bootstrap CI over utterances (2000 resamples, seed 0). Rows with numbers are
kept (their spelling differences cancel in the paired difference) and also reported without them.
Writes spikes/results/s11_cer.json.
"""

from __future__ import annotations

import s11_common as c

import argparse  # noqa: E402
import http.client  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import signal  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import unicodedata  # noqa: E402
import uuid  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import s11_chain as ch  # noqa: E402
import s11_run as run  # noqa: E402

PORT = 8112
HOST = "127.0.0.1"
MODEL = "mlx-community/Qwen3-ASR-1.7B-8bit"
LANG = {"zh": "Chinese", "en": "English"}
B2_DONE = c.RESULTS / "s3s4s9_report.md"
LOG_DIR = Path(os.environ.get("S11_ASR_LOGDIR", str(c.CACHE / "asr_server")))

# ------------------------------------------------------------------------------------- scoring

CJK = "㐀-䶿一-鿿豈-﫿"
_SPACE_NEXT_TO_CJK = re.compile(rf"(?<=[{CJK}])\s+|\s+(?=[{CJK}])")


def norm_text(text: str) -> str:
    """spikes/datasets/a2_data.py norm_text (= spikes/audio/b2_common.norm_text)."""
    s = unicodedata.normalize("NFKC", text).replace("’", "'").replace("‘", "'").lower()
    out = []
    for i, chh in enumerate(s):
        cat = unicodedata.category(chh)
        if chh == "'" and 0 < i < len(s) - 1 and s[i - 1].isalpha() and s[i + 1].isalpha():
            out.append(chh)
        elif cat[0] in "PS":
            out.append(" ")
        else:
            out.append(chh)
    s = re.sub(r"\s+", " ", "".join(out)).strip()
    return _SPACE_NEXT_TO_CJK.sub("", s)


def tokens(t: str, kind: str) -> list[str]:
    return [x for x in t if not x.isspace()] if kind == "cer" else t.split()


def edits(ref: list[str], hyp: list[str]) -> int:
    n, m = len(ref), len(hyp)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ref[i - 1] != hyp[j - 1]))
        prev = cur
    return prev[m]


# ------------------------------------------------------------------------------------- server


def port_open() -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex((HOST, PORT)) == 0


def start_server() -> tuple[subprocess.Popen, dict]:
    if port_open():
        raise SystemExit(f"port {PORT} is already in use")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    tmp = LOG_DIR / "tmp"
    tmp.mkdir(exist_ok=True)
    cmd = ["uv", "run", "--locked", "--project", str(c.SPIKES / "audio"), "python", "-m", "mlx_audio.server",
           "--host", HOST, "--port", str(PORT), "--log-dir", str(LOG_DIR / "logs")]
    env = dict(os.environ, HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1", TMPDIR=str(tmp))
    log = open(LOG_DIR / "server.log", "w")
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(c.REPO), start_new_session=True)
    while time.perf_counter() - t0 < 300:
        if proc.poll() is not None:
            raise SystemExit(f"ASR server exited early, see {LOG_DIR / 'server.log'}")
        if port_open():
            try:
                conn = http.client.HTTPConnection(HOST, PORT, timeout=5)
                conn.request("GET", "/v1/models")
                if conn.getresponse().status == 200:
                    break
            except OSError:
                pass
        time.sleep(0.5)
    else:
        raise SystemExit("ASR server not ready in 300 s")
    desc = {"cmd": " ".join(cmd).replace(str(c.SPIKES), "spikes"), "env": {"HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1"},
            "seconds_to_ready": round(time.perf_counter() - t0, 1)}
    return proc, desc


def stop_server(proc: subprocess.Popen) -> dict:
    if proc.poll() is not None:
        return {"already_exited": proc.returncode}
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=30)
        how = "SIGTERM"
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)
        how = "SIGKILL"
    for _ in range(40):
        if not port_open():
            break
        time.sleep(0.25)
    return {"stopped_with": how, "returncode": proc.returncode, "port_free_after": not port_open()}


def transcribe(wav: bytes, language: str) -> tuple[dict | None, float, str | None]:
    boundary = "s11" + uuid.uuid4().hex
    fields = {"model": MODEL, "language": language, "response_format": "verbose_json"}
    body = b"".join([f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
                     for k, v in fields.items()])
    body += (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="clip.wav"\r\n'
             "Content-Type: audio/wav\r\n\r\n").encode() + wav + b"\r\n" + f"--{boundary}--\r\n".encode()
    t0 = time.perf_counter_ns()
    try:
        conn = http.client.HTTPConnection(HOST, PORT, timeout=120)
        conn.request("POST", "/v1/audio/transcriptions", body, {"Content-Type": f"multipart/form-data; boundary={boundary}"})
        r = conn.getresponse()
        data = r.read()
        ms = (time.perf_counter_ns() - t0) / 1e6
        if r.status != 200:
            return None, ms, f"HTTP {r.status}: {data[:300]!r}"
        return json.loads(data), ms, None
    except Exception as e:  # noqa: BLE001 - recorded per request
        return None, (time.perf_counter_ns() - t0) / 1e6, repr(e)


# ------------------------------------------------------------------------------------- main


def utterances() -> list[dict]:
    rows = []
    for cond in run.grid("dt"):
        _, meta = ch.load(ch.cache_path(cond))
        for k, u in enumerate(meta["near"]):
            rows.append({"cid": cond.cid, "k": k, "utt": u["utt"], "set": u["set"], "text_norm": u["text_norm"],
                         "has_number": u["has_number"], "erl": cond.erl, "near_rel_db": cond.near_rel_db,
                         "erl_eff": meta["erl_eff"], "path": cond.path})
    return rows


def pooled(rs: list[dict]) -> dict:
    ref = sum(r["ref"] for r in rs)
    ed = sum(r["edits"] for r in rs)
    return {"n_utts": len(rs), "ref_tokens": ref, "edits": ed, "rate": round(ed / ref, 4) if ref else None}


def boot_delta(a: list[dict], b: list[dict], n: int = 2000) -> dict:
    """Paired bootstrap of (rate_a - rate_b) over utterances; a and b aligned by index."""
    ea = np.array([r["edits"] for r in a], dtype=float)
    eb = np.array([r["edits"] for r in b], dtype=float)
    rf = np.array([r["ref"] for r in a], dtype=float)
    rng = np.random.default_rng(0)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(ea), len(ea))
        d.append((ea[i].sum() - eb[i].sum()) / rf[i].sum())
    point = (ea.sum() - eb.sum()) / rf.sum()
    return {"delta_points": round(100 * point, 2), "ci95_points": [round(100 * float(np.percentile(d, 2.5)), 2),
                                                                     round(100 * float(np.percentile(d, 97.5)), 2)]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--variants", default=None, help="comma list of asr dirs (default: every dir under asr/)")
    a = ap.parse_args()
    if not B2_DONE.exists() and not a.force:
        raise SystemExit(f"{B2_DONE} does not exist yet: B2 is still using the GPU; dCER stays pending")
    variants = a.variants.split(",") if a.variants else sorted(p.name for p in run.ASR_DIR.iterdir() if p.is_dir())
    rows = utterances()
    mj = json.loads((c.RESULTS / "models.json").read_text()) if (c.RESULTS / "models.json").exists() else {}
    entry = next((m for m in mj.get("models", []) if m.get("repo") == MODEL), {})
    res: dict = {"spike": "S11 part 2", "part": "cer", "environment": c.environment(), "model": MODEL,
                 "model_revision": entry.get("sha_hub"),
                 "protocol": __doc__.split("ASR:")[1].strip(), "load_before": c.loadavg(), "variants": {}}
    proc, desc = start_server()
    res["server"] = desc
    try:
        warm = sorted((run.ASR_DIR / "clean").glob("*.wav"))[:3]
        res["warmup_ms"] = [round(transcribe(w.read_bytes(), LANG["zh" if "_zh" in w.name else "en"])[1], 1) for w in warm]
        for v in variants:
            recs = []
            t0 = time.perf_counter()
            for r in rows:
                f = run.ASR_DIR / v / f"{r['cid']}__{r['k']:02d}.wav"
                if not f.exists():
                    recs.append({**r, "missing": True})
                    continue
                js, ms, err = transcribe(f.read_bytes(), LANG[r["set"]])
                hyp = (js or {}).get("text") or ""
                kind = "cer" if r["set"] == "zh" else "wer"
                ref_t = tokens(r["text_norm"], kind)
                recs.append({**r, "kind": kind, "hyp": hyp, "ms": round(ms, 1), "error": err,
                             "ref": len(ref_t), "edits": edits(ref_t, tokens(norm_text(hyp), kind))})
            res["variants"][v] = {"records": recs, "wall_s": round(time.perf_counter() - t0, 1), "load_after": c.loadavg()}
            print(f"{v}: {len(recs)} utts in {time.perf_counter() - t0:.0f} s", flush=True)
    finally:
        res["server_stop"] = stop_server(proc)
    # ---- summary
    summ = {}
    clean = {(r["cid"], r["k"]): r for r in res["variants"].get("clean", {}).get("records", []) if "edits" in r}
    for v, d in res["variants"].items():
        ok = [r for r in d["records"] if "edits" in r and not r.get("error")]
        s = {}
        for lang in ("zh", "en"):
            for erl in (6, 15):
                for rel in (0, -15):
                    sel = [r for r in ok if r["set"] == lang and r["erl"] == erl and r["near_rel_db"] == rel]
                    key = f"{lang}|erl{erl}|near{rel}"
                    s[key] = pooled(sel)
                    pairs = [(r, clean[(r["cid"], r["k"])]) for r in sel if (r["cid"], r["k"]) in clean]
                    if pairs and v != "clean":
                        s[key]["vs_clean"] = boot_delta([p[0] for p in pairs], [p[1] for p in pairs])
                        nn = [p for p in pairs if not p[0]["has_number"]]
                        if nn:
                            s[key]["vs_clean_no_numbers"] = boot_delta([p[0] for p in nn], [p[1] for p in nn])
        for rel in (0, -15):
            for lang in ("zh", "en"):
                sel = [r for r in ok if r["set"] == lang and r["near_rel_db"] == rel]
                pairs = [(r, clean[(r["cid"], r["k"])]) for r in sel if (r["cid"], r["k"]) in clean]
                key = f"{lang}|near{rel}"
                s[key] = pooled(sel)
                if pairs and v != "clean":
                    s[key]["vs_clean"] = boot_delta([p[0] for p in pairs], [p[1] for p in pairs])
        s["errors"] = sum(1 for r in d["records"] if r.get("error"))
        s["missing"] = sum(1 for r in d["records"] if r.get("missing"))
        s["latency_ms"] = c.stats([r["ms"] for r in ok], 1)
        summ[v] = s
    res["summary"] = summ
    res["load_after"] = c.loadavg()
    p = c.save_json("s11_cer.json", res)
    print(f"wrote {p}")
    for v, s in summ.items():
        print(v, {k: (x.get("rate"), x.get("vs_clean", {}).get("delta_points")) for k, x in s.items()
                  if isinstance(x, dict) and "|near" in k and "erl" not in k})


if __name__ == "__main__":
    main()
