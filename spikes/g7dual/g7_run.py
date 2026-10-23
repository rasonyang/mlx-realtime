"""G7 dual path: double-talk dCER of the gateway's ASR input, and the cancellers' CPU. Throwaway.

    cd spikes/g7dual
    uv run --locked python g7_run.py gen                  # 2 layouts x 32 calls -> spikes/data/g7dual/
    uv run --locked python g7_run.py run                  # every variant through the gateway's echo path
    uv run --locked python g7_run.py asr                  # the real ASR (GPU) -> spikes/results/g7dual_cer.json
    uv run --locked python g7_run.py cpu                  # -> spikes/results/g7dual_cpu.json
    uv run --locked python g7_run.py runs                 # the echo path's counters -> g7dual_runs.json

Calls: g7_chain.py (B4's double-talk grid in two layouts, `dt` and `bargein`). Each call goes
through the gateway's own echo path (g7_gateway.py) once per variant (an `aec` configuration);
each near-end utterance +-300 ms is cut from the tracks the InputBuffer would hold:

- `raw` (asr_input raw), `aec3` (asr_input aec3) and `speex` (asr_input speex: the linear filter
  with the 3.8 kHz pre-filter, no preprocessor), from one pass;
- `speex_pp`: Speex with its preprocessor (denoise and residual-echo suppression on): the
  gateway's default after this measurement;
- `speex_nopf`: Speex without the 3.8 kHz pre-filter;
- `speex_f128`: a 128 ms Speex filter, the echo kept 0-108 ms behind its reference;
- `speex_pp_nopf`: Speex with its preprocessor, without the pre-filter;
- `clean`: the same telephone chain without echo, through the same decoder (the reference).

ASR (`asr`): Qwen3-ASR-1.7B-8bit @a8379a2 on the gateway's own audio backend (backends/audio,
mlx_audio.server), started here on 127.0.0.1:8112 with HF_HUB_OFFLINE=1 and stopped at the end;
each clip transcribed by the gateway's MlxAudioASR provider with the utterance's language, one
request at a time, after 3 discarded warm-up requests. Scoring as B4 (s11_cer.py): the manifest's
text_norm rule, CER for zh, WER for en, micro-averaged; dCER = error(variant) - error(clean),
paired by utterance, 95 % bootstrap CI (2000 resamples, seed 0).
"""

from __future__ import annotations

import g7_chain as gc  # first: it puts spikes/aec on the path and caps BLAS threads

import argparse  # noqa: E402
import asyncio  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import pickle  # noqa: E402
import signal  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import g7_gateway as gg  # noqa: E402
import s11_cer as cer  # noqa: E402
import s11_common as c  # noqa: E402
from mlx_realtime.config import AecConfig, MlxAudioAsrConfig, SpeexConfig  # noqa: E402
from mlx_realtime.providers.mlx_audio_asr import MlxAudioASR  # noqa: E402

LAYOUTS = ("dt", "bargein")
VARIANTS: dict[str, tuple[AecConfig, dict[str, str]]] = {
    # name: (the aec config, {track in the run's output: the ASR input it stands for}). Each
    # Speex setting is explicit: `pp` is the gateway's default after this measurement.
    "linear": (AecConfig(speex=SpeexConfig(preprocess=False)), {"raw": "raw", "aec3": "aec3", "asr": "speex"}),
    "pp": (AecConfig(speex=SpeexConfig(preprocess=True)), {"asr": "speex_pp"}),
    "nopf": (AecConfig(speex=SpeexConfig(preprocess=False, prefilter_hz=None)), {"asr": "speex_nopf"}),
    "f128": (AecConfig(speex=SpeexConfig(preprocess=False, filter_ms=128, max_delay_ms=108)), {"asr": "speex_f128"}),
    "pp_nopf": (AecConfig(speex=SpeexConfig(preprocess=True, prefilter_hz=None)), {"asr": "speex_pp_nopf"}),
}
CACHE = gc.CACHE
ASR_DIR = CACHE / "asr"
RUN_DIR = CACHE / "runs"
REPO = c.REPO
PORT = 8112
HOST = "127.0.0.1"
MODEL = "mlx-community/Qwen3-ASR-1.7B-8bit"
LANG = {"zh": "zh", "en": "en"}


def call_path(layout: str, cid: str) -> Path:
    return CACHE / "calls" / layout / f"{cid}.pkl"


def load_call(layout: str, cid: str) -> gc.Call:
    with call_path(layout, cid).open("rb") as f:
        return pickle.load(f)


def cmd_gen() -> None:
    t0 = time.perf_counter()
    for layout in LAYOUTS:
        for i, cond in enumerate(gc.conds()):
            p = call_path(layout, cond.cid)
            if p.exists():
                continue
            call = gc.build(cond, layout)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.with_suffix(".tmp").open("wb") as f:
                pickle.dump(call, f)
            p.with_suffix(".tmp").rename(p)
            clean = gg.decode(call.up24_clean)
            write_clips(clean, call, ASR_DIR / layout / "clean")
            print(f"gen {layout} [{i + 1}/32] {cond.cid} {time.perf_counter() - t0:.0f} s", flush=True)


def write_clips(track: np.ndarray, call: gc.Call, out_dir: Path) -> None:
    for k, u in enumerate(call.near):
        a = max(0, int((u["start"] - 0.3) * c.SR))
        b = min(len(track), int((u["end"] + 0.3) * c.SR))
        c.write_wav(out_dir / f"{call.cond.cid}__{k:02d}.wav", track[a:b], c.SR)


def cmd_run(variants: list[str]) -> None:
    for name in variants:
        aec, tracks = VARIANTS[name]
        for layout in LAYOUTS:
            out = RUN_DIR / layout / f"{name}.json"
            if out.exists():
                print(f"{name} {layout}: done already")
                continue
            summaries = {}
            t0 = time.perf_counter()
            with ThreadPoolExecutor(1) as executor:
                for cond in gc.conds():
                    call = load_call(layout, cond.cid)
                    res = asyncio.run(gg.run_call(call, aec, executor))
                    for key, track in tracks.items():
                        write_clips(res[key], call, ASR_DIR / layout / track)
                    summaries[cond.cid] = res["summary"]
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"variant": name, "aec": aec.model_dump(), "tracks": tracks,
                                       "wall_s": round(time.perf_counter() - t0, 1), "calls": summaries}, indent=1))
            print(f"{name} {layout}: {len(summaries)} calls in {time.perf_counter() - t0:.0f} s", flush=True)


# ------------------------------------------------------------------------------------------- ASR


def port_open() -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex((HOST, PORT)) == 0


def start_server() -> subprocess.Popen:
    if port_open():
        raise SystemExit(f"port {PORT} is in use")
    log_dir = CACHE / "asr_server"
    log_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["uv", "run", "--locked", "--project", str(REPO / "backends" / "audio"), "mlx_audio.server",
           "--host", HOST, "--port", str(PORT), "--log-dir", str(log_dir / "logs")]
    env = dict(os.environ, HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1")
    log = (log_dir / "server.log").open("w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(REPO), start_new_session=True)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 300:
        if proc.poll() is not None:
            raise SystemExit(f"the ASR server exited, see {log_dir / 'server.log'}")
        if port_open():
            return proc
        time.sleep(0.5)
    raise SystemExit("the ASR server was not ready in 300 s")


def stop_server(proc: subprocess.Popen) -> str:
    if proc.poll() is not None:
        return f"exited {proc.returncode}"
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
    return f"{how}, port free: {not port_open()}"


async def transcribe_all(jobs: list[tuple[str, str, Path]]) -> dict[tuple[str, str], dict]:
    asr = MlxAudioASR(MlxAudioAsrConfig(type="mlx_audio_transcription", base_url=f"http://{HOST}:{PORT}", model=MODEL))
    await asr.warmup()
    out: dict[tuple[str, str], dict] = {}
    try:
        warm = jobs[:3]
        for _, lang, path in warm:
            await asr.transcribe(read_pcm(path), language=lang)
        t0 = time.perf_counter()
        for i, (key, lang, path) in enumerate(jobs):
            began = time.perf_counter()
            try:
                transcript = await asr.transcribe(read_pcm(path), language=lang)
                out[(key, path.stem)] = {"hyp": transcript.text, "ms": round((time.perf_counter() - began) * 1000, 1)}
            except Exception as error:  # noqa: BLE001 - recorded per request
                out[(key, path.stem)] = {"hyp": "", "error": repr(error)}
            if (i + 1) % 200 == 0:
                print(f"asr {i + 1}/{len(jobs)} {time.perf_counter() - t0:.0f} s", flush=True)
    finally:
        await asr.aclose()
    return out


def read_pcm(path: Path) -> bytes:
    x, sr = c.read_wav(path)
    assert sr == c.SR
    return x.astype("<i2").tobytes()


def cmd_asr(tracks: list[str] | None) -> None:
    rows = []
    for layout in LAYOUTS:
        for cond in gc.conds():
            call = load_call(layout, cond.cid)
            for k, u in enumerate(call.near):
                rows.append({"layout": layout, "cid": cond.cid, "k": k, "utt": u["utt"], "set": u["set"],
                             "text_norm": u["text_norm"], "has_number": u["has_number"], "erl": cond.erl,
                             "near_rel_db": cond.near_rel_db})
    names = tracks or sorted({p.name for layout in LAYOUTS for p in (ASR_DIR / layout).iterdir() if p.is_dir()})
    jobs = []
    for name in names:
        for r in rows:
            path = ASR_DIR / r["layout"] / name / f"{r['cid']}__{r['k']:02d}.wav"
            if path.exists():
                jobs.append((f"{r['layout']}/{name}", LANG[r["set"]], path))
    print(f"{len(jobs)} clips, tracks {names}", flush=True)
    res: dict = {"spike": "G7 dual path", "environment": c.environment(), "model": MODEL,
                 "model_revision": "a8379a2e2f9e313c9292cdf1af4055ab56d50d55", "protocol": __doc__.split("ASR (`asr`):")[1].strip(),
                 "load_before": c.loadavg()}
    proc = start_server()
    t0 = time.perf_counter()
    try:
        heard = asyncio.run(transcribe_all(jobs))
    finally:
        res["server_stop"] = stop_server(proc)
    res["asr_wall_s"] = round(time.perf_counter() - t0, 1)
    records: dict[str, list[dict]] = {}
    for key, lang, path in jobs:
        layout, name = key.split("/")
        r = next(x for x in rows if x["layout"] == layout and f"{x['cid']}__{x['k']:02d}" == path.stem)
        h = heard[(key, path.stem)]
        kind = "cer" if r["set"] == "zh" else "wer"
        ref = cer.tokens(r["text_norm"], kind)
        records.setdefault(key, []).append({**r, "kind": kind, "hyp": h["hyp"], "error": h.get("error"), "ms": h.get("ms"),
                                            "ref": len(ref), "edits": cer.edits(ref, cer.tokens(cer.norm_text(h["hyp"]), kind))})
    out = c.RESULTS / "g7dual_cer.json"
    if tracks and out.exists():  # a partial run adds its tracks to the earlier ones
        earlier = json.loads(out.read_text())
        res["earlier_runs"] = earlier.get("earlier_runs", []) + [
            {k: earlier.get(k) for k in ("environment", "asr_wall_s", "server_stop", "load_before", "load_after")}]
        records = {**earlier["records"], **records}
    res["records"] = records
    res["summary"] = summarize(records)
    res["load_after"] = c.loadavg()
    p = c.save_json("g7dual_cer.json", res)
    print(f"wrote {p}")
    print_table(res["summary"])


def summarize(records: dict[str, list[dict]]) -> dict:
    out: dict = {}
    for key, recs in records.items():
        layout, name = key.split("/")
        clean = {(r["cid"], r["k"]): r for r in records.get(f"{layout}/clean", [])}
        ok = [r for r in recs if not r.get("error")]
        s: dict = {"errors": sum(1 for r in recs if r.get("error"))}
        for lang in ("zh", "en"):
            for rel in (0, -15):
                for erl in (None, 6, 15):
                    sel = [r for r in ok if r["set"] == lang and r["near_rel_db"] == rel and (erl is None or r["erl"] == erl)]
                    k = f"{lang}|near{rel}" + ("" if erl is None else f"|erl{erl}")
                    s[k] = cer.pooled(sel)
                    pairs = [(r, clean[(r["cid"], r["k"])]) for r in sel if (r["cid"], r["k"]) in clean]
                    if pairs and name != "clean":
                        s[k]["vs_clean"] = cer.boot_delta([p[0] for p in pairs], [p[1] for p in pairs])
        out[key] = s
    return out


def print_table(summary: dict) -> None:
    for key, s in sorted(summary.items()):
        cells = []
        for lang in ("zh", "en"):
            for rel in (0, -15):
                x = s.get(f"{lang}|near{rel}", {})
                d = x.get("vs_clean", {}).get("delta_points")
                cells.append(f"{lang}{rel:+d}: {x.get('rate')} ({d:+.1f})" if d is not None else f"{lang}{rel:+d}: {x.get('rate')}")
        print(f"{key:24s} " + " | ".join(cells))


# ------------------------------------------------------------------------------------------- CPU


def cmd_cpu() -> None:
    """CPU of the gateway's echo path per session-second (AEC3, the Speex path with its pre-filter,
    the estimator), single session over 4 calls of each layout; then N sessions at once on the
    gateway's canceller executor (aec.workers = 2), N = 1, 3 (the capacity target) and 8."""
    aec = AecConfig()
    res: dict = {"spike": "G7 dual path", "part": "cpu", "environment": c.environment(), "load_before": c.loadavg()}
    single = []
    for layout in LAYOUTS:
        for cond in gc.conds()[:16:4]:
            call = load_call(layout, cond.cid)
            single.append({"cid": call.cid, **gg.run_call_sync(call, aec)["summary"]["cpu_ms_per_audio_s"]})
    res["single"] = single
    res["single_mean"] = {k: round(float(np.mean([x[k] for x in single])), 3) for k in ("aec3", "speex_path", "executor_total")}
    calls = [load_call("bargein", cond.cid) for cond in gc.conds()[:8]]
    conc = {}
    for n in (1, 3, 8):
        async def many(n: int = n) -> list[dict]:
            with ThreadPoolExecutor(aec.workers) as executor:
                return await asyncio.gather(*(gg.run_call(calls[i % len(calls)], aec, executor) for i in range(n)))
        t0 = time.perf_counter()
        outs = asyncio.run(many())
        wall = time.perf_counter() - t0
        audio = sum(o["summary"]["audio_s"] for o in outs)
        conc[str(n)] = {"wall_s": round(wall, 2), "audio_s": round(audio, 1), "x_realtime": round(audio / wall, 1),
                        "executor_ms_per_audio_s": round(float(np.mean([o["summary"]["cpu_ms_per_audio_s"]["executor_total"] for o in outs])), 3),
                        "load_after": c.loadavg()}
        print(n, conc[str(n)], flush=True)
    res["concurrency"] = conc
    res["note"] = ("Thread CPU inside the canceller executor's work (EchoStage._run), per second of call audio. "
                   "Concurrency: N simulated calls at once, as fast as possible, on one ThreadPoolExecutor(2); "
                   "x_realtime = total audio / wall.")
    p = c.save_json("g7dual_cpu.json", res)
    print(f"wrote {p}", json.dumps(res["single_mean"]))


def cmd_runs() -> None:
    """The echo path's own counters per variant and layout -> spikes/results/g7dual_runs.json."""
    out: dict = {"spike": "G7 dual path", "part": "runs", "variants": {}}
    for layout in LAYOUTS:
        for path in sorted((RUN_DIR / layout).glob("*.json")):
            d = json.loads(path.read_text())
            calls = list(d["calls"].values())

            def med(key: str) -> float | None:
                xs = [x[key] for x in calls if x[key] is not None]
                return round(float(np.median(xs)), 2) if xs else None

            out["variants"][f"{layout}/{d['variant']}"] = {
                "calls": len(calls), "tracks": d["tracks"],
                "speex_aligned_at_s_p50": med("speex_aligned_at_s"),
                "speex_realignments": sum(x["speex_realignments"] for x in calls),
                "speex_resets": sum(x["speex_resets"] for x in calls),
                "anchors": sum(x["anchors"] for x in calls), "aec3_realignments": sum(x["realignments"] for x in calls),
                "bypasses": sum(x["bypasses"] for x in calls),
                "erle_db_p50": med("erle_db"), "speex_erle_db_p50": med("speex_erle_db"),
                "cpu_ms_per_audio_s_mean": {k: round(float(np.mean([x["cpu_ms_per_audio_s"][k] for x in calls])), 3)
                                            for k in ("aec3", "speex_path", "executor_total")},
                "note": "ERLE: median over 0.5 s windows with an active reference, double talk included (the dt layout is double talk throughout)",
            }
    p = c.save_json("g7dual_runs.json", out)
    print(f"wrote {p}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "run", "asr", "cpu", "runs"])
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--tracks", default=None)
    a = ap.parse_args()
    if a.cmd == "gen":
        cmd_gen()
    elif a.cmd == "run":
        cmd_run(a.variants.split(","))
    elif a.cmd == "asr":
        cmd_asr(a.tracks.split(",") if a.tracks else None)
    elif a.cmd == "runs":
        cmd_runs()
    else:
        cmd_cpu()


if __name__ == "__main__":
    sys.exit(main())
