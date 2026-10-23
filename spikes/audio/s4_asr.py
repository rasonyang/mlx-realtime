"""M0 spike S4 (ASR): Qwen3-ASR on mlx_audio.server 0.6.1 at 127.0.0.1:8102. Throwaway (work package B2).

Run from the repository root:

    B2_SCRATCH=<dir> uv run --locked --project spikes/audio python spikes/audio/s4_asr.py <step> [--model ...]

Steps:
  clips     build the 2 / 5 / 10 s latency clips from the manifest (wideband copies), 20 per set and length:
            utterances of one set are joined with 0.3 s of silence until long enough, then cut to exactly
            T seconds with a 10 ms fade-out. Written to spikes/data/s4/clips/ (git-ignored) + clips.json.
  latency   warm latency, n = 20 per (set, length, language given or not), cells interleaved
  accuracy  every manifest utterance, wideband and telephone copy, with and without `language`
  shape     output shape: response formats, language spellings, a > 30 s clip (server chunk_duration)

Request (ARCHITECTURE §5.3): multipart `file` (WAV 16 kHz mono s16), `model`, `language` (omitted for
auto-detect), `response_format: verbose_json`. Language names are the model's own (config
support_languages): Chinese, English. Mixed (code-switched) speech is sent as Chinese, as the gateway
would tag it zh.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time

import numpy as np

import b2_common as c

CLIP_DIR = c.DATA / "s4" / "clips"
SETS = ["zh", "en", "mix"]
DURATIONS = [2, 5, 10]


def manifest() -> list[dict]:
    return [json.loads(line) for line in (c.DATA / "manifest.jsonl").read_text().splitlines() if line.strip()]


def wav_of(path_rel: str) -> bytes:
    return (c.DATA / path_rel).read_bytes()


def step_clips(a) -> None:
    rows = manifest()
    out = []
    CLIP_DIR.mkdir(parents=True, exist_ok=True)
    gap = np.zeros(int(0.3 * c.ASR_RATE), dtype=np.float32)
    fade = np.linspace(1.0, 0.0, int(0.01 * c.ASR_RATE), dtype=np.float32)
    for s in SETS:
        utts = [r for r in rows if r["set"] == s]
        for T in DURATIONS:
            for k in range(20):
                start = (k * 2 + T) % len(utts)  # spread the starting utterance over the set
                parts, ids, i = [], [], start
                while sum(p.size for p in parts) < T * c.ASR_RATE:
                    pcm, rate = c.read_wav(c.DATA / utts[i % len(utts)]["path_16k"])
                    assert rate == c.ASR_RATE
                    if parts:
                        parts.append(gap)
                    parts.append(c.pcm_to_float(pcm))
                    ids.append(utts[i % len(utts)]["id"])
                    i += 1
                x = np.concatenate(parts)[: T * c.ASR_RATE].copy()
                x[-fade.size:] *= fade
                name = f"{s}_{T:02d}s_{k:02d}.wav"
                c.write_wav(CLIP_DIR / name, c.float_to_pcm(x), c.ASR_RATE)
                out.append({"file": name, "set": s, "seconds": T, "source_ids": ids})
    (CLIP_DIR / "clips.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {len(out)} clips to {CLIP_DIR}")


def transcribe(wav: bytes, model: str, language: str | None, response_format: str = "verbose_json") -> dict:
    fields = {"model": model, "response_format": response_format}
    if language is not None:
        fields["language"] = language
    la = c.loadavg1()
    r, js = c.asr_request(wav, fields)
    rec = {"status": r.status, "ms": r.end_ms, "headers_ms": r.headers_ms, "connect_ms": r.connect_ms,
           "error": r.error, "loadavg1_before": la, "content_type": r.headers.get("content-type")}
    if js is not None:
        rec["json"] = js
    else:
        rec["body"] = r.body.decode("utf-8", "replace")[:2000]
    return rec


def header(step: str, srv: c.Server, model: str, **conditions) -> dict:
    return {"spike": "S4", "step": step, "started_at": c.now_iso(), "loadavg1_start": c.loadavg1(),
            "environment": c.environment(), "model": c.model_revision(model),
            "server": srv.describe(), "conditions": conditions}


def finish(res: dict, srv: c.Server) -> None:
    res["loadavg1_end"] = c.loadavg1()
    res["finished_at"] = c.now_iso()
    res["server_footprint_bytes_end"] = srv.footprint()
    res["server_log_interesting"] = c.interesting_log(srv.lines, srv.started_ns)[-60:]
    res["server_stop"] = srv.stop()


def warm(model: str) -> list:
    out = []
    for f in ("zh_05s_00.wav", "en_05s_00.wav", "mix_02s_00.wav"):
        r = transcribe((CLIP_DIR / f).read_bytes(), model, None)
        out.append({"file": f, "ms": r["ms"], "status": r["status"], "text": (r.get("json") or {}).get("text")})
    return out


def step_latency(a) -> None:
    clips = json.loads((CLIP_DIR / "clips.json").read_text())
    if a.limit:
        clips = [cl for cl in clips if int(cl["file"][-6:-4]) < a.limit]
    srv = c.Server(f"asr_latency{a.tag}", c.ASR_PORT).start()
    res = header("latency", srv, a.model, n_per_cell=20, concurrency=1, language_values=c.ASR_LANG,
                 order="cells interleaved in a seeded random order; one warm request per cell first, discarded",
                 timing="client side, from connect() to the end of the JSON body")
    t0 = time.perf_counter_ns()
    res["cold_first_request"] = transcribe((CLIP_DIR / "zh_05s_01.wav").read_bytes(), a.model, None)["ms"]
    res["warmup"] = warm(a.model)
    jobs = []
    for cl in clips:
        for lang_given in (False, True):
            jobs.append((cl, lang_given))
    rng = random.Random(a.seed)
    rng.shuffle(jobs)
    # one discarded request per cell
    seen, recs = set(), []
    for cl, lang_given in jobs:
        key = (cl["set"], cl["seconds"], lang_given)
        if key in seen:
            continue
        seen.add(key)
        transcribe((CLIP_DIR / cl["file"]).read_bytes(), a.model, c.ASR_LANG[cl["set"]] if lang_given else None)
    for cl, lang_given in jobs:
        r = transcribe((CLIP_DIR / cl["file"]).read_bytes(), a.model, c.ASR_LANG[cl["set"]] if lang_given else None)
        js = r.get("json") or {}
        recs.append({"file": cl["file"], "set": cl["set"], "seconds": cl["seconds"], "language_given": lang_given,
                     "status": r["status"], "error": r["error"], "ms": r["ms"], "loadavg1_before": r["loadavg1_before"],
                     "server_total_time_s": js.get("total_time"), "prompt_tokens": js.get("prompt_tokens"),
                     "generation_tokens": js.get("generation_tokens"), "text": js.get("text"),
                     "language": js.get("language")})
    res["records"] = recs
    res["log_during_run"] = c.interesting_log(srv.log_since(t0), t0)[:40]
    summ = {}
    for s in SETS:
        for T in DURATIONS:
            for g in (False, True):
                sel = [r for r in recs if r["set"] == s and r["seconds"] == T and r["language_given"] == g]
                summ[f"{s}|{T}s|{'lang' if g else 'auto'}"] = {
                    "ms": c.stats([r["ms"] for r in sel]),
                    "server_total_time_ms": c.stats([r["server_total_time_s"] * 1000 for r in sel
                                                     if r["server_total_time_s"] is not None]),
                    "generation_tokens": c.stats([r["generation_tokens"] for r in sel]),
                    "errors": sum(1 for r in sel if r["status"] != 200 or r["error"]),
                    "loadavg1": [sel[0]["loadavg1_before"], sel[-1]["loadavg1_before"]] if sel else None}
    for T in DURATIONS:
        for g in (False, True):
            sel = [r for r in recs if r["seconds"] == T and r["language_given"] == g]
            summ[f"all|{T}s|{'lang' if g else 'auto'}"] = {"ms": c.stats([r["ms"] for r in sel])}
    res["summary"] = summ
    finish(res, srv)
    print(c.save_json(f"s4_latency{a.tag}.json", res))
    for k, v in summ.items():
        print(k, v["ms"])


def score_rows(recs: list[dict]) -> dict:
    out = {}
    for s in SETS:
        for copy in ("16k", "tel16k"):
            for g in (False, True):
                sel = [r for r in recs if r["set"] == s and r["copy"] == copy and r["language_given"] == g]
                key = f"{s}|{copy}|{'lang' if g else 'auto'}"
                out[key] = {
                    "metric": c.KIND_FOR_SET[s],
                    "no_number": c.pooled([r["err"] for r in sel if not r["has_number"]]),
                    "has_number": c.pooled([r["err"] for r in sel if r["has_number"]]),
                    "all": c.pooled([r["err"] for r in sel]),
                    "language_detected": {str(k): sum(1 for r in sel if str(r["language"]) == str(k))
                                          for k in {str(r["language"]) for r in sel}},
                    "errors": sum(1 for r in sel if r["status"] != 200),
                }
    for s in SETS:
        for g in ("auto", "lang"):
            wb, tel = out[f"{s}|16k|{g}"], out[f"{s}|tel16k|{g}"]
            out[f"{s}|telephone_delta|{g}"] = {
                part: (round(tel[part]["rate"] - wb[part]["rate"], 4)
                       if tel[part]["rate"] is not None and wb[part]["rate"] is not None else None)
                for part in ("no_number", "has_number", "all")}
    return out


def step_accuracy(a) -> None:
    rows = manifest()
    if a.limit:
        rows = [r for s in SETS for r in [x for x in rows if x["set"] == s][: a.limit]]
    srv = c.Server(f"asr_accuracy{a.tag}", c.ASR_PORT).start()
    res = header("accuracy", srv, a.model, references="manifest text_norm; hypotheses normalized with the same "
                 "rule (b2_common.norm_text)", metrics="zh CER (characters), en WER (words), mix MER (one token "
                 "per CJK character + one per Latin/digit word)", copies=["path_16k", "path_tel_16k"],
                 numbers="text_norm does not verbalize numbers; rows with has_number are scored separately")
    warm(a.model)
    recs = []
    for r in rows:
        for copy, key in (("16k", "path_16k"), ("tel16k", "path_tel_16k")):
            for g in (False, True):
                t = transcribe(wav_of(r[key]), a.model, c.ASR_LANG[r["set"]] if g else None)
                js = t.get("json") or {}
                hyp = js.get("text") or ""
                recs.append({"id": r["id"], "set": r["set"], "copy": copy, "language_given": g,
                             "has_number": r["has_number"], "duration_s": r["duration_s"], "ref": r["text_norm"],
                             "hyp": hyp, "language": js.get("language"), "segments": len(js.get("segments") or []),
                             "status": t["status"], "ms": t["ms"],
                             "err": c.error_rate(r["text_norm"], hyp, c.KIND_FOR_SET[r["set"]], ref_is_norm=True)})
        print(r["id"], recs[-4]["err"]["rate"], recs[-4]["hyp"][:60])
    res["records"] = recs
    res["summary"] = score_rows(recs)
    finish(res, srv)
    print(c.save_json(f"s4_accuracy{a.tag}.json", res))
    for k, v in res["summary"].items():
        print(k, {kk: (vv["rate"] if isinstance(vv, dict) and "rate" in vv else vv) for kk, vv in v.items()})


def step_shape(a) -> None:
    rows = manifest()
    srv = c.Server(f"asr_shape{a.tag}", c.ASR_PORT).start()
    res = header("shape", srv, a.model)
    warm(a.model)
    zh = next(r for r in rows if r["id"] == "zh_018")  # has a date and digits
    en = next(r for r in rows if r["id"] == "en_028")
    mix = next(r for r in rows if r["id"] == "mix_015")
    probes = []
    for row in (zh, en, mix):
        for fmt in ("verbose_json", "json", "text"):
            t = transcribe(wav_of(row["path_16k"]), a.model, None, fmt)
            probes.append({"id": row["id"], "response_format": fmt, **t})
    for row in (zh, en, mix):
        for lang in ("zh", "Chinese", "chinese", "en", "English", "Cantonese"):
            t = transcribe(wav_of(row["path_16k"]), a.model, lang)
            probes.append({"id": row["id"], "language": lang, **t})
    # > 30 s: the server passes chunk_duration=30.0 (its form default) to Qwen3-ASR's generate
    parts, ids = [], []
    for r in [x for x in rows if x["set"] == "zh"][20:26]:
        pcm, _ = c.read_wav(c.DATA / r["path_16k"])
        parts.append(c.pcm_to_float(pcm))
        parts.append(np.zeros(int(0.5 * c.ASR_RATE), dtype=np.float32))
        ids.append(r["id"])
    x = np.concatenate(parts)
    long_wav = c.wav_bytes(c.float_to_pcm(x), c.ASR_RATE)
    t = transcribe(long_wav, a.model, None)
    probes.append({"id": "+".join(ids), "seconds": round(x.size / c.ASR_RATE, 2), "note": "long clip", **t})
    res["probes"] = probes
    res["references"] = {row["id"]: row["text"] for row in (zh, en, mix)}
    res["long_clip_reference"] = " ".join(r["text"] for r in rows if r["id"] in ids)
    finish(res, srv)
    print(c.save_json(f"s4_shape{a.tag}.json", res))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["clips", "latency", "accuracy", "shape"])
    ap.add_argument("--model", default=c.ASR)
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=20261023)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: fewer clips / rows")
    a = ap.parse_args()
    if a.model != c.ASR and not c.model_ok(a.model):
        raise SystemExit(f"{a.model} is not status ok in models.json; skipped")
    try:
        {"clips": step_clips, "latency": step_latency, "accuracy": step_accuracy, "shape": step_shape}[a.step](a)
    finally:
        c.stop_all()


if __name__ == "__main__":
    main()
