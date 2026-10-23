"""M0 spike S3 (TTS) over HTTP: mlx_audio.server 0.6.1 on 127.0.0.1:8103 (ASR on :8102 where a step
back-transcribes). Throwaway spike code (work package B2).

Run from the repository root, one step at a time:

    B2_SCRATCH=<dir> uv run --locked --project spikes/audio python spikes/audio/s3_tts.py <step> [options]

Steps (each writes spikes/results/s3_<step>[_<tag>].json):
  cold       first request after a fresh server start (lazy model load), n restarts
  stream     streaming check: chunk arrival times and sizes, stream vs no stream
  sweep      first-chunk latency grid: streaming_interval x lang x length x speaker, warm
  pinning    warmup, then 50 identical requests (2nd vs 50th), idle gaps, log/footprint/swap
  cancel     close after the first chunk, send a new request at once (traced server)
  speakers   speaker comparison for the zh and en defaults (back-transcribed)
  guardrail  100 runs over 10 sentences: speaking-rate distribution, 30 back-transcribed
  kokoro     Kokoro-82M vs Qwen3-TTS on 20 zh / en / mixed sentences (D13), back-transcribed
  concurrency  k = 1, 2, 4 simultaneous streaming requests: interleaved or one after another?

Timing: time.perf_counter_ns on the client side of the HTTP call; t0 is taken just before connect().
TTFC = first audio byte. RTF here = processing time / audio duration (< 1 is faster than real time).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import time
from pathlib import Path

import b2_common as c
import b2_texts as T

SPEAKERS_SWEEP = ["vivian", "ryan"]  # one zh and one en speaker from the model's own list
INTERVALS = [0.16, 0.32, 0.8]
LANGS = ["zh", "en", "mix"]
LENGTHS = ["short", "medium", "long"]
BYTES_PER_S = c.TTS_RATE * 2


# ------------------------------------------------------------------------------------------------
def tts_payload(text: str, voice: str, lang: str, interval: float | None = 0.32, *, model: str = c.TTS,
                stream: bool = True, kokoro: bool = False, **extra) -> dict:
    p = {"model": model, "input": text, "voice": voice,
         "lang_code": (c.KOKORO_LANG if kokoro else c.QWEN_LANG)[lang],
         "response_format": "pcm", "stream": stream}
    if interval is not None:
        p["streaming_interval"] = interval
    return p | extra


def measure(r: c.HttpResult) -> dict:
    """Derived per-request numbers from one TTS response."""
    rec = r.summary()
    audio_s = len(r.body) / BYTES_PER_S
    rec["audio_s"] = round(audio_s, 3)
    rec["ttfc_ms"] = r.first_byte_ms
    rec["total_ms"] = r.end_ms
    rec["chunks"] = [[ch[1], ch[2]] for ch in r.chunks]  # [ms complete, bytes]
    if r.chunks and audio_s > 0 and r.end_ms:
        rec["rtf"] = round(r.end_ms / 1000 / audio_s, 4)
        first_audio = r.chunks[0][2] / BYTES_PER_S
        rest = audio_s - first_audio
        last = r.chunks[-1][1]
        rec["steady_rtf"] = round((last - r.chunks[0][1]) / 1000 / rest, 4) if rest > 0.05 else None
        rec["first_chunk_audio_s"] = round(first_audio, 3)
        # Playback that starts at the first chunk stalls if chunk k arrives after the audio before it
        # has played out: slack_k = ttfc + audio_before_k - arrival_k.
        slack, cum = [], 0.0
        for k, (t_first, _t_done, n) in enumerate(r.chunks):
            if k > 0:
                slack.append(r.first_byte_ms + cum * 1000 - t_first)
            cum += n / BYTES_PER_S
        rec["min_slack_ms"] = round(min(slack), 1) if slack else None
        gaps = [b[0] - a[1] for a, b in zip(r.chunks, r.chunks[1:])]
        rec["max_gap_ms"] = round(max(gaps), 1) if gaps else None
    return rec


def call(payload: dict, **kw) -> tuple[dict, bytes]:
    la = c.loadavg1()
    r = c.tts_request(payload, **kw)
    rec = measure(r)
    rec["loadavg1_before"] = la
    if r.status != 200 or r.error:
        rec["body_text"] = r.body[:500].decode("utf-8", "replace")
    return rec, r.body


def back_transcribe(pcm: bytes, language: str | None = None, model: str = c.ASR) -> dict:
    fields = {"model": model, "response_format": "verbose_json"}
    if language:
        fields["language"] = language
    r, js = c.asr_request(c.tts_pcm_to_asr_wav(pcm), fields)
    return {"status": r.status, "ms": r.end_ms, "text": (js or {}).get("text"),
            "language": (js or {}).get("language"), "error": r.error}


def cell_stats(recs: list[dict]) -> dict:
    ok = [r for r in recs if r.get("status") == 200 and not r.get("error") and r.get("ttfc_ms") is not None]
    out = {k: c.stats([r.get(k) for r in ok]) for k in
           ("ttfc_ms", "total_ms", "rtf", "steady_rtf", "audio_s", "min_slack_ms", "max_gap_ms")}
    sizes = [ch[1] for r in ok for ch in r["chunks"][:-1]]
    out["chunk_audio_s_mode"] = (round(statistics.mode(sizes) / BYTES_PER_S, 3) if sizes else None)
    out["n_errors"] = len(recs) - len(ok)
    out["stalls"] = sum(1 for r in ok if (r.get("min_slack_ms") or 0) < 0)
    out["loadavg1"] = {"before_first": recs[0].get("loadavg1_before") if recs else None,
                       "before_last": recs[-1].get("loadavg1_before") if recs else None}
    return out


def header(step: str, servers: list[c.Server], **conditions) -> dict:
    return {"spike": "S3", "step": step, "started_at": c.now_iso(), "loadavg1_start": c.loadavg1(),
            "environment": c.environment(),
            "models": {m: c.model_revision(m) for m in (c.TTS, c.ASR, c.KOKORO, c.TTS_4BIT, c.TTS_17B)},
            "servers": [s.describe() for s in servers], "conditions": conditions}


def finish(res: dict, servers: list[c.Server]) -> None:
    res["loadavg1_end"] = c.loadavg1()
    res["finished_at"] = c.now_iso()
    for s in servers:
        d = s.describe()
        d["footprint_bytes_end"] = s.footprint()
        d["log_interesting"] = c.interesting_log(s.lines, s.started_ns)[-80:]
        d["stop"] = s.stop()
        res.setdefault("servers_end", []).append(d)


def warmup(texts=(("zh", "vivian"), ("en", "ryan"), ("mix", "vivian")), model: str = c.TTS) -> list:
    out = []
    for lang, spk in texts:
        rec, _ = call(tts_payload(T.SWEEP[lang]["medium"][0], spk, lang, 0.32, model=model))
        out.append({"lang": lang, "speaker": spk, **{k: rec.get(k) for k in ("status", "ttfc_ms", "total_ms",
                                                                              "audio_s", "error")}})
    return out


# ------------------------------------------------------------------------------------------------
def step_cold(a) -> None:
    """Fresh server, first request (includes the lazy model load), then a second, warm, request."""
    runs = []
    model = a.model
    for i in range(a.n):
        srv = c.Server(f"tts_cold{i}", c.TTS_PORT).start()
        t_ready = time.perf_counter_ns()
        first, _ = call(tts_payload(T.SWEEP["zh"]["medium"][0], "vivian", "zh", 0.32, model=model))
        second, _ = call(tts_payload(T.SWEEP["zh"]["medium"][1], "vivian", "zh", 0.32, model=model))
        runs.append({"i": i, "seconds_to_ready": srv.ready_s, "footprint_after": srv.footprint(),
                     "first": {k: v for k, v in first.items() if k != "chunks"},
                     "second": {k: v for k, v in second.items() if k != "chunks"},
                     "log": c.interesting_log(srv.log_since(t_ready), t_ready)[:20], "stop": srv.stop()})
        print(f"cold {i}: ready {srv.ready_s}s first ttfc {first['ttfc_ms']} total {first['total_ms']} "
              f"| second ttfc {second['ttfc_ms']}")
        time.sleep(2)
    res = header("cold", [], model=model, text="zh medium #0, then #1", voice="vivian", streaming_interval=0.32,
                 note="each run starts a fresh server; 'first' includes the lazy model load (preflight)")
    res["runs"] = runs
    res["summary"] = {
        "seconds_to_ready": c.stats([r["seconds_to_ready"] for r in runs]),
        "first_ttfc_ms": c.stats([r["first"]["ttfc_ms"] for r in runs]),
        "first_total_ms": c.stats([r["first"]["total_ms"] for r in runs]),
        "second_ttfc_ms": c.stats([r["second"]["ttfc_ms"] for r in runs]),
    }
    res["loadavg1_end"] = c.loadavg1()
    print(c.save_json(f"s3_cold{a.tag}.json", res))


def step_stream(a) -> None:
    srv = c.Server("tts_stream", c.TTS_PORT).start()
    res = header("stream", [srv], model=c.TTS, voice="vivian")
    res["warmup"] = warmup()
    runs = []
    for rep in range(a.n):
        for length in ("medium", "long"):
            text = T.SWEEP["zh"][length][rep % 3]
            for stream, interval in ((True, 0.32), (True, None), (False, None)):
                rec, body = call(tts_payload(text, "vivian", "zh", interval, stream=stream))
                rec.update(length=length, stream=stream, streaming_interval=interval if interval else "default (2.0)",
                           rep=rep, even_bytes=len(body) % 2 == 0)
                runs.append(rec)
                print(f"{length} stream={stream} interval={interval}: ttfc {rec['ttfc_ms']} total {rec['total_ms']} "
                      f"chunks {rec['n_chunks']} bytes {rec['bytes']}")
    res["runs"] = runs
    finish(res, [srv])
    print(c.save_json(f"s3_stream{a.tag}.json", res))


def step_sweep(a) -> None:
    model = a.model
    speakers = a.speakers.split(",")
    intervals = [float(x) for x in a.intervals.split(",")]
    langs, lengths = a.langs.split(","), a.lengths.split(",")
    cells = [(iv, lang, ln, spk) for iv in intervals for lang in langs for ln in lengths for spk in speakers]
    rng = random.Random(a.seed)
    srv = c.Server(f"tts_sweep{a.tag}", c.TTS_PORT).start()
    res = header("sweep", [srv], model=model, speakers=speakers, intervals=intervals, langs=langs,
                 lengths=lengths, n_per_cell=a.n, order="one discarded first-in-cell pass, then n rounds, "
                 "each visiting every cell once in a seeded random order", seed=a.seed,
                 texts="3 texts per (lang, length), rotated by round", concurrency=1,
                 lang_code=c.QWEN_LANG)
    res["text_units"] = {f"{lang}/{ln}": [c.units(t) for t in T.SWEEP[lang][ln]] for lang in langs for ln in lengths}
    t_start = time.perf_counter_ns()
    res["warmup"] = warmup(model=model)
    recs = []

    def run(cell, rnd, phase):
        iv, lang, ln, spk = cell
        text_i = rnd % 3
        rec, _ = call(tts_payload(T.SWEEP[lang][ln][text_i], spk, lang, iv, model=model))
        rec.update(interval=iv, lang=lang, length=ln, speaker=spk, text_i=text_i, round=rnd, phase=phase)
        recs.append(rec)
        return rec

    if a.topup:
        # Replacement requests for sweep requests excluded afterwards (see s3_sweep_perturbation.json):
        # the same (cell, text) pairs, in the listed order, after the warmup.
        todo = json.loads(Path(a.topup).read_text())["replace"]
        for t in todo:
            cell = (t["interval"], t["lang"], t["length"], t["speaker"])
            iv, lang, ln, spk = cell
            rec, _ = call(tts_payload(T.SWEEP[lang][ln][t["text_i"]], spk, lang, iv, model=model))
            rec.update(interval=iv, lang=lang, length=ln, speaker=spk, text_i=t["text_i"], round=t["round"],
                       phase="topup")
            recs.append(rec)
        res["topup_of"] = a.topup
        res["records"] = recs
        res["log_during_run"] = c.interesting_log(srv.log_since(t_start), t_start)[:50]
        finish(res, [srv])
        print(c.save_json(f"s3_sweep{a.tag}.json", res))
        return
    order = cells[:]
    rng.shuffle(order)
    for cell in order:
        run(cell, 0, "first_in_cell")
    round_load = []
    for rnd in range(a.n):
        la0 = c.loadavg1()
        rng.shuffle(order)
        for cell in order:
            run(cell, rnd, "measured")
        round_load.append([la0, c.loadavg1()])
        meas = [r for r in recs if r["phase"] == "measured" and r["round"] == rnd]
        print(f"round {rnd}: load {la0}->{round_load[-1][1]}, median ttfc "
              f"{statistics.median([r['ttfc_ms'] for r in meas if r['ttfc_ms']]):.1f} ms, errors "
              f"{sum(1 for r in meas if r.get('error') or r.get('status') != 200)}")
    res["round_loadavg1"] = round_load
    res["log_during_run"] = c.interesting_log(srv.log_since(t_start), t_start)[:50]
    res["n_requests"] = len(recs)
    summary = {}
    for cell in cells:
        iv, lang, ln, spk = cell
        sel = [r for r in recs if (r["interval"], r["lang"], r["length"], r["speaker"]) == cell]
        meas = [r for r in sel if r["phase"] == "measured"]
        first = [r for r in sel if r["phase"] == "first_in_cell"]
        summary[f"{iv}|{lang}|{ln}|{spk}"] = cell_stats(meas) | {
            "first_in_cell_ttfc_ms": first[0]["ttfc_ms"] if first else None}
    res["summary"] = summary
    for key in ("interval", "length", "lang", "speaker"):
        groups = {}
        for r in recs:
            if r["phase"] == "measured":
                groups.setdefault(str(r[key]), []).append(r)
        res[f"by_{key}"] = {k: cell_stats(v) for k, v in groups.items()}
    res["records"] = recs
    finish(res, [srv])
    print(c.save_json(f"s3_sweep{a.tag}.json", res))


def step_pinning(a) -> None:
    srv = c.Server("tts_pinning", c.TTS_PORT, traced=True).start()
    res = header("pinning", [srv], model=c.TTS, voice="vivian", streaming_interval=0.32,
                 note="traced server (trace lines only); model.load lines show every real load")
    t0 = time.perf_counter_ns()
    res["models_before"] = c.get_json(c.TTS_PORT, "/v1/models")
    res["swap_before"] = c.environment()["sysctl"]["vm.swapusage"]
    res["footprint_before_warmup"] = srv.footprint()
    w, _ = call(tts_payload(T.SWEEP["zh"]["medium"][0], "vivian", "zh", 0.32))
    res["warmup"] = {k: v for k, v in w.items() if k != "chunks"}
    res["footprint_after_warmup"] = srv.footprint()
    seq = []
    text = T.SWEEP["zh"]["medium"][1]
    for i in range(a.n):
        rec, _ = call(tts_payload(text, "vivian", "zh", 0.32))
        seq.append({"i": i + 2, **{k: rec.get(k) for k in ("status", "ttfc_ms", "total_ms", "rtf", "audio_s",
                                                            "loadavg1_before", "error")}})
    res["sequence"] = seq
    res["footprint_after_sequence"] = srv.footprint()
    res["sequence_summary"] = {"request_2": seq[0], f"request_{a.n + 1}": seq[-1],
                               "ttfc_ms": c.stats([s["ttfc_ms"] for s in seq]),
                               "first10_ttfc_ms": c.stats([s["ttfc_ms"] for s in seq[:10]]),
                               "last10_ttfc_ms": c.stats([s["ttfc_ms"] for s in seq[-10:]])}
    idle = []
    for gap in [float(x) for x in a.gaps.split(",")] if a.gaps else []:
        for rep in range(a.gap_reps):
            time.sleep(gap)
            rec, _ = call(tts_payload(text, "vivian", "zh", 0.32))
            rec2, _ = call(tts_payload(text, "vivian", "zh", 0.32))
            idle.append({"idle_s": gap, "rep": rep, "after_idle_ttfc_ms": rec["ttfc_ms"],
                         "after_idle_total_ms": rec["total_ms"], "next_ttfc_ms": rec2["ttfc_ms"],
                         "next_total_ms": rec2["total_ms"], "loadavg1_before": rec["loadavg1_before"]})
            print(f"idle {gap}s rep {rep}: ttfc {rec['ttfc_ms']} then {rec2['ttfc_ms']}")
    res["idle_gaps"] = idle
    res["models_after"] = c.get_json(c.TTS_PORT, "/v1/models")
    res["swap_after"] = c.environment()["sysctl"]["vm.swapusage"]
    res["model_load_trace_lines"] = [f"{(t - t0) / 1e6:.1f} ms: {s}" for t, s in srv.lines if "model.load" in s]
    res["load_lines_stock"] = [s for _, s in srv.lines if "Loaded speech tokenizer" in s or "KokoroPipeline" in s]
    finish(res, [srv])
    print(c.save_json(f"s3_pinning{a.tag}.json", res))


def parse_trace(lines: list[tuple[int, str]]) -> list[dict]:
    out = []
    for t_tee, s in lines:
        m = re.match(r"\[trace (\d+)\] (\S+) (.*)", s)
        if m:
            out.append({"t_ns": int(m.group(1)), "event": m.group(2), "rest": m.group(3)})
    return out


def step_cancel(a) -> None:
    """Close the stream after its first chunk, then send a new request at once (trial), or send the
    new request after a normally completed one (control). Traced server: when does the old request's
    generation stop?"""
    srv = c.Server("tts_cancel", c.TTS_PORT, traced=True).start()
    res = header("cancel", [srv], model=c.TTS, voice="vivian", old_text="zh long", new_text="zh medium",
                 note="trial: close after 1st chunk, new request immediately; control: new request right "
                      "after a completed old request (short old text)")
    res["warmup"] = warmup()
    trials = []
    for iv in [float(x) for x in a.intervals.split(",")]:
        for rep in range(a.n):
            for kind in ("trial", "control"):
                old_text = T.SWEEP["zh"]["long"][rep % 3] if kind == "trial" else T.SWEEP["zh"]["short"][rep % 3]
                new_text = T.SWEEP["zh"]["medium"][rep % 3]
                mark = time.perf_counter_ns()
                old = c.tts_request(tts_payload(old_text, "vivian", "zh", iv),
                                    close_after_chunks=1 if kind == "trial" else None)
                t_close = old.close_ns or time.perf_counter_ns()
                new = c.tts_request(tts_payload(new_text, "vivian", "zh", iv))
                time.sleep(0.3)  # let the trace lines arrive
                tr = parse_trace(srv.log_since(mark))
                rids = []
                for e in tr:
                    m = re.search(r"rid=(\w+)", e["rest"])
                    if e["event"] == "tts.start" and m:
                        rids.append(m.group(1))
                rec = {"interval": iv, "rep": rep, "kind": kind, "old_ttfc_ms": old.first_byte_ms,
                       "old_closed_early": old.closed_early, "old_error": old.error,
                       "new_ttfc_ms": new.first_byte_ms, "new_total_ms": new.end_ms, "new_error": new.error,
                       "new_status": new.status, "loadavg1": c.loadavg1()}
                if len(rids) >= 2:
                    o, n = rids[0], rids[1]
                    ev_o = [e for e in tr if f"rid={o}" in e["rest"]]
                    ev_n = [e for e in tr if f"rid={n}" in e["rest"]]
                    emits_o = [e for e in ev_o if e["event"] == "tts.emit"]
                    end_o = next((e for e in ev_o if e["event"] == "tts.end"), None)
                    cancel_o = next((e for e in ev_o if e["event"] == "handle.cancel"), None)
                    start_n = next((e for e in ev_n if e["event"] == "tts.start"), None)
                    rel = lambda e: round((e["t_ns"] - t_close) / 1e6, 2) if e else None  # noqa: E731
                    rec.update({
                        "old_chunks_emitted": len(emits_o),
                        "old_chunks_after_close": sum(1 for e in emits_o if e["t_ns"] > t_close),
                        "old_handle_cancel_ms_after_close": rel(cancel_o),
                        "old_generation_end_ms_after_close": rel(end_o),
                        "old_end_cancelled_flag": ("cancelled=True" in end_o["rest"]) if end_o else None,
                        "new_broker_start_ms_after_close": rel(start_n),
                    })
                else:
                    rec["trace_problem"] = [f"{e['event']} {e['rest']}" for e in tr]
                trials.append(rec)
                print({k: rec.get(k) for k in ("interval", "kind", "new_ttfc_ms", "old_generation_end_ms_after_close",
                                               "old_chunks_after_close", "new_broker_start_ms_after_close")})
                time.sleep(0.2)
    res["trials"] = trials
    summ = {}
    for iv in sorted({t["interval"] for t in trials}):
        for kind in ("trial", "control"):
            sel = [t for t in trials if t["interval"] == iv and t["kind"] == kind]
            summ[f"{iv}|{kind}"] = {k: c.stats([t.get(k) for t in sel]) for k in
                                   ("new_ttfc_ms", "old_generation_end_ms_after_close",
                                    "old_handle_cancel_ms_after_close", "old_chunks_after_close",
                                    "new_broker_start_ms_after_close")}
    res["summary"] = summ
    finish(res, [srv])
    print(c.save_json(f"s3_cancel{a.tag}.json", res))


def step_concurrency(a) -> None:
    """k simultaneous streaming requests (as k sessions would send): does the server interleave them,
    or run each to completion in turn? (Streaming TTS requests are not batched by mlx-audio 0.6.1:
    Qwen3-TTS supports_tts_batch returns False when stream is set.)"""
    import threading

    srv = c.Server("tts_concurrency", c.TTS_PORT).start()
    res = header("concurrency", [srv], model=c.TTS, voice="vivian", text="zh medium", streaming_interval=0.32,
                 ks=[1, 2, 4], reps=a.n)
    res["warmup"] = warmup()
    rows = []
    for k in (1, 2, 4):
        for rep in range(a.n):
            barrier = threading.Barrier(k)
            out: list = [None] * k

            def worker(i: int) -> None:
                barrier.wait()
                rec, _ = call(tts_payload(T.SWEEP["zh"]["medium"][(rep + i) % 3], "vivian", "zh", 0.32))
                out[i] = rec

            ths = [threading.Thread(target=worker, args=(i,)) for i in range(k)]
            for t in ths:
                t.start()
            for t in ths:
                t.join()
            by_ttfc = sorted(out, key=lambda r: r["ttfc_ms"] or 1e9)
            rows.append({"k": k, "rep": rep, "ttfc_ms_sorted": [r["ttfc_ms"] for r in by_ttfc],
                         "total_ms_sorted": [r["total_ms"] for r in by_ttfc],
                         "errors": [r.get("error") for r in out if r.get("error")]})
        print(k, rows[-1])
    res["rows"] = rows
    res["summary"] = {f"k{k}": {f"rank{j}_ttfc_ms": c.stats([r["ttfc_ms_sorted"][j] for r in rows if r["k"] == k])
                                for j in range(k)} for k in (1, 2, 4)}
    finish(res, [srv])
    print(c.save_json(f"s3_concurrency{a.tag}.json", res))


# ------------------------------------------------------------------------------------------------
def kind_for(lang: str) -> str:
    return {"zh": "cer", "en": "wer", "mix": "mer"}[lang]


def step_speakers(a) -> None:
    asr = c.Server("asr_speakers", c.ASR_PORT).start()
    srv = c.Server("tts_speakers", c.TTS_PORT).start()
    res = header("speakers", [asr, srv], model=c.TTS, streaming_interval=0.32, runs_per_sentence=a.n,
                 back_transcription="ASR 1.7B, no language field (auto-detect)",
                 note="dylan and eric are left out: with lang_code chinese/auto, mlx-audio switches them to "
                      "their dialect language id (beijing_dialect / sichuan_dialect)")
    res["warmup"] = warmup()
    back_transcribe(b"\0\0" * 24000)  # warm the ASR
    plan = {"zh": ["vivian", "serena", "uncle_fu", "ryan", "aiden"],
            "en": ["ryan", "aiden", "vivian", "serena"],
            "mix": ["vivian", "serena", "uncle_fu"]}
    recs = []
    for lang, spks in plan.items():
        sents = T.D13[lang][: a.sentences]
        for rep in range(a.n):
            for spk in spks:
                for si, text in enumerate(sents):
                    rec, pcm = call(tts_payload(text, spk, lang, 0.32))
                    bt = back_transcribe(pcm)
                    err = c.error_rate(text, bt["text"] or "", kind_for(lang))
                    span = c.voiced_span(pcm)
                    recs.append({"lang": lang, "speaker": spk, "sentence": si, "rep": rep,
                                 "ttfc_ms": rec["ttfc_ms"], "total_ms": rec["total_ms"], "audio_s": rec["audio_s"],
                                 "voiced_s": span["voiced_s"], "units": c.units(text),
                                 "asr_text": bt["text"], "asr_language": bt["language"], "err": err})
                    if rep == 0 and si < 3:
                        c.write_wav(c.DATA / "s3" / "speakers" / f"{lang}_{spk}_{si:02d}.wav", pcm, c.TTS_RATE)
            print(lang, rep, "done")
    res["records"] = recs
    summ = {}
    for lang, spks in plan.items():
        for spk in spks:
            sel = [r for r in recs if r["lang"] == lang and r["speaker"] == spk]
            summ[f"{lang}|{spk}"] = {
                "error": c.pooled([r["err"] for r in sel]),
                "ttfc_ms": c.stats([r["ttfc_ms"] for r in sel]),
                "units_per_voiced_s": c.stats([r["units"] / r["voiced_s"] for r in sel if r["voiced_s"]]),
                "asr_language_counts": {str(k): sum(1 for r in sel if str(r["asr_language"]) == k)
                                        for k in {str(r["asr_language"]) for r in sel}},
            }
    res["summary"] = summ
    finish(res, [srv, asr])
    print(c.save_json(f"s3_speakers{a.tag}.json", res))


def step_guardrail(a) -> None:
    asr = c.Server("asr_guardrail", c.ASR_PORT).start()
    srv = c.Server("tts_guardrail", c.TTS_PORT).start()
    voice = {"zh": a.zh_voice, "en": a.en_voice}
    model = a.model
    res = header("guardrail", [asr, srv], model=model, voices=voice, streaming_interval=0.32,
                 runs_per_sentence=a.n, sampling="server defaults = generation_config.json "
                 "(temperature 0.9, top_p 1.0, top_k 50, repetition_penalty 1.05)",
                 rate_unit="zh: CJK characters, en: words, per second of voiced audio (leading/trailing "
                           "silence removed by a 10 ms frame energy threshold: max-35 dB, floor -55 dBFS)",
                 outlier_rule="rate < 0.6 x the median rate of the same sentence")
    res["warmup"] = warmup(model=model)
    back_transcribe(b"\0\0" * 24000)
    runs = []
    out_dir = c.DATA / "s3" / f"guardrail{a.tag}"
    for rep in range(a.n):
        for si, (lang, text) in enumerate(T.GUARDRAIL):
            rec, pcm = call(tts_payload(text, voice[lang], lang, 0.32, model=model))
            span = c.voiced_span(pcm)
            u = len(c.tokens(c.norm_text(text), "cer" if lang == "zh" else "wer"))
            path = out_dir / f"s{si:02d}_{lang}_r{rep:02d}.wav"
            c.write_wav(path, pcm, c.TTS_RATE)
            runs.append({"sentence": si, "lang": lang, "rep": rep, "units": u, **span,
                         "rate": round(u / span["voiced_s"], 3) if span["voiced_s"] else None,
                         "ttfc_ms": rec["ttfc_ms"], "total_ms": rec["total_ms"], "audio_s": rec["audio_s"],
                         "status": rec["status"], "error": rec.get("error"), "wav": str(path.relative_to(c.REPO))})
        print("rep", rep, "done")
    for si in range(len(T.GUARDRAIL)):
        sel = [r for r in runs if r["sentence"] == si and r["rate"]]
        med = statistics.median([r["rate"] for r in sel])
        for r in sel:
            r["ratio_to_median"] = round(r["rate"] / med, 3)
            r["outlier"] = r["ratio_to_median"] < 0.6
    outliers = [r for r in runs if r.get("outlier")]
    picks = outliers + [r for r in runs if r["rep"] in (0, 4, 8) and not r.get("outlier")]
    picks = picks[:30]
    for r in picks:
        pcm, _ = c.read_wav(c.REPO / r["wav"])
        lang = r["lang"]
        bt = back_transcribe(pcm)
        r["asr_text"] = bt["text"]
        r["err"] = c.error_rate(T.GUARDRAIL[r["sentence"]][1], bt["text"] or "", "cer" if lang == "zh" else "wer")
    res["runs"] = runs
    ratios = [r["ratio_to_median"] for r in runs if "ratio_to_median" in r]
    res["summary"] = {
        "n_runs": len(runs),
        "ratio_to_median": c.stats(ratios),
        "ratio_histogram": {f"{lo:.1f}-{lo + 0.1:.1f}": sum(1 for x in ratios if lo <= x < lo + 0.1)
                            for lo in [i / 10 for i in range(3, 16)]},
        "outliers_below_0.6": [{k: r[k] for k in ("sentence", "lang", "rep", "rate", "ratio_to_median", "wav")}
                               for r in outliers],
        "rate_by_lang": {lang: c.stats([r["rate"] for r in runs if r["lang"] == lang]) for lang in ("zh", "en")},
        "per_sentence": {si: {"lang": T.GUARDRAIL[si][0],
                              "rate": c.stats([r["rate"] for r in runs if r["sentence"] == si]),
                              "voiced_s": c.stats([r["voiced_s"] for r in runs if r["sentence"] == si])}
                         for si in range(len(T.GUARDRAIL))},
        "back_transcribed": {
            "n": len(picks),
            "zh_cer": c.pooled([r["err"] for r in picks if r["lang"] == "zh"]),
            "en_wer": c.pooled([r["err"] for r in picks if r["lang"] == "en"]),
            "outliers_included": len([r for r in picks if r.get("outlier")]),
        },
        "ttfc_ms": c.stats([r["ttfc_ms"] for r in runs]),
    }
    finish(res, [srv, asr])
    print(c.save_json(f"s3_guardrail{a.tag}.json", res))


EN_WORD = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)*")


def en_word_recall(text: str, hyp: str) -> dict:
    """English words of a mixed sentence found again in the ASR text (case-insensitive; 'Wi-Fi' also
    matches 'wifi')."""
    h = c.norm_text(hyp or "")
    h_compact = h.replace(" ", "")
    words = [w.lower() for w in EN_WORD.findall(text)]
    found = []
    for w in words:
        w_norm = c.norm_text(w)
        ok = bool(re.search(rf"(?<![a-z]){re.escape(w_norm)}(?![a-z])", h)) or w_norm.replace(" ", "") in h_compact
        found.append([w, ok])
    return {"words": found, "n": len(words), "found": sum(1 for _, ok in found if ok)}


def step_kokoro(a) -> None:
    asr = c.Server("asr_kokoro", c.ASR_PORT).start()
    srv = c.Server("tts_kokoro", c.TTS_PORT).start()
    kvoice = {"zh": a.kokoro_zh_voice, "en": a.kokoro_en_voice, "mix": a.kokoro_zh_voice}
    qvoice = {"zh": a.zh_voice, "en": a.en_voice, "mix": a.zh_voice}
    res = header("kokoro", [asr, srv], kokoro=c.KOKORO, kokoro_voices=kvoice, kokoro_lang_code=c.KOKORO_LANG,
                 qwen=c.TTS, qwen_voices=qvoice, qwen_lang_code=c.QWEN_LANG, qwen_streaming_interval=0.32,
                 voice_resolution="Kokoro voices passed as .safetensors paths inside the "
                                  "mlx-community/Kokoro-82M-bf16 snapshot (offline; see b2_common.kokoro_voice_path)",
                 back_transcription="ASR 1.7B, no language field (auto-detect)")
    back_transcribe(b"\0\0" * 24000)
    res["qwen_warmup"] = warmup()
    # Kokoro cold: first request per lang pipeline (model load, then pipeline / G2P creation)
    t0 = time.perf_counter_ns()
    cold = []
    for lang in ("en", "zh"):
        rec, _ = call(tts_payload(T.D13[lang][0], c.kokoro_voice_path(kvoice[lang]), lang, None,
                                  model=c.KOKORO, kokoro=True))
        cold.append({"lang": lang, **{k: rec.get(k) for k in ("status", "ttfc_ms", "total_ms", "audio_s", "error")}})
    res["kokoro_cold"] = cold
    res["kokoro_cold_log"] = c.interesting_log(srv.log_since(t0), t0)[:30]
    recs = []
    out = c.DATA / "s3" / "d13"
    for engine in ("kokoro", "qwen"):
        for lang in ("zh", "en", "mix"):
            for si, text in enumerate(T.D13[lang][: a.sentences]):
                if engine == "kokoro":
                    p = tts_payload(text, c.kokoro_voice_path(kvoice[lang]), lang, None, model=c.KOKORO, kokoro=True)
                else:
                    p = tts_payload(text, qvoice[lang], lang, 0.32)
                rec, pcm = call(p)
                path = out / f"{engine}_{lang}_{si:02d}.wav"
                c.write_wav(path, pcm, c.TTS_RATE)
                bt = back_transcribe(pcm)
                r = {"engine": engine, "lang": lang, "sentence": si, "text": text,
                     **{k: rec.get(k) for k in ("status", "error", "ttfc_ms", "total_ms", "audio_s", "rtf",
                                                "n_chunks", "loadavg1_before")},
                     "voiced_s": c.voiced_span(pcm)["voiced_s"], "asr_text": bt["text"],
                     "asr_language": bt["language"], "err": c.error_rate(text, bt["text"] or "", kind_for(lang)),
                     "wav": str(path.relative_to(c.REPO))}
                if lang == "mix":
                    r["en_recall"] = en_word_recall(text, bt["text"])
                recs.append(r)
            print(engine, lang, "done")
    res["records"] = recs
    summ = {}
    for engine in ("kokoro", "qwen"):
        for lang in ("zh", "en", "mix"):
            sel = [r for r in recs if r["engine"] == engine and r["lang"] == lang]
            s = {"error": c.pooled([r["err"] for r in sel]), "ttfc_ms": c.stats([r["ttfc_ms"] for r in sel]),
                 "total_ms": c.stats([r["total_ms"] for r in sel]), "rtf": c.stats([r["rtf"] for r in sel]),
                 "audio_s": c.stats([r["audio_s"] for r in sel]),
                 "sentences_with_errors": sum(1 for r in sel if r["err"]["edits"] > 0)}
            if lang == "mix":
                s["en_word_recall"] = {"found": sum(r["en_recall"]["found"] for r in sel),
                                       "total": sum(r["en_recall"]["n"] for r in sel)}
            summ[f"{engine}|{lang}"] = s
    res["summary"] = summ
    finish(res, [srv, asr])
    print(c.save_json(f"s3_kokoro{a.tag}.json", res))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="step", required=True)
    p = sub.add_parser("cold")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--model", default=c.TTS)
    p = sub.add_parser("stream")
    p.add_argument("--n", type=int, default=3)
    p = sub.add_parser("sweep")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--model", default=c.TTS)
    p.add_argument("--speakers", default=",".join(SPEAKERS_SWEEP))
    p.add_argument("--intervals", default=",".join(map(str, INTERVALS)))
    p.add_argument("--langs", default=",".join(LANGS))
    p.add_argument("--lengths", default=",".join(LENGTHS))
    p.add_argument("--seed", type=int, default=20261023)
    p.add_argument("--topup", default=None, help="JSON with a 'replace' list: re-run only those requests")
    p = sub.add_parser("pinning")
    p.add_argument("--n", type=int, default=49, help="requests after the warmup (request #2 .. #n+1)")
    p.add_argument("--gaps", default="5,30,120", help="idle gaps in seconds (empty: none)")
    p.add_argument("--gap-reps", type=int, default=2)
    p = sub.add_parser("cancel")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--intervals", default=",".join(map(str, INTERVALS)))
    p = sub.add_parser("concurrency")
    p.add_argument("--n", type=int, default=10)
    p = sub.add_parser("speakers")
    p.add_argument("--n", type=int, default=2, help="runs per sentence")
    p.add_argument("--sentences", type=int, default=10)
    for name in ("guardrail", "kokoro"):
        p = sub.add_parser(name)
        p.add_argument("--n", type=int, default=10)
        p.add_argument("--zh-voice", default="vivian")
        p.add_argument("--en-voice", default="ryan")
        p.add_argument("--kokoro-zh-voice", default="zf_xiaobei")
        p.add_argument("--kokoro-en-voice", default="af_heart")
        p.add_argument("--sentences", type=int, default=20, help="kokoro: sentences per language")
        p.add_argument("--model", default=c.TTS, help="guardrail: TTS model (comparison runs)")
    for p in sub.choices.values():
        p.add_argument("--tag", default="", help="suffix for the result file name")
    a = ap.parse_args()
    try:
        STEPS[a.step](a)
    finally:
        c.stop_all()


STEPS = {"cold": step_cold, "stream": step_stream, "sweep": step_sweep, "pinning": step_pinning,
     "cancel": step_cancel, "speakers": step_speakers, "guardrail": step_guardrail,
     "kokoro": step_kokoro, "concurrency": step_concurrency}


if __name__ == "__main__":
    main()
