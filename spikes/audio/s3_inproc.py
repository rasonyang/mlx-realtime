"""M0 spike S3.3 (and an S4 counterpart): the in-process lower bound, to get the HTTP overhead.

The model is loaded in this process with mlx_audio.utils.load_model, which is what mlx_audio.server's
ModelProvider calls. TTS: model.generate(stream=True) is called with exactly the kwargs that
mlx_audio.server's TTSExecutionAdapter.run_serial builds for the same request (the server's own
SpeechRequest and _resolve_sampling_params are used), and every chunk goes through the same
audio_write(..., format="pcm") call as the server's _emit_audio. Cells, texts and order follow the
HTTP sweep (s3_tts.py sweep), speaker vivian only.

ASR: model.generate on the s4 latency clips with the kwargs the server's STTExecutionAdapter passes
(language, chunk_duration=30.0, prefill_step_size=2048, max_tokens=8192, ...), from a WAV file path as
the server does.

    B2_SCRATCH=<dir> uv run --locked --project spikes/audio python spikes/audio/s3_inproc.py tts|asr [--n 20]

No server may run at the same time (they would share the GPU).
"""

from __future__ import annotations

import argparse
import inspect
import io
import json
import random
import statistics
import tempfile
import time
from pathlib import Path

import b2_common as c
import b2_texts as T


def ms_since(t0: int) -> float:
    return round((time.perf_counter_ns() - t0) / 1e6, 2)


def step_tts(a) -> None:
    from mlx_audio.audio_io import write as audio_write
    from mlx_audio.server import SpeechRequest, _resolve_sampling_params
    from mlx_audio.utils import load_model

    if a.model != c.TTS and not c.model_ok(a.model):
        raise SystemExit(f"{a.model} not status ok in models.json; skipped")
    t0 = time.perf_counter_ns()
    model = load_model(a.model)
    load_ms = ms_since(t0)

    def gen(text: str, voice: str, lang: str, interval: float) -> dict:
        req = SpeechRequest(model=a.model, input=text, voice=voice, lang_code=c.QWEN_LANG[lang],
                            response_format="pcm", stream=True, streaming_interval=interval)
        kwargs = {
            "voice": req.voice, "speed": req.speed, "gender": req.gender, "pitch": req.pitch,
            "instruct": req.instruct, "lang_code": req.lang_code, "ref_audio": None, "ref_text": req.ref_text,
            **_resolve_sampling_params(model, req),
            "stream": req.stream, "streaming_interval": req.streaming_interval,
            "max_tokens": req.max_tokens, "verbose": req.verbose,
        }
        la = c.loadavg1()
        t0 = time.perf_counter_ns()
        chunks, nbytes, first_yield = [], 0, None
        for result in model.generate(req.input, **kwargs):
            t_y = ms_since(t0)
            buf = io.BytesIO()
            audio_write(buf, result.audio, result.sample_rate, format="pcm")
            n = len(buf.getvalue())
            t_p = ms_since(t0)
            if first_yield is None:
                first_yield = t_y
            chunks.append([t_y, t_p, n])
            nbytes += n
        total = ms_since(t0)
        audio_s = nbytes / (c.TTS_RATE * 2)
        return {"first_yield_ms": first_yield, "ttfc_ms": chunks[0][1] if chunks else None, "total_ms": total,
                "audio_s": round(audio_s, 3), "rtf": round(total / 1000 / audio_s, 4) if audio_s else None,
                "n_chunks": len(chunks), "pcm_convert_first_ms": round(chunks[0][1] - chunks[0][0], 3) if chunks else None,
                "chunks": [[ch[1], ch[2]] for ch in chunks], "loadavg1_before": la}

    intervals = [float(x) for x in a.intervals.split(",")]
    langs, lengths, speakers = a.langs.split(","), a.lengths.split(","), a.speakers.split(",")
    cells = [(iv, lang, ln, spk) for iv in intervals for lang in langs for ln in lengths for spk in speakers]
    rng = random.Random(a.seed)
    res = {"spike": "S3", "step": "inproc_tts", "started_at": c.now_iso(), "loadavg1_start": c.loadavg1(),
           "environment": c.environment(), "model": c.model_revision(a.model), "load_model_ms": load_ms,
           "conditions": {"speakers": speakers, "intervals": intervals, "langs": langs, "lengths": lengths,
                          "n_per_cell": a.n, "seed": a.seed, "thread": "main thread, default MLX streams",
                          "kwargs": "as mlx_audio.server run_serial builds them (max_tokens 1200, sampling from "
                                    "generation_config.json)"}}
    t_first = time.perf_counter_ns()
    cold = gen(T.SWEEP["zh"]["medium"][0], "vivian", "zh", 0.32)
    res["cold_first_generate"] = {k: v for k, v in cold.items() if k != "chunks"}
    res["warmup"] = [gen(T.SWEEP[lang]["medium"][0], spk, lang, 0.32)["ttfc_ms"]
                     for lang, spk in (("en", "ryan"), ("mix", "vivian"), ("zh", "vivian"))]
    recs = []
    order = cells[:]
    rng.shuffle(order)
    for cell in order:
        iv, lang, ln, spk = cell
        r = gen(T.SWEEP[lang][ln][0], spk, lang, iv)
        recs.append(r | {"interval": iv, "lang": lang, "length": ln, "speaker": spk, "round": 0, "text_i": 0,
                         "phase": "first_in_cell"})
    for rnd in range(a.n):
        rng.shuffle(order)
        for cell in order:
            iv, lang, ln, spk = cell
            r = gen(T.SWEEP[lang][ln][rnd % 3], spk, lang, iv)
            recs.append(r | {"interval": iv, "lang": lang, "length": ln, "speaker": spk, "round": rnd,
                             "text_i": rnd % 3, "phase": "measured"})
        meas = [r for r in recs if r["round"] == rnd and r["phase"] == "measured"]
        print(f"round {rnd}: median ttfc {statistics.median([r['ttfc_ms'] for r in meas]):.1f} ms, load {c.loadavg1()}")
    summ = {}
    for cell in cells:
        sel = [r for r in recs if (r["interval"], r["lang"], r["length"], r["speaker"]) == cell
               and r["phase"] == "measured"]
        summ["|".join(map(str, cell))] = {k: c.stats([r[k] for r in sel]) for k in
                                          ("first_yield_ms", "ttfc_ms", "total_ms", "rtf", "pcm_convert_first_ms")}
    res["summary"] = summ
    res["records"] = recs
    res["elapsed_s"] = round((time.perf_counter_ns() - t_first) / 1e9, 1)
    res["loadavg1_end"] = c.loadavg1()
    res["finished_at"] = c.now_iso()
    print(c.save_json(f"s3_inproc{a.tag}.json", res))


def step_asr(a) -> None:
    from mlx_audio.utils import load_model

    clip_dir = c.DATA / "s4" / "clips"
    clips = json.loads((clip_dir / "clips.json").read_text())
    t0 = time.perf_counter_ns()
    model = load_model(a.model)
    load_ms = ms_since(t0)
    sig = inspect.signature(model.generate)
    # TranscriptionRequest defaults, filtered by the generate signature as STTExecutionAdapter does;
    # max_tokens: the model's own default (8192) since it exceeds the server's 1024 floor.
    base = {"verbose": False, "chunk_duration": 30.0, "stream": False, "prefill_step_size": 2048,
            "word_timestamps": False, "max_tokens": sig.parameters["max_tokens"].default}

    tmpdir = Path(tempfile.mkdtemp(dir=c.SCRATCH))

    def run(cl: dict, lang_given: bool) -> dict:
        path = tmpdir / cl["file"]
        if not path.exists():
            path.write_bytes((clip_dir / cl["file"]).read_bytes())
        kw = dict(base)
        if lang_given:
            kw["language"] = c.ASR_LANG[cl["set"]]
        la = c.loadavg1()
        t0 = time.perf_counter_ns()
        out = model.generate(str(path), **kw)
        return {"file": cl["file"], "set": cl["set"], "seconds": cl["seconds"], "language_given": lang_given,
                "ms": ms_since(t0), "text": out.text, "loadavg1_before": la}

    res = {"spike": "S4", "step": "inproc_asr", "started_at": c.now_iso(), "environment": c.environment(),
           "model": c.model_revision(a.model), "load_model_ms": load_ms, "kwargs": base,
           "loadavg1_start": c.loadavg1()}
    first = run(clips[0], False)
    res["cold_first_generate_ms"] = first["ms"]
    for f in ("zh_05s_00.wav", "en_05s_00.wav", "mix_02s_00.wav"):
        run(next(x for x in clips if x["file"] == f), False)
    jobs = [(cl, g) for cl in clips for g in (False, True)]
    random.Random(a.seed).shuffle(jobs)
    seen = set()
    for cl, g in jobs:  # one discarded run per cell, as in the HTTP run
        if (cl["set"], cl["seconds"], g) not in seen:
            seen.add((cl["set"], cl["seconds"], g))
            run(cl, g)
    recs = [run(cl, g) for cl, g in jobs]
    summ = {}
    for s in ("zh", "en", "mix"):
        for T_ in (2, 5, 10):
            for g in (False, True):
                sel = [r for r in recs if r["set"] == s and r["seconds"] == T_ and r["language_given"] == g]
                summ[f"{s}|{T_}s|{'lang' if g else 'auto'}"] = {"ms": c.stats([r["ms"] for r in sel])}
    res["summary"] = summ
    res["records"] = recs
    res["loadavg1_end"] = c.loadavg1()
    res["finished_at"] = c.now_iso()
    print(c.save_json(f"s4_inproc{a.tag}.json", res))
    for k, v in summ.items():
        print(k, v["ms"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["tts", "asr"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--model", default=None)
    ap.add_argument("--speakers", default="vivian")
    ap.add_argument("--intervals", default="0.16,0.32,0.8")
    ap.add_argument("--langs", default="zh,en,mix")
    ap.add_argument("--lengths", default="short,medium,long")
    ap.add_argument("--seed", type=int, default=20261023)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    for port in (c.ASR_PORT, c.TTS_PORT):
        if c.port_open(port):
            raise SystemExit(f"a server is listening on {port}; stop it first (GPU sharing)")
    if a.step == "tts":
        a.model = a.model or c.TTS
        step_tts(a)
    else:
        a.model = a.model or c.ASR
        step_asr(a)


if __name__ == "__main__":
    main()
