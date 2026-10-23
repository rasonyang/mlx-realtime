"""M0 spike S9 (G5): TTS text-normalization probe. Throwaway (work package B2).

1. synth: every b2_texts.S9 sentence is synthesized RAW by Qwen3-TTS-0.6B-CustomVoice-8bit (default
   zh speaker vivian, lang_code chinese, stream, streaming_interval 0.32), --runs times, because
   sampling is on (temperature 0.9). The intended spoken form (numbers written out the way a
   normalizer would) is synthesized once as a reference, and so is a plausible wrong reading
   (b2_texts.S9_ALT) where one exists, to check that the ASR writes the two differently. Every WAV is
   back-transcribed by
   Qwen3-ASR-1.7B-8bit, both without `language` (auto) and with language=Chinese. The 6 English
   extras (b2_texts.S9_EN) use speaker ryan and lang_code english.
   WAVs: spikes/data/s9/ (git-ignored), for a human to listen to later.
2. score: per sentence and run, every key item (number, ID, amount) is checked against the ASR text
   with the rules in S9_RULES below, written after reading the ASR's conventions on the spoken
   references (e.g. it writes digit strings for digit-by-digit readings). Verdict per run: correct
   (all items read as intended), partial (some), wrong (none). Per category: counts of correct /
   partial / wrong.

    B2_SCRATCH=<dir> uv run --locked --project spikes/audio python spikes/audio/s9_norm_probe.py synth|score
"""

from __future__ import annotations

import argparse
import json
import re
import time

import b2_common as c
import b2_texts as T

OUT = c.DATA / "s9"


def tts(text: str, voice: str, lang_code: str) -> tuple[dict, bytes]:
    payload = {"model": c.TTS, "input": text, "voice": voice, "lang_code": lang_code, "response_format": "pcm",
               "stream": True, "streaming_interval": 0.32}
    r = c.tts_request(payload)
    return {"status": r.status, "error": r.error, "ttfc_ms": r.first_byte_ms, "total_ms": r.end_ms,
            "audio_s": round(len(r.body) / (2 * c.TTS_RATE), 3)}, r.body


def asr(pcm: bytes, language: str | None) -> dict:
    fields = {"model": c.ASR, "response_format": "verbose_json"}
    if language:
        fields["language"] = language
    r, js = c.asr_request(c.tts_pcm_to_asr_wav(pcm), fields)
    return {"text": (js or {}).get("text"), "language": (js or {}).get("language"), "status": r.status,
            "error": r.error}


def step_synth(a) -> None:
    asr_srv = c.Server("asr_s9", c.ASR_PORT).start()
    tts_srv = c.Server("tts_s9", c.TTS_PORT).start()
    res = {"spike": "S9", "step": "synth", "started_at": c.now_iso(), "loadavg1_start": c.loadavg1(),
           "environment": c.environment(),
           "models": {m: c.model_revision(m) for m in (c.TTS, c.ASR)},
           "servers": [asr_srv.describe(), tts_srv.describe()],
           "conditions": {"zh": {"voice": "vivian", "lang_code": "chinese"},
                          "en_extra": {"voice": "ryan", "lang_code": "english"},
                          "streaming_interval": 0.32, "raw_runs": a.runs, "spoken_runs": 1,
                          "sampling": "server defaults (generation_config.json: temperature 0.9, top_k 50, "
                                      "top_p 1.0, repetition_penalty 1.05)",
                          "asr": "verbose_json, once without language and once with the language name"}}
    # warm both servers
    _, pcm = tts("您好，请问有什么可以帮您？", "vivian", "chinese")
    asr(pcm, None)
    items = [dict(s, set="zh") for s in T.S9] + [dict(s, set="en") for s in T.S9_EN]
    if a.limit:
        items = items[: a.limit]
    recs = []
    for s in items:
        voice, lang_code, lang_name = (("vivian", "chinese", "Chinese") if s["set"] == "zh"
                                       else ("ryan", "english", "English"))
        variants = [("raw", s["text"], a.runs), ("spoken", s["spoken"], 1)]
        if s["id"] in T.S9_ALT:
            variants.append(("alt", T.S9_ALT[s["id"]], 1))
        for variant, text, runs in variants:
            for run in range(runs):
                meta, pcm = tts(text, voice, lang_code)
                path = OUT / f"{s['id']}_{variant}{run if variant == 'raw' else ''}.wav"
                c.write_wav(path, pcm, c.TTS_RATE)
                recs.append({"id": s["id"], "cat": s["cat"], "set": s["set"], "variant": variant, "run": run,
                             "input": text, **meta, "wav": str(path.relative_to(c.REPO)),
                             "asr_auto": asr(pcm, None), "asr_lang": asr(pcm, lang_name)})
        last = recs[-1]
        print(s["id"], "|", s["text"], "->", [r["asr_auto"]["text"] for r in recs if r["id"] == s["id"]])
    res["records"] = recs
    res["loadavg1_end"] = c.loadavg1()
    res["finished_at"] = c.now_iso()
    for srv in (tts_srv, asr_srv):
        res.setdefault("server_logs", {})[srv.name] = c.interesting_log(srv.lines, srv.started_ns)[-30:]
        srv.stop()
    print(c.save_json(f"s9_transcripts{a.tag}.json", res))


# ------------------------------------------------------------------------------------------------
# Scoring rules: per sentence, the key items and, for each, a regex that matches the ASR text when the
# item was read as intended. Filled in after reading the transcripts of the spoken references.
# Applied (case-insensitive) to the ASR text after canon(): NFKC, whitespace and separator punctuation
# removed. One = [一幺1]. The ASR writes what it hears in characters (digit by digit as 一二八, cardinals
# as 一百二十八), and the wrong-reading references (S9_ALT) came back distinct from the intended ones,
# so these patterns separate the readings.
O = "[一幺1]"
S9_RULES: dict[str, list[tuple[str, str]]] = {
    "num1": [("3笔", r"(三|3)笔")],
    "num2": [("128人", r"(一百二十八|128)人")],
    "num3": [("15600分", r"(一万五千六百|15600)分")],
    "num4": [("8.5折", r"(八点五|8\.5|八五)折")],
    "num5": [("99.6%", r"百分之九十九点六|99\.6%")],
    "num6": [("第2位", r"第(二|2)位")],
    "money1": [("¥1,280.50", r"一千[二两]百八十(点五零?元|元五角|块五)|1280\.50?元")],
    "money2": [("100元", r"(一百|100)元")],
    "money3": [("12.5元", r"(十二点五|12\.5)元")],
    "money4": [("$29.99", r"(二十九点九九|29\.99)美元")],
    "money5": [("满200", r"满(二百|两百|200)"), ("减30", r"减(三十|30)")],
    "money6": [("5%", r"百分之五"), ("2500元", r"(两千五百|二千五百|2500)元")],
    "date1": [("2024年", r"二零二四年"), ("3月15日", r"三月十五[日号]"), ("2:30", r"下午[两二]点(三十|半)")],
    "date2": [("12月31日", r"十二月三十一[日号]"), ("24:00", r"二十四点。")],
    "date3": [("9:00", r"九点到"), ("21:00", r"到二十一点。")],
    "date4": [("2025-01-01", r"二零二五年一月一[日号]")],
    "date5": [("15:45", r"十五点四十五分?之前")],
    "date6": [("5号", r"每月(五|5)号")],
    "phone1": [("400-820-8820", r"热线四零零八二零八八二零。")],
    "phone2": [("95588", r"电话九五五八八。")],
    "phone3": [("138 0013 8000", rf"手机号是{O}三八零零{O}三八零零零对吗")],
    "phone4": [("5678", r"尾号是五六七八。")],
    "phone5": [("010-62345678", rf"电话是零{O}零六二三四五六七八。")],
    "phone6": [("021", rf"回拨零二{O}"), ("12345678", rf"{O}二三四五六七八转"), ("转8001", rf"转八零零{O}分机")],
    "phone7": [("13912345678", rf"电话是{O}三九{O}二三四五六七八。")],
    "id1": [("A20241015001", rf"A二零二四{O}零{O}五零零{O}。")],
    "id2": [("SF1234567890", rf"SF{O}二三四五六七八九零。")],
    "id3": [("TK-2024-0789", r"TK二零二四零七八九。")],
    "id4": [("386425", r"是三八六四二五五分钟")],
    "id5": [("VIP8866", r"VIP八八六六。")],
    "id6": [("RT20241102A3", rf"RT二零二四{O}{O}零二A三。")],
    "unit1": [("2.5kg", r"二点五(千克|公斤)")],
    "unit2": [("15GB", r"十五(GB|G)")],
    "unit3": [("5km", r"五(公里|千米)")],
    "unit4": [("6.1英寸", r"六点一英寸")],
    "unit5": [("26℃", r"二十六(摄氏)?度")],
    "unit6": [("500Mbps", r"五百兆")],
    "mix1": [("App", r"App")],
    "mix2": [("www.example.com", r"www(\.|点|dot)*example(\.|点|dot)*com")],
    "mix3": [("@", r"@|艾特|at"), ("example.com", r"example(\.|点|dot)com")],
    "mix4": [("iPhone 15", r"iPhone(十五|15)")],
    "mix5": [("Wi-Fi", r"Wi-?Fi")],
    "mix6": [("PayPal", r"PayPal"), ("Apple Pay", r"ApplePay")],
    "en_money": [("$1,280.50", r"onethousandtwohundred(and)?eightydollarsandfiftycents|\$1,?280\.50")],
    "en_phone": [("1-800-555-0199", r"one(eighthundred|eightzerozero)fivefivefivezerooneninenine")],
    "en_date": [("03/15/2024", r"march(fifteenth|fifteen)(twentytwentyfour|2024)"),
                ("2:30 PM", r"twothirtyp\.?m")],
    "en_id": [("A20241015001", r"atwozerotwofouronezeroonefivezerozeroone")],
    "en_unit": [("2.5 kg", r"(2\.5|twopointfive)kilograms?")],
    "en_percent": [("15%", r"15%|fifteenpercent"), ("$200", r"\$200|twohundreddollars")],
}


def canon(text: str) -> str:
    """ASR text, NFKC-folded, without whitespace, quotes and separator punctuation (case kept)."""
    import unicodedata

    t = unicodedata.normalize("NFKC", text or "")
    return re.sub(r"[\s,，、·:：\-–—'’“”\"]+", "", t)


def step_score(a) -> None:
    src = json.loads((c.RESULTS / "s9_transcripts.json").read_text())
    recs = src["records"]
    out, per_cat = [], {}
    for r in recs:
        rules = S9_RULES.get(r["id"])
        if not rules:
            continue
        for which in ("asr_auto", "asr_lang"):
            hyp = canon(r[which]["text"])
            hits = [[name, bool(re.search(rx, hyp, re.IGNORECASE))] for name, rx in rules]
            n_ok = sum(ok for _, ok in hits)
            verdict = "correct" if n_ok == len(hits) else ("wrong" if n_ok == 0 else "partial")
            out.append({"id": r["id"], "cat": r["cat"], "set": r["set"], "variant": r["variant"], "run": r["run"],
                        "asr": which, "hyp": r[which]["text"], "items": hits, "verdict": verdict})
    for o in out:
        if o["variant"] != "raw":
            continue
        key = f"{o['set']}|{o['cat']}|{o['asr']}"
        d = per_cat.setdefault(key, {"correct": 0, "partial": 0, "wrong": 0})
        d[o["verdict"]] += 1
    # Per sentence over its raw runs (ASR auto): correct in every run, wrong in every run, or mixed.
    per_sentence: dict = {}
    for o in out:
        if o["variant"] == "raw" and o["asr"] == "asr_auto":
            per_sentence.setdefault(o["id"], {"set": o["set"], "cat": o["cat"], "runs": []})["runs"].append(o["verdict"])
    per_cat_sent: dict = {}
    for sid, v in per_sentence.items():
        runs = v["runs"]
        v["summary"] = ("correct" if all(x == "correct" for x in runs)
                        else "wrong" if all(x == "wrong" for x in runs) else "mixed")
        d = per_cat_sent.setdefault(f"{v['set']}|{v['cat']}", {"correct": 0, "mixed": 0, "wrong": 0})
        d[v["summary"]] += 1
    checks = {"spoken_not_correct": [o["id"] for o in out if o["variant"] == "spoken" and o["asr"] == "asr_auto"
                                     and o["verdict"] != "correct"],
              "alt_items_matching_intended": [[o["id"], [n for n, ok in o["items"] if ok]] for o in out
                                              if o["variant"] == "alt" and o["asr"] == "asr_auto" and o["verdict"] != "wrong"],
              "auto_vs_lang_disagreements": sum(1 for o in out if o["asr"] == "asr_auto" and o["verdict"] != next(
                  x["verdict"] for x in out if x["asr"] == "asr_lang" and (x["id"], x["variant"], x["run"]) ==
                  (o["id"], o["variant"], o["run"])))}
    res = {"spike": "S9", "step": "score", "scored_at": c.now_iso(), "rules": S9_RULES, "checks": checks,
           "per_category_runs": per_cat, "per_category_sentences": per_cat_sent, "per_sentence": per_sentence,
           "results": out}
    print(c.save_json("s9_scores.json", res))


def step_show(a) -> None:
    """Print every sentence with the ASR text of each variant, for writing S9_RULES."""
    src = json.loads((c.RESULTS / f"s9_transcripts{a.tag}.json").read_text())
    by_id: dict[str, list] = {}
    for r in src["records"]:
        by_id.setdefault(r["id"], []).append(r)
    for sid, rs in by_id.items():
        print(f"\n== {sid} [{rs[0]['cat']}] {rs[0]['input'] if rs[0]['variant'] == 'raw' else ''}")
        for r in rs:
            a_, l_ = r["asr_auto"]["text"], r["asr_lang"]["text"]
            same = "=" if a_ == l_ else f" | lang: {l_}"
            print(f"  {r['variant']}{r['run'] if r['variant'] == 'raw' else ''} ({r['audio_s']} s): {a_}{same}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["synth", "score", "show"])
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    try:
        {"synth": step_synth, "score": step_score, "show": step_show}[a.step](a)
    finally:
        c.stop_all()


if __name__ == "__main__":
    main()
