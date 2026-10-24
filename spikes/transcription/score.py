"""Score a ``harness.py cer`` run: CER per leg, junk segments (committed audio with no reference
speech: hold music, DTMF, line noise), missed utterances, clipped starts.

    uv run --locked --project spikes/transcription python spikes/transcription/score.py RUN.json [--examples 12]

CER (pooled per leg kind): the reference of a leg is every utterance's text in time order, the hypothesis
every ``completed`` transcript in commit order, both normalized (common.norm), character edit distance.
Junk transcripts are in the hypothesis (a hallucination on hold music costs CER); ``CER*`` leaves the junk
segments out, to show how much of the error is the speech itself. A segment is junk when no reference
utterance overlaps [speech start - 1 s, commit]. A reference utterance is missed when no segment overlaps
it. Clipped start: none of the first 3 normalized characters of an utterance (6+ characters) is in the first 8
characters of the transcript of the segment that holds its start.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import sys

import common as c

sys.path.insert(0, str(c.HERE.parent / "asr_context"))
import terms as T  # noqa: E402

PROMPT_TERMS = sorted({c.norm(t) for t in T.BRANDS + T.PRODUCTS + T.JARGON}, key=len, reverse=True)


def prompt_copy(text: str | None) -> bool:
    """The transcript is made of the prompt's terms only (the ASR copied its context)."""
    rest = c.norm(text or "")
    if not rest:
        return False
    for t in PROMPT_TERMS:
        rest = rest.replace(t, "")
    return not rest


def has_prompt_term(text: str | None) -> bool:
    n = c.norm(text or "")
    return any(t in n for t in PROMPT_TERMS)


def overlaps(a0: float, a1: float, b0: float, b1: float) -> bool:
    return a0 < b1 and b0 < a1


def head_found(head: str, hyp: str) -> bool:
    """Some of the utterance's first 3 characters are in the first 8 of the transcript."""
    return any(ch in hyp[:8] for ch in head[:3])


def score_leg(name: str, leg: dict, meta: dict) -> dict:
    kind = name.rsplit("_", 1)[1]
    utts = [u for u in meta["utts"] if u["leg"] == kind]
    segs = leg["segments"]
    failed = [s for s in segs if s.get("failed")]
    # an utterance whose segment failed (backend_busy, ...) is out of the reference: the failure is counted apart
    utts = [u for u in utts if not any(overlaps(s["speech_t"] - 1.0, s["commit_t"], u["start"], u["end"]) for s in failed)]
    segs = [s for s in segs if not s.get("failed")]
    ref = "".join(c.norm(u["text"]) for u in utts)
    hyp_all = "".join(c.norm(s["transcript"] or "") for s in segs)
    junk, clipped, missed = [], [], []
    keep_hyp = ""
    for s in segs:
        span = (s["speech_t"] - 1.0, s["commit_t"])
        if any(overlaps(*span, u["start"], u["end"]) for u in utts):
            keep_hyp += c.norm(s["transcript"] or "")
        else:
            what = "music" if any(overlaps(*span, h["start"], h["end"]) and h["audio"] == "music" for h in meta["holds"]) else \
                "dtmf" if any(overlaps(*span, d["start"], d["end"]) for d in meta["dtmf"]) else "noise"
            junk.append({"what": what, "audio_s": round(s["commit_t"] - s["speech_t"], 1), "text": s["transcript"]})
    for u in utts:
        holders = [s for s in segs if overlaps(s["speech_t"] - 1.0, s["commit_t"], u["start"], u["end"])]
        if not holders:
            missed.append(u["text"])
            continue
        first = min(holders, key=lambda s: s["speech_t"])
        head = c.norm(u["text"])[:4]
        if len(c.norm(u["text"])) >= 6 and first["speech_t"] - 1.0 <= u["start"] <= first["commit_t"]:
            seg_hyp = c.norm(first["transcript"] or "")
            # the utterance is at the start of its segment only when no other utterance precedes it there
            earlier = [v for v in utts if v is not u and v["start"] < u["start"] and overlaps(first["speech_t"] - 1.0, first["commit_t"], v["start"], v["end"])]
            if not earlier and not head_found(head, seg_hyp):
                clipped.append({"ref": u["text"], "hyp": first["transcript"]})
    return {"ref_chars": len(ref), "edits": c.edit_distance(ref, hyp_all), "edits_nojunk": c.edit_distance(ref, keep_hyp),
            "commits": len(segs) + len(failed), "failed": len(failed), "finals": sum(s["transcript"] is not None for s in segs),
            "junk": junk, "missed": missed, "clipped": clipped, "ended_by": leg["ended_by"],
            "latency_ms": [s["latency_ms"] for s in segs if s["latency_ms"] is not None]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--examples", type=int, default=12)
    ap.add_argument("--calls", default="", help="only these calls (comma-separated)")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    run = json.load(open(a.run))
    metas: dict[str, dict] = {}
    scores: dict[str, dict] = {}
    for name, leg in run["legs"].items():
        if a.calls and name.rsplit("_", 1)[0] not in a.calls.split(","):
            continue
        call = name.rsplit("_", 1)[0]
        metas.setdefault(call, json.loads((c.DATA / "legs" / f"{call}.json").read_text()))
        scores[name] = score_leg(name, leg, metas[call])
    print(f"## {a.run} (prompt {run.get('prompt')}, speed {run.get('speed')})")
    tot = defaultdict(lambda: defaultdict(int))
    for name, s in scores.items():
        kind = name.rsplit("_", 1)[1]
        for k in ("ref_chars", "edits", "edits_nojunk", "commits", "finals", "failed"):
            tot[kind][k] += s[k]
        tot[kind]["junk"] += len(s["junk"])
        tot[kind]["missed"] += len(s["missed"])
        tot[kind]["clipped"] += len(s["clipped"])
        tot[kind]["ended_early"] += s["ended_by"] != "audio_done"
        tot[kind]["legs"] += 1
    print("leg        legs  ref_chars  CER     CER*(no junk)  commits  junk  missed  clipped(head)  failed")
    for kind in ("agent", "customer", "both"):
        t = tot[kind] if kind != "both" else {k: sum(tot[x][k] for x in ("agent", "customer")) for k in tot["agent"]}
        print(f"{kind:9} {t['legs']:5} {t['ref_chars']:10} {100 * t['edits'] / t['ref_chars']:6.2f}% {100 * t['edits_nojunk'] / t['ref_chars']:8.2f}%"
              f"      {t['commits']:8} {t['junk']:5} {t['missed']:7} {t['clipped']:8} {t['failed']:11}")
    junk = [(n, j) for n, s in scores.items() for j in s["junk"]]
    by = defaultdict(list)
    for n, j in junk:
        by[j["what"]].append(j)
    print("junk segments by source:", {k: len(v) for k, v in by.items()})
    for what, items in by.items():
        empty = sum(1 for j in items if not c.norm(j["text"] or ""))
        copies = sum(prompt_copy(j["text"]) for j in items)
        withterm = sum(has_prompt_term(j["text"]) for j in items)
        print(f"  {what}: {len(items)} segments, {len(items) - empty} non-empty, {copies} only prompt terms, {withterm} with a prompt term; examples:")
        for j in items[: a.examples // 3 + 1]:
            print(f"    {j['audio_s']} s -> {j['text']!r}")
    for n, s in scores.items():
        for m in s["missed"][:2]:
            print("missed", n, m)
        for k in s["clipped"][:2]:
            print("clipped", n, k)
    if a.json:
        json.dump(scores, open(a.json, "w"), ensure_ascii=False)


if __name__ == "__main__":
    main()
