"""List what a context mode fixed and broke relative to `none`, and hallucinated terms on ASCEND.

    uv run --locked --project spikes/asr_context python spikes/asr_context/analyze.py RESULTS.json [MODE]
"""
import json
import sys
from collections import Counter

import common as c
import terms as T

d = json.load(open(sys.argv[1]))
mode = sys.argv[2] if len(sys.argv) > 2 else "context"
rows = {r["id"]: r for r in d["rows"]}
fixed, broke = [], []
for cid, per in d["clips"].items():
    r = rows[cid]
    for t in r["terms"]:
        a, b = c.term_hit(t, per["none"]["hyp"]), c.term_hit(t, per[mode]["hyp"])
        if b and not a:
            fixed.append((cid, r["category"], r["degradation"], t, per["none"]["hyp"], per[mode]["hyp"]))
        if a and not b:
            broke.append((cid, r["category"], r["degradation"], t, per["none"]["hyp"], per[mode]["hyp"]))
print(f"## {mode}: fixed {len(fixed)}, broke {len(broke)}")
print("fixed by term:", dict(Counter(f[3] for f in fixed).most_common()))
print("broke by term:", dict(Counter(f[3] for f in broke).most_common()))
print("broke by degradation:", dict(Counter(f[2] for f in broke)))
for tag, lst in (("FIXED", fixed[:12]), ("BROKE", broke)):
    print(tag)
    for x in lst:
        print("  ", x[0], x[2], x[3], "| none:", x[4], "| ctx:", x[5])
# ASCEND: hypotheses that differ, and context terms that appear in the hypothesis but not the reference
vocab = [t for t in T.BRANDS + T.PRODUCTS + T.JARGON]
print("ASCEND diffs")
n = 0
for cid, per in d["clips"].items():
    r = rows[cid]
    if r["set"] != "ascend" or per["none"]["hyp"] == per[mode]["hyp"]:
        continue
    n += 1
    halluc = [t for t in vocab if c.term_hit(t, per[mode]["hyp"]) and not c.term_hit(t, r["text"])
              and not c.term_hit(t, per["none"]["hyp"])]
    e0, e1 = c.score(r["text"], per["none"]["hyp"], [])["edits"], c.score(r["text"], per[mode]["hyp"], [])["edits"]
    print(f"  {cid} edits {e0}->{e1} halluc={halluc} | ref {r['text']} | none {per['none']['hyp']} | ctx {per[mode]['hyp']}")
print("ascend clips that differ:", n)
