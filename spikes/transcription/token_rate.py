"""Tokens per second of the corpus' real utterances (and the fastest ones), to size the ASR token cap.

    cd spikes/transcription && uv run --locked --project ../.. python token_rate.py   # the root environment has transformers

The tokenizer is Qwen3-ASR's own (from the local Hugging Face cache). The audio the ASR gets for an
utterance is the utterance plus the pump's 0.6 s of silence and up to 1 s of lead: the rate is also
shown over that.
"""

from __future__ import annotations

import json
import os

import common as c

os.environ.setdefault("HF_HUB_OFFLINE", "1")
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained("mlx-community/Qwen3-ASR-1.7B-8bit")
rows = []
for call in c.load_calls():
    meta = json.loads((c.DATA / "legs" / f"{call['id']}.json").read_text())
    for u in meta["utts"]:
        n = len(tok(u["text"])["input_ids"])
        d = u["end"] - u["start"]
        rows.append((n / d, n, d, n / (d + 1.6), u["text"][:30]))
rows.sort(reverse=True)
print("utterances", len(rows))
print("tokens/s of speech: max %.1f, p99 %.1f, median %.1f" % (rows[0][0], rows[len(rows) // 100][0], rows[len(rows) // 2][0]))
print("tokens/s over speech + 1.6 s: max %.1f" % max(r[3] for r in rows))
print("highest:", [(round(r[0], 1), r[1], round(r[2], 1), r[4]) for r in rows[:5]])
short = sorted(rows, key=lambda r: -(r[1] / (r[2] + 1.6)))[:3]
print("highest over speech+1.6:", [(round(r[3], 1), r[1], round(r[2], 1), r[4]) for r in short])
# the cap 12 + 8 s on audio = speech + 1.6 s: smallest margin
print("min margin cap/needed: %.2f" % min((12 + 8 * (r[2] + 1.6)) / max(r[1], 1) for r in rows))
