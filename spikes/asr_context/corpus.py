"""Build the corpus manifest ``corpus.jsonl`` (versioned; audio is generated from it).

    uv run --locked --project spikes/asr_context python spikes/asr_context/corpus.py

Rows (one per clip): id, text (the reference: what the speaker says, terms as written), tts (what
is fed to the TTS when it differs: order numbers are spoken character by character), terms,
category, set, voice, degradation, plus ``source`` for human recordings.

- 240 synthetic utterances (set ``tts``), categories brand 60, product 70, order 50, codeswitch 60.
  Each has one voice and one degradation, assigned by a cycle over a seeded shuffle inside its
  category: degradation = j % 5, voice = (j // 5) % 4 (j: rank in the category), so every
  category spreads over all 20 (voice, degradation) cells and the cells hold 11-13 clips.
- 84 human rows (set ``ascend``): the 42 ASCEND mixed-language test recordings, each as the
  untouched wideband copy (``clean``) and through the AICC G.711 path (``g711``). They contain no
  term of the list: the no-term regression set.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import terms as T

HERE = Path(__file__).resolve().parent
VOICES = ["vivian", "serena", "uncle_fu", "aiden"]
DEGRADATIONS = ["clean", "g711", "noise", "onset", "echo"]
SEED = 20241024

BRAND_T = ["{t} 是做什么的？", "我说，{t} 是做什么的？", "你们是 {t} 的客服吗？", "我想问一下 {t} 的营业厅在哪里？",
           "{t} 的客服电话是多少？", "请问 {t} 和别的运营商有什么区别？", "我听朋友推荐了 {t}，想了解一下。",
           "{t} 最近有什么优惠活动吗？", "我是 {t} 的老用户了，想问个问题。", "帮我转接到 {t} 的人工客服。"]
PRODUCT_T = ["我想办一个 {t} 的套餐。", "{t} 每个月多少钱？", "请问 {t} 怎么开通？", "你们的 {t} 有什么优惠吗？",
             "我想把现在的套餐换成 {t}。", "{t} 覆盖我们小区吗？", "我朋友用的 {t}，我也想试试。"]
CODESWITCH_T = ["我这个 {t} 老是出问题。", "能不能帮我看看 {t} 怎么设置？", "我家的 {t} 一直不行，你们能远程处理吗？",
                "{t} 是不是要重新弄一下？", "刚才你说的 {t} 我没听懂，再讲一遍。", "昨天开始 {t} 就不能用了。"]
ORDER_T = ["我的订单号是 {c}，帮我查一下进度。", "工单号 {c}，什么时候能处理？", "我要取消订单 {c}。",
           "维修单 {c} 的师傅怎么还没来？", "麻烦核对一下 {c} 这个单号。", "{c} 这个订单的收货地址我想改一下。"]
CODESWITCH2_T = ["我家的 {a} 连不上 {b} 的网络。", "{b} 的 {a} 我昨天刚装好，还是不行。"]
CN_DIGIT = "零一二三四五六七八九"


def spoken(code: str) -> str:
    """Order numbers are read out character by character: letters as letters, digits as Chinese digits."""
    out = []
    for ch in code:
        if ch.isdigit():
            out.append(CN_DIGIT[int(ch)])
        elif ch.isalpha():
            out.append(ch)
        else:
            out.append(" ")  # a hyphen is a short pause
    return " ".join("".join(out).split())


def spaced(spoken_code: str) -> str:
    return " ".join(spoken_code.replace(" ", ""))


def make_codes(rnd: random.Random, n: int) -> list[str]:
    patterns = [
        lambda: f"NV2024{rnd.randint(1, 12):02d}{rnd.randint(1, 28):02d}",
        lambda: f"TK-{rnd.randint(10000, 99999)}",
        lambda: f"SO-{rnd.randint(100, 999)}-{rnd.randint(100, 999)}",
        lambda: f"RMA-{rnd.randint(1000, 9999)}",
        lambda: f"NN{rnd.randint(1000, 9999)}{rnd.choice('XYZABK')}",
    ]
    out: list[str] = []
    while len(out) < n:
        code = patterns[len(out) % 5]()
        if code not in out and code not in T.CODE_EXAMPLES:
            out.append(code)
    return out


def utterances() -> list[dict]:
    rnd = random.Random(SEED)
    rows: list[dict] = []
    for b in T.BRANDS:
        for tpl in BRAND_T:
            rows.append({"text": tpl.format(t=b), "terms": [b], "category": "brand"})
    for j, p in enumerate(T.PRODUCTS):
        for k in range(5):
            tpl = PRODUCT_T[(j + k) % len(PRODUCT_T)]
            rows.append({"text": tpl.format(t=p), "terms": [p], "category": "product"})
    for j, g in enumerate(T.JARGON):
        for k in range(4):
            tpl = CODESWITCH_T[(j + k) % len(CODESWITCH_T)]
            rows.append({"text": tpl.format(t=g), "terms": [g], "category": "codeswitch"})
    pairs = [(T.PRODUCTS[i % len(T.PRODUCTS)], T.BRANDS[i % len(T.BRANDS)]) for i in range(12)]
    for i, (a, b) in enumerate(pairs):
        rows.append({"text": CODESWITCH2_T[i % 2].format(a=a, b=b), "terms": [a, b], "category": "codeswitch"})
    for i, code in enumerate(make_codes(rnd, 50)):
        tpl = ORDER_T[i % len(ORDER_T)]
        sp = spoken(code)
        rows.append({"text": tpl.format(c=code), "tts": tpl.format(c=spaced(sp)).replace("  ", " "),
                     "terms": [code], "category": "order"})
    assert len({r["text"] for r in rows}) == len(rows) == 240, len(rows)
    rnd.shuffle(rows)
    return rows


def assign(rows: list[dict]) -> list[dict]:
    """degradation = j % 5, voice = (j // 5) % 4 with j the rank inside the category: every
    category spreads over all cells, and the cells stay within one clip of each other."""
    seen: dict[str, int] = {}
    for r in rows:
        j = seen.get(r["category"], 0)
        seen[r["category"]] = j + 1
        r["degradation"], r["voice"] = DEGRADATIONS[j % 5], VOICES[(j // 5) % 4]
    return rows


def human_rows() -> list[dict]:
    manifest = [json.loads(line) for line in (HERE.parent / "datasets" / "manifest.jsonl").read_text().splitlines()]
    out = []
    for r in manifest:
        if r["set"] != "mix":
            continue
        for deg in ("clean", "g711"):
            out.append({"id": f"{r['id']}_{deg}", "text": r["text"], "terms": [], "category": "ascend",
                        "set": "ascend", "voice": "human", "degradation": deg,
                        "source": {"corpus": "ASCEND test (mixed)", "id": r["id"], "path_16k": r["path_16k"]}})
    return out


def main() -> None:
    rows = []
    for i, u in enumerate(assign(utterances())):
        row = {"id": f"u{i + 1:03d}", "text": u["text"]}
        if "tts" in u:
            row["tts"] = u["tts"]
        row.update(terms=u["terms"], category=u["category"], set="tts", voice=u["voice"],
                   degradation=u["degradation"])
        rows.append(row)
    rows += human_rows()
    path = HERE / "corpus.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    (HERE / "context.txt").write_text(T.CONTEXT + "\n")
    from collections import Counter
    tts = [r for r in rows if r["set"] == "tts"]
    print(len(rows), "rows;", len(tts), "synthetic")
    for key in ("category", "voice", "degradation"):
        print(key, dict(Counter(r[key] for r in rows)))
    print("cells", dict(Counter((r["voice"], r["degradation"]) for r in tts)))
    print("cat x deg", dict(Counter((r["category"], r["degradation"]) for r in tts)))


if __name__ == "__main__":
    main()
