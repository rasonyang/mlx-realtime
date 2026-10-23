"""S7.7 — G.711 lookup tables vs stdlib audioop (Python 3.12), exhaustively.

    uv run python s7_7_g711.py

Builds mu-law and A-law tables (256-entry decode, 65 536-entry encode) from the ITU-T G.711
segment algorithm as written in CPython's Modules/audioop.c (st_14linear2ulaw / st_linear2alaw,
fed `sample >> 2` and `sample >> 3` for width 2) and Sun's g711.c (ulaw2linear / alaw2linear).
Every int16 input and every code is checked against audioop's lin2ulaw / lin2alaw / ulaw2lin /
alaw2lin (width 2). For information it also checks Sun's later 16-bit-input linear2ulaw /
linear2alaw, which differ from audioop for some negative inputs.
Writes spikes/results/s7_7_g711.json.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import environment, save_json, stats_ms  # noqa: E402

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import audioop  # deprecated in 3.12, removed in 3.13: the test oracle only

QUANT_MASK = 0x0F
SEG_MASK = 0x70
SEG_SHIFT = 4
SIGN_BIT = 0x80
BIAS = 0x84  # 132
SEG_UEND = (0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF)  # 14-bit mu-law segment ends
SEG_AEND = (0x1F, 0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF)  # 13-bit A-law segment ends
SEG_END16 = (0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF, 0x3FFF, 0x7FFF)  # Sun 16-bit variant


def _search(val: int, table: tuple[int, ...]) -> int:
    for i, end in enumerate(table):
        if val <= end:
            return i
    return len(table)


# ---- decode (Sun g711.c; CPython's _st_*2linear16 tables hold the same values)


def ulaw2linear(u: int) -> int:
    u = ~u & 0xFF
    t = ((u & QUANT_MASK) << 3) + BIAS
    t <<= (u & SEG_MASK) >> SEG_SHIFT
    return (BIAS - t) if (u & SIGN_BIT) else (t - BIAS)


def alaw2linear(a: int) -> int:
    a ^= 0x55
    t = (a & QUANT_MASK) << 4
    seg = (a & SEG_MASK) >> SEG_SHIFT
    if seg == 0:
        t += 8
    elif seg == 1:
        t += 0x108
    else:
        t += 0x108
        t <<= seg - 1
    return t if (a & SIGN_BIT) else -t


# ---- encode, audioop.c flavour (14-bit mu-law input, 13-bit A-law input)


def st_14linear2ulaw(pcm_val: int) -> int:
    """CPython audioop.c: pcm_val is the 14-bit sample (int16 >> 2, arithmetic shift)."""
    if pcm_val < 0:
        pcm_val = -pcm_val
        mask = 0x7F
    else:
        mask = 0xFF
    if pcm_val > 32635:  # CLIP; unreachable for 14-bit input, kept to mirror the C code
        pcm_val = 32635
    pcm_val += BIAS >> 2
    seg = _search(pcm_val, SEG_UEND)
    if seg >= 8:
        return 0x7F ^ mask
    return ((seg << 4) | ((pcm_val >> (seg + 1)) & 0xF)) ^ mask


def st_linear2alaw(pcm_val: int) -> int:
    """CPython audioop.c: pcm_val is the 13-bit sample (int16 >> 3, arithmetic shift)."""
    if pcm_val >= 0:
        mask = 0xD5
    else:
        mask = 0x55
        pcm_val = -pcm_val - 1
    seg = _search(pcm_val, SEG_AEND)
    if seg >= 8:
        return 0x7F ^ mask
    aval = seg << SEG_SHIFT
    aval |= ((pcm_val >> 1) if seg < 2 else (pcm_val >> seg)) & QUANT_MASK
    return aval ^ mask


# ---- encode, Sun g711.c later 16-bit-input flavour (information only)


def sun16_linear2ulaw(pcm_val: int) -> int:
    if pcm_val < 0:
        pcm_val = BIAS - pcm_val
        mask = 0x7F
    else:
        pcm_val += BIAS
        mask = 0xFF
    seg = _search(pcm_val, SEG_END16)
    if seg >= 8:
        return 0x7F ^ mask
    return ((seg << 4) | ((pcm_val >> (seg + 3)) & 0xF)) ^ mask


def sun16_linear2alaw(pcm_val: int) -> int:
    if pcm_val >= 0:
        mask = 0xD5
    else:
        mask = 0x55
        pcm_val = -pcm_val - 8
    seg = _search(pcm_val, SEG_END16)
    if seg >= 8:
        return 0x7F ^ mask
    aval = seg << SEG_SHIFT
    aval |= ((pcm_val >> 4) if seg < 2 else (pcm_val >> (seg + 3))) & QUANT_MASK
    return aval ^ mask


# ---- tables

ALL_INT16 = np.arange(-32768, 32768, dtype=np.int32)  # index = sample + 32768
ALL_CODES = np.arange(256, dtype=np.int32)


def build_tables() -> dict[str, np.ndarray]:
    return {
        "ulaw_decode": np.array([ulaw2linear(c) for c in ALL_CODES], dtype=np.int16),
        "alaw_decode": np.array([alaw2linear(c) for c in ALL_CODES], dtype=np.int16),
        "ulaw_encode": np.array([st_14linear2ulaw(int(s) >> 2) for s in ALL_INT16], dtype=np.uint8),
        "alaw_encode": np.array([st_linear2alaw(int(s) >> 3) for s in ALL_INT16], dtype=np.uint8),
        "ulaw_encode_sun16": np.array([sun16_linear2ulaw(int(s)) for s in ALL_INT16], dtype=np.uint8),
        "alaw_encode_sun16": np.array([sun16_linear2alaw(int(s)) for s in ALL_INT16], dtype=np.uint8),
    }


def lut_encode(table: np.ndarray, pcm: np.ndarray) -> np.ndarray:
    return table[pcm.astype(np.int32) + 32768]


def compare(name: str, ours: np.ndarray, ref: np.ndarray, inputs: np.ndarray) -> dict:
    bad = np.flatnonzero(ours != ref)
    return {
        "check": name,
        "inputs": int(ref.size),
        "mismatches": int(bad.size),
        "examples": [
            {"input": int(inputs[i]), "ours": int(ours[i]), "audioop": int(ref[i])} for i in bad[:8]
        ],
    }


def main() -> None:
    t0 = time.perf_counter()
    T = build_tables()
    build_s = time.perf_counter() - t0

    pcm_all = ALL_INT16.astype(np.int16)
    ref_ulaw_enc = np.frombuffer(audioop.lin2ulaw(pcm_all.tobytes(), 2), dtype=np.uint8)
    ref_alaw_enc = np.frombuffer(audioop.lin2alaw(pcm_all.tobytes(), 2), dtype=np.uint8)
    codes = bytes(range(256))
    ref_ulaw_dec = np.frombuffer(audioop.ulaw2lin(codes, 2), dtype=np.int16)
    ref_alaw_dec = np.frombuffer(audioop.alaw2lin(codes, 2), dtype=np.int16)

    checks = [
        compare("ulaw encode (audioop.c algorithm) vs lin2ulaw", T["ulaw_encode"], ref_ulaw_enc, ALL_INT16),
        compare("alaw encode (audioop.c algorithm) vs lin2alaw", T["alaw_encode"], ref_alaw_enc, ALL_INT16),
        compare("ulaw decode (g711.c ulaw2linear) vs ulaw2lin", T["ulaw_decode"], ref_ulaw_dec, ALL_CODES),
        compare("alaw decode (g711.c alaw2linear) vs alaw2lin", T["alaw_decode"], ref_alaw_dec, ALL_CODES),
        # Through the public LUT functions on a non-contiguous order, to catch indexing slips.
        compare(
            "ulaw encode via lut_encode on shuffled input",
            lut_encode(T["ulaw_encode"], pcm_all[::-1]),
            ref_ulaw_enc[::-1],
            ALL_INT16[::-1],
        ),
    ]
    info = []
    for law, ref in (("ulaw", ref_ulaw_enc), ("alaw", ref_alaw_enc)):
        ours = T[f"{law}_encode_sun16"]
        c = compare(f"{law} encode, Sun 16-bit linear2{law} vs lin2{law}", ours, ref, ALL_INT16)
        bad = ALL_INT16[ours != ref]
        c["mismatch_input_range"] = [int(bad.min()), int(bad.max())] if bad.size else None
        c["mismatch_inputs_negative_only"] = bool(bad.size == 0 or bad.max() < 0)
        info.append(c)

    # Round-trip properties of the tables (and therefore of audioop).
    props = {}
    for law in ("ulaw", "alaw"):
        enc, dec = T[f"{law}_encode"], T[f"{law}_decode"]
        rt = dec[enc[ALL_INT16 + 32768]].astype(np.int32)
        err = np.abs(rt - ALL_INT16)
        codes_rt = enc[dec.astype(np.int32) + 32768]
        props[law] = {
            "max_abs_roundtrip_error": int(err.max()),
            "codes_not_fixed_points": [int(c) for c in np.flatnonzero(codes_rt != ALL_CODES)],
            "decode_min_max": [int(dec.min()), int(dec.max())],
        }

    # Throughput: 20 ms at 8 kHz (160 samples) and 1 s, LUT (numpy) vs audioop.
    rng = np.random.default_rng(7)
    timing = {}
    for label, n in (("20ms_160", 160), ("1s_8000", 8000)):
        x = rng.integers(-32768, 32768, n, dtype=np.int16)
        xb = x.tobytes()
        u = lut_encode(T["ulaw_encode"], x).tobytes()
        runs = {"lut_encode": [], "audioop_encode": [], "lut_decode": [], "audioop_decode": []}
        for i in range(1200):
            a = time.perf_counter_ns()
            lut_encode(T["ulaw_encode"], np.frombuffer(xb, dtype=np.int16)).tobytes()
            b = time.perf_counter_ns()
            audioop.lin2ulaw(xb, 2)
            c = time.perf_counter_ns()
            T["ulaw_decode"][np.frombuffer(u, dtype=np.uint8)].tobytes()
            d = time.perf_counter_ns()
            audioop.ulaw2lin(u, 2)
            e = time.perf_counter_ns()
            if i >= 200:
                runs["lut_encode"].append(b - a)
                runs["audioop_encode"].append(c - b)
                runs["lut_decode"].append(d - c)
                runs["audioop_decode"].append(e - d)
        timing[label] = {k: stats_ms(v) for k, v in runs.items()}

    ok = all(c["mismatches"] == 0 for c in checks)
    payload = {
        "spike": "S7.7 G.711 LUT vs audioop",
        "env": environment(),
        "build_tables_s": round(build_s, 3),
        "checks": checks,
        "all_checks_pass": ok,
        "information_sun_16bit_variant": info,
        "roundtrip_properties": props,
        "timing_ms": timing,
    }
    save_json("s7_7_g711.json", payload)
    for c in checks + info:
        print(f"{c['mismatches']:6d} mismatches / {c['inputs']:6d}  {c['check']}")
    print("ALL PASS" if ok else "MISMATCH FOUND")


if __name__ == "__main__":
    main()
