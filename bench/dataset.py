"""The benchmark's input: utterances with reference transcripts, as 24 kHz client audio.

Two layouts:

- a manifest (``.jsonl``), one utterance per row, paths relative to the manifest's directory. The
  M0 manifest (``spikes/data/manifest.jsonl``) has ``id``, ``set``, ``text``, ``text_norm``,
  ``has_number`` and three copies per row; ``variant`` picks one: ``wide`` (``path_16k``),
  ``tel16k`` (``path_tel_16k``) or ``tel8k`` (``path_8k_ulaw`` / ``path_8k_alaw``: headerless
  8 kHz G.711). Any other manifest gives ``audio`` (a WAV path) and ``text``, and optionally
  ``id``, ``set``, ``text_norm``, ``has_number``;
- a directory of WAVs, each with its transcript in a ``.txt`` beside it.

Every WAV is PCM16 at any rate (8 kHz telephone audio included), mixed to mono and resampled to
24 kHz for the ``audio/pcm`` path. A row's ``set`` (``zh``, ``en``, ``mix``) picks how it is scored;
without one it is guessed from the text. ``has_number`` defaults to whether the text has a digit.
"""

from __future__ import annotations

import json
import re
import wave
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
import numpy.typing as npt

from bench.scoring import norm_text
from mlx_realtime.audio.resample import Resampler

WIRE_RATE = 24_000
FRAME_MS = 20
type Variant = Literal["wide", "tel16k", "tel8k"]
type Law = Literal["ulaw", "alaw"]

_LATIN = re.compile(r"[A-Za-z]")
_CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿]")


@dataclass(frozen=True, slots=True)
class Utterance:
    id: str
    set: str  # zh | en | mix
    text: str
    text_norm: str
    has_number: bool
    pcm: bytes  # 24 kHz mono s16le
    speech_end_ms: int  # where the speech in the audio ends (trailing silence excluded)
    source_rate: int
    g711: tuple[Law, bytes] | None = None  # the original 8 kHz G.711 stream, if that was the input

    @property
    def duration_ms(self) -> int:
        return len(self.pcm) // 2 * 1000 // WIRE_RATE


def load(
    source: Path,
    *,
    variant: Variant = "tel8k",
    sets: Sequence[str] = (),
    ids: str | None = None,
    numbers: Literal["include", "exclude", "only"] = "include",
    limit: int | None = None,
) -> list[Utterance]:
    """The utterances of a manifest or a directory, filtered: ``sets`` (empty: all), ``ids`` (a
    regular expression searched in the id), ``numbers`` and ``limit``."""
    rows = _manifest_rows(source) if source.is_file() else _directory_rows(source)
    pattern = re.compile(ids) if ids else None
    chosen: list[Utterance] = []
    for row in rows:
        if sets and row.set not in sets:
            continue
        if pattern is not None and not pattern.search(row.id):
            continue
        if (numbers == "exclude" and row.has_number) or (numbers == "only" and not row.has_number):
            continue
        chosen.append(row.load(variant))
        if limit is not None and len(chosen) >= limit:
            break
    return chosen


@dataclass(frozen=True, slots=True)
class _Row:
    id: str
    set: str
    text: str
    text_norm: str
    has_number: bool
    paths: dict[str, Path]  # variant -> file; "audio" for a plain WAV
    law: Law | None

    def load(self, variant: Variant) -> Utterance:
        if "audio" in self.paths:
            path, kind = self.paths["audio"], "wav"
        elif variant in self.paths:
            path, kind = self.paths[variant], "g711" if variant == "tel8k" else "wav"
        else:
            raise ValueError(f"{self.id}: no {variant} copy in the manifest")
        g711: tuple[Law, bytes] | None = None
        if kind == "g711":
            assert self.law is not None
            data = path.read_bytes()
            g711 = (self.law, data)
            samples, rate = g711_decode(data, self.law), 8000
        else:
            samples, rate = read_wav(path)
        pcm = to_wire(samples, rate)
        return Utterance(
            id=self.id,
            set=self.set,
            text=self.text,
            text_norm=self.text_norm,
            has_number=self.has_number,
            pcm=pcm,
            speech_end_ms=speech_end_ms(pcm),
            source_rate=rate,
            g711=g711,
        )


def _manifest_rows(path: Path) -> list[_Row]:
    rows: list[_Row] = []
    base = path.parent
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = cast(dict[str, object], json.loads(line))
        text = str(raw["text"])
        paths: dict[str, Path] = {}
        law: Law | None = None
        if "audio" in raw:
            paths["audio"] = base / str(raw["audio"])
        for key, variant in (("path_16k", "wide"), ("path_tel_16k", "tel16k")):
            if key in raw:
                paths[variant] = base / str(raw[key])
        for key, found in (("path_8k_ulaw", "ulaw"), ("path_8k_alaw", "alaw")):
            if key in raw:
                paths["tel8k"] = base / str(raw[key])
                law = cast(Law, found)
        rows.append(
            _Row(
                id=str(raw.get("id", Path(str(raw.get("audio", len(rows)))).stem)),
                set=str(raw.get("set") or guess_set(text)),
                text=text,
                text_norm=str(raw.get("text_norm") or norm_text(text)),
                has_number=bool(raw.get("has_number", _has_digit(text))),
                paths=paths,
                law=law,
            )
        )
    return rows


def _directory_rows(directory: Path) -> list[_Row]:
    rows: list[_Row] = []
    for wav in sorted(directory.glob("*.wav")):
        transcript = wav.with_suffix(".txt")
        if not transcript.is_file():
            continue
        text = transcript.read_text(encoding="utf-8").strip()
        rows.append(
            _Row(
                id=wav.stem,
                set=guess_set(text),
                text=text,
                text_norm=norm_text(text),
                has_number=_has_digit(text),
                paths={"audio": wav},
                law=None,
            )
        )
    return rows


def guess_set(text: str) -> str:
    """``zh``, ``en`` or ``mix``, from the scripts the text uses."""
    cjk, latin = bool(_CJK.search(text)), bool(_LATIN.search(text))
    if cjk and latin:
        return "mix"
    return "zh" if cjk else "en"


def _has_digit(text: str) -> bool:
    return any(char.isdigit() for char in text)


def read_wav(path: Path) -> tuple[npt.NDArray[np.int16], int]:
    """A PCM16 WAV as mono samples and its rate."""
    with wave.open(str(path)) as file:
        if file.getsampwidth() != 2:
            raise ValueError(f"{path}: only 16-bit PCM WAV is supported")
        channels, rate = file.getnchannels(), file.getframerate()
        frames = file.readframes(file.getnframes())
    samples = np.frombuffer(frames, dtype="<i2")
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1).astype(np.int16)
    return samples.astype(np.int16), rate


def to_wire(samples: npt.NDArray[np.int16], rate: int) -> bytes:
    """Mono samples at ``rate`` as 24 kHz s16le."""
    pcm = samples.astype("<i2").tobytes()
    if rate == WIRE_RATE:
        return pcm
    resampler = Resampler(rate, WIRE_RATE)
    return resampler.process(pcm) + resampler.flush()


def speech_end_ms(pcm: bytes, *, floor_dbfs: float = -50.0, below_peak_db: float = 35.0) -> int:
    """The end of the last 20 ms frame of speech: louder than ``floor_dbfs`` and within
    ``below_peak_db`` of the loudest frame. The client's own estimate of the acoustic speech end
    (the server's is the VAD's ``speech_end``)."""
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float64) / 32768
    size = WIRE_RATE * FRAME_MS // 1000
    count = len(samples) // size
    if count == 0:
        return 0
    frames = samples[: count * size].reshape(count, size)
    rms_db = 10 * np.log10(np.mean(frames**2, axis=1) + 1e-12)
    threshold = max(floor_dbfs, float(rms_db.max()) - below_peak_db)
    voiced = np.flatnonzero(rms_db >= threshold)
    return 0 if len(voiced) == 0 else int(voiced[-1] + 1) * FRAME_MS


# G.711 (ITU-T G.711), for the 8 kHz telephone copies; the gateway's own codec arrives in M7.

_ULAW_BIAS = 0x84


def _ulaw_table() -> npt.NDArray[np.int16]:
    codes = ~np.arange(256, dtype=np.int32) & 0xFF
    sign = codes & 0x80
    exponent = (codes >> 4) & 0x07
    mantissa = codes & 0x0F
    magnitude = (((mantissa << 3) + _ULAW_BIAS) << exponent) - _ULAW_BIAS
    return np.where(sign != 0, -magnitude, magnitude).astype(np.int16)


def _alaw_table() -> npt.NDArray[np.int16]:
    codes = np.arange(256, dtype=np.int32) ^ 0x55
    sign = codes & 0x80
    exponent = (codes >> 4) & 0x07
    mantissa = codes & 0x0F
    magnitude = np.where(
        exponent == 0, (mantissa << 4) + 8, ((mantissa << 4) + 0x108) << (exponent - 1)
    )
    return np.where(sign != 0, magnitude, -magnitude).astype(np.int16)


_DECODE: dict[Law, npt.NDArray[np.int16]] = {"ulaw": _ulaw_table(), "alaw": _alaw_table()}


def g711_decode(data: bytes, law: Law) -> npt.NDArray[np.int16]:
    return _DECODE[law][np.frombuffer(data, dtype=np.uint8)]


_ULAW_SEG_END = np.array([0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF])
_ALAW_SEG_END = np.array([0x1F, 0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF])


def g711_encode(samples: npt.NDArray[np.int16], law: Law) -> bytes:
    """The standard G.711 encoders (the reference C code that ``audioop`` uses)."""
    values = samples.astype(np.int32)
    if law == "ulaw":
        scaled = values >> 2
        mask = np.where(scaled < 0, 0x7F, 0xFF)
        magnitude = np.minimum(np.abs(scaled), 8159) + (_ULAW_BIAS >> 2)
        segment = np.searchsorted(_ULAW_SEG_END, magnitude)
        code = (segment << 4) | ((magnitude >> (segment + 1)) & 0xF)
    else:
        scaled = values >> 3
        mask = np.where(scaled >= 0, 0xD5, 0x55)
        magnitude = np.where(scaled >= 0, scaled, -scaled - 1)
        segment = np.searchsorted(_ALAW_SEG_END, magnitude)
        shift = np.where(segment < 2, 1, segment)
        code = (segment << 4) | ((magnitude >> shift) & 0xF)
    code = np.where(segment >= 8, 0x7F, code)
    return (code ^ mask).astype(np.uint8).tobytes()
