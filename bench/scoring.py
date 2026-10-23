"""ASR error rates, scored as M0 scored them (S4; the ``text_norm`` rule of spikes/data/README.md).

Normalization (the manifest's ``text_norm``): NFKC with curly apostrophes mapped to ``'``,
lower-case, every punctuation (P*) or symbol (S*) character replaced by a space except an
apostrophe between two letters, whitespace collapsed, and whitespace next to a CJK character
removed. Numbers are not verbalized: rows with numbers are reported separately.

Tokens:

- ``cer``: characters, spaces excluded (Chinese);
- ``wer``: whitespace-separated words (English);
- ``mer``: one token per CJK character and one per Latin or digit word (code-switched speech).

Rates are pooled over a corpus: total edits over total reference tokens.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

type Kind = Literal["cer", "wer", "mer"]

KIND_FOR_SET: dict[str, Kind] = {"zh": "cer", "en": "wer", "mix": "mer"}

_CJK = "㐀-䶿一-鿿豈-﫿"
_SPACE_NEXT_TO_CJK = re.compile(rf"(?<=[{_CJK}])\s+|\s+(?=[{_CJK}])")
_MER_TOKEN = re.compile(rf"[{_CJK}]|[^\s{_CJK}]+")
_SPACES = re.compile(r"\s+")


def norm_text(text: str) -> str:
    """The manifest's ``text_norm`` rule."""
    folded = unicodedata.normalize("NFKC", text).replace("’", "'").replace("‘", "'")
    folded = folded.lower()
    out: list[str] = []
    for index, char in enumerate(folded):
        between_letters = 0 < index < len(folded) - 1 and (
            folded[index - 1].isalpha() and folded[index + 1].isalpha()
        )
        if char == "'" and between_letters:
            out.append(char)
        elif unicodedata.category(char)[0] in "PS":
            out.append(" ")
        else:
            out.append(char)
    collapsed = _SPACES.sub(" ", "".join(out)).strip()
    return _SPACE_NEXT_TO_CJK.sub("", collapsed)


def tokens(text_norm: str, kind: Kind) -> list[str]:
    match kind:
        case "cer":
            return [char for char in text_norm if not char.isspace()]
        case "wer":
            return text_norm.split()
        case "mer":
            return _MER_TOKEN.findall(text_norm)


@dataclass(frozen=True, slots=True)
class Edits:
    ref: int  # reference tokens
    sub: int
    dels: int
    ins: int

    @property
    def edits(self) -> int:
        return self.sub + self.dels + self.ins

    @property
    def rate(self) -> float | None:
        return self.edits / self.ref if self.ref else None


def align(ref: Sequence[str], hyp: Sequence[str]) -> Edits:
    """Levenshtein alignment of two token sequences, with its substitutions, deletions and
    insertions (on a tie: substitution, then deletion, then insertion, as M0 counted)."""
    rows, cols = len(ref), len(hyp)
    table = [[0] * (cols + 1) for _ in range(rows + 1)]
    for i in range(rows + 1):
        table[i][0] = i
    for j in range(cols + 1):
        table[0][j] = j
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            table[i][j] = min(
                table[i - 1][j] + 1,
                table[i][j - 1] + 1,
                table[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]),
            )
    i, j = rows, cols
    sub = dels = ins = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and table[i][j] == table[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            sub += ref[i - 1] != hyp[j - 1]
            i, j = i - 1, j - 1
        elif i > 0 and table[i][j] == table[i - 1][j] + 1:
            dels += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return Edits(rows, sub, dels, ins)


def error(ref_norm: str, hypothesis: str, kind: Kind) -> Edits:
    """One utterance: ``ref_norm`` is already normalized; the hypothesis is normalized here."""
    return align(tokens(ref_norm, kind), tokens(norm_text(hypothesis), kind))


def pooled(edits: Iterable[Edits]) -> Edits:
    total = Edits(0, 0, 0, 0)
    for one in edits:
        total = Edits(
            total.ref + one.ref,
            total.sub + one.sub,
            total.dels + one.dels,
            total.ins + one.ins,
        )
    return total
