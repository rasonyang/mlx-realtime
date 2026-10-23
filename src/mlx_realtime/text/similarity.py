"""Normalized overlap of a transcript with text we spoke, for echo-turn suppression
(ARCHITECTURE §6.5, D17).

A user turn that is the client's echo of our own speech transcribes to (most of) what we said.
``overlap(transcript, played)`` is the share of the transcript's units that appear, in order, in
the played text: the longest common subsequence of the two unit sequences over the
transcript's length. Units are CJK characters and Latin words (letters and digits, NFKC,
case-folded), as the segmenter counts them: Latin text compared letter by letter would make
any two English sentences look alike. Punctuation, spacing and full-width forms do not count.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Hashable, Sequence

from mlx_realtime.text.lang import is_cjk


def units(text: str) -> list[str]:
    """CJK characters and Latin words, NFKC-normalized and case-folded."""
    folded = unicodedata.normalize("NFKC", text).casefold()
    found: list[str] = []
    word: list[str] = []
    for char in folded:
        if is_cjk(char) or not char.isalnum():
            if word:
                found.append("".join(word))
                word.clear()
            if is_cjk(char):
                found.append(char)
        else:
            word.append(char)
    if word:
        found.append("".join(word))
    return found


def overlap(transcript: str, played: str) -> float:
    """The share of ``transcript``'s units found in order in ``played``."""
    needle, haystack = units(transcript), units(played)
    if not needle:
        return 0.0
    return lcs_length(needle, haystack) / len(needle)


def lcs_length(first: Sequence[Hashable], second: Sequence[Hashable]) -> int:
    """The length of the longest common subsequence (Hyyrö's bit-parallel algorithm, with the
    positions of ``first`` as the bits of one integer: one pass over ``second``)."""
    if not first or not second:
        return 0
    masks: dict[Hashable, int] = {}
    for index, item in enumerate(first):
        masks[item] = masks.get(item, 0) | 1 << index
    full = (1 << len(first)) - 1
    row = full
    for item in second:
        common = row & masks.get(item, 0)
        row = ((row + common) | (row - common)) & full
    return len(first) - row.bit_count()
