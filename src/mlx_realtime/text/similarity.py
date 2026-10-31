"""Normalized overlap of a transcript with text we spoke, for echo-turn suppression
(ARCHITECTURE §6.5, D17).

A user turn that is the client's echo of our own speech transcribes to (most of) what we said.
``overlap(transcript, played)`` is the share of the transcript's units that appear, in order, in
the played text: the longest common subsequence of the two unit sequences over the
transcript's length. Units are CJK characters and Latin words (letters and digits, NFKC,
case-folded), as the segmenter counts them: Latin text compared letter by letter would make
any two English sentences look alike. Punctuation, spacing and full-width forms do not count.

The echo-turn rule uses ``contiguity`` instead: the share of the transcript's units that lie in
runs of ``MIN_RUN`` or more units that are also contiguous in the played text. Echo is our text
transcribed in long runs, ASR errors apart; a caller who paraphrases us reuses only short
fragments, which the in-order overlap still counts (live line, issue #9: "为什么天是蓝的" against
"比如天为什么是蓝色的" overlaps 6/7, contiguity 3/7).

``after_echo`` cuts the leading echo off a transcript, for a caller who spoke into the echo within
the same turn (issue #8).
"""

from __future__ import annotations

import unicodedata
from collections.abc import Hashable, Sequence

from mlx_realtime.text.lang import is_cjk

MIN_RUN = 3  # units: the shortest run that is more than a reused fragment


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


def contiguity(transcript: str, played: str, min_run: int = MIN_RUN) -> float:
    """The share of ``transcript``'s units that lie in a run of ``min_run`` or more units that
    also appears, contiguous and in order, in ``played``."""
    needle = units(transcript)
    if not needle:
        return 0.0
    return sum(_in_runs(needle, units(played), min_run)) / len(needle)


def _in_runs(needle: Sequence[str], haystack: Sequence[str], min_run: int) -> list[bool]:
    """Per unit of ``needle``: whether it lies in a run of ``min_run`` or more units that also
    appears contiguous in ``haystack``."""
    where: dict[str, list[int]] = {}
    for index, item in enumerate(haystack):
        where.setdefault(item, []).append(index)
    covered = [False] * len(needle)
    run: dict[int, int] = {}  # played position -> length of the common run ending there
    for index, item in enumerate(needle):
        step: dict[int, int] = {}
        for position in where.get(item, ()):
            length = run.get(position - 1, 0) + 1
            step[position] = length
            if length >= min_run:
                covered[index - length + 1 : index + 1] = [True] * length
        run = step
    return covered


def after_echo(transcript: str, played: str, min_run: int = MIN_RUN) -> str:
    """``transcript`` without its leading echo of ``played``, or whole if it has none.

    The echo prefix is the leading stretch made of runs of ``min_run`` or more units that are
    contiguous in ``played`` (the runs ``contiguity`` counts), joined across gaps shorter than
    ``min_run`` units: an ASR error inside the echo. It ends after the last such run, before a
    gap of ``min_run`` units or more, and so cannot reach into what a caller added. A
    transcript that does not start in a run has no echo prefix. What is left starts at the
    first letter or digit after the prefix (``""`` when the whole transcript is echo)."""
    ends = _unit_ends(transcript)
    if ends is None:
        return transcript  # the units cannot be told apart in the text: nothing to cut
    covered = _in_runs([unit for unit, _ in ends], units(played), min_run)
    prefix, gap = 0, 0
    for index, flag in enumerate(covered):
        if flag:
            prefix, gap = index + 1, 0
        else:
            gap += 1
            if gap >= min_run:
                break
    if prefix == 0:
        return transcript
    rest = transcript[ends[prefix - 1][1] :]
    for index, char in enumerate(rest):
        if char.isalnum():
            return rest[index:]
    return ""


def _unit_ends(text: str) -> list[tuple[str, int]] | None:
    """The units of ``text`` with the index in ``text`` just past each one; None if
    normalizing character by character gives other units than ``units`` does."""
    found: list[tuple[str, int]] = []
    word: list[str] = []
    for index, original in enumerate(text):
        for char in unicodedata.normalize("NFKC", original).casefold():
            if is_cjk(char) or not char.isalnum():
                if word:
                    found.append(("".join(word), index))
                    word.clear()
                if is_cjk(char):
                    found.append((char, index + 1))
            else:
                word.append(char)
    if word:
        found.append(("".join(word), len(text)))
    return found if [unit for unit, _ in found] == units(text) else None


def unmatched(transcript: str, played: str) -> int:
    """How many of ``transcript``'s units are not part of the longest in-order match with
    ``played``."""
    needle = units(transcript)
    return len(needle) - lcs_length(needle, units(played))


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
