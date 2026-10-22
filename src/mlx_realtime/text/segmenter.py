"""Splits streamed LLM text into segments for synthesis, one TTS request each (ARCHITECTURE §7).

Rules:

- Hard boundaries: ``。！？；…``, newline, and ``.!?;`` when followed by whitespace or CJK text.
  A ``.`` after an abbreviation, an initial or a list number is not a boundary.
- The first segment may also end at ``，、：`` (or ``,:`` followed by whitespace) once it holds
  ``first_min_chars``, or at the best boundary so far once ``first_max_wait_ms`` has passed
  since its first token.
- Later segments end at a hard boundary once they hold ``min_chars``.
- A segment longer than ``max_chars`` splits at its latest soft boundary, else a space, else
  between two CJK characters.
- Length counts CJK characters and Latin words.
- ASCII punctuation counts only when followed by whitespace or CJK text, so a segment never
  ends inside a number, decimal, URL, e-mail address or ID; whitespace between two digit groups
  ("138 0013 8000") is never a cut either, so a spaced phone number reaches the normalizer
  whole.

Segments are slices of the input: joined, they reproduce it exactly.
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass

from mlx_realtime.text.lang import is_cjk

_HARD_CJK = "。！？；…"
_HARD_ASCII = ".!?;"
_SOFT_CJK = "，、："
_SOFT_ASCII = ",:"
_CLOSERS = "\"'”’」』）)]}》〉"
_ABBREVIATIONS = frozenset(
    {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e", "inc", "ltd"}
    | {"co", "no", "fig", "approx", "dept", "est", "u.s", "u.k"}
)
_LATIN_WORD = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z]+)?")


class _Kind(enum.IntEnum):
    """Cut strength, weakest first."""

    CJK_PAIR = 0
    SPACE = 1
    SOFT = 2
    HARD = 3


@dataclass(frozen=True, slots=True)
class _Cut:
    end: int
    kind: _Kind


def text_units(text: str) -> int:
    """Segment length: CJK characters plus Latin words."""
    return sum(1 for char in text if is_cjk(char)) + len(_LATIN_WORD.findall(text))


class Segmenter:
    """Incremental: feed it with ``push``, ``poll`` it on a timer, ``flush`` it at the end."""

    def __init__(
        self, *, first_min_chars: int, first_max_wait_ms: int, min_chars: int, max_chars: int
    ) -> None:
        self._first_min_chars = first_min_chars
        self._first_max_wait_ms = first_max_wait_ms
        self._min_chars = min_chars
        self._max_chars = max_chars
        self._buffer = ""
        self._first_done = False
        self._first_token_ms: float | None = None

    def push(self, text: str, now_ms: float) -> list[str]:
        """Add streamed text; return the segments it completes."""
        if not text:
            return []
        if self._first_token_ms is None:
            self._first_token_ms = now_ms
        self._buffer += text
        return self._drain(now_ms, final=False)

    def poll(self, now_ms: float) -> list[str]:
        """Release the first segment once its wait has expired, if a boundary exists."""
        return self._drain(now_ms, final=False)

    def flush(self) -> list[str]:
        """End of text: return every remaining segment."""
        segments = self._drain(None, final=True)
        if self._buffer:
            segments.append(self._buffer)
            self._buffer = ""
        return segments

    def deadline_ms(self) -> float | None:
        """When ``poll`` may next release the first segment, or ``None``."""
        if self._first_done or self._first_token_ms is None:
            return None
        return self._first_token_ms + self._first_max_wait_ms

    def _drain(self, now_ms: float | None, *, final: bool) -> list[str]:
        segments: list[str] = []
        while (end := self._next_cut(now_ms, final)) is not None:
            segments.append(self._buffer[:end])
            self._buffer = self._buffer[end:]
            self._first_done = True
        return segments

    def _next_cut(self, now_ms: float | None, final: bool) -> int | None:
        text = self._buffer
        cuts = [cut for cut in _cuts(text, final) if text[: cut.end].strip()]
        if not self._first_done:
            for cut in cuts:
                if cut.kind is _Kind.HARD:
                    return cut.end
                if cut.kind is _Kind.SOFT and text_units(text[: cut.end]) >= self._first_min_chars:
                    return cut.end
            deadline = self.deadline_ms()
            if now_ms is not None and deadline is not None and now_ms >= deadline:
                best = _latest(cuts, _Kind.SOFT) or _latest(cuts, _Kind.SPACE)
                if best is not None:
                    return best.end
        else:
            for cut in cuts:
                if cut.kind is _Kind.HARD and text_units(text[: cut.end]) >= self._min_chars:
                    return cut.end
        if text_units(text) > self._max_chars:
            within = [cut for cut in cuts if text_units(text[: cut.end]) <= self._max_chars]
            for floor in (_Kind.SOFT, _Kind.SPACE, _Kind.CJK_PAIR):
                best = _latest(within, floor)
                if best is not None:
                    return best.end
        return None


def _latest(cuts: list[_Cut], floor: _Kind) -> _Cut | None:
    """The last cut at least as strong as ``floor``."""
    return next((cut for cut in reversed(cuts) if cut.kind >= floor), None)


def _cuts(text: str, final: bool) -> list[_Cut]:
    """Every confirmed cut point in ``text``, in order of position."""
    cuts: list[_Cut] = []
    for index, char in enumerate(text):
        if char == "\n" or char in _HARD_CJK:
            cuts.append(_Cut(_extend(text, index + 1), _Kind.HARD))
        elif char in _SOFT_CJK:
            cuts.append(_Cut(_extend(text, index + 1), _Kind.SOFT))
        elif char in _HARD_ASCII or char in _SOFT_ASCII:
            after = _skip_closers(text, index + 1)
            if not _ends_clause(text, after, final):
                continue
            if char == "." and (_is_abbreviation(text, index) or _is_list_number(text, index)):
                continue
            kind = _Kind.HARD if char in _HARD_ASCII else _Kind.SOFT
            cuts.append(_Cut(_extend(text, after), kind))
        elif char.isspace():
            if (
                index > 0
                and not text[index - 1].isspace()
                and not _between_digit_groups(text, index, final)
            ):
                cuts.append(_Cut(_extend(text, index), _Kind.SPACE))
        elif index > 0 and is_cjk(char) and is_cjk(text[index - 1]):
            cuts.append(_Cut(index, _Kind.CJK_PAIR))
    return cuts


def _ends_clause(text: str, index: int, final: bool) -> bool:
    """Whether ASCII punctuation just before ``index`` ends a clause (it may still be inside a
    number or URL while nothing follows it)."""
    if index == len(text):
        return final
    char = text[index]
    return char.isspace() or is_cjk(char) or char in _HARD_CJK or char in _SOFT_CJK


def _between_digit_groups(text: str, space: int, final: bool) -> bool:
    """Whether the whitespace at ``space`` separates digit groups ("138 0013 8000"): after a
    digit it counts as one until the next character shows otherwise."""
    if not text[space - 1].isdigit():
        return False
    after = space
    while after < len(text) and text[after].isspace():
        after += 1
    return not final if after == len(text) else text[after].isdigit()


def _skip_closers(text: str, index: int) -> int:
    while index < len(text) and text[index] in _CLOSERS:
        index += 1
    return index


def _extend(text: str, index: int) -> int:
    """Move a cut past trailing CJK punctuation, closing quotes and whitespace."""
    while index < len(text) and (text[index] in _CLOSERS or text[index] in _HARD_CJK):
        index += 1
    while index < len(text) and text[index].isspace():
        index += 1
    return index


def _is_abbreviation(text: str, dot: int) -> bool:
    start = dot
    while start > 0 and (
        text[start - 1] == "." or (text[start - 1].isascii() and text[start - 1].isalpha())
    ):
        start -= 1
    word = text[start:dot].lower()
    return (len(word) == 1 and word.isalpha()) or word in _ABBREVIATIONS


def _is_list_number(text: str, dot: int) -> bool:
    """``1.`` at the start of a line enumerates; it does not end a sentence."""
    start = dot
    while start > 0 and text[start - 1].isdigit():
        start -= 1
    return start < dot and (start == 0 or text[start - 1] == "\n")
