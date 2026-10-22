"""The language a match is read in, decided by its neighbourhood (ARCHITECTURE §7).

Rules:

- Look outwards from the match to the nearest letter on the left and on the right, skipping
  digits, spaces, punctuation and symbols.
- If either letter is CJK, read in Chinese; else if there is a letter at all, read in English;
  else fall back to the segment's language tag.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass

from mlx_realtime.text.lang import is_cjk
from mlx_realtime.text.normalize.common import Lang


@dataclass(frozen=True, slots=True)
class Context:
    """One text and where its letters are, to decide the language of each match in it."""

    text: str
    fallback: Lang
    letters: tuple[int, ...]  # the index of every letter in ``text``, ascending

    @classmethod
    def of(cls, text: str, fallback: Lang) -> Context:
        letters = tuple(index for index, char in enumerate(text) if char.isalpha())
        return cls(text, fallback, letters)

    def lang(self, start: int, end: int) -> Lang:
        """The language to read ``text[start:end]`` in."""
        sides = (bisect_left(self.letters, start) - 1, bisect_left(self.letters, end))
        nearest = [self.text[self.letters[side]] for side in sides if 0 <= side < len(self.letters)]
        if any(is_cjk(letter) for letter in nearest):
            return "zh"
        return "en" if nearest else self.fallback
