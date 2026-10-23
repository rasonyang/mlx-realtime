"""Shared pieces of the normalizer: the match type, masking and the number guards.

Rules:

- Each category finds ``Match`` es in the masked text: spans that an earlier category claimed are
  replaced by ``MASK`` characters, so no later pattern can match inside them.
- A number is a whole token. It is not glued to a letter or digit, directly or through one of
  ``-_/.,:``, so ``Qwen3``, ``COVID-19``, ``v1.2`` and ``1.2.3`` hold no free-standing number.
- ``-`` before a number is a minus sign (``温度-5°C``, ``x = -5``), except right after a letter,
  a digit, a claimed span, a closing bracket, ``%``, or a number with a unit character (日, 号, 月,
  年, 天, 点, 分, 秒, 时, 元, 块, 个, 次, 周, 岁, 度): there it is a hyphen or a range dash
  (``10月23日-25日``, ``10度-15度``).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

Lang = Literal["zh", "en"]

MASK = "\x00"
"""Stands in for each character of a claimed span."""

NUMBER = r"(?>(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?)"
"""A written number: a plain or comma-grouped integer with optional decimals, matched whole."""

FREE_LEFT = r"(?<![A-Za-z0-9_])(?<![A-Za-z0-9][-_/.,:])"
"""Nothing glues the token that starts here to a letter or digit on its left."""

FREE_RIGHT = r"(?![A-Za-z0-9_])(?![-_/.,:][A-Za-z0-9])"
"""Nothing glues the token that ends here to a letter or digit on its right."""

_MINUS = "-−－"
_BEFORE_DASH = frozenset(MASK + ")]}）】」》%％")
_UNIT_CHARS = frozenset("日号月年天点分秒时元块个次周岁度")
_DIGIT_OR_MASK = frozenset("0123456789" + MASK)
_KEYWORD_WINDOW = 40


@dataclass(frozen=True, slots=True)
class Match:
    """A recognized span ``text[start:end]`` and how to read it in a language.

    The optional facts let a category that is off still read some of its spans (G5): the
    amount of money, whether a date is numeric, the digits of a plain cardinal.
    """

    start: int
    end: int
    read: Callable[[Lang], str]
    amount: Decimal | None = None  # money: the largest value of the amount, scale applied
    numeric_date: bool = False  # dates: written all in digits (2024-10-23, 03/15/2024)
    digits: int = 0  # numbers: the integer digits of a plain cardinal (0 for anything else)


def is_minus(text: str, index: int) -> bool:
    """Whether ``text[index]`` is a minus sign (see the module rules)."""
    if index < 0 or text[index] not in _MINUS:
        return False
    if index == 0:
        return True
    before = text[index - 1]
    if (before.isascii() and before.isalnum()) or before in _BEFORE_DASH:
        return False
    return not (before in _UNIT_CHARS and index > 1 and text[index - 2] in _DIGIT_OR_MASK)


def ends_with(pattern: re.Pattern[str], text: str, end: int) -> bool:
    """Whether ``pattern`` (anchored by ``\\Z``) matches ``text`` right before ``end``. Only the
    last ``_KEYWORD_WINDOW`` characters are searched; look-behinds still see past them."""
    return pattern.search(text, max(0, end - _KEYWORD_WINDOW), end) is not None


def mask(text: str, spans: Iterable[tuple[int, int]]) -> str:
    """``text`` with each span replaced by as many ``MASK`` characters."""
    chars = list(text)
    for start, end in spans:
        chars[start:end] = MASK * (end - start)
    return "".join(chars)


def first_wins(matches: Iterable[Match]) -> list[Match]:
    """The matches in text order, without those that overlap an earlier one (at the same start,
    the longer one wins)."""
    kept: list[Match] = []
    for match in sorted(matches, key=lambda match: (match.start, -match.end)):
        if not kept or match.start >= kept[-1].end:
            kept.append(match)
    return kept
