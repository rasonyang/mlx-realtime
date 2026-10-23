"""Text normalization for TTS input; transcripts keep the original text (ARCHITECTURE §7).

Scope (G5, M0 S9): only what Qwen3-TTS reads wrongly by itself is normalized by default: phone
numbers in every form (with 尾号 and extensions), alphanumeric and dashed IDs, e-mail addresses,
cardinals of ``min_cardinal_digits`` (5) or more digits, and in English text numeric dates and
amounts of money of ``en_amount_min`` (1000) or more. Money, decimals, percentages, ordinals,
dates and times, units, URLs and brand names stay as written; each category can still be
switched on.

Rules:

- Spans are recognized in priority order: URLs and e-mail addresses (e-mail addresses are read
  by ``emails``, URLs always left as written), phones, IDs, money, dates and times, units (a
  number with a unit), then free-standing numbers (percent, decimal, cardinal, range, ordinal).
  Each category has its own module and rules.
- A span recognized by one category is never re-read by a later one, even when its own category
  is off: then it stays exactly as written, unless a narrower rule reads it (English amounts and
  numeric dates, long cardinals). ``numbers`` covers only free-standing numbers; the other
  categories read their digits themselves.
- Each span is read in Chinese or English by its neighbourhood (``context``); ``lang``, the
  segment's tag, decides only when the neighbourhood has no letter. "English text" in the scope
  means a span read in English.
- zh/en spacing runs last, on the normalized text.
- The output is plain text for the TTS, never SSML. Text with nothing to normalize comes back
  unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import partial

from mlx_realtime.text.lang import detect_lang
from mlx_realtime.text.normalize import (
    dates,
    emails,
    ids,
    money,
    numbers,
    phones,
    protected,
    units,
)
from mlx_realtime.text.normalize.common import Lang, Match, mask
from mlx_realtime.text.normalize.context import Context
from mlx_realtime.text.normalize.spacing import space_with_map

__all__ = ["NormalizerOptions", "normalize", "normalize_with_map"]

_Finder = Callable[[str], Iterable[Match]]
_ReadIf = Callable[[Match, Lang], bool]


@dataclass(frozen=True, slots=True)
class NormalizerOptions:
    """Which categories are read; a category that is off leaves its spans as written. The
    field names are the keys of ``text.normalizer`` in the configuration."""

    enabled: bool = True  # master switch: False returns the text unchanged
    numbers: bool = False  # every free-standing number: cardinals, decimals, %, ranges, ordinals
    cardinals: bool = True  # free-standing plain cardinals of min_cardinal_digits or more
    min_cardinal_digits: int = 5
    money: bool = False
    dates: bool = False  # dates and times
    phones: bool = True  # with 尾号 and extensions
    ids: bool = True
    emails: bool = True  # "@" and "." spoken
    units: bool = False  # a unit after a number
    spacing: bool = False  # zh/en spacing
    en_dates: bool = True  # numeric dates read in English, even with dates off
    en_amounts: bool = True  # money read in English from en_amount_min, even with money off
    en_amount_min: int = 1000
    phone_one_as_yao: bool = True  # read 1 as 幺 in Chinese phone numbers


_DEFAULTS = NormalizerOptions()


def normalize(
    text: str, options: NormalizerOptions = _DEFAULTS, *, lang: Lang | None = None
) -> str:
    """The text to synthesize for ``text``. ``lang`` is the segment's tag (``detect_lang`` when
    ``None``), used only where a span's neighbourhood holds no letter."""
    return normalize_with_map(text, options, lang=lang)[0]


def normalize_with_map(
    text: str, options: NormalizerOptions = _DEFAULTS, *, lang: Lang | None = None
) -> tuple[str, tuple[int, ...]]:
    """``normalize(text)``, and where each spoken character comes from in ``text``: entry ``i``
    is the index in ``text`` of spoken character ``i``, and the last entry is ``len(text)``.

    Unchanged text maps one to one; the reading of a recognized span spreads evenly over the
    span it replaces (truncation measures what was heard in spoken characters, ARCHITECTURE
    §6.6). A space the spacing rules insert maps to the character after it.
    """
    if not options.enabled:
        return text, tuple(range(len(text) + 1))
    context = Context.of(text, lang or detect_lang(text))
    readings: list[tuple[int, int, str]] = []
    if options.emails:
        readings += _readings(emails.find(text), context, lambda _match, _lang: True)
    masked = mask(text, protected.find(text))
    for read_if, find in _categories(options):
        matches = list(find(masked))
        readings += _readings(matches, context, read_if)
        masked = mask(masked, ((match.start, match.end) for match in matches))
    spoken, source = _splice(text, readings)
    if options.spacing:
        spoken, spaced = space_with_map(spoken)
        source = [source[index] for index in spaced]
    return spoken, (*source, len(text))


def _readings(
    matches: Iterable[Match], context: Context, read_if: _ReadIf
) -> list[tuple[int, int, str]]:
    """The reading of each match that ``read_if`` admits, in the language of its context."""
    readings: list[tuple[int, int, str]] = []
    for match in matches:
        lang = context.lang(match.start, match.end)
        if read_if(match, lang):
            readings.append((match.start, match.end, match.read(lang)))
    return readings


def _categories(options: NormalizerOptions) -> list[tuple[_ReadIf, _Finder]]:
    """Each category's rule for reading a match, and its finder, in priority order."""

    def always(enabled: bool) -> _ReadIf:
        return lambda _match, _lang: enabled

    def money_read_if(match: Match, lang: Lang) -> bool:
        if options.money:
            return True
        large = match.amount is not None and match.amount >= options.en_amount_min
        return options.en_amounts and lang == "en" and large

    def dates_read_if(match: Match, lang: Lang) -> bool:
        return options.dates or (options.en_dates and lang == "en" and match.numeric_date)

    def numbers_read_if(match: Match, _lang: Lang) -> bool:
        long_cardinal = match.digits >= max(options.min_cardinal_digits, 1)
        return options.numbers or (options.cardinals and long_cardinal)

    return [
        (always(options.phones), partial(phones.find, one_as_yao=options.phone_one_as_yao)),
        (always(options.ids), ids.find),
        (money_read_if, money.find),
        (dates_read_if, dates.find),
        (always(options.units), partial(units.find, read_numbers=options.numbers)),
        (numbers_read_if, numbers.find),
    ]


def _splice(text: str, readings: list[tuple[int, int, str]]) -> tuple[str, list[int]]:
    """``text`` with each ``(start, end)`` span replaced by its reading, and for each output
    character the index in ``text`` it comes from."""
    pieces: list[str] = []
    source: list[int] = []
    position = 0
    for start, end, reading in sorted(readings):
        pieces += [text[position:start], reading]
        source += range(position, start)
        source += (start + offset * (end - start) // len(reading) for offset in range(len(reading)))
        position = end
    pieces.append(text[position:])
    source += range(position, len(text))
    return "".join(pieces), source
