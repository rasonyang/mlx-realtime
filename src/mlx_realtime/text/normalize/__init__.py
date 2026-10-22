"""Text normalization for TTS input; transcripts keep the original text (ARCHITECTURE §7).

Rules:

- Spans are recognized in priority order: URLs and e-mail addresses (always left as written),
  phones, IDs, money, dates and times, units (a number with a unit), then free-standing numbers
  (percent, decimal, cardinal, range, ordinal). Each category has its own module and rules.
- A span recognized by one category is never re-read by a later one, even when its own category
  is off: then it stays exactly as written. ``numbers`` covers only free-standing numbers; the
  other categories read their digits themselves.
- Each span is read in Chinese or English by its neighbourhood (``context``); ``lang``, the
  segment's tag, decides only when the neighbourhood has no letter.
- zh/en spacing runs last, on the normalized text.
- The output is plain text for the TTS, never SSML. Text with nothing to normalize comes back
  unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import partial

from mlx_realtime.text.lang import detect_lang
from mlx_realtime.text.normalize import dates, ids, money, numbers, phones, protected, units
from mlx_realtime.text.normalize.common import Lang, Match, mask
from mlx_realtime.text.normalize.context import Context
from mlx_realtime.text.normalize.spacing import space

__all__ = ["NormalizerOptions", "normalize"]

_Finder = Callable[[str], Iterable[Match]]


@dataclass(frozen=True, slots=True)
class NormalizerOptions:
    """Which categories are read; a category that is off leaves its spans as written."""

    enabled: bool = True  # master switch: False returns the text unchanged
    numbers: bool = False  # free-standing cardinals, decimals, percents, ranges, ordinals
    money: bool = False
    dates: bool = False  # dates and times
    phones: bool = True
    ids: bool = True
    units: bool = False  # a unit after a number
    spacing: bool = False  # zh/en spacing
    phone_one_as_yao: bool = True  # read 1 as 幺 in Chinese phone numbers


_DEFAULTS = NormalizerOptions()


def normalize(
    text: str, options: NormalizerOptions = _DEFAULTS, *, lang: Lang | None = None
) -> str:
    """The text to synthesize for ``text``. ``lang`` is the segment's tag (``detect_lang`` when
    ``None``), used only where a span's neighbourhood holds no letter."""
    if not options.enabled:
        return text
    context = Context.of(text, lang or detect_lang(text))
    masked = mask(text, protected.find(text))
    readings: list[tuple[int, int, str]] = []
    for enabled, find in _categories(options):
        matches = list(find(masked))
        if enabled:
            readings += [(match.start, match.end, _read(match, context)) for match in matches]
        masked = mask(masked, ((match.start, match.end) for match in matches))
    spoken = _splice(text, readings)
    return space(spoken) if options.spacing else spoken


def _read(match: Match, context: Context) -> str:
    return match.read(context.lang(match.start, match.end))


def _categories(options: NormalizerOptions) -> list[tuple[bool, _Finder]]:
    """Each category's switch and finder, in priority order."""
    return [
        (options.phones, partial(phones.find, one_as_yao=options.phone_one_as_yao)),
        (options.ids, ids.find),
        (options.money, money.find),
        (options.dates, dates.find),
        (options.units, partial(units.find, read_numbers=options.numbers)),
        (options.numbers, numbers.find),
    ]


def _splice(text: str, readings: list[tuple[int, int, str]]) -> str:
    """``text`` with each ``(start, end)`` span replaced by its reading."""
    pieces: list[str] = []
    position = 0
    for start, end, reading in sorted(readings):
        pieces += [text[position:start], reading]
        position = end
    pieces.append(text[position:])
    return "".join(pieces)
