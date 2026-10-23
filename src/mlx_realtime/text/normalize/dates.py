"""Dates and times (ARCHITECTURE §7).

Rules:

- Dates: ``2024-10-23``, ``2024/10/23``, ``2024.10.23``, ``2024年10月23日`` (or 号), ``2024年10月``,
  ``10月23日``, ``10月`` and a year ``2024年`` (19xx or 20xx). The day must exist.
  zh: the year digit by digit, month and day as numbers: 二零二四年十月二十三日.
  en: ``October twenty-third, twenty twenty-four``. Years are read in pairs (nineteen ninety-nine,
  nineteen oh five, twenty twenty-four), except two thousand to two thousand nine.
- Times: ``14:30``, ``09:05``, ``14:30:15``, ``9:30 pm``, ``9am`` and a range of two times
  (``9:00-17:30``). Hours 0-23, or 1-12 with am/pm; minutes and seconds 0-59.
  zh: 十四点三十分, 九点零五分, 八点, 两点, 十四点三十分十五秒; am/pm become 上午/下午 in front.
  en: fourteen thirty, nine oh five, ten o'clock (fourteen hundred past noon, zero hundred at
  midnight), fourteen thirty and fifteen seconds; am/pm are spelled A M / P M.
- US numeric dates with a four-digit year: ``03/15/2024`` is month/day/year; when the month
  cannot be one (``15/03/2024``) it is read day/month/year.
- Ambiguous forms such as ``10/11/12`` are left as written.
- Numeric dates (``2024-10-23``, ``03/15/2024``) are marked ``Match.numeric_date``: English text
  reads them even with the category off (G5).
"""

from __future__ import annotations

import calendar
import re
from collections.abc import Iterator
from dataclasses import dataclass
from functools import partial

from mlx_realtime.text.normalize.common import FREE_LEFT, FREE_RIGHT, Lang, Match, first_wins
from mlx_realtime.text.normalize.readers import (
    en_cardinal,
    en_ordinal,
    zh_cardinal,
    zh_digits,
)

_MONTHS = (
    *("January", "February", "March", "April", "May", "June", "July", "August"),
    *("September", "October", "November", "December"),
)
_NUMERIC_DATE = re.compile(
    rf"{FREE_LEFT}(?P<year>[12][0-9]{{3}})(?P<sep>[-/.])(?P<month>[0-9]{{1,2}})(?P=sep)"
    rf"(?P<day>[0-9]{{1,2}}){FREE_RIGHT}"
)
_US_DATE = re.compile(
    rf"{FREE_LEFT}(?P<first>[0-9]{{1,2}})/(?P<second>[0-9]{{1,2}})/(?P<year>[12][0-9]{{3}})"
    rf"{FREE_RIGHT}"
)
_ZH_DATE = re.compile(
    rf"{FREE_LEFT}(?:(?P<year>[12][0-9]{{3}}) ?年 ?)?(?P<month>[0-9]{{1,2}}) ?月"
    r"(?: ?(?P<day>[0-9]{1,2}) ?(?P<day_word>[日号]))?"
)
_ZH_YEAR = re.compile(rf"{FREE_LEFT}(?P<year>(?:19|20)[0-9]{{2}}) ?年")
_AMPM = r"[AaPp](?:\.[Mm]\.|[Mm])(?![A-Za-z])"


def _clock_pattern(prefix: str) -> str:
    """A clock time with minutes, am/pm or both; its groups are named ``{prefix}hour`` etc."""
    return (
        f"(?P<{prefix}hour>[0-9]{{1,2}})(?=[:：][0-9]{{2}}| ?{_AMPM})"
        f"(?:[:：](?P<{prefix}minute>[0-9]{{2}})(?:[:：](?P<{prefix}second>[0-9]{{2}}))?)?"
        f"(?: ?(?P<{prefix}ampm>{_AMPM}))?"
    )


_TIME = re.compile(
    rf"{FREE_LEFT}{_clock_pattern('')}(?: ?[-~～–] ?{_clock_pattern('to_')})?"
    r"(?![A-Za-z0-9_])(?![.,:：][0-9])"
)


@dataclass(frozen=True, slots=True)
class _Date:
    year: int | None
    month: int | None
    day: int | None
    day_word: str  # 日 or 号


@dataclass(frozen=True, slots=True)
class _Clock:
    hour: int
    minute: int
    second: int | None
    ampm: str  # "", "a" or "p"


def find(text: str) -> list[Match]:
    """Every date and time in ``text``."""
    return first_wins([*_dates(text), *_us_dates(text), *_times(text)])


def _dates(text: str) -> Iterator[Match]:
    for pattern in (_NUMERIC_DATE, _ZH_DATE, _ZH_YEAR):
        for found in pattern.finditer(text):
            groups = found.groupdict()
            year, month, day = (
                int(groups[name]) if groups.get(name) else None for name in ("year", "month", "day")
            )
            if _exists(year, month, day):
                date = _Date(year, month, day, groups.get("day_word") or "日")
                numeric = pattern is _NUMERIC_DATE
                read = partial(_read_date, date)
                yield Match(found.start(), found.end(), read, numeric_date=numeric)


def _us_dates(text: str) -> Iterator[Match]:
    """``03/15/2024``: month/day/year, or day/month/year when the first cannot be a month."""
    for found in _US_DATE.finditer(text):
        year, first, second = int(found["year"]), int(found["first"]), int(found["second"])
        for month, day in ((first, second), (second, first)):
            if _exists(year, month, day):
                date = _Date(year, month, day, "日")
                read = partial(_read_date, date)
                yield Match(found.start(), found.end(), read, numeric_date=True)
                break


def _exists(year: int | None, month: int | None, day: int | None) -> bool:
    if month is None:
        return year is not None
    if not 1 <= month <= 12:
        return False
    leap_year = 2000  # without a year, 2月29日 exists
    return day is None or 1 <= day <= calendar.monthrange(year or leap_year, month)[1]


def _times(text: str) -> Iterator[Match]:
    for found in _TIME.finditer(text):
        clocks = [_clock(found, "")]
        if found["to_hour"] is not None:
            clocks.append(_clock(found, "to_"))
        valid = [clock for clock in clocks if clock is not None]
        if len(valid) == len(clocks):
            dotted = found.group().endswith(".")
            yield Match(found.start(), found.end(), partial(_read_times, valid, dotted=dotted))


def _clock(found: re.Match[str], prefix: str) -> _Clock | None:
    hour, minute, second = (found[prefix + name] for name in ("hour", "minute", "second"))
    ampm = (found[prefix + "ampm"] or "")[:1].lower()
    clock = _Clock(int(hour), int(minute or 0), None if second is None else int(second), ampm)
    if clock.minute > 59 or (clock.second or 0) > 59:
        return None
    if ampm:
        return clock if 1 <= clock.hour <= 12 else None
    return clock if clock.hour <= 23 else None


def _read_date(date: _Date, lang: Lang) -> str:
    if lang == "zh":
        spoken = "" if date.year is None else zh_digits(str(date.year)) + "年"
        if date.month is not None:
            spoken += zh_cardinal(date.month) + "月"
        if date.day is not None:
            spoken += zh_cardinal(date.day) + date.day_word
        return spoken
    words: list[str] = []
    if date.month is not None:
        words.append(_MONTHS[date.month - 1])
    if date.day is not None:
        words.append(en_ordinal(date.day) + ("," if date.year is not None else ""))
    if date.year is not None:
        words.append(_en_year(date.year))
    return " ".join(words)


def _en_year(year: int) -> str:
    high, low = divmod(year, 100)
    if year % 1000 < 10 and high % 10 == 0:
        return en_cardinal(year)
    if low == 0:
        return f"{en_cardinal(high)} hundred"
    return f"{en_cardinal(high)} {'oh ' if low < 10 else ''}{en_cardinal(low)}"


def _read_times(clocks: list[_Clock], lang: Lang, *, dotted: bool) -> str:
    if lang == "zh":
        return "到".join(_zh_clock(clock) for clock in clocks)
    return " to ".join(_en_clock(clock) for clock in clocks) + ("." if dotted else "")


def _zh_clock(clock: _Clock) -> str:
    spoken = {"": "", "a": "上午", "p": "下午"}[clock.ampm]
    spoken += ("两" if clock.hour == 2 else zh_cardinal(clock.hour)) + "点"
    if clock.minute or clock.second:
        spoken += _zh_sixty(clock.minute) + "分"
    if clock.second:
        spoken += _zh_sixty(clock.second) + "秒"
    return spoken


def _zh_sixty(value: int) -> str:
    """Minutes and seconds: 5 → 零五, 30 → 三十."""
    return ("零" if 0 < value < 10 else "") + zh_cardinal(value)


def _en_clock(clock: _Clock) -> str:
    hour = en_cardinal(clock.hour)
    if clock.minute:
        minute = en_cardinal(clock.minute)
        spoken = f"{hour} {'oh ' if clock.minute < 10 else ''}{minute}"
    elif clock.ampm:
        spoken = hour
    else:
        spoken = f"{hour} o'clock" if 1 <= clock.hour <= 12 else f"{hour} hundred"
    if clock.second:
        spoken += f" and {en_cardinal(clock.second)} second{'' if clock.second == 1 else 's'}"
    if clock.ampm:
        spoken += " A M" if clock.ampm == "a" else " P M"
    return spoken
