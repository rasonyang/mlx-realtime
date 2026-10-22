"""Free-standing numbers: cardinals, decimals, percents, ranges and ordinals (ARCHITECTURE §7).

Rules:

- A number is a whole token (see ``common``): ``Qwen3``, ``COVID-19``, ``5G`` and ``1.2.3`` hold
  none. A token that is neither one number nor a range (``10/11/12``, ``1,23``) is left as written,
  and so is an integer longer than 16 digits.
- zh: 1,234,567 → 一百二十三万四千五百六十七; 3.14 → 三点一四; 50% → 百分之五十; -5 → 负五;
  3-5 and 3~5 → 三到五 (with ``-`` only when the first is smaller); 21st → 第二十一.
  2 right before a measure word (个, 位, 件, 次, 天, 小时, 分钟, 张, 本, 台, 万 …) is 两, but never
  after 第.
- en: 1234 → one thousand two hundred thirty-four; 3.14 → three point one four; 50% → fifty
  percent; -5 → minus five; 3-5 → three to five; 21st → twenty-first (the suffix must fit).
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from functools import partial

from mlx_realtime.text.normalize.common import FREE_LEFT, FREE_RIGHT, Lang, Match, is_minus
from mlx_realtime.text.normalize.readers import (
    MAX_DIGITS,
    ascending,
    en_number,
    en_ordinal,
    readable,
    zh_cardinal,
    zh_number,
)

_FREE = re.compile(
    rf"{FREE_LEFT}(?P<body>(?>[0-9]+(?:[-~～.,:/][0-9]+)*))"
    rf"(?P<suffix>(?: ?[%％]|st|nd|rd|th)?+){FREE_RIGHT}"
)
_SINGLE = r"(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?"
_ONE = re.compile(_SINGLE)
_RANGE = re.compile(rf"(?P<first>{_SINGLE})(?P<dash>[-~～])(?P<second>{_SINGLE})")
_MEASURE = re.compile(
    "个|位|件|次|天|张|本|台|岁|周|只|条|份|杯|瓶|箱|包|种|层|间|套|辆|家|页|节|场|门|双|对|部|首"
    "|颗|根|支|把|片|盒|袋|斤|人|名|遍|趟|倍|成|点|秒|米|克|升|千|万|亿|小时|星期|礼拜|公里|公斤"
    "|分(?!之)|年(?!级)"
)
_ORDINAL_SUFFIXES = ("st", "nd", "rd", "th")


@dataclass(frozen=True, slots=True)
class _Number:
    numbers: tuple[str, ...]  # one, or two for a range
    negative: bool
    percent: bool
    liang: bool  # a 2 reads 两


def find(text: str) -> Iterator[Match]:
    """Every free-standing number in ``text``."""
    for found in _FREE.finditer(text):
        ordinal = found["suffix"] in _ORDINAL_SUFFIXES
        match = _ordinal(text, found) if ordinal else _number(text, found)
        if match is not None:
            yield match


def _number(text: str, found: re.Match[str]) -> Match | None:
    body, start, end = found["body"], found.start(), found.end()
    if _ONE.fullmatch(body):
        numbers: tuple[str, ...] = (body,)
    elif (pair := _RANGE.fullmatch(body)) and (
        pair["dash"] != "-" or ascending(pair["first"], pair["second"])
    ):
        numbers = (pair["first"], pair["second"])
    else:
        return None
    if not all(readable(number) for number in numbers):
        return None
    percent = bool(found["suffix"])
    negative = len(numbers) == 1 and is_minus(text, start - 1)
    liang = not (negative or percent or _after_di(text, start)) and bool(_MEASURE.match(text, end))
    number = _Number(numbers, negative, percent, liang)
    return Match(start - 1 if negative else start, end, partial(_read, number))


def _after_di(text: str, start: int) -> bool:
    return start > 0 and text[start - 1] == "第"


def _ordinal(text: str, found: re.Match[str]) -> Match | None:
    body = found["body"]
    if not body.isdigit() or len(body) > MAX_DIGITS or _suffix(int(body)) != found["suffix"]:
        return None
    read = partial(_read_ordinal, int(body), di=not _after_di(text, found.start()))
    return Match(found.start(), found.end(), read)


def _suffix(value: int) -> str:
    if value % 100 in (11, 12, 13):
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")


def _read(number: _Number, lang: Lang) -> str:
    if lang == "zh":
        percent = "百分之" if number.percent else ""
        words = [percent + (zh_number(n, liang=number.liang) or n) for n in number.numbers]
        return ("负" if number.negative else "") + "到".join(words)
    words = " to ".join(en_number(n) or n for n in number.numbers)
    return ("minus " if number.negative else "") + words + (" percent" if number.percent else "")


def _read_ordinal(value: int, lang: Lang, *, di: bool) -> str:
    """zh: 第 and the number (no second 第 after one); en: the ordinal word."""
    if lang == "zh":
        return ("第" if di else "") + zh_cardinal(value)
    return en_ordinal(value)
