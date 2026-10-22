"""Phone numbers, read digit by digit in groups (ARCHITECTURE §7).

Rules:

- Mobile: 11 digits ``1[3-9]…``, plain or split 3-4-4 by spaces or hyphens, optionally after
  ``+86``, ``0086`` or ``86`` (read 加八六 / plus eight six). Groups 3-4-4.
- Landline: an area code (``010``, ``02x`` or ``0[3-9]xx``, in parentheses or not), then 7 or 8
  digits, plain or split 3-4 / 4-4. Groups: the area code, then the local number 3-4 or 4-4.
- Service: ``400`` or ``800`` and 7 more digits, plain or split 3-3-4, 3-4-3 or 4-3-3.
  Groups 3-3-4.
- Hotline: five digits ``95xxx`` or ``96xxx``, only right after 电话, 热线, 拨打, 致电, call, dial,
  tel, telephone, phone or hotline (spaces, colons, 是, 为 or ``is`` may come between). One group.
- zh: 零一二三四五六七八九, 1 → 幺 when ``one_as_yao``; groups joined by "，".
  en: zero … nine; groups joined by ", ".
- A phone is never part of a longer digit run, a decimal, a time, an amount or a word.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from functools import partial

from mlx_realtime.text.normalize.common import Lang, Match, ends_with, first_wins
from mlx_realtime.text.normalize.readers import en_digits, zh_digits

_LEFT = r"(?<![A-Za-z0-9_$¥￥€£])(?<![0-9][.,:])"
_RIGHT = r"(?![A-Za-z0-9_])(?![.,:：][0-9])(?! ?(?:[%％元块]|美元|美金|欧元|英镑|日元|港元|港币))"
_MOBILE = (
    r"(?:(?P<prefix>(?:\+|00)?86)[ -]?)?"
    r"(?P<mobile>1[3-9][0-9](?:[0-9]{8}|[ -][0-9]{4}[ -][0-9]{4}))"
)
_SERVICE = (
    r"(?P<service>[48]00(?:[0-9]{7}|[ -][0-9]{3}[ -][0-9]{4}|[ -][0-9]{4}[ -][0-9]{3}"
    r"|[0-9][ -][0-9]{3}[ -][0-9]{3}))"
)
_AREA = r"010|02[0-9]|0[3-9][0-9]{2}"
_LANDLINE = (
    rf"(?:\((?P<parea>{_AREA})\) ?|(?P<area>{_AREA})[ -]?)"
    r"(?P<local>[0-9]{7,8}|[0-9]{3,4}[ -][0-9]{4})"
)
_PHONE = re.compile(f"{_LEFT}(?:{_MOBILE}|{_SERVICE}|{_LANDLINE}){_RIGHT}")
_HOTLINE = re.compile(f"{_LEFT}9[56][0-9]{{3}}{_RIGHT}")
_HOTLINE_KEYWORD = re.compile(
    r"(?:电话|热线|拨打|致电|(?<![A-Za-z])(?:call|dial|tel|telephone|phone|hotline))"
    r"(?:[ :：]|是|为|(?<![A-Za-z])is(?![A-Za-z]))*\Z",
    re.IGNORECASE,
)
_SEPARATORS = re.compile(r"[ -]")


def find(text: str, *, one_as_yao: bool) -> list[Match]:
    """Every phone number in ``text``."""
    return first_wins([*_numbers(text, one_as_yao), *_hotlines(text, one_as_yao)])


def _numbers(text: str, one_as_yao: bool) -> Iterator[Match]:
    for found in _PHONE.finditer(text):
        groups = _groups(found)
        read = partial(_read, groups, prefixed=bool(found["prefix"]), one_as_yao=one_as_yao)
        yield Match(found.start(), found.end(), read)


def _hotlines(text: str, one_as_yao: bool) -> Iterator[Match]:
    for found in _HOTLINE.finditer(text):
        if ends_with(_HOTLINE_KEYWORD, text, found.start()):
            read = partial(_read, [found.group()], prefixed=False, one_as_yao=one_as_yao)
            yield Match(found.start(), found.end(), read)


def _groups(found: re.Match[str]) -> list[str]:
    if found["mobile"]:
        digits = _SEPARATORS.sub("", found["mobile"])
        return [digits[:3], digits[3:7], digits[7:]]
    if found["service"]:
        digits = _SEPARATORS.sub("", found["service"])
        return [digits[:3], digits[3:6], digits[6:]]
    local = _SEPARATORS.sub("", found["local"])
    split = 4 if len(local) == 8 else 3
    return [found["area"] or found["parea"], local[:split], local[split:]]


def _read(groups: list[str], lang: Lang, *, prefixed: bool, one_as_yao: bool) -> str:
    if lang == "zh":
        spoken = [zh_digits(group, yao=one_as_yao) for group in groups]
        return "，".join(["加八六", *spoken] if prefixed else spoken)
    spoken = [en_digits(group) for group in groups]
    return ", ".join(["plus eight six", *spoken] if prefixed else spoken)
