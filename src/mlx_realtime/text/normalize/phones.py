"""Phone numbers, read digit by digit in groups (ARCHITECTURE §7, G5).

Rules:

- Mobile: 11 digits ``1[3-9]…``, plain or split 3-4-4 by spaces or hyphens, optionally after
  ``+86``, ``0086`` or ``86`` (read 加八六 / plus eight six). Groups 3-4-4.
- Landline: an area code (``010``, ``02x`` or ``0[3-9]xx``, in parentheses or not), then 7 or 8
  digits, plain or split 3-4 / 4-4. Groups: the area code, then the local number 3-4 or 4-4.
- Service: ``400`` or ``800`` and 7 more digits, plain or split 3-3-4, 3-4-3 or 4-3-3.
  Groups 3-3-4.
- North American: ``800-555-0199``, ``(800) 555-0199``, ``800.555.0199``, optionally after ``1``
  or ``+1`` (read 幺 / one, 加幺 / plus one); the area code and exchange start with 2-9, and the
  groups must be separated. Groups 3-3-4.
- Short numbers: 3 to 6 digits (``110``, ``95588``, ``10086``) right after 电话, 热线, 拨打,
  致电, call, dial, tel, telephone, phone or hotline (spaces, colons, 是, 为 or ``is`` may come
  between), and the well-known service numbers 10086, 12306, 12315 and 12366 anywhere (not
  10000, 10010 or 12345, which are as often quantities). One group.
- Last four digits (尾号): four digits right after 尾号, 末四位, 后四位, ending in, ends with or
  last four (digits) (是, 为, is, are, spaces or colons between). One group; the keyword stays.
- Extensions: one to six digits after 转, 分机, 分机号, ext, ext., extension or x right after a
  phone number (one space allowed), or after 分机, 分机号, ext, ext. or extension anywhere. zh:
  a Chinese keyword is kept, a Latin one becomes 分机; en: ``extension`` and the digits.
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
_NORTH_AMERICA = (
    r"(?:(?P<us_prefix>\+?1)[ .-]?)?"
    r"(?:\((?P<us_parea>[2-9][0-9]{2})\) ?|(?P<us_area>[2-9][0-9]{2})[ .-])"
    r"(?P<us_exchange>[2-9][0-9]{2})[ .-](?P<us_line>[0-9]{4})"
)
_PHONE = re.compile(f"{_LEFT}(?:{_MOBILE}|{_SERVICE}|{_LANDLINE}|{_NORTH_AMERICA}){_RIGHT}")
_HOTLINE = re.compile(f"{_LEFT}[0-9]{{3,6}}{_RIGHT}")
_WELL_KNOWN = re.compile(f"{_LEFT}(?:10086|12306|12315|12366){_RIGHT}")
_HOTLINE_KEYWORD = re.compile(
    r"(?:电话|热线|拨打|致电|(?<![A-Za-z])(?:call|dial|tel|telephone|phone|hotline))"
    r"(?:[ :：]|是|为|(?<![A-Za-z])is(?![A-Za-z]))*\Z",
    re.IGNORECASE,
)
_TAIL = re.compile(f"{_LEFT}[0-9]{{4}}{_RIGHT}")
_TAIL_KEYWORD = re.compile(
    r"(?:尾号|末四位|后四位"
    r"|(?<![A-Za-z])(?:ending (?:in|with)|ends (?:in|with)|last (?:four|4)(?: digits)?))"
    r"(?:[ :：]|是|为|(?<![A-Za-z])(?:is|are)(?![A-Za-z]))*\Z",
    re.IGNORECASE,
)
_DIGITS_AFTER = rf"(?P<digits>[0-9]{{1,6}}){_RIGHT}"
_EXTENSION_AFTER_PHONE = re.compile(
    rf" ?(?P<keyword>转|分机号?|(?i:extension|ext\.?|x)) ?{_DIGITS_AFTER}"
)
_EXTENSION = re.compile(
    r"(?P<keyword>分机号?|(?<![A-Za-z])(?i:extension|ext\.?)(?![A-Za-z]))"
    rf"(?P<gap>(?:[ :：#]|是|为)*){_DIGITS_AFTER}"
)
_SEPARATORS = re.compile(r"[ .-]")


_CHINA = ("加八六", "plus eight six")


def find(text: str, *, one_as_yao: bool) -> list[Match]:
    """Every phone number in ``text``, with its extension, and every last-four and extension."""
    return first_wins(
        [
            *_numbers(text, one_as_yao),
            *_hotlines(text, one_as_yao),
            *_tails(text, one_as_yao),
            *_extensions(text, one_as_yao),
        ]
    )


def _numbers(text: str, one_as_yao: bool) -> Iterator[Match]:
    for found in _PHONE.finditer(text):
        groups, prefix = _groups(found)
        read = partial(_read, groups, prefix=prefix, one_as_yao=one_as_yao)
        yield Match(found.start(), found.end(), read)
        if extension := _EXTENSION_AFTER_PHONE.match(text, found.end()):
            yield _extension(extension, "", one_as_yao)


def _hotlines(text: str, one_as_yao: bool) -> Iterator[Match]:
    keyed = (
        found
        for found in _HOTLINE.finditer(text)
        if ends_with(_HOTLINE_KEYWORD, text, found.start())
    )
    for found in (*keyed, *_WELL_KNOWN.finditer(text)):
        read = partial(_read, [found.group()], prefix=None, one_as_yao=one_as_yao)
        yield Match(found.start(), found.end(), read)


def _tails(text: str, one_as_yao: bool) -> Iterator[Match]:
    for found in _TAIL.finditer(text):
        if ends_with(_TAIL_KEYWORD, text, found.start()):
            read = partial(_read, [found.group()], prefix=None, one_as_yao=one_as_yao)
            yield Match(found.start(), found.end(), read)


def _extensions(text: str, one_as_yao: bool) -> Iterator[Match]:
    for found in _EXTENSION.finditer(text):
        yield _extension(found, found["gap"], one_as_yao)


def _extension(found: re.Match[str], gap: str, one_as_yao: bool) -> Match:
    """The keyword through the digits: ``转8001``, ``分机号是 23``, ``ext. 123``."""
    keyword, digits = found["keyword"], found["digits"]
    read = partial(_read_extension, keyword, gap, digits, one_as_yao=one_as_yao)
    return Match(found.start("keyword"), found.end(), read)


def _groups(found: re.Match[str]) -> tuple[list[str], tuple[str, str] | None]:
    """The digit groups, and how the country prefix is read (zh, en) if there is one."""
    if found["mobile"]:
        digits = _SEPARATORS.sub("", found["mobile"])
        return [digits[:3], digits[3:7], digits[7:]], _CHINA if found["prefix"] else None
    if found["service"]:
        digits = _SEPARATORS.sub("", found["service"])
        return [digits[:3], digits[3:6], digits[6:]], None
    if found["us_exchange"]:
        area = found["us_area"] or found["us_parea"]
        groups = [area, found["us_exchange"], found["us_line"]]
        prefix = found["us_prefix"]
        if prefix is None:
            return groups, None
        return groups, ("加幺", "plus one") if prefix.startswith("+") else ("幺", "one")
    local = _SEPARATORS.sub("", found["local"])
    split = 4 if len(local) == 8 else 3
    return [found["area"] or found["parea"], local[:split], local[split:]], None


def _read(
    groups: list[str], lang: Lang, *, prefix: tuple[str, str] | None, one_as_yao: bool
) -> str:
    if lang == "zh":
        spoken = [zh_digits(group, yao=one_as_yao) for group in groups]
        if prefix is not None:
            spoken.insert(0, prefix[0] if one_as_yao else prefix[0].replace("幺", "一"))
        return "，".join(spoken)
    spoken = [en_digits(group) for group in groups]
    return ", ".join([prefix[1], *spoken] if prefix is not None else spoken)


def _read_extension(keyword: str, gap: str, digits: str, lang: Lang, *, one_as_yao: bool) -> str:
    if lang == "zh":
        chinese = not keyword.isascii()
        head = keyword + gap.strip() if chinese else "分机"
        return head + zh_digits(digits, yao=one_as_yao)
    return "extension " + en_digits(digits)
