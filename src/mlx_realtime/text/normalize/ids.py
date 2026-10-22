"""Order, tracking and reference IDs, read character by character (ARCHITECTURE §7).

Rules:

- An alphanumeric token (letters and digits joined by single ``-_/.``) is an ID when it holds at
  least 6 letters and digits, at least 4 digits and at least 1 letter, and no run of more than 5
  letters (a longer run is a word: ``report2024``). A quantity is not an ID: a number with a unit,
  an ordinal suffix, a decade ``s``, am/pm or a currency code (``1000km``, ``1990s``,
  ``10am-12pm``, ``USD2000``), or a size (``1920x1080``); nor is a name with a version number
  (``Qwen2.5-72B``, ``Llama-3.1-70B``). A leading ``No.`` is not part of the ID.
- A run of 6 or more digits (joined by single ``-_/.``, or by spaces before groups of 3 or more)
  is an ID only right after an ID keyword: 订单号, 订单编号, 单号, 运单号, 快递单号, 物流单号,
  工单号, 编号, 流水号, 账号, 卡号, 会员号, 验证码, order number, order no., order #,
  tracking number, reference number, ref, account number, confirmation code, case number, ID.
  Spaces, colons, ``#``, ``.``, ``No.``, 是, 为 and ``is`` may come between.
- Reading: letters as capitals separated by spaces; each digit on its own (zh 零一二…九, 1 is 一,
  never 幺; en zero … nine); a letter run and a digit run are separated by a space. A digit run
  longer than 4 is split in fours from the left. Each separator in the ID and each split is a
  pause: "，" in zh, ", " in en.
  ``ORD-2024-001`` → zh ``O R D，二零二四，零零一``, en ``O R D, two zero two four, zero zero one``;
  ``SF1234567890`` → zh ``S F 一二三四，五六七八，九零``.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from functools import partial

from mlx_realtime.text.normalize.common import Lang, Match, ends_with, first_wins
from mlx_realtime.text.normalize.money import CODES
from mlx_realtime.text.normalize.readers import en_digits, zh_digits
from mlx_realtime.text.normalize.units import UNIT_PATTERN

_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_])(?<![A-Za-z0-9][-_/.])"
    r"(?>[A-Za-z0-9]+(?:[-_/.][A-Za-z0-9]+)*)"
    r"(?![-_/.:]?[A-Za-z0-9])"
)
_DIGIT_RUN = re.compile(
    r"(?<![A-Za-z0-9_])(?<![0-9][-_/.,:])"
    r"(?>[0-9]+(?:[-_/.][0-9]+| [0-9]{3,})*)"
    r"(?![A-Za-z0-9_])(?![-_/.,:][A-Za-z0-9])"
)
_KEYWORD = re.compile(
    r"(?:订单编号|订单号|快递单号|物流单号|运单号|工单号|单号|编号|流水号|账号|卡号|会员号|验证码"
    r"|(?<![A-Za-z])(?:order (?:number|no\.|#)|tracking number|reference number|ref"
    r"|account number|confirmation code|case number|ID))"
    r"(?:[ :：#.]|是|为|No\.|(?<![A-Za-z])is(?![A-Za-z]))*\Z",
    re.IGNORECASE,
)
_QUANTITY = re.compile(
    rf"[0-9][0-9.,]*(?:{UNIT_PATTERN}|st|nd|rd|th|s|am|pm|AM|PM|{CODES})?"
    rf"|(?:{CODES})[0-9][0-9.,]*|[0-9]+[xX][0-9]+"
)
_QUANTITY_PARTS = re.compile(r"[-/]")
_NUMERO = re.compile(r"no\.", re.IGNORECASE)
_WORD = re.compile(r"[A-Za-z]{6,}")
_VERSION = re.compile(r"(?<![0-9])[0-9]{1,2}\.[0-9]{1,2}(?![0-9])")
_SEPARATOR = re.compile(r"[-_/. ]")
_RUN = re.compile(r"[A-Za-z]+|[0-9]+")


def find(text: str) -> list[Match]:
    """Every ID in ``text``."""
    return first_wins([*_alphanumeric(text), *_keyed_digits(text)])


def _alphanumeric(text: str) -> Iterator[Match]:
    for found in _TOKEN.finditer(text):
        start, token = found.start(), found.group()
        if numero := _NUMERO.match(token):  # No.123456: the ID is what follows No.
            start, token = start + numero.end(), token[numero.end() :]
        if _is_id(token):
            yield Match(start, found.end(), partial(_read, token))


def _keyed_digits(text: str) -> Iterator[Match]:
    for found in _DIGIT_RUN.finditer(text):
        token = found.group()
        digits = sum(char.isdigit() for char in token)
        if digits >= 6 and ends_with(_KEYWORD, text, found.start()):
            yield Match(found.start(), found.end(), partial(_read, token))


def _is_id(token: str) -> bool:
    digits = sum(char.isdigit() for char in token)
    letters = sum(char.isalpha() for char in token)
    return (
        digits >= 4
        and letters >= 1
        and digits + letters >= 6
        and not _WORD.search(token)
        and not _VERSION.search(token)
        and not _is_quantity(token)
    )


def _is_quantity(token: str) -> bool:
    if _QUANTITY.fullmatch(token):
        return True
    return all(_QUANTITY.fullmatch(part) for part in _QUANTITY_PARTS.split(token))


def _read(token: str, lang: Lang) -> str:
    pause = "，" if lang == "zh" else ", "
    return pause.join(_read_part(part, lang, pause) for part in _SEPARATOR.split(token))


def _read_part(part: str, lang: Lang, pause: str) -> str:
    runs: list[str] = []
    for run in _RUN.findall(part):
        if run.isalpha():
            runs.append(" ".join(run.upper()))
        else:
            chunks = [run[start : start + 4] for start in range(0, len(run), 4)]
            spell = zh_digits if lang == "zh" else en_digits
            runs.append(pause.join(spell(chunk) for chunk in chunks))
    return " ".join(runs)
