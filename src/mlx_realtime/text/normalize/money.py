"""Amounts of money (ARCHITECTURE §7).

Rules:

- Forms: a symbol or code before the amount (``¥1,234.56``, ``￥88``, ``$12.99``, ``US$5``,
  ``HK$10``, ``€5``, ``£3``, ``USD 20``), or a word or code after it (``12.5元``, ``5块``,
  ``12.99美元``, ``20 USD``). ¥ and ￥ are yuan. A range shares one currency (``¥100-200``,
  ``10-20美元``). A scale may follow the amount: 万, 亿 or 万亿, or thousand, million, billion or
  trillion after a symbol or code.
- zh: yuan in 元角分 (块毛分 after 块): ``¥1,234.56`` → 一千二百三十四元五角六分, ``12.5元`` →
  十二元五角, ``￥88`` → 八十八元. Other currencies read the number, then the name: ``$12.99`` →
  十二点九九美元, ``USD 20`` → 二十美元. An amount of exactly 2 is 两 (两元, 两美元).
- en: dollars, euros and pounds read whole units and cents (pence): ``$12.99`` → twelve dollars
  and ninety-nine cents, ``$1`` → one dollar. Yuan and yen read the number: ``¥100`` → one
  hundred yuan.
- A scale from the other language is multiplied out: ``$1.5 million`` in zh → 一百五十万美元,
  ``5万元`` in en → fifty thousand yuan.
- ``3块5`` (块 followed by a digit) is not an amount. An amount glued to a letter (``$5k``) is
  left as written.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from decimal import Decimal
from functools import partial

from mlx_realtime.text.normalize.common import FREE_LEFT, NUMBER, Lang, Match, first_wins
from mlx_realtime.text.normalize.readers import (
    ZH_DIGITS,
    ascending,
    en_cardinal,
    en_number,
    plain,
    split_number,
    value,
    zh_number,
)

CODES = "USD|RMB|CNY|EUR|GBP|JPY|HKD"
"""Currency codes, as a regex alternation."""

_BY_SYMBOL = {
    "$": "USD",
    "US$": "USD",
    "HK$": "HKD",
    "¥": "CNY",
    "￥": "CNY",
    "€": "EUR",
    "£": "GBP",
}
_BY_CODE = {code: code for code in CODES.split("|")} | {"RMB": "CNY"}
_BY_WORD = {
    "美元": "USD",
    "美金": "USD",
    "欧元": "EUR",
    "英镑": "GBP",
    "日元": "JPY",
    "港元": "HKD",
    "港币": "HKD",
    "元": "CNY",
    "块": "CNY",
}
_ZH_NAMES = {"CNY": "元", "USD": "美元", "EUR": "欧元", "GBP": "英镑", "JPY": "日元", "HKD": "港元"}
_EN_NAMES = {
    "CNY": ("yuan", "yuan"),
    "USD": ("dollar", "dollars"),
    "EUR": ("euro", "euros"),
    "GBP": ("pound", "pounds"),
    "JPY": ("yen", "yen"),
    "HKD": ("Hong Kong dollar", "Hong Kong dollars"),
}
_EN_CENTS = {
    "USD": ("cent", "cents"),
    "EUR": ("cent", "cents"),
    "HKD": ("cent", "cents"),
    "GBP": ("penny", "pence"),
}
_SCALES = {
    "万": Decimal(10**4),
    "亿": Decimal(10**8),
    "万亿": Decimal(10**12),
    "thousand": Decimal(10**3),
    "million": Decimal(10**6),
    "billion": Decimal(10**9),
    "trillion": Decimal(10**12),
}
_ZH_SCALES = frozenset({"万", "亿", "万亿"})

_SYMBOL = r"US\$|HK\$|[$¥￥€£]"
_WORD = r"美元|美金|欧元|英镑|日元|港元|港币|元|块(?![0-9])"
_PREFIXED = re.compile(
    rf"(?<![A-Za-z0-9_])(?:(?P<symbol>{_SYMBOL})|(?P<code>{CODES}) ?)"
    rf"(?P<first>{NUMBER})(?: ?(?P<dash>[-~～]) ?(?:{_SYMBOL})?(?P<second>{NUMBER}))?"
    r"(?P<scale>万亿|亿|万| (?:thousand|million|billion|trillion)(?![A-Za-z]))?"
    rf"(?:{_WORD})?"  # a redundant word (¥100元) is dropped
    r"(?![A-Za-z0-9_])(?![-.,][0-9])"
)
_SUFFIXED = re.compile(
    rf"{FREE_LEFT}(?P<first>{NUMBER})(?: ?(?P<dash>[-~～]) ?(?P<second>{NUMBER}))?"
    rf"(?P<scale>万亿|亿|万)? ?(?:(?P<word>{_WORD})|(?P<code>{CODES})(?![A-Za-z0-9_]))"
)


@dataclass(frozen=True, slots=True)
class _Amount:
    written: str
    currency: str  # ISO 4217 code
    numbers: tuple[str, ...]  # one, or two for a range
    scale: str  # "", 万, 亿, 万亿, thousand, million, billion or trillion
    word: str | None  # the zh word written after the amount


def find(text: str) -> list[Match]:
    """Every amount of money in ``text``."""
    return first_wins([*_find(_PREFIXED, text), *_find(_SUFFIXED, text)])


def _find(pattern: re.Pattern[str], text: str) -> Iterator[Match]:
    for found in pattern.finditer(text):
        groups = found.groupdict()
        first, second = found["first"], found["second"]
        if second is not None and found["dash"] == "-" and not ascending(first, second):
            continue
        if symbol := groups.get("symbol"):
            currency = _BY_SYMBOL[symbol]
        elif code := groups.get("code"):
            currency = _BY_CODE[code]
        else:
            currency = _BY_WORD[found["word"]]
        numbers = (first,) if second is None else (first, second)
        scale = (found["scale"] or "").strip()
        amount = _Amount(found.group(), currency, numbers, scale, groups.get("word"))
        yield Match(found.start(), found.end(), partial(_read, amount))


def _read(amount: _Amount, lang: Lang) -> str:
    spoken = _read_zh(amount) if lang == "zh" else _read_en(amount)
    return spoken or amount.written


def _read_zh(amount: _Amount) -> str | None:
    name = amount.word or _ZH_NAMES[amount.currency]
    if name in ("元", "块") and len(amount.numbers) == 1 and not amount.scale:
        spoken = _zh_yuan(amount.numbers[0], name)
        if spoken is not None:
            return spoken
    if amount.scale in _ZH_SCALES or not amount.scale:
        readings = [zh_number(number, liang=True) for number in amount.numbers]
        suffix = amount.scale + name
    else:
        scale = _SCALES[amount.scale]
        readings = [zh_number(plain(value(number) * scale)) for number in amount.numbers]
        suffix = name
    words = [reading for reading in readings if reading is not None]
    return "到".join(words) + suffix if len(words) == len(readings) else None


def _zh_yuan(written: str, unit: str) -> str | None:
    """元角分, or 块毛分 after 块; ``None`` past two decimals."""
    integer, fraction = split_number(written)
    whole = zh_number(integer, liang=True)
    if whole is None or len(fraction) > 2:
        return None
    jiao, fen = (int(digit) for digit in fraction.ljust(2, "0"))
    spoken = whole + unit if int(integer) or not (jiao or fen) else ""
    if jiao:
        spoken += ZH_DIGITS[jiao] + ("毛" if unit == "块" else "角")
    if fen:
        spoken += ("零" if spoken and not jiao else "") + ZH_DIGITS[fen] + "分"
    return spoken


def _read_en(amount: _Amount) -> str | None:
    if len(amount.numbers) == 1 and not amount.scale and amount.currency in _EN_CENTS:
        spoken = _en_cents(amount.numbers[0], amount.currency)
        if spoken is not None:
            return spoken
    readings = [_en_scaled(number, amount.scale) for number in amount.numbers]
    words = [reading for reading in readings if reading is not None]
    if len(words) < len(readings):
        return None
    singular, plural = _EN_NAMES[amount.currency]
    one = amount.numbers == ("1",) and not amount.scale
    return " to ".join(words) + " " + (singular if one else plural)


def _en_scaled(written: str, scale: str) -> str | None:
    if scale in _ZH_SCALES:
        return en_number(plain(value(written) * _SCALES[scale]))
    spoken = en_number(written)
    return f"{spoken} {scale}" if spoken is not None and scale else spoken


def _en_cents(written: str, currency: str) -> str | None:
    """Whole units and cents: twelve dollars and ninety-nine cents; ``None`` past two decimals."""
    integer, fraction = split_number(written)
    whole = en_number(integer)
    if whole is None or len(fraction) > 2:
        return None
    cents = int(fraction.ljust(2, "0"))
    (singular, plural), (cent, cents_name) = _EN_NAMES[currency], _EN_CENTS[currency]
    parts: list[str] = []
    if int(integer) or not cents:
        parts.append(f"{whole} {singular if integer == '1' else plural}")
    if cents:
        parts.append(f"{en_cardinal(cents)} {cent if cents == 1 else cents_name}")
    return " and ".join(parts)
