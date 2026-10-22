"""Number readers: digits, cardinals, ordinals and decimals in Chinese (zh) and English (en).

Rules:

- zh cardinals group by 万 and 亿, with one 零 for each gap: 10 → 十, 11 → 十一, 110 → 一百一十,
  1005 → 一千零五, 10010 → 一万零一十, 100000 → 十万, 1,234,567 → 一百二十三万四千五百六十七.
- en cardinals use no "and": 1234 → one thousand two hundred thirty-four.
- A decimal reads its integer part as a cardinal and each fraction digit on its own:
  3.14 → 三点一四 / three point one four, 0.5 → 零点五 / zero point five.
- An integer written with a leading zero (``007``) is read digit by digit.
- Integers longer than ``MAX_DIGITS`` digits are not read: the readers return ``None`` and the
  caller leaves the text as written.
"""

from __future__ import annotations

from decimal import Decimal

MAX_DIGITS = 16
ZH_DIGITS = "零一二三四五六七八九"
EN_DIGITS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
_EN_SMALL = (
    *EN_DIGITS,
    *("ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen"),
    *("eighteen", "nineteen"),
)
_EN_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
_EN_SCALES = ("", "thousand", "million", "billion", "trillion", "quadrillion")
_EN_ORDINALS = {
    "one": "first",
    "two": "second",
    "three": "third",
    "five": "fifth",
    "eight": "eighth",
    "nine": "ninth",
    "twelve": "twelfth",
}


def zh_digits(digits: str, *, yao: bool = False) -> str:
    """Digit by digit: ``2024`` → 二零二四; with ``yao``, 1 is 幺."""
    spoken = "".join(ZH_DIGITS[int(digit)] for digit in digits)
    return spoken.replace("一", "幺") if yao else spoken


def en_digits(digits: str) -> str:
    """Digit by digit: ``2024`` → two zero two four."""
    return " ".join(EN_DIGITS[int(digit)] for digit in digits)


def zh_cardinal(value: int) -> str:
    """0 ≤ value < 10**16."""
    if value == 0:
        return "零"
    high, low = divmod(value, 10**8)
    spoken = _zh_below_1e8(high) + "亿" + _zh_tail(low, 10**7) if high else _zh_below_1e8(low)
    return spoken[1:] if spoken.startswith("一十") else spoken


def _zh_below_1e8(value: int) -> str:
    high, low = divmod(value, 10**4)
    return _zh_below_1e4(high) + "万" + _zh_tail(low, 1000) if high else _zh_below_1e4(low)


def _zh_tail(value: int, full: int) -> str:
    """What follows 万 or 亿: 零 first when its leading place (``full``) is empty."""
    if not value:
        return ""
    return ("零" if value < full else "") + _zh_below_1e8(value)


def _zh_below_1e4(value: int) -> str:
    """1 ≤ value < 10**4, keeping the 一 of 一十 (``zh_cardinal`` drops it at the very start)."""
    spoken = ""
    gap = False
    for power, unit in ((1000, "千"), (100, "百"), (10, "十"), (1, "")):
        digit = value // power % 10
        if digit:
            spoken += ("零" if gap else "") + ZH_DIGITS[digit] + unit
            gap = False
        elif spoken:
            gap = True
    return spoken


def en_cardinal(value: int) -> str:
    """0 ≤ value < 10**18."""
    if value == 0:
        return "zero"
    words: list[str] = []
    for scale in reversed(range(len(_EN_SCALES))):
        group = value // 1000**scale % 1000
        if group:
            words.append(_en_below_1000(group) + (" " + _EN_SCALES[scale] if scale else ""))
    return " ".join(words)


def _en_below_1000(value: int) -> str:
    hundreds, rest = divmod(value, 100)
    words = [f"{_EN_SMALL[hundreds]} hundred"] if hundreds else []
    if rest >= 20:
        tens, ones = divmod(rest, 10)
        words.append(_EN_TENS[tens] + (f"-{_EN_SMALL[ones]}" if ones else ""))
    elif rest:
        words.append(_EN_SMALL[rest])
    return " ".join(words)


def en_ordinal(value: int) -> str:
    """1 → first, 12 → twelfth, 21 → twenty-first, 100 → one hundredth."""
    cardinal = en_cardinal(value)
    cut = max(cardinal.rfind(" "), cardinal.rfind("-")) + 1
    word = cardinal[cut:]
    if word in _EN_ORDINALS:
        ordinal = _EN_ORDINALS[word]
    elif word.endswith("y"):
        ordinal = word[:-1] + "ieth"
    else:
        ordinal = word + "th"
    return cardinal[:cut] + ordinal


def split_number(written: str) -> tuple[str, str]:
    """``1,234.50`` → (``1234``, ``50``)."""
    integer, _, fraction = written.replace(",", "").partition(".")
    return integer, fraction


def readable(written: str) -> bool:
    """Whether the integer part of a written number is short enough to read."""
    return len(split_number(written)[0]) <= MAX_DIGITS


def value(written: str) -> Decimal:
    """The value of a written number: ``1,234.5`` → 1234.5."""
    return Decimal(written.replace(",", ""))


def ascending(first: str, second: str) -> bool:
    """Whether written ``first`` is below ``second``: ``3-5`` is a range, ``5-3`` is not."""
    return value(first) < value(second)


def plain(number: Decimal) -> str:
    """A value written without exponent or trailing zeros: 1.5E+6 → ``1500000``."""
    return format(number.normalize(), "f")


def zh_number(written: str, *, liang: bool = False) -> str | None:
    """A written number (``1,234.5``) in Chinese; with ``liang``, a plain 2 is 两."""
    integer, fraction = split_number(written)
    if len(integer) > MAX_DIGITS:
        return None
    if liang and integer == "2" and not fraction:
        return "两"
    if len(integer) > 1 and integer.startswith("0"):
        head = zh_digits(integer)
    else:
        head = zh_cardinal(int(integer))
    return head + ("点" + zh_digits(fraction) if fraction else "")


def en_number(written: str) -> str | None:
    """A written number (``1,234.5``) in English."""
    integer, fraction = split_number(written)
    if len(integer) > MAX_DIGITS:
        return None
    if len(integer) > 1 and integer.startswith("0"):
        head = en_digits(integer)
    else:
        head = en_cardinal(int(integer))
    return head + (" point " + en_digits(fraction) if fraction else "")
