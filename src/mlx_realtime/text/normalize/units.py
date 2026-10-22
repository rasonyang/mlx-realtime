"""Units of measure right after a number (ARCHITECTURE §7).

Rules:

- A unit counts only right after a number or a range, with at most one space between:
  ``5km``, ``2.5 kg``, ``-5°C``, ``3-5km``. Units are case-sensitive and end the token, so ``5G``,
  ``5min`` and ``5ms`` hold no unit.
- zh names: km 公里, m 米, cm 厘米, mm 毫米, kg 公斤, g 克, mg 毫克, L/l 升, ml/mL 毫升,
  °C/℃ 摄氏度, °F/℉ 华氏度, km/h 公里每小时, m/s 米每秒, ㎡/m² 平方米, m³ 立方米, kWh 千瓦时,
  kW 千瓦, W 瓦, mAh 毫安时, Hz 赫兹, kHz 千赫兹, MHz 兆赫兹, GHz 吉赫兹. Data sizes are spelled
  as letters: KB → K B, MB → M B, GB → G B, TB → T B.
- en names: kilometers, meters, …, degrees Celsius, kilometers per hour, square meters, kilowatt
  hours, gigahertz, gigabytes, …; singular after exactly ``1``.
- With ``numbers`` on, the number is read too (zh 2 is 两, except before a temperature); with it
  off the number stays as written: ``5km`` → zh ``5公里``, en ``5 kilometers``.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from functools import partial

from mlx_realtime.text.normalize.common import FREE_LEFT, NUMBER, Lang, Match, is_minus
from mlx_realtime.text.normalize.readers import ascending, en_number, zh_number

_NAMES: dict[str, tuple[str, str, str]] = {
    "km/h": ("公里每小时", "kilometer per hour", "kilometers per hour"),
    "m/s": ("米每秒", "meter per second", "meters per second"),
    "km": ("公里", "kilometer", "kilometers"),
    "m": ("米", "meter", "meters"),
    "cm": ("厘米", "centimeter", "centimeters"),
    "mm": ("毫米", "millimeter", "millimeters"),
    "㎡": ("平方米", "square meter", "square meters"),
    "m²": ("平方米", "square meter", "square meters"),
    "m³": ("立方米", "cubic meter", "cubic meters"),
    "kg": ("公斤", "kilogram", "kilograms"),
    "g": ("克", "gram", "grams"),
    "mg": ("毫克", "milligram", "milligrams"),
    "L": ("升", "liter", "liters"),
    "l": ("升", "liter", "liters"),
    "ml": ("毫升", "milliliter", "milliliters"),
    "mL": ("毫升", "milliliter", "milliliters"),
    "°C": ("摄氏度", "degree Celsius", "degrees Celsius"),
    "℃": ("摄氏度", "degree Celsius", "degrees Celsius"),
    "°F": ("华氏度", "degree Fahrenheit", "degrees Fahrenheit"),
    "℉": ("华氏度", "degree Fahrenheit", "degrees Fahrenheit"),
    "kWh": ("千瓦时", "kilowatt hour", "kilowatt hours"),
    "kW": ("千瓦", "kilowatt", "kilowatts"),
    "W": ("瓦", "watt", "watts"),
    "mAh": ("毫安时", "milliamp hour", "milliamp hours"),
    "Hz": ("赫兹", "hertz", "hertz"),
    "kHz": ("千赫兹", "kilohertz", "kilohertz"),
    "MHz": ("兆赫兹", "megahertz", "megahertz"),
    "GHz": ("吉赫兹", "gigahertz", "gigahertz"),
    "KB": ("K B", "kilobyte", "kilobytes"),
    "MB": ("M B", "megabyte", "megabytes"),
    "GB": ("G B", "gigabyte", "gigabytes"),
    "TB": ("T B", "terabyte", "terabytes"),
}
_TEMPERATURES = frozenset({"°C", "℃", "°F", "℉"})

UNIT_PATTERN = "|".join(re.escape(unit) for unit in sorted(_NAMES, key=len, reverse=True))
"""Every unit, longest first."""

_UNIT = re.compile(
    rf"{FREE_LEFT}(?P<number>(?P<first>{NUMBER})(?: ?(?P<dash>[-~～]) ?(?P<second>{NUMBER}))?)"
    rf" ?(?P<unit>{UNIT_PATTERN})(?![A-Za-z0-9_²³])"
)


@dataclass(frozen=True, slots=True)
class _Quantity:
    written: str  # the number as written, sign included
    first: str
    second: str | None
    negative: bool
    unit: str


def find(text: str, *, read_numbers: bool) -> Iterator[Match]:
    """Every number with a unit in ``text``."""
    for found in _UNIT.finditer(text):
        first, second = found["first"], found["second"]
        if second is not None and found["dash"] == "-" and not ascending(first, second):
            continue
        negative = second is None and is_minus(text, found.start() - 1)
        start = found.start() - 1 if negative else found.start()
        written = text[start : found.end("number")]
        quantity = _Quantity(written, first, second, negative, found["unit"])
        yield Match(start, found.end(), partial(_read, quantity, read_numbers=read_numbers))


def _read(quantity: _Quantity, lang: Lang, *, read_numbers: bool) -> str:
    zh_name, singular, plural = _NAMES[quantity.unit]
    spoken = (_read_numbers(quantity, lang) if read_numbers else None) or quantity.written
    if lang == "zh":
        return spoken + zh_name
    one = quantity.second is None and quantity.first == "1"
    return f"{spoken} {singular if one else plural}"


def _read_numbers(quantity: _Quantity, lang: Lang) -> str | None:
    numbers = [quantity.first] if quantity.second is None else [quantity.first, quantity.second]
    if lang == "zh":
        liang = quantity.unit not in _TEMPERATURES and not quantity.negative
        readings = [zh_number(number, liang=liang) for number in numbers]
        sign, joiner = "负", "到"
    else:
        readings = [en_number(number) for number in numbers]
        sign, joiner = "minus ", " to "
    words = [reading for reading in readings if reading is not None]
    if len(words) < len(readings):
        return None
    return (sign if quantity.negative else "") + joiner.join(words)
