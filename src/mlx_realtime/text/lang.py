"""Character classes and the per-segment language tag (ARCHITECTURE §7)."""

from __future__ import annotations

from typing import Literal

_CJK_RANGES = (
    (0x3040, 0x30FF),  # Hiragana, Katakana
    (0x3400, 0x4DBF),  # CJK Extension A
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0xAC00, 0xD7AF),  # Hangul syllables
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
    (0x20000, 0x2FA1F),  # CJK Extensions B-F and supplements
)


def is_cjk(char: str) -> bool:
    """True for a CJK ideograph, kana or hangul syllable (not for CJK punctuation)."""
    code = ord(char)
    return any(low <= code <= high for low, high in _CJK_RANGES)


def detect_lang(text: str) -> Literal["zh", "en"]:
    """``zh`` if the text holds any CJK character (mixed text is tagged ``zh``), else ``en``."""
    return "zh" if any(is_cjk(char) for char in text) else "en"
