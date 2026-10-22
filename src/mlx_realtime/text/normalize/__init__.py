"""Text normalization for TTS input; transcripts keep the original text (ARCHITECTURE §7).

M1 passes text through unchanged. The categories (numbers, money, dates, phones, IDs, units,
zh/en spacing) land once M0 shows which ones the TTS reads wrongly.
"""

from __future__ import annotations


def normalize(text: str) -> str:
    return text
