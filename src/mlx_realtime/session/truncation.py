"""Truncation arithmetic: which text the client heard (ARCHITECTURE §6.6).

Positions are on the item's output-audio timeline. Segments that end at or before the cut are
heard whole; the segment playing at the cut contributes a prefix proportional to how much of
its audio played, measured in characters and snapped back to a word boundary in Latin text.
"""

from __future__ import annotations

from collections.abc import Sequence

from mlx_realtime.session.conversation import SpokenSegment


def clamp_audio_end(audio_end_ms: int, sent_ms: int) -> int:
    """A cut beyond the audio sent is clamped to it (D18): GA would answer with an error."""
    return min(audio_end_ms, sent_ms)


def heard_text(segments: Sequence[SpokenSegment], audio_end_ms: int) -> str:
    parts: list[str] = []
    for segment in segments:
        if segment.start_ms >= audio_end_ms:
            break
        if segment.start_ms + segment.audio_ms <= audio_end_ms:
            parts.append(segment.text)
            continue
        played = (audio_end_ms - segment.start_ms) / segment.audio_ms
        parts.append(_prefix(segment.text, played))
        break
    return "".join(parts)


def _prefix(text: str, fraction: float) -> str:
    cut = int(len(text) * fraction)
    if 0 < cut < len(text) and _latin(text[cut - 1]) and _latin(text[cut]):
        while cut > 0 and _latin(text[cut - 1]):
            cut -= 1
    return text[:cut]


def _latin(char: str) -> bool:
    return char.isascii() and char.isalnum()
