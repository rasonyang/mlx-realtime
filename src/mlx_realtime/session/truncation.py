"""Truncation arithmetic: which text the client heard (ARCHITECTURE §6.6).

Positions are on the item's output-audio timeline. Segments that end at or before the cut are
heard whole; a segment's speech ends before the keepalive silence the audio reserve may have
sent after it while the next segment waited (a cut in that silence heard the segment whole).
The segment playing at the cut contributes a prefix proportional to how much of its speech
played: the audio speaks the segment's normalized text, so the share is taken in
spoken characters and mapped back through the normalizer's character map to the original text
(a phone number read digit by digit is heard as far as its digits were read), then snapped back
to a word boundary in Latin text.
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
        if segment.start_ms + segment.speech_ms <= audio_end_ms:
            parts.append(segment.text)
            continue
        played = (audio_end_ms - segment.start_ms) / segment.speech_ms
        parts.append(_prefix(segment, played))
        break
    return "".join(parts)


def heard_position(segment: SpokenSegment, fraction: float) -> int:
    """How far into ``segment.text`` the client heard after ``fraction`` of its audio."""
    if segment.source is None:
        return int(len(segment.text) * fraction)
    spoken = segment.spoken
    said = spoken.strip()  # what the TTS was asked to say
    lead = len(spoken) - len(spoken.lstrip())
    index = min(lead + int(len(said) * fraction), len(spoken))
    return segment.source[index]


def _prefix(segment: SpokenSegment, fraction: float) -> str:
    text = segment.text
    cut = heard_position(segment, fraction)
    if 0 < cut < len(text) and _latin(text[cut - 1]) and _latin(text[cut]):
        while cut > 0 and _latin(text[cut - 1]):
            cut -= 1
    return text[:cut]


def _latin(char: str) -> bool:
    return char.isascii() and char.isalnum()
