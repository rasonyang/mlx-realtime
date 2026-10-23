"""Session voice names and what each TTS model is asked for (D19).

A session names a voice in ``audio.output.voice``. The voice map in the configuration says,
for each name, which speaker the primary model uses, how its lang_code is chosen, and which
voice the fallback model uses for each segment language. A name that is not in the map is never
rejected: it resolves to the default voice, and the session echoes the effective name.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from mlx_realtime.config import VoicesConfig
from mlx_realtime.providers.base import Lang

SEGMENT_LANG = "segment"
"""A voice's lang policy that sends each segment's own language tag, mapped per model."""


@dataclass(frozen=True, slots=True)
class Voice:
    name: str
    speaker: str  # the primary model's speaker
    lang: str  # SEGMENT_LANG, or the lang_code the primary model always gets
    fallback: Mapping[Lang, str]  # the fallback model's voice per segment language
    speakers: Mapping[Lang, str] = field(default_factory=dict[Lang, str])  # per-language speaker

    def speaker_for(self, lang: Lang) -> str:
        """The primary model's speaker for a segment in ``lang``."""
        return self.speakers.get(lang, self.speaker)


class VoiceBook:
    def __init__(self, config: VoicesConfig) -> None:
        self.default = config.default
        self._voices = {
            name: Voice(name, entry.speaker, entry.lang, dict(entry.fallback), dict(entry.speakers))
            for name, entry in config.map.items()
        }

    def __contains__(self, name: object) -> bool:
        return name in self._voices

    def names(self) -> tuple[str, ...]:
        return tuple(self._voices)

    def effective(self, name: str) -> str:
        """``name`` if it is configured, else the default voice."""
        return name if name in self._voices else self.default

    def voice(self, name: str) -> Voice:
        return self._voices[self.effective(name)]

    def voices(self) -> tuple[Voice, ...]:
        return tuple(self._voices.values())
