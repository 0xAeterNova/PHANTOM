"""Optional local speech-synthesis boundary."""

from __future__ import annotations

from typing import Protocol


class SpeechSynthesizer(Protocol):
    def speak(self, text: str, *, speed: float = 1.0) -> None: ...


class TextOnlySpeechOutput:
    """Safe fallback that exposes text without requiring a TTS dependency."""

    def __init__(self) -> None:
        self.last_text = ""

    def speak(self, text: str, *, speed: float = 1.0) -> None:
        if not 0.5 <= speed <= 2.0:
            raise ValueError("speed must be between 0.5 and 2.0")
        self.last_text = text
