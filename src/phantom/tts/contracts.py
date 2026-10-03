"""Text-to-speech contracts with explicit audio-unavailable states."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable


class TTSMode(str, Enum):
    REAL = "real"
    TEXT_ONLY = "text-only"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class TTSInfo:
    name: str
    mode: TTSMode
    voice: str | None = None
    local: bool = True


@dataclass(frozen=True, slots=True)
class SpeechResult:
    provider: TTSInfo
    audio_wav: bytes | None
    mime_type: str | None
    audio_available: bool
    text_only: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.audio_available:
            if self.audio_wav is None or self.mime_type != "audio/wav":
                raise ValueError(
                    "available speech output requires WAV bytes and audio/wav MIME type"
                )
            if self.text_only:
                raise ValueError("available audio output cannot also be text-only")
        elif self.audio_wav is not None or self.mime_type is not None:
            raise ValueError("unavailable speech output cannot contain audio data")


class TTSError(RuntimeError):
    """Base TTS error whose message must not include the synthesized text."""


class TTSUnavailableError(TTSError):
    """The selected synthesis backend or dependency is unavailable."""


class TTSSynthesisError(TTSError):
    """The selected backend failed to create valid bounded WAV output."""


class TTSInputError(TTSError):
    """The requested text is empty or exceeds the configured safety bound."""


@runtime_checkable
class TTSProvider(Protocol):
    @property
    def info(self) -> TTSInfo: ...

    def synthesize(self, text: str) -> SpeechResult: ...

    def stop(self) -> None: ...


def validate_tts_text(text: str, max_characters: int) -> str:
    clean = text.strip()
    if not clean:
        raise TTSInputError("text-to-speech input cannot be empty")
    if len(clean) > max_characters:
        raise TTSInputError(f"text-to-speech input exceeds the {max_characters}-character limit")
    return clean


class TextOnlyTTS:
    """Explicit operator-selected backend that deliberately produces no audio."""

    def __init__(self, reason: str = "text-only output was selected") -> None:
        self.reason = reason
        self._info = TTSInfo(name="text-only", mode=TTSMode.TEXT_ONLY)

    @property
    def info(self) -> TTSInfo:
        return self._info

    def synthesize(self, text: str) -> SpeechResult:
        validate_tts_text(text, 20_000)
        return SpeechResult(
            provider=self.info,
            audio_wav=None,
            mime_type=None,
            audio_available=False,
            text_only=True,
            reason=self.reason,
        )

    def stop(self) -> None:
        return None


class UnavailableTTS:
    """Explicit state for a requested audio backend that could not be initialized."""

    def __init__(self, reason: str = "text-to-speech is unavailable") -> None:
        self.reason = reason
        self._info = TTSInfo(name="unavailable", mode=TTSMode.UNAVAILABLE)

    @property
    def info(self) -> TTSInfo:
        return self._info

    def synthesize(self, text: str) -> SpeechResult:
        validate_tts_text(text, 20_000)
        return SpeechResult(
            provider=self.info,
            audio_wav=None,
            mime_type=None,
            audio_available=False,
            text_only=True,
            reason=self.reason,
        )

    def stop(self) -> None:
        return None
