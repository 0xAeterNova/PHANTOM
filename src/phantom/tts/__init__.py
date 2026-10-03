"""Local text-to-speech providers and explicit text-only behavior."""

from phantom.tts.contracts import (
    SpeechResult,
    TextOnlyTTS,
    TTSError,
    TTSInfo,
    TTSInputError,
    TTSMode,
    TTSProvider,
    TTSSynthesisError,
    TTSUnavailableError,
    UnavailableTTS,
)
from phantom.tts.espeak_backend import EspeakNGTTS
from phantom.tts.piper_backend import PiperArabicHybridTTS, contains_arabic_letters
from phantom.tts.pyttsx3_backend import Pyttsx3WindowsTTS

__all__ = [
    "EspeakNGTTS",
    "PiperArabicHybridTTS",
    "Pyttsx3WindowsTTS",
    "SpeechResult",
    "TTSError",
    "TTSInfo",
    "TTSInputError",
    "TTSMode",
    "TTSProvider",
    "TTSSynthesisError",
    "TTSUnavailableError",
    "TextOnlyTTS",
    "UnavailableTTS",
    "contains_arabic_letters",
]
