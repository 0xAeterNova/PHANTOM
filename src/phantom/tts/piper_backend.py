"""Local Arabic neural TTS with platform-specific local English speech output."""

from __future__ import annotations

import hashlib
import platform
import threading
import wave
from collections.abc import Callable
from importlib import import_module
from io import BytesIO
from pathlib import Path
from typing import Any

from phantom.tts.contracts import (
    SpeechResult,
    TTSInfo,
    TTSMode,
    TTSSynthesisError,
    TTSUnavailableError,
    validate_tts_text,
)
from phantom.tts.espeak_backend import EspeakNGTTS
from phantom.tts.pyttsx3_backend import Pyttsx3WindowsTTS

PiperVoiceLoader = Callable[[Path, Path], Any]


def contains_arabic_letters(text: str) -> bool:
    """Return whether text contains a letter from an Arabic Unicode block."""

    ranges = (
        ("\u0600", "\u06ff"),
        ("\u0750", "\u077f"),
        ("\u08a0", "\u08ff"),
        ("\ufb50", "\ufdff"),
        ("\ufe70", "\ufeff"),
    )
    return any(
        character.isalpha() and start <= character <= end
        for character in text
        for start, end in ranges
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_audible_wav(data: bytes, *, max_wav_bytes: int) -> bool:
    """Validate bounded uncompressed PCM with at least 20 ms of nonzero audio."""

    if (
        len(data) < 44
        or len(data) > max_wav_bytes
        or not data.startswith(b"RIFF")
        or data[8:12] != b"WAVE"
    ):
        return False
    try:
        with wave.open(BytesIO(data), "rb") as source:
            frame_rate = source.getframerate()
            frame_count = source.getnframes()
            sample_width = source.getsampwidth()
            channels = source.getnchannels()
            if (
                source.getcomptype() != "NONE"
                or frame_rate <= 0
                or channels <= 0
                or sample_width not in {1, 2, 3, 4}
                or frame_count < max(1, int(frame_rate * 0.02))
            ):
                return False
            samples = source.readframes(frame_count)
    except (EOFError, OSError, wave.Error):
        return False

    if sample_width == 1:
        return any(abs(sample - 128) >= 2 for sample in samples)
    zero_sample = b"\x00" * sample_width
    return any(
        samples[offset : offset + sample_width] != zero_sample
        for offset in range(0, len(samples) - sample_width + 1, sample_width)
    )


class PiperArabicHybridTTS:
    """Route Arabic/mixed text to Piper and English to local SAPI or eSpeak NG.

    Piper is loaded lazily and kept in memory after the first Arabic utterance.
    The caller must provision the configured model and JSON pair explicitly;
    this runtime never downloads executable/model content during a user turn.
    """

    def __init__(
        self,
        model_path: str | Path,
        config_path: str | Path,
        *,
        model_sha256: str,
        config_sha256: str,
        english_voice_id: str | None = None,
        english_rate: int = 175,
        max_text_characters: int = 8_000,
        max_wav_bytes: int = 20 * 1024 * 1024,
        voice_loader: PiperVoiceLoader | None = None,
        english_provider: Pyttsx3WindowsTTS | EspeakNGTTS | None = None,
    ) -> None:
        if not 128 <= max_text_characters <= 20_000:
            raise ValueError("max_text_characters must be between 128 and 20000")
        if not 44 <= max_wav_bytes <= 100 * 1024 * 1024:
            raise ValueError("max_wav_bytes must be between 44 bytes and 100 MiB")
        for label, digest in (
            ("model_sha256", model_sha256),
            ("config_sha256", config_sha256),
        ):
            if len(digest) != 64 or any(
                character not in "0123456789abcdefABCDEF" for character in digest
            ):
                raise ValueError(f"{label} must be a 64-character hexadecimal digest")

        self.model_path = Path(model_path).expanduser().resolve()
        self.config_path = Path(config_path).expanduser().resolve()
        self.model_sha256 = model_sha256.lower()
        self.config_sha256 = config_sha256.lower()
        self.max_text_characters = max_text_characters
        self.max_wav_bytes = max_wav_bytes
        self._voice_loader = voice_loader
        self._voice: Any | None = None
        self._lock = threading.RLock()
        provider_class = EspeakNGTTS if platform.system() == "Linux" else Pyttsx3WindowsTTS
        self._english = english_provider or provider_class(
            voice_id=english_voice_id,
            rate=english_rate,
            max_text_characters=max_text_characters,
            max_wav_bytes=max_wav_bytes,
        )
        linux_english = isinstance(self._english, EspeakNGTTS)
        self._info = TTSInfo(
            name="piper-arabic+espeak-ng" if linux_english else "piper-arabic+pyttsx3-windows",
            mode=TTSMode.REAL,
            voice="ar_JO-kareem-low / eSpeak NG"
            if linux_english
            else "ar_JO-kareem-low / Windows SAPI",
            local=True,
        )

    @property
    def info(self) -> TTSInfo:
        return self._info

    def _validate_artifact(self, path: Path, expected_sha256: str, label: str) -> None:
        if not path.is_file():
            raise TTSUnavailableError(
                f"the local Piper {label} is missing; run scripts/setup_piper_tts.py"
            )
        try:
            actual_sha256 = _sha256(path)
        except OSError as exc:
            raise TTSUnavailableError(f"the local Piper {label} could not be read") from exc
        if actual_sha256 != expected_sha256:
            raise TTSUnavailableError(f"the local Piper {label} failed its SHA-256 integrity check")

    def _load(self) -> Any:
        with self._lock:
            if self._voice is not None:
                return self._voice
            self._validate_artifact(self.model_path, self.model_sha256, "model")
            self._validate_artifact(self.config_path, self.config_sha256, "configuration")
            try:
                loader = self._voice_loader
                if loader is None:
                    piper_voice = import_module("piper.voice").PiperVoice
                    voice = piper_voice.load(self.model_path, config_path=self.config_path)
                else:
                    voice = loader(self.model_path, self.config_path)
            except (ImportError, ModuleNotFoundError) as exc:
                raise TTSUnavailableError(
                    "piper-tts 1.6.0 is not installed or could not load its local runtime"
                ) from exc
            except Exception as exc:
                raise TTSUnavailableError(
                    "the local Piper Arabic voice could not be loaded"
                ) from exc
            self._voice = voice
            return voice

    def probe(self) -> None:
        """Load the Arabic model and validate the local English route."""

        self._load()
        self._english.probe()

    def _synthesize_arabic(self, clean: str) -> SpeechResult:
        voice = self._load()
        output = BytesIO()
        try:
            with self._lock, wave.open(output, "wb") as wav_file:
                voice.synthesize_wav(clean, wav_file)
            data = output.getvalue()
        except Exception as exc:
            raise TTSSynthesisError("local Piper Arabic speech synthesis failed") from exc
        if not _is_audible_wav(data, max_wav_bytes=self.max_wav_bytes):
            if len(data) > self.max_wav_bytes:
                raise TTSSynthesisError("the synthesized WAV exceeded its configured size limit")
            raise TTSSynthesisError("the Piper engine returned invalid or inaudible WAV data")
        return SpeechResult(
            provider=self.info,
            audio_wav=data,
            mime_type="audio/wav",
            audio_available=True,
            text_only=False,
        )

    def synthesize(self, text: str) -> SpeechResult:
        clean = validate_tts_text(text, self.max_text_characters)
        if contains_arabic_letters(clean):
            return self._synthesize_arabic(clean)
        result = self._english.synthesize(clean)
        return SpeechResult(
            provider=self.info,
            audio_wav=result.audio_wav,
            mime_type=result.mime_type,
            audio_available=result.audio_available,
            text_only=result.text_only,
            reason=result.reason,
        )

    def stop(self) -> None:
        self._english.stop()
