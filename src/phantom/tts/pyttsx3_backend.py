"""Lazy Windows pyttsx3 adapter that returns transient WAV bytes."""

from __future__ import annotations

import os
import platform
import sys
import tempfile
import threading
import wave
from array import array
from collections.abc import Callable
from contextlib import suppress
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

EngineFactory = Callable[[], Any]


def _contains_arabic_letters(text: str) -> bool:
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


def _voice_supports_arabic(voice: Any) -> bool:
    """Inspect pyttsx3's inconsistent str/bytes language metadata safely."""

    for raw_language in getattr(voice, "languages", None) or ():
        if isinstance(raw_language, bytes):
            language = raw_language.decode("utf-8", errors="ignore")
        else:
            language = str(raw_language)
        normalized = (
            language.lstrip("\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09").replace("_", "-").lower()
        )
        if normalized == "ar" or normalized.startswith("ar-"):
            return True

    # Some Windows tokens expose no language collection. Their stable token or
    # display name still includes either "Arabic" or an ``AR-xx`` locale.
    description = (
        " ".join(str(getattr(voice, field, "") or "") for field in ("name", "id"))
        .replace("_", "-")
        .lower()
    )
    return "arabic" in description or "-ar-" in description


def _pcm_peak(samples: bytes, sample_width: int) -> tuple[int, int]:
    """Return absolute PCM peak and full scale for a standard WAV sample width."""

    usable = len(samples) - (len(samples) % sample_width)
    if usable <= 0:
        return 0, 1
    samples = samples[:usable]
    if sample_width == 1:
        return max(abs(sample - 128) for sample in samples), 128
    if sample_width in {2, 4}:
        typecode = "h" if sample_width == 2 else "i"
        values = array(typecode)
        values.frombytes(samples)
        if sys.byteorder != "little":
            values.byteswap()
        return max((abs(sample) for sample in values), default=0), 1 << (8 * sample_width - 1)
    if sample_width == 3:
        peak = max(
            (
                abs(int.from_bytes(samples[offset : offset + 3], "little", signed=True))
                for offset in range(0, usable, 3)
            ),
            default=0,
        )
        return peak, 1 << 23
    return 0, 1


class Pyttsx3WindowsTTS:
    """Use Windows SAPI through pyttsx3 and remove each temporary WAV immediately."""

    def __init__(
        self,
        *,
        voice_id: str | None = None,
        rate: int = 175,
        volume: float = 1.0,
        max_text_characters: int = 8_000,
        max_wav_bytes: int = 20 * 1024 * 1024,
        temporary_directory: str | Path | None = None,
        engine_factory: EngineFactory | None = None,
    ) -> None:
        if not 80 <= rate <= 400:
            raise ValueError("pyttsx3 rate must be between 80 and 400")
        if not 0.0 <= volume <= 1.0:
            raise ValueError("pyttsx3 volume must be between 0 and 1")
        if not 128 <= max_text_characters <= 20_000:
            raise ValueError("max_text_characters must be between 128 and 20000")
        if not 44 <= max_wav_bytes <= 100 * 1024 * 1024:
            raise ValueError("max_wav_bytes must be between 44 bytes and 100 MiB")
        selected_temp = Path(temporary_directory).resolve() if temporary_directory else None
        if selected_temp is not None and not selected_temp.is_dir():
            raise ValueError("temporary_directory must be an existing directory")
        self.voice_id = voice_id
        self.rate = rate
        self.volume = volume
        self.max_text_characters = max_text_characters
        self.max_wav_bytes = max_wav_bytes
        self.temporary_directory = selected_temp
        self._engine_factory = engine_factory
        self._engine: Any | None = None
        self._lock = threading.RLock()
        self._info = TTSInfo(
            name="pyttsx3-windows",
            mode=TTSMode.REAL,
            voice=voice_id,
            local=True,
        )

    @property
    def info(self) -> TTSInfo:
        return self._info

    def _load_engine(self) -> Any:
        if platform.system() != "Windows":
            raise TTSUnavailableError("pyttsx3 Windows synthesis requires Windows")
        with self._lock:
            if self._engine is not None:
                return self._engine
            try:
                factory = self._engine_factory
                if factory is None:
                    pyttsx3 = import_module("pyttsx3")
                    # ``pyttsx3.init`` globally caches the SAPI engine.  On
                    # Windows, that cached engine can block forever on a
                    # second ``save_to_file`` call.  A fresh lightweight SAPI
                    # engine per utterance keeps multi-turn audio reliable;
                    # neural inference models are unaffected and stay loaded.
                    factory = pyttsx3.engine.Engine
                engine = factory()
                engine.setProperty("rate", self.rate)
                engine.setProperty("volume", self.volume)
                if self.voice_id:
                    engine.setProperty("voice", self.voice_id)
            except (ImportError, ModuleNotFoundError) as exc:
                raise TTSUnavailableError(
                    "pyttsx3 is not installed or cannot load its Windows speech driver"
                ) from exc
            except Exception as exc:
                raise TTSUnavailableError(
                    "the Windows text-to-speech engine could not be initialized"
                ) from exc
            self._engine = engine
            return engine

    @staticmethod
    def _is_wav(data: bytes) -> bool:
        return len(data) >= 44 and data.startswith(b"RIFF") and data[8:12] == b"WAVE"

    @classmethod
    def _has_audible_pcm(cls, data: bytes) -> bool:
        """Reject header-only and effectively silent PCM returned by SAPI."""

        if not cls._is_wav(data):
            return False
        try:
            with wave.open(BytesIO(data), "rb") as source:
                frame_rate = source.getframerate()
                frame_count = source.getnframes()
                sample_width = source.getsampwidth()
                if (
                    source.getcomptype() != "NONE"
                    or frame_rate <= 0
                    or sample_width not in {1, 2, 3, 4}
                    or frame_count < max(1, int(frame_rate * 0.02))
                ):
                    return False
                peak, full_scale = _pcm_peak(source.readframes(frame_count), sample_width)
        except (EOFError, OSError, wave.Error):
            return False
        # This threshold is intentionally far below normal speech. It catches
        # an empty/all-zero file without rejecting a legitimately quiet voice.
        return peak >= max(2, int(full_scale * 0.0001))

    def _select_compatible_voice(self, engine: Any, text: str) -> None:
        """Use an installed Arabic voice for Arabic text or fail explicitly."""

        if not _contains_arabic_letters(text):
            return
        try:
            voices = list(engine.getProperty("voices") or ())
            selected_id = str(engine.getProperty("voice") or "")
        except Exception:
            # Older/custom SAPI adapters may not expose voice metadata. In
            # that case waveform validation remains the final truth check.
            return
        selected = next(
            (voice for voice in voices if str(getattr(voice, "id", "")) == selected_id),
            None,
        )
        if selected is not None and _voice_supports_arabic(selected):
            return
        compatible = next((voice for voice in voices if _voice_supports_arabic(voice)), None)
        if compatible is not None and self.voice_id is None:
            engine.setProperty("voice", compatible.id)
            return
        if compatible is not None:
            raise TTSUnavailableError(
                "the configured Windows SAPI voice does not support Arabic speech"
            )
        if voices:
            raise TTSUnavailableError("no installed Windows SAPI voice supports Arabic speech")

    def probe(self) -> None:
        """Validate SAPI on the current thread, then release its COM engine.

        Windows SAPI objects are apartment-threaded.  Keeping an engine created
        during ASGI lifespan warm-up and later using it from a request worker
        can block indefinitely.  Neural models remain resident; this lightweight
        engine is deliberately recreated in the thread that synthesizes speech.
        """

        engine = self._load_engine()
        with self._lock:
            self._engine = None
            try:
                engine.stop()
            except Exception as exc:
                raise TTSUnavailableError(
                    "the Windows text-to-speech engine failed its startup probe"
                ) from exc

    def synthesize(self, text: str) -> SpeechResult:
        clean = validate_tts_text(text, self.max_text_characters)
        engine = self._load_engine()
        path: Path | None = None
        with self._lock:
            try:
                self._select_compatible_voice(engine, clean)
                descriptor, raw_path = tempfile.mkstemp(
                    prefix="phantom_tts_",
                    suffix=".wav",
                    dir=str(self.temporary_directory) if self.temporary_directory else None,
                )
                os.close(descriptor)
                path = Path(raw_path)
                engine.save_to_file(clean, str(path))
                engine.runAndWait()
                if not path.is_file():
                    raise TTSSynthesisError("the speech engine did not create a WAV file")
                size = path.stat().st_size
                if size > self.max_wav_bytes:
                    raise TTSSynthesisError(
                        "the synthesized WAV exceeded its configured size limit"
                    )
                data = path.read_bytes()
                if not self._is_wav(data):
                    raise TTSSynthesisError("the speech engine returned invalid WAV data")
                if not self._has_audible_pcm(data):
                    raise TTSSynthesisError("the speech engine returned no audible speech")
                return SpeechResult(
                    provider=self.info,
                    audio_wav=data,
                    mime_type="audio/wav",
                    audio_available=True,
                    text_only=False,
                )
            except (TTSSynthesisError, TTSUnavailableError):
                self._engine = None
                raise
            except Exception as exc:
                self._engine = None
                raise TTSSynthesisError("Windows text-to-speech synthesis failed") from exc
            finally:
                if path is not None:
                    with suppress(OSError):
                        path.unlink(missing_ok=True)
                with suppress(Exception):
                    engine.stop()
                if self._engine is engine:
                    self._engine = None

    def stop(self) -> None:
        with self._lock:
            engine = self._engine
            self._engine = None
            if engine is not None:
                try:
                    engine.stop()
                except Exception as exc:
                    raise TTSSynthesisError("the speech engine could not be stopped") from exc
