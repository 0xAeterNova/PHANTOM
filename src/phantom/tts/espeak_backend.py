"""Headless local eSpeak NG speech output for Linux, without audio-device access."""

from __future__ import annotations

import shutil

# Run local synthesis without a shell or an audio playback device.
import subprocess  # nosec B404
import tempfile
import wave
from io import BytesIO

from phantom.tts.contracts import (
    SpeechResult,
    TTSInfo,
    TTSMode,
    TTSSynthesisError,
    TTSUnavailableError,
    validate_tts_text,
)
from phantom.tts.pyttsx3_backend import Pyttsx3WindowsTTS


class EspeakNGTTS:
    """Generate bounded PCM WAV bytes using the installed eSpeak NG executable."""

    def __init__(
        self,
        *,
        voice_id: str | None = None,
        rate: int = 175,
        max_text_characters: int = 8_000,
        max_wav_bytes: int = 20 * 1024 * 1024,
        timeout_seconds: float = 30.0,
    ) -> None:
        if not 80 <= rate <= 400:
            raise ValueError("eSpeak rate must be between 80 and 400")
        if not 128 <= max_text_characters <= 20_000:
            raise ValueError("max_text_characters must be between 128 and 20000")
        if not 44 <= max_wav_bytes <= 100 * 1024 * 1024:
            raise ValueError("max_wav_bytes must be between 44 bytes and 100 MiB")
        if not 1.0 <= timeout_seconds <= 120.0:
            raise ValueError("timeout_seconds must be between 1 and 120")
        self.voice_id = voice_id or "en"
        self.rate = rate
        self.max_text_characters = max_text_characters
        self.max_wav_bytes = max_wav_bytes
        self.timeout_seconds = timeout_seconds
        self._info = TTSInfo("espeak-ng", TTSMode.REAL, voice=self.voice_id, local=True)

    @property
    def info(self) -> TTSInfo:
        return self._info

    def probe(self) -> None:
        """Check executable, voice, and WAV generation with non-personal text."""

        self.synthesize("PHANTOM local speech check.")

    def synthesize(self, text: str) -> SpeechResult:
        clean = validate_tts_text(text, self.max_text_characters)
        executable = shutil.which("espeak-ng")
        if executable is None:
            raise TTSUnavailableError("local English speech requires the espeak-ng package")
        try:
            # Text travels through stdin, never shell arguments or logs. Output
            # uses a deleted temporary file (Compose /tmp is memory-backed), so
            # oversized output is rejected before it is read into Python memory.
            with tempfile.TemporaryFile() as output:
                subprocess.run(  # nosec B603
                    [
                        executable,
                        "--stdout",
                        "--stdin",
                        "-v",
                        self.voice_id,
                        "-s",
                        str(self.rate),
                    ],
                    input=clean.encode("utf-8"),
                    stdout=output,
                    stderr=subprocess.DEVNULL,
                    check=True,
                    timeout=self.timeout_seconds,
                )
                if output.tell() > self.max_wav_bytes:
                    raise TTSSynthesisError(
                        "the synthesized WAV exceeded its configured size limit"
                    )
                output.seek(0)
                data = output.read(self.max_wav_bytes + 1)
        except (OSError, subprocess.SubprocessError) as exc:
            raise TTSSynthesisError("local eSpeak NG speech synthesis failed") from exc

        if not Pyttsx3WindowsTTS._has_audible_pcm(data):
            raise TTSSynthesisError("the eSpeak NG engine returned invalid or inaudible WAV data")
        # eSpeak's stdout header has placeholder lengths. Rewrite the header so
        # browsers see the actual duration and can play repeated responses.
        normalized = BytesIO()
        with wave.open(BytesIO(data), "rb") as source, wave.open(normalized, "wb") as target:
            target.setnchannels(source.getnchannels())
            target.setsampwidth(source.getsampwidth())
            target.setframerate(source.getframerate())
            target.writeframes(source.readframes(source.getnframes()))
        return SpeechResult(self.info, normalized.getvalue(), "audio/wav", True, False)

    def stop(self) -> None:
        # This adapter does not play sound or keep a speech engine between calls.
        return None
