from __future__ import annotations

import hashlib
import math
import struct
import wave
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest

from phantom.config import load_config
from phantom.tts import (
    PiperArabicHybridTTS,
    Pyttsx3WindowsTTS,
    SpeechResult,
    TTSInfo,
    TTSMode,
    TTSSynthesisError,
    TTSUnavailableError,
    contains_arabic_letters,
)


def _wav_bytes(*, audible: bool = True) -> bytes:
    output = BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16_000)
        samples = (
            struct.pack("<h", round(4_000 * math.sin(2 * math.pi * 220 * frame / 16_000)))
            if audible
            else b"\x00\x00"
            for frame in range(3_200)
        )
        wav_file.writeframes(b"".join(samples))
    return output.getvalue()


class FakePiperVoice:
    def __init__(self, *, audible: bool = True) -> None:
        self.audible = audible
        self.texts: list[str] = []

    def synthesize_wav(self, text: str, wav_file: Any) -> None:
        self.texts.append(text)
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16_000)
        data = _wav_bytes(audible=self.audible)
        with wave.open(BytesIO(data), "rb") as source:
            wav_file.writeframes(source.readframes(source.getnframes()))


class FakeEnglishTTS(Pyttsx3WindowsTTS):
    def __init__(self) -> None:
        self.synthesized: list[str] = []
        self.probed = False
        self.stopped = False
        self._fake_info = TTSInfo("fake-english", TTSMode.REAL, voice="en-US")

    @property
    def info(self) -> TTSInfo:
        return self._fake_info

    def probe(self) -> None:
        self.probed = True

    def synthesize(self, text: str) -> SpeechResult:
        self.synthesized.append(text)
        return SpeechResult(self.info, _wav_bytes(), "audio/wav", True, False)

    def stop(self) -> None:
        self.stopped = True


def _artifact_pair(tmp_path: Path) -> tuple[Path, Path, str, str]:
    model = tmp_path / "voice.onnx"
    config = tmp_path / "voice.onnx.json"
    model.write_bytes(b"model fixture")
    config.write_bytes(b'{"fixture": true}')
    return (
        model,
        config,
        hashlib.sha256(model.read_bytes()).hexdigest(),
        hashlib.sha256(config.read_bytes()).hexdigest(),
    )


def _provider(
    tmp_path: Path,
    voice: FakePiperVoice,
    english: FakeEnglishTTS,
) -> PiperArabicHybridTTS:
    model, config, model_hash, config_hash = _artifact_pair(tmp_path)
    return PiperArabicHybridTTS(
        model,
        config,
        model_sha256=model_hash,
        config_sha256=config_hash,
        voice_loader=lambda _model, _config: voice,
        english_provider=english,
    )


def test_arabic_letter_detection_ignores_english_and_punctuation() -> None:
    assert contains_arabic_letters("مرحباً")
    assert contains_arabic_letters("Project PHANTOM يعمل محلياً")
    assert not contains_arabic_letters("Project PHANTOM works locally.")
    assert not contains_arabic_letters("، 123 !")


def test_hybrid_routes_arabic_and_mixed_to_piper_and_english_to_sapi(
    tmp_path: Path,
) -> None:
    piper = FakePiperVoice()
    english = FakeEnglishTTS()
    provider = _provider(tmp_path, piper, english)

    arabic = provider.synthesize("مرحباً بك")
    mixed = provider.synthesize("Project PHANTOM يعمل محلياً")
    english_result = provider.synthesize("Project PHANTOM works locally")

    assert piper.texts == ["مرحباً بك", "Project PHANTOM يعمل محلياً"]
    assert english.synthesized == ["Project PHANTOM works locally"]
    assert arabic.audio_available and mixed.audio_available and english_result.audio_available
    assert arabic.provider.name == "piper-arabic+pyttsx3-windows"
    assert english_result.provider.name == "piper-arabic+pyttsx3-windows"


def test_probe_loads_piper_once_and_checks_english_route(tmp_path: Path) -> None:
    piper = FakePiperVoice()
    english = FakeEnglishTTS()
    load_count = 0
    model, config, model_hash, config_hash = _artifact_pair(tmp_path)

    def loader(_model: Path, _config: Path) -> FakePiperVoice:
        nonlocal load_count
        load_count += 1
        return piper

    provider = PiperArabicHybridTTS(
        model,
        config,
        model_sha256=model_hash,
        config_sha256=config_hash,
        voice_loader=loader,
        english_provider=english,
    )

    provider.probe()
    provider.probe()

    assert load_count == 1
    assert english.probed


def test_tampered_model_fails_closed_before_piper_loader(tmp_path: Path) -> None:
    model, config, model_hash, config_hash = _artifact_pair(tmp_path)
    model.write_bytes(b"tampered")
    loaded = False

    def loader(_model: Path, _config: Path) -> FakePiperVoice:
        nonlocal loaded
        loaded = True
        return FakePiperVoice()

    provider = PiperArabicHybridTTS(
        model,
        config,
        model_sha256=model_hash,
        config_sha256=config_hash,
        voice_loader=loader,
        english_provider=FakeEnglishTTS(),
    )

    with pytest.raises(TTSUnavailableError, match="SHA-256 integrity check"):
        provider.synthesize("مرحباً")

    assert not loaded


def test_silent_piper_output_is_rejected(tmp_path: Path) -> None:
    provider = _provider(tmp_path, FakePiperVoice(audible=False), FakeEnglishTTS())

    with pytest.raises(TTSSynthesisError, match="inaudible WAV"):
        provider.synthesize("هذا صوت صامت")


def test_piper_hybrid_configuration_loads_explicit_artifact_contract(tmp_path: Path) -> None:
    config_path = tmp_path / "piper.yaml"
    config_path.write_text(
        "tts:\n"
        "  backend: piper-hybrid\n"
        "  piper_model_path: custom/voice.onnx\n"
        "  piper_config_path: custom/voice.onnx.json\n"
        f"  piper_model_sha256: {'a' * 64}\n"
        f"  piper_config_sha256: {'b' * 64}\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.tts.backend == "piper-hybrid"
    assert config.tts.piper_model_path == Path("custom/voice.onnx")
    assert config.tts.piper_config_path == Path("custom/voice.onnx.json")
    assert config.tts.piper_model_sha256 == "a" * 64
    assert config.tts.piper_config_sha256 == "b" * 64
