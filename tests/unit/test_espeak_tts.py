from __future__ import annotations

import struct
import subprocess
import wave
from io import BytesIO
from typing import Any

import pytest

from phantom.tts import EspeakNGTTS, TTSInputError, TTSSynthesisError, TTSUnavailableError


def _wav(*, audible: bool = True, streaming_header: bool = False) -> bytes:
    output = BytesIO()
    with wave.open(output, "wb") as speech:
        speech.setnchannels(1)
        speech.setsampwidth(2)
        speech.setframerate(16_000)
        speech.writeframes(struct.pack("<h", 2_000 if audible else 0) * 3_200)
    data = bytearray(output.getvalue())
    if streaming_header:
        data[4:8] = struct.pack("<I", 0x7FFFFFFF)
        data[40:44] = struct.pack("<I", 0x7FFFFFFF - 36)
    return bytes(data)


def _mock_engine(monkeypatch: pytest.MonkeyPatch, data: bytes) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr("phantom.tts.espeak_backend.shutil.which", lambda _: "/usr/bin/espeak-ng")

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        calls.append({"command": command, **kwargs})
        kwargs["stdout"].write(data)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("phantom.tts.espeak_backend.subprocess.run", run)
    return calls


def test_headless_speech_uses_stdin_and_normalizes_streaming_wav(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _mock_engine(monkeypatch, _wav(streaming_header=True))
    provider = EspeakNGTTS(rate=170)
    text = "--output /not-a-command; PHANTOM speech check."

    for _ in range(2):
        result = provider.synthesize(text)
        assert result.provider.name == "espeak-ng"
        assert result.provider.local is True
        assert result.audio_available and not result.text_only
        assert result.audio_wav is not None
        with wave.open(BytesIO(result.audio_wav), "rb") as speech:
            assert speech.getnframes() == 3_200
            assert speech.getframerate() == 16_000

    assert len(calls) == 2
    for call in calls:
        assert call["command"] == [
            "/usr/bin/espeak-ng",
            "--stdout",
            "--stdin",
            "-v",
            "en",
            "-s",
            "170",
        ]
        assert text not in call["command"]
        assert call["input"] == text.encode("utf-8")
        assert call["stderr"] == subprocess.DEVNULL
        assert call["check"] is True
        assert call["timeout"] == 30.0
        assert "shell" not in call


def test_missing_executable_remains_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("phantom.tts.espeak_backend.shutil.which", lambda _: None)

    with pytest.raises(TTSUnavailableError, match="espeak-ng package"):
        EspeakNGTTS().synthesize("Generated test sentence.")


@pytest.mark.parametrize("data", [b"not a WAV", _wav(audible=False), _wav()[:44]])
def test_invalid_or_silent_output_is_rejected(monkeypatch: pytest.MonkeyPatch, data: bytes) -> None:
    _mock_engine(monkeypatch, data)

    with pytest.raises(TTSSynthesisError, match="invalid or inaudible"):
        EspeakNGTTS().synthesize("Generated test sentence.")


def test_oversized_speech_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_engine(monkeypatch, _wav())

    with pytest.raises(TTSSynthesisError, match="size limit"):
        EspeakNGTTS(max_wav_bytes=100).synthesize("Generated test sentence.")


@pytest.mark.parametrize("failure", ["exit", "timeout"])
def test_engine_failure_does_not_expose_text_or_return_synthetic_speech(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    monkeypatch.setattr("phantom.tts.espeak_backend.shutil.which", lambda _: "/usr/bin/espeak-ng")

    def fail(command: list[str], **_kwargs: Any) -> None:
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 30)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr("phantom.tts.espeak_backend.subprocess.run", fail)
    text = "This generated sentence must not appear in an error."
    with pytest.raises(TTSSynthesisError) as error:
        EspeakNGTTS().synthesize(text)
    assert text not in str(error.value)


def test_empty_input_is_rejected_before_engine_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden_lookup(_executable: str) -> None:
        pytest.fail("invalid text must not reach the engine")

    monkeypatch.setattr("phantom.tts.espeak_backend.shutil.which", forbidden_lookup)
    with pytest.raises(TTSInputError):
        EspeakNGTTS().synthesize("  ")


def test_probe_generates_only_fixed_non_personal_text(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _mock_engine(monkeypatch, _wav())
    provider = EspeakNGTTS()
    provider.probe()
    provider.stop()
    assert calls[0]["input"] == b"PHANTOM local speech check."
