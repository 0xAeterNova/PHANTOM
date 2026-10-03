from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

import pytest

from phantom.tts import (
    Pyttsx3WindowsTTS,
    TextOnlyTTS,
    TTSInputError,
    TTSMode,
    TTSSynthesisError,
    TTSUnavailableError,
    UnavailableTTS,
)


@dataclass
class FakeVoice:
    id: str
    languages: list[str | bytes]
    name: str = "Test voice"


class FakeEngine:
    def __init__(
        self,
        *,
        fail: bool = False,
        silent: bool = False,
        voices: list[FakeVoice] | None = None,
        selected_voice: str | None = None,
    ) -> None:
        self.fail = fail
        self.silent = silent
        self.voices = voices or []
        self.selected_voice = selected_voice or (self.voices[0].id if self.voices else None)
        self.properties: dict[str, object] = {}
        self.pending_path: Path | None = None
        self.pending_text: str | None = None
        self.stop_calls = 0

    def setProperty(self, name: str, value: object) -> None:
        self.properties[name] = value
        if name == "voice":
            self.selected_voice = str(value)

    def getProperty(self, name: str) -> object:
        if name == "voices":
            return self.voices
        if name == "voice":
            return self.selected_voice
        return self.properties.get(name)

    def save_to_file(self, text: str, path: str) -> None:
        self.pending_text = text
        self.pending_path = Path(path)

    def runAndWait(self) -> None:
        if self.fail:
            raise RuntimeError("synthetic engine failure")
        assert self.pending_path is not None
        with wave.open(str(self.pending_path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16_000)
            sample = b"\x00\x00" if self.silent else b"\xe8\x03"
            output.writeframes(sample * 1_600)

    def stop(self) -> None:
        self.stop_calls += 1


def test_text_only_and_unavailable_backends_are_explicit() -> None:
    text_only = TextOnlyTTS().synthesize("Visible assistant response")
    assert text_only.provider.mode is TTSMode.TEXT_ONLY
    assert text_only.audio_available is False
    assert text_only.text_only is True
    assert text_only.audio_wav is None

    unavailable = UnavailableTTS("speech engine missing").synthesize("Visible assistant response")
    assert unavailable.provider.mode is TTSMode.UNAVAILABLE
    assert unavailable.audio_available is False
    assert unavailable.reason == "speech engine missing"


def test_pyttsx3_windows_backend_returns_wav_bytes_recreates_engine_and_cleans_temp(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("phantom.tts.pyttsx3_backend.platform.system", lambda: "Windows")
    engines: list[FakeEngine] = []
    factory_calls = 0

    def factory() -> FakeEngine:
        nonlocal factory_calls
        factory_calls += 1
        engine = FakeEngine()
        engines.append(engine)
        return engine

    provider = Pyttsx3WindowsTTS(
        engine_factory=factory,
        temporary_directory=tmp_path,
        rate=180,
        volume=0.8,
    )
    first = provider.synthesize("First response")
    second = provider.synthesize("Second response")

    assert factory_calls == 2
    assert first.audio_available is True
    assert first.audio_wav is not None and first.audio_wav.startswith(b"RIFF")
    assert first.audio_wav[8:12] == b"WAVE"
    assert second.provider.mode is TTSMode.REAL
    assert engines[0].properties["rate"] == 180
    assert engines[1].properties["volume"] == 0.8
    assert [engine.stop_calls for engine in engines] == [1, 1]
    assert list(tmp_path.glob("phantom_tts_*.wav")) == []

    provider.stop()
    assert [engine.stop_calls for engine in engines] == [1, 1]


def test_pyttsx3_startup_probe_releases_thread_bound_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("phantom.tts.pyttsx3_backend.platform.system", lambda: "Windows")
    engine = FakeEngine()
    provider = Pyttsx3WindowsTTS(engine_factory=lambda: engine)

    provider.probe()

    assert engine.stop_calls == 1
    assert provider._engine is None


def test_pyttsx3_failure_removes_temp_and_does_not_echo_text(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("phantom.tts.pyttsx3_backend.platform.system", lambda: "Windows")
    private_text = "PRIVATE_ASSISTANT_RESPONSE"
    provider = Pyttsx3WindowsTTS(
        engine_factory=lambda: FakeEngine(fail=True),
        temporary_directory=tmp_path,
    )

    with pytest.raises(TTSSynthesisError) as captured:
        provider.synthesize(private_text)
    assert private_text not in str(captured.value)
    assert list(tmp_path.glob("phantom_tts_*.wav")) == []


def test_pyttsx3_rejects_structurally_valid_but_silent_wav(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("phantom.tts.pyttsx3_backend.platform.system", lambda: "Windows")
    provider = Pyttsx3WindowsTTS(
        engine_factory=lambda: FakeEngine(silent=True),
        temporary_directory=tmp_path,
    )

    with pytest.raises(TTSSynthesisError, match="no audible speech"):
        provider.synthesize("Visible assistant response")
    assert list(tmp_path.glob("phantom_tts_*.wav")) == []


def test_pyttsx3_selects_installed_arabic_voice_for_arabic_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("phantom.tts.pyttsx3_backend.platform.system", lambda: "Windows")
    engine = FakeEngine(
        voices=[
            FakeVoice("english", ["en-US"]),
            FakeVoice("arabic", [b"\x05ar-SA"]),
        ],
        selected_voice="english",
    )
    provider = Pyttsx3WindowsTTS(engine_factory=lambda: engine)

    result = provider.synthesize("أفهم شعورك")

    assert result.audio_available is True
    assert engine.properties["voice"] == "arabic"


def test_pyttsx3_reports_missing_arabic_voice_instead_of_empty_wav(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("phantom.tts.pyttsx3_backend.platform.system", lambda: "Windows")
    engine = FakeEngine(
        voices=[FakeVoice("english", ["en-US"], "English test voice")],
        selected_voice="english",
    )
    provider = Pyttsx3WindowsTTS(engine_factory=lambda: engine)

    with pytest.raises(TTSUnavailableError, match="no installed Windows SAPI voice"):
        provider.synthesize("أفهم شعورك")


def test_pyttsx3_is_explicitly_unavailable_outside_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("phantom.tts.pyttsx3_backend.platform.system", lambda: "Linux")
    provider = Pyttsx3WindowsTTS(engine_factory=FakeEngine)
    with pytest.raises(TTSUnavailableError, match="requires Windows"):
        provider.synthesize("Hello")


def test_tts_input_is_bounded() -> None:
    provider = Pyttsx3WindowsTTS(max_text_characters=128, engine_factory=FakeEngine)
    with pytest.raises(TTSInputError, match="128-character"):
        provider.synthesize("x" * 129)
