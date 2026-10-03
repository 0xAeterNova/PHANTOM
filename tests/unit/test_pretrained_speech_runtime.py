from __future__ import annotations

import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from phantom.audio.inference import FasterWhisperSpeechToTextAdapter
from phantom.audio.model import HuggingFaceSpeechEmotionAdapter
from phantom.audio.preprocessing import AudioBuffer
from phantom.schemas import EmotionLabel


def test_faster_whisper_is_lazy_reused_and_uses_vad_for_arabic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loads: list[tuple[str, dict[str, Any]]] = []
    calls: list[dict[str, Any]] = []

    class FakeWhisperModel:
        def __init__(self, name: str, **options: Any) -> None:
            loads.append((name, options))

        def transcribe(self, samples: Any, **options: Any) -> tuple[list[Any], object]:
            calls.append({"samples": samples, **options})
            return [SimpleNamespace(text="  مرحباً  "), SimpleNamespace(text="بك")], object()

    fake_module = ModuleType("faster_whisper")
    fake_module.WhisperModel = FakeWhisperModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)

    adapter = FasterWhisperSpeechToTextAdapter(language="AR")
    assert loads == []
    assert adapter.provenance == {
        "backend": "faster-whisper",
        "model_name_or_path": "base",
        "device": "cpu",
        "compute_type": "int8",
        "language": "ar",
        "local_files_only": False,
    }

    audio = AudioBuffer(np.zeros(16_000, dtype=np.float32), 16_000)
    assert adapter.transcribe(audio) == "مرحباً بك"
    assert adapter.transcribe(audio) == "مرحباً بك"

    assert loads == [
        (
            "base",
            {"device": "cpu", "compute_type": "int8", "local_files_only": False},
        )
    ]
    assert len(calls) == 2
    assert all(call["vad_filter"] is True for call in calls)
    assert all(call["language"] == "ar" for call in calls)
    assert all(call["beam_size"] == 5 for call in calls)


def test_faster_whisper_returns_none_for_empty_transcript(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeWhisperModel:
        def __init__(self, name: str, **options: Any) -> None:
            del name, options

        def transcribe(self, samples: Any, **options: Any) -> tuple[list[Any], object]:
            del samples, options
            return [SimpleNamespace(text="   ")], object()

    fake_module = ModuleType("faster_whisper")
    fake_module.WhisperModel = FakeWhisperModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)

    adapter = FasterWhisperSpeechToTextAdapter()
    assert adapter.transcribe(AudioBuffer(np.zeros(8_000, dtype=np.float32), 8_000)) is None


def test_faster_whisper_missing_dependency_fails_clearly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable_module(name: str) -> Any:
        assert name == "faster_whisper"
        raise ImportError("not installed")

    monkeypatch.setattr("phantom.audio.inference.import_module", unavailable_module)
    adapter = FasterWhisperSpeechToTextAdapter(local_files_only=True)
    with pytest.raises(RuntimeError, match="requires the optional 'faster-whisper' package"):
        adapter.transcribe(AudioBuffer(np.zeros(16_000, dtype=np.float32), 16_000))


def test_superb_emotion_aliases_and_provenance_are_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTensor:
        def __init__(self, values: list[float] | None = None) -> None:
            self.values = values

        def to(self, device: str) -> FakeTensor:
            del device
            return self

        def detach(self) -> FakeTensor:
            return self

        def cpu(self) -> FakeTensor:
            return self

        def tolist(self) -> list[float]:
            assert self.values is not None
            return self.values

    class FakeExtractor:
        def __call__(self, samples: Any, **options: Any) -> dict[str, FakeTensor]:
            del samples, options
            return {"input_values": FakeTensor()}

    class FakeModel:
        config = SimpleNamespace(id2label={0: "neu", 1: "hap", 2: "ang", 3: "sad"})

        def __call__(self, **inputs: Any) -> Any:
            del inputs
            return SimpleNamespace(logits=[FakeTensor([0.1, 0.2, 0.6, 0.1])])

    fake_torch = ModuleType("torch")
    fake_torch.inference_mode = nullcontext  # type: ignore[attr-defined]
    fake_torch.softmax = lambda tensor, dim: tensor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    adapter = HuggingFaceSpeechEmotionAdapter(
        "superb/wav2vec2-base-superb-er",
        revision="verified-revision",
        device="cpu",
    )
    monkeypatch.setattr(adapter, "_load", lambda: (FakeExtractor(), FakeModel()))

    probabilities = adapter.predict_waveform(np.zeros(16_000, dtype=np.float32), 16_000)
    assert max(probabilities, key=probabilities.get) is EmotionLabel.ANGRY
    assert probabilities[EmotionLabel.ANGRY] == pytest.approx(0.6)
    assert probabilities[EmotionLabel.FEARFUL] == 0.0
    assert adapter.provenance == {
        "backend": "huggingface-audio-classification",
        "model_name_or_path": "superb/wav2vec2-base-superb-er",
        "revision": "verified-revision",
        "device": "cpu",
        "local_files_only": True,
        "evaluation_dataset": "IEMOCAP four-class protocol",
        "source_language": "English",
        "source_labels": "neutral,happy,angry,sad",
        "arabic_validation_documented": False,
    }


def test_unmapped_speech_emotion_labels_raise_instead_of_falling_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTensor:
        def to(self, device: str) -> FakeTensor:
            del device
            return self

        def detach(self) -> FakeTensor:
            return self

        def cpu(self) -> FakeTensor:
            return self

        def tolist(self) -> list[float]:
            return [1.0]

    class FakeExtractor:
        def __call__(self, samples: Any, **options: Any) -> dict[str, FakeTensor]:
            del samples, options
            return {"input_values": FakeTensor()}

    class FakeModel:
        config = SimpleNamespace(id2label={0: "unsupported-label"})

        def __call__(self, **inputs: Any) -> Any:
            del inputs
            return SimpleNamespace(logits=[FakeTensor()])

    fake_torch = ModuleType("torch")
    fake_torch.inference_mode = nullcontext  # type: ignore[attr-defined]
    fake_torch.softmax = lambda tensor, dim: tensor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    adapter = HuggingFaceSpeechEmotionAdapter("local-model", revision="verified")
    monkeypatch.setattr(adapter, "_load", lambda: (FakeExtractor(), FakeModel()))
    with pytest.raises(RuntimeError, match="labels do not map"):
        adapter.predict_waveform(np.zeros(16_000, dtype=np.float32), 16_000)
