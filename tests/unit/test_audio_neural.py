from __future__ import annotations

import csv
import io
import json
import math
import sys
import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from phantom.audio.inference import AudioAnalyzer
from phantom.config import AudioModelConfig
from phantom.schemas import FUSION_EMOTIONS
from scripts import export_models, train_audio

torch = pytest.importorskip(
    "torch",
    reason="spectrogram CNN/CRNN tests require the optional ML dependency set",
)
torch_models = pytest.importorskip("phantom.audio.torch_models")
neural = pytest.importorskip("phantom.audio.neural")
training = pytest.importorskip("phantom.audio.training")

build_spectrogram_model = torch_models.build_spectrogram_model
pooled_frame_lengths = torch_models.pooled_frame_lengths
SpectrogramEmotionAdapter = neural.SpectrogramEmotionAdapter
AudioManifestEntry = training.AudioManifestEntry
DatasetSplit = training.DatasetSplit
LogMelConfig = training.LogMelConfig
SpectrogramDataset = training.SpectrogramDataset
speaker_independent_split = training.speaker_independent_split
write_split_manifest = training.write_split_manifest
resolve_torch_device = neural.resolve_torch_device

pytestmark = pytest.mark.optional_ml


def test_speaker_split_is_deterministic_and_has_no_identity_leakage() -> None:
    entries = tuple(
        AudioManifestEntry(
            sample_id=f"speaker-{speaker}-{label.value}",
            path=Path(f"speaker-{speaker}-{label.value}.wav"),
            manifest_path=f"speaker-{speaker}-{label.value}.wav",
            label=label.value,
            speaker_id=f"speaker-{speaker}",
        )
        for speaker in range(4)
        for label in FUSION_EMOTIONS
    )

    first = speaker_independent_split(entries, seed=17)
    second = speaker_independent_split(entries, seed=17)
    speaker_sets = [
        {entry.speaker_id for entry in part} for part in (first.train, first.validation, first.test)
    ]

    assert [entry.sample_id for entry in first.train] == [entry.sample_id for entry in second.train]
    assert speaker_sets[0].isdisjoint(speaker_sets[1])
    assert speaker_sets[0].isdisjoint(speaker_sets[2])
    assert speaker_sets[1].isdisjoint(speaker_sets[2])
    assert {entry.label for entry in first.train} == {label.value for label in FUSION_EMOTIONS}


def test_frozen_split_paths_are_relative_to_their_output_manifest(tmp_path: Path) -> None:
    source_directory = tmp_path / "source"
    source_directory.mkdir()
    entries = []
    for index, split_name in enumerate(("train", "validation", "test")):
        audio_path = source_directory / f"{split_name}.wav"
        audio_path.write_bytes(b"synthetic-placeholder")
        entries.append(
            AudioManifestEntry(
                sample_id=split_name,
                path=audio_path.resolve(),
                manifest_path=audio_path.name,
                label=FUSION_EMOTIONS[index].value,
                speaker_id=f"speaker-{index}",
            )
        )
    split = DatasetSplit((entries[0],), (entries[1],), (entries[2],))
    output = tmp_path / "exports" / "split.csv"

    write_split_manifest(output, split)

    with output.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_id = {entry.sample_id: entry.path for entry in entries}
    for row in rows:
        assert (output.parent / row["path"]).resolve() == by_id[row["sample_id"]]


@pytest.mark.parametrize("architecture", ["cnn", "crnn"])
def test_spectrogram_models_return_finite_class_logits(architecture: str) -> None:
    torch.manual_seed(7)
    model = build_spectrogram_model(
        architecture,
        n_mels=40,
        num_classes=len(FUSION_EMOTIONS),
    ).eval()
    spectrograms = torch.randn(2, 1, 40, 64)
    lengths = torch.tensor([64, 41], dtype=torch.long)

    with torch.inference_mode():
        logits = model(spectrograms, lengths)

    assert tuple(logits.shape) == (2, len(FUSION_EMOTIONS))
    assert bool(torch.isfinite(logits).all())


def test_crnn_contains_a_bidirectional_recurrent_layer() -> None:
    model = build_spectrogram_model(
        "crnn",
        n_mels=40,
        num_classes=len(FUSION_EMOTIONS),
    )

    recurrent_layers = [
        module for module in model.modules() if isinstance(module, (torch.nn.GRU, torch.nn.LSTM))
    ]

    assert recurrent_layers
    assert all(layer.bidirectional for layer in recurrent_layers)


def test_pooled_lengths_follow_two_ceil_mode_time_pools() -> None:
    lengths = torch.tensor([1, 2, 3, 4, 5, 8, 9, 64], dtype=torch.long)

    pooled = pooled_frame_lengths(lengths)

    assert pooled.tolist() == [1, 1, 1, 1, 2, 2, 3, 16]


def test_automatic_device_selection_falls_back_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if hasattr(torch.backends, "mps"):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    assert str(resolve_torch_device("auto")) == "cpu"


@pytest.mark.parametrize("architecture", ["cnn", "crnn"])
def test_right_padding_does_not_change_clip_logits(architecture: str) -> None:
    torch.manual_seed(11)
    model = build_spectrogram_model(
        architecture,
        n_mels=40,
        num_classes=len(FUSION_EMOTIONS),
    ).eval()
    clip = torch.randn(1, 1, 40, 37)
    padded = torch.nn.functional.pad(clip, (0, 24))
    lengths = torch.tensor([37], dtype=torch.long)

    with torch.inference_mode():
        unpadded_logits = model(clip, lengths)
        padded_logits = model(padded, lengths)

    assert torch.allclose(unpadded_logits, padded_logits, atol=1e-5, rtol=1e-5)


def _checkpoint_payload(architecture: str = "crnn") -> dict[str, Any]:
    model = build_spectrogram_model(
        architecture,
        n_mels=40,
        num_classes=len(FUSION_EMOTIONS),
    )
    return {
        "format_version": 1,
        "architecture": architecture,
        "model_config": {},
        "feature_config": {
            "sample_rate": 16_000,
            "n_mels": 40,
            "n_fft": 512,
            "frame_ms": 25.0,
            "hop_ms": 10.0,
            "f_min": 20.0,
            "f_max": 8_000.0,
        },
        "labels": [label.value for label in FUSION_EMOTIONS],
        "normalization": {"mean": [0.0] * 40, "std": [1.0] * 40},
        "temperature": 1.0,
        "state_dict": model.state_dict(),
        "training": {"pretrained": False, "held_out_metrics_reported": False},
    }


def _write_checkpoint(path: Path, architecture: str = "crnn") -> None:
    torch.save(_checkpoint_payload(architecture), path)


def _wav_bytes(samples: np.ndarray, sample_rate: int = 16_000) -> bytes:
    buffer = io.BytesIO()
    encoded = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(encoded)
    return buffer.getvalue()


def test_training_crop_is_independent_of_model_initialization_rng(tmp_path: Path) -> None:
    sample_rate = 16_000
    time = np.arange(round(0.6 * sample_rate), dtype=np.float32) / sample_rate
    samples = (0.25 * np.sin(2 * math.pi * 220 * time)).astype(np.float32)
    audio_path = tmp_path / "long.wav"
    audio_path.write_bytes(_wav_bytes(samples))
    entry = AudioManifestEntry(
        sample_id="stable-sample",
        path=audio_path,
        manifest_path=audio_path.name,
        label="neutral",
        speaker_id="speaker-1",
    )
    dataset = SpectrogramDataset(
        (entry,),
        LogMelConfig(),
        10,
        np.zeros(40, dtype=np.float32),
        np.ones(40, dtype=np.float32),
        training=True,
        seed=23,
    )

    first = dataset[0][0]
    build_spectrogram_model("cnn", n_mels=40, num_classes=len(FUSION_EMOTIONS))
    build_spectrogram_model("crnn", n_mels=40, num_classes=len(FUSION_EMOTIONS))
    second = dataset[0][0]

    assert torch.equal(first, second)


def test_training_predictions_apply_declared_classifier_abstention(tmp_path: Path) -> None:
    output = tmp_path / "predictions.csv"
    evaluation = SimpleNamespace(
        logits=torch.tensor([[4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        targets=torch.tensor([0]),
        sample_ids=("sample-1",),
        speaker_ids=("speaker-1",),
    )

    train_audio._write_predictions(
        output,
        evaluation,
        temperature=1.0,
        labels=[label.value for label in FUSION_EMOTIONS],
        abstention_threshold=0.95,
    )

    with output.open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["raw_predicted_label"] == "neutral"
    assert row["predicted_label"] == "uncertain"
    assert row["abstained"] == "True"
    assert float(row["abstention_threshold"]) == pytest.approx(0.95)


def test_adapter_loads_checkpoint_on_cpu_and_returns_canonical_distribution(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "audio_crnn.pt"
    _write_checkpoint(checkpoint)
    adapter = SpectrogramEmotionAdapter(checkpoint, device="cpu")
    features = np.random.default_rng(13).normal(size=(48, 40)).astype(np.float32)

    probabilities = adapter.predict(features)

    assert adapter.architecture == "crnn"
    assert adapter.n_mels == 40
    assert adapter.device == "cpu"
    assert {parameter.device.type for parameter in adapter.model.parameters()} == {"cpu"}
    assert set(probabilities) == set(FUSION_EMOTIONS)
    assert all(np.isfinite(value) and 0.0 <= value <= 1.0 for value in probabilities.values())
    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_direct_adapter_composition_uses_the_checkpoint_temperature(tmp_path: Path) -> None:
    checkpoint = tmp_path / "calibrated_crnn.pt"
    payload = _checkpoint_payload()
    payload["temperature"] = 2.25
    torch.save(payload, checkpoint)

    adapter = SpectrogramEmotionAdapter(checkpoint, device="cpu")
    analyzer = AudioAnalyzer(model=adapter)

    assert analyzer.calibrator.temperature == pytest.approx(2.25)


def test_adapter_rejects_an_unexpected_checkpoint_architecture(tmp_path: Path) -> None:
    checkpoint = tmp_path / "audio_cnn.pt"
    _write_checkpoint(checkpoint, "cnn")

    with pytest.raises(ValueError, match="architecture"):
        SpectrogramEmotionAdapter(
            checkpoint,
            device="cpu",
            expected_architecture="crnn",
        )


def test_export_manifest_safely_reconstructs_a_spectrogram_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "audio_crnn.pt"
    output = tmp_path / "audio_crnn.export.json"
    _write_checkpoint(checkpoint)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_models.py",
            str(checkpoint),
            "--validate-spectrogram",
            "--output",
            str(output),
        ],
    )

    assert export_models.main() == 0
    manifest = json.loads(output.read_text(encoding="utf-8"))
    validation = manifest["spectrogram_validation"]
    assert validation["architecture"] == "crnn"
    assert validation["labels"] == [label.value for label in FUSION_EMOTIONS]
    assert validation["safe_weights_only_load"] is True
    assert validation["strict_state_reconstruction"] is True


def test_spectrogram_export_reports_a_clear_missing_dependency_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "audio_crnn.pt"
    checkpoint.write_bytes(b"placeholder")
    monkeypatch.setattr(
        neural,
        "_require_torch",
        lambda: (_ for _ in ()).throw(RuntimeError("PyTorch is unavailable")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["export_models.py", str(checkpoint), "--validate-spectrogram"],
    )

    with pytest.raises(SystemExit, match="audio-ml"):
        export_models.main()


def test_adapter_rejects_noncanonical_checkpoint_labels(tmp_path: Path) -> None:
    checkpoint = tmp_path / "bad_labels.pt"
    payload = _checkpoint_payload()
    payload["labels"] = list(reversed(payload["labels"]))
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match=r"labels|label order"):
        SpectrogramEmotionAdapter(checkpoint, device="cpu")


@pytest.mark.parametrize(
    ("model_config", "message"),
    [
        ({"channels": [32, 64, 1_000_000]}, "channel counts"),
        ({"projection_size": 100_000}, "projection_size"),
        ({"recurrent_hidden_size": 100_000}, "recurrent_hidden_size"),
        ({"recurrent_layers": 100}, "recurrent_layers"),
        ({"recurrent_type": 7}, "recurrent_type"),
    ],
)
def test_adapter_rejects_unbounded_or_mistyped_model_configuration_before_allocation(
    tmp_path: Path,
    model_config: dict[str, Any],
    message: str,
) -> None:
    checkpoint = tmp_path / "unsafe_config.pt"
    payload = _checkpoint_payload()
    payload["model_config"] = model_config
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match=message):
        SpectrogramEmotionAdapter(checkpoint, device="cpu")


def test_adapter_rejects_nonfinite_checkpoint_weights_at_startup(tmp_path: Path) -> None:
    checkpoint = tmp_path / "nan_weights.pt"
    payload = _checkpoint_payload()
    first_name = next(iter(payload["state_dict"]))
    corrupted = payload["state_dict"][first_name].clone()
    corrupted.reshape(-1)[0] = float("nan")
    payload["state_dict"][first_name] = corrupted
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="non-finite"):
        SpectrogramEmotionAdapter(checkpoint, device="cpu")


def test_adapter_rejects_finite_weights_that_would_overflow_during_dtype_cast(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "overflow_weights.pt"
    payload = _checkpoint_payload()
    first_name = next(iter(payload["state_dict"]))
    original = payload["state_dict"][first_name]
    payload["state_dict"][first_name] = torch.full(
        original.shape,
        1e100,
        dtype=torch.float64,
    )
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="dtype"):
        SpectrogramEmotionAdapter(checkpoint, device="cpu")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("mean", [1e100] * 40, "means exceed"),
        ("std", [1e-100] * 40, "standard deviations"),
    ],
)
def test_adapter_rejects_normalization_that_is_not_float32_safe(
    tmp_path: Path,
    field: str,
    value: list[float],
    message: str,
) -> None:
    checkpoint = tmp_path / "unsafe_normalization.pt"
    payload = _checkpoint_payload()
    payload["normalization"][field] = value
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match=message):
        SpectrogramEmotionAdapter(checkpoint, device="cpu")


def test_configured_crnn_analyzer_honors_the_checkpoint_training_frame_cap(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "bounded_crnn.pt"
    payload = _checkpoint_payload()
    payload["feature_config"]["max_frames"] = 32
    torch.save(payload, checkpoint)
    analyzer = AudioAnalyzer.from_model_config(
        AudioModelConfig(
            backend="crnn",
            checkpoint=checkpoint,
            device="cpu",
            n_mels=40,
            max_frames=80,
        )
    )
    sample_rate = 16_000
    time = np.arange(round(0.6 * sample_rate), dtype=np.float32) / sample_rate
    envelope = 0.55 + 0.35 * np.sin(2 * math.pi * 3 * time)
    samples = (0.30 * envelope * np.sin(2 * math.pi * 220 * time)).astype(np.float32)

    result = analyzer.analyze_bytes(_wav_bytes(samples), "tone.wav", "audio/wav")

    assert analyzer.max_feature_frames == 32
    assert result.metadata["model_backend"] == "crnn"
    assert result.metadata["feature_frames"] == 32
    assert result.metadata["feature_truncated"] is True
    assert result.metadata["feature_crop_start"] > 0


@pytest.mark.parametrize(
    "features",
    [
        np.zeros(40, dtype=np.float32),
        np.empty((0, 40), dtype=np.float32),
        np.zeros((10, 39), dtype=np.float32),
        np.full((10, 40), np.nan, dtype=np.float32),
    ],
)
def test_adapter_rejects_invalid_spectrograms(tmp_path: Path, features: np.ndarray) -> None:
    checkpoint = tmp_path / "audio_crnn.pt"
    _write_checkpoint(checkpoint)
    adapter = SpectrogramEmotionAdapter(checkpoint, device="cpu")

    with pytest.raises(ValueError):
        adapter.predict(features)
