from __future__ import annotations

from pathlib import Path

import pytest

from phantom.config import load_config


def test_realtime_cpu_profile_bounds_local_decoder_latency() -> None:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "realtime_cpu.yaml"

    config = load_config(config_path)

    assert config.runtime.mode == "real"
    assert config.llm.backend == "transformers"
    assert config.llm.model_name == "Qwen/Qwen2.5-0.5B-Instruct"
    assert config.llm.quantization == "none"
    assert config.llm.max_new_tokens == 48
    assert config.llm.temperature == 0.0


def test_default_audio_gates_match_the_balanced_four_label_policy(tmp_path: Path) -> None:
    config_path = tmp_path / "defaults.yaml"
    config_path.write_text("{}\n", encoding="utf-8")

    config = load_config(config_path)

    assert config.audio.confidence_floor == 0.36
    assert config.audio.decision_margin_floor == 0.07
    assert config.audio.angry_confidence_floor == 0.62
    assert config.audio.angry_decision_margin_floor == 0.22


def test_neural_audio_backend_requires_an_explicit_checkpoint(tmp_path: Path) -> None:
    config_path = tmp_path / "missing-checkpoint.yaml"
    config_path.write_text("audio:\n  backend: crnn\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"audio\.checkpoint is required"):
        load_config(config_path)


def test_crnn_audio_configuration_is_loaded_without_enabling_random_weights(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "crnn.yaml"
    config_path.write_text(
        "audio:\n"
        "  backend: crnn\n"
        "  checkpoint: models/audio_crnn.pt\n"
        "  device: auto\n"
        "  n_mels: 64\n"
        "  max_frames: 1500\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.audio.backend == "crnn"
    assert config.audio.checkpoint == Path("models/audio_crnn.pt")
    assert config.audio.device == "auto"
    assert config.audio.n_mels == 64
    assert config.audio.max_frames == 1500


def test_audio_decision_gates_are_explicit_and_anger_is_more_conservative(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "audio-gates.yaml"
    config_path.write_text(
        "audio:\n"
        "  confidence_floor: 0.44\n"
        "  decision_margin_floor: 0.11\n"
        "  angry_confidence_floor: 0.63\n"
        "  angry_decision_margin_floor: 0.23\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.audio.confidence_floor == 0.44
    assert config.audio.decision_margin_floor == 0.11
    assert config.audio.angry_confidence_floor == 0.63
    assert config.audio.angry_decision_margin_floor == 0.23


@pytest.mark.parametrize(
    ("setting", "value", "message"),
    [
        ("confidence_floor", "1.1", "audio.confidence_floor"),
        ("decision_margin_floor", "-0.1", "audio.decision_margin_floor"),
        (
            "angry_confidence_floor",
            "0.2",
            "cannot be lower than audio.confidence_floor",
        ),
        (
            "angry_decision_margin_floor",
            "0.01",
            "cannot be lower than audio.decision_margin_floor",
        ),
    ],
)
def test_invalid_audio_decision_gates_fail_closed(
    tmp_path: Path, setting: str, value: str, message: str
) -> None:
    config_path = tmp_path / f"invalid-{setting}.yaml"
    config_path.write_text(f"audio:\n  {setting}: {value}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_config(config_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("backend", "transformer", "audio.backend"),
        ("device", "tpu", "audio.device"),
        ("n_mels", "4", "audio.n_mels"),
        ("max_frames", "10", "audio.max_frames"),
        ("checkpoint", "[]", "audio.checkpoint"),
    ],
)
def test_invalid_audio_configuration_fails_closed(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    config_path = tmp_path / f"invalid-{field}.yaml"
    config_path.write_text(
        f"audio:\n  backend: heuristic\n  {field}: {value}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_config(config_path)


def test_llm_dynamic_int8_configuration_is_explicit_and_cpu_only(tmp_path: Path) -> None:
    config_path = tmp_path / "dynamic-int8.yaml"
    config_path.write_text(
        "llm:\n  backend: transformers\n  device: cpu\n  quantization: dynamic-int8\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.llm.quantization == "dynamic-int8"


@pytest.mark.parametrize(
    ("backend", "device", "quantization", "message"),
    [
        ("transformers", "cpu", "int8", "llm.quantization must be"),
        ("ollama", "cpu", "dynamic-int8", "only by the transformers backend"),
        ("transformers", "cuda", "dynamic-int8", "requires llm.device cpu"),
    ],
)
def test_invalid_llm_quantization_fails_closed(
    tmp_path: Path,
    backend: str,
    device: str,
    quantization: str,
    message: str,
) -> None:
    config_path = tmp_path / f"invalid-llm-{backend}-{device}-{quantization}.yaml"
    config_path.write_text(
        f"llm:\n  backend: {backend}\n  device: {device}\n  quantization: {quantization}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_config(config_path)


def test_default_fusion_weights_prioritize_the_users_words(tmp_path: Path) -> None:
    config = load_config(tmp_path / "not-present.yaml")

    assert config.fusion.weights == {"audio": 0.8, "vision": 0.6, "text": 1.6}
