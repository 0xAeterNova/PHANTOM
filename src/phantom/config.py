"""YAML and environment-backed runtime configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class FusionConfig:
    weights: dict[str, float] = field(
        default_factory=lambda: {"audio": 0.8, "vision": 0.6, "text": 1.6}
    )
    confidence_threshold: float = 0.45
    contradiction_threshold: float = 0.65
    quality_floor: float = 0.2


@dataclass(frozen=True, slots=True)
class SafetyConfig:
    country_code: str = "GLOBAL"
    emergency_message: str = (
        "If you may act on these thoughts or are in immediate danger, contact local emergency "
        "services now or ask a trusted nearby person to stay with you."
    )
    resources: tuple[str, ...] = (
        "Use a verified local crisis directory or your country's emergency service.",
    )


@dataclass(frozen=True, slots=True)
class AudioModelConfig:
    """Local acoustic-model selection.

    The neural backends are deliberately opt-in: PHANTOM never enables random
    weights and requires an explicit, locally trained checkpoint.
    """

    backend: str = "heuristic"
    checkpoint: Path | None = None
    device: str = "cpu"
    n_mels: int = 40
    max_frames: int = 3_000
    model_name: str = "superb/wav2vec2-base-superb-er"
    revision: str = "main"
    local_files_only: bool = False
    confidence_floor: float = 0.36
    decision_margin_floor: float = 0.07
    angry_confidence_floor: float = 0.62
    angry_decision_margin_floor: float = 0.22


@dataclass(frozen=True, slots=True)
class SpeechToTextConfig:
    backend: str = "none"
    model_name: str = "base"
    device: str = "cpu"
    compute_type: str = "int8"
    language: str | None = "ar"
    beam_size: int = 3


@dataclass(frozen=True, slots=True)
class VisionModelConfig:
    backend: str = "baseline"
    detector: str = "opencv-haar"
    analyze_interval_seconds: float = 1.25
    age_interval_seconds: float = 20.0
    max_staleness_seconds: float = 3.0


@dataclass(frozen=True, slots=True)
class LLMConfig:
    backend: str = "demo"
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    revision: str = "main"
    device: str = "cpu"
    quantization: str = "none"
    base_url: str = "http://127.0.0.1:11434"
    api_key_env: str = "PHANTOM_LLM_API_KEY"
    max_new_tokens: int = 180
    temperature: float = 0.55
    max_history_turns: int = 8
    timeout_seconds: float = 60.0


@dataclass(frozen=True, slots=True)
class TTSConfig:
    backend: str = "none"
    voice_id: str | None = None
    rate: int = 175
    piper_model_path: Path = Path("models/tts/ar_JO-kareem-low.onnx")
    piper_config_path: Path = Path("models/tts/ar_JO-kareem-low.onnx.json")
    piper_model_sha256: str = "2887e9d68b125965c747e1371fa21e1cef19555ea98d0795a0d5d71188b13890"
    piper_config_sha256: str = "da328e52896826135508f797c1c77b45b35117e967c71befc377d654f100f328"


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    mode: str = "demo"
    preload_models: bool = False
    save_audio: bool = False
    save_camera: bool = False


@dataclass(frozen=True, slots=True)
class AppConfig:
    environment: str = "development"
    mock_mode: bool = True
    log_level: str = "INFO"
    max_upload_bytes: int = 10 * 1024 * 1024
    request_timeout_seconds: float = 30.0
    session_ttl_seconds: int = 1800
    max_sessions: int = 1000
    privacy_strict: bool = True
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    audio: AudioModelConfig = field(default_factory=AudioModelConfig)
    stt: SpeechToTextConfig = field(default_factory=SpeechToTextConfig)
    vision: VisionModelConfig = field(default_factory=VisionModelConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)


def _nested(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key, {})
    return value if isinstance(value, dict) else {}


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load configuration while preserving privacy-safe non-overridable defaults."""

    try:
        from dotenv import load_dotenv
    except ImportError:
        pass
    else:
        load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)

    selected_source: str | Path = (
        path if path is not None else os.getenv("PHANTOM_CONFIG") or "configs/default.yaml"
    )
    selected = Path(selected_source)
    data: dict[str, Any] = {}
    if selected.exists():
        loaded = yaml.safe_load(selected.read_text(encoding="utf-8"))
        if loaded is not None and not isinstance(loaded, dict):
            raise ValueError("configuration root must be a mapping")
        data = loaded or {}

    audio_data = _nested(data, "audio")
    stt_data = _nested(data, "stt")
    vision_data = _nested(data, "vision")
    llm_data = _nested(data, "llm")
    tts_data = _nested(data, "tts")
    runtime_data = _nested(data, "runtime")
    fusion_data = _nested(data, "fusion")
    safety_data = _nested(data, "safety")
    upload_mb = int(data.get("max_upload_mb", 10))
    if not 1 <= upload_mb <= 100:
        raise ValueError("max_upload_mb must be between 1 and 100")

    weights = fusion_data.get("weights", {"audio": 0.8, "vision": 0.6, "text": 1.6})
    if not isinstance(weights, dict) or set(weights) - {"audio", "vision", "text"}:
        raise ValueError("fusion weights may only contain audio, vision, and text")
    normalized_weights = {str(key): float(value) for key, value in weights.items()}
    if any(value < 0.0 for value in normalized_weights.values()):
        raise ValueError("fusion weights cannot be negative")

    audio_backend = str(audio_data.get("backend", "heuristic")).strip().lower()
    if audio_backend not in {"heuristic", "cnn", "crnn", "huggingface"}:
        raise ValueError("audio.backend must be one of: heuristic, cnn, crnn, huggingface")
    audio_device = str(audio_data.get("device", "cpu")).strip().lower()
    if audio_device not in {"auto", "cpu", "cuda"}:
        raise ValueError("audio.device must be one of: auto, cpu, cuda")
    n_mels = int(audio_data.get("n_mels", 40))
    if not 8 <= n_mels <= 256:
        raise ValueError("audio.n_mels must be between 8 and 256")
    max_frames = int(audio_data.get("max_frames", 3_000))
    if not 25 <= max_frames <= 30_000:
        raise ValueError("audio.max_frames must be between 25 and 30000")
    checkpoint_value = audio_data.get("checkpoint")
    if checkpoint_value is None or checkpoint_value == "":
        checkpoint = None
    elif not isinstance(checkpoint_value, str):
        raise ValueError("audio.checkpoint must be a file path string or null")
    else:
        checkpoint = Path(checkpoint_value).expanduser()
    if audio_backend in {"cnn", "crnn"} and checkpoint is None:
        raise ValueError("audio.checkpoint is required when audio.backend is cnn or crnn")

    audio_confidence_floor = float(audio_data.get("confidence_floor", 0.36))
    audio_decision_margin_floor = float(audio_data.get("decision_margin_floor", 0.07))
    angry_confidence_floor = float(audio_data.get("angry_confidence_floor", 0.62))
    angry_decision_margin_floor = float(audio_data.get("angry_decision_margin_floor", 0.22))
    audio_decision_settings = {
        "audio.confidence_floor": audio_confidence_floor,
        "audio.decision_margin_floor": audio_decision_margin_floor,
        "audio.angry_confidence_floor": angry_confidence_floor,
        "audio.angry_decision_margin_floor": angry_decision_margin_floor,
    }
    for setting, value in audio_decision_settings.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{setting} must be between 0 and 1")
    if angry_confidence_floor < audio_confidence_floor:
        raise ValueError("audio.angry_confidence_floor cannot be lower than audio.confidence_floor")
    if angry_decision_margin_floor < audio_decision_margin_floor:
        raise ValueError(
            "audio.angry_decision_margin_floor cannot be lower than audio.decision_margin_floor"
        )

    runtime_mode = (
        str(os.getenv("PHANTOM_RUNTIME_MODE", runtime_data.get("mode", "demo"))).strip().lower()
    )
    if runtime_mode not in {"demo", "real"}:
        raise ValueError("runtime.mode must be demo or real")
    save_audio = bool(runtime_data.get("save_audio", False))
    save_camera = bool(runtime_data.get("save_camera", False))
    if save_audio or save_camera:
        raise ValueError("raw audio and camera persistence are not implemented in this build")

    stt_backend = str(stt_data.get("backend", "none")).strip().lower()
    if stt_backend not in {"none", "faster-whisper"}:
        raise ValueError("stt.backend must be none or faster-whisper")
    stt_device = str(stt_data.get("device", "cpu")).strip().lower()
    if stt_device not in {"auto", "cpu", "cuda"}:
        raise ValueError("stt.device must be auto, cpu, or cuda")
    stt_language_value = stt_data.get("language", "ar")
    stt_language = None if stt_language_value in {None, "", "auto"} else str(stt_language_value)
    stt_beam_size = int(stt_data.get("beam_size", 3))
    if not 1 <= stt_beam_size <= 10:
        raise ValueError("stt.beam_size must be between 1 and 10")

    vision_backend = str(vision_data.get("backend", "baseline")).strip().lower()
    if vision_backend not in {"baseline", "deepface"}:
        raise ValueError("vision.backend must be baseline or deepface")
    vision_detector = str(vision_data.get("detector", "opencv-haar")).strip().lower()
    if vision_detector != "opencv-haar":
        raise ValueError("vision.detector must be opencv-haar in this build")
    analyze_interval = float(vision_data.get("analyze_interval_seconds", 1.25))
    age_interval = float(vision_data.get("age_interval_seconds", 20.0))
    max_staleness = float(vision_data.get("max_staleness_seconds", 3.0))
    if not 0.5 <= analyze_interval <= 5.0:
        raise ValueError("vision.analyze_interval_seconds must be between 0.5 and 5")
    if not 5.0 <= age_interval <= 300.0:
        raise ValueError("vision.age_interval_seconds must be between 5 and 300")
    if not 1.0 <= max_staleness <= 10.0:
        raise ValueError("vision.max_staleness_seconds must be between 1 and 10")

    llm_backend = (
        str(os.getenv("PHANTOM_LLM_PROVIDER", llm_data.get("backend", "demo"))).strip().lower()
    )
    if llm_backend not in {"demo", "transformers", "ollama", "openai-compatible"}:
        raise ValueError("llm.backend must be demo, transformers, ollama, or openai-compatible")
    if runtime_mode == "real" and llm_backend == "demo":
        raise ValueError("real runtime mode cannot use the demo LLM provider")
    llm_device = str(llm_data.get("device", "cpu")).strip().lower()
    if llm_device not in {"auto", "cpu", "cuda"}:
        raise ValueError("llm.device must be auto, cpu, or cuda")
    llm_quantization = str(llm_data.get("quantization", "none")).strip().lower()
    if llm_quantization not in {"none", "dynamic-int8"}:
        raise ValueError("llm.quantization must be none or dynamic-int8")
    if llm_quantization != "none" and llm_backend != "transformers":
        raise ValueError("llm.quantization is supported only by the transformers backend")
    if llm_quantization == "dynamic-int8" and llm_device != "cpu":
        raise ValueError("llm.quantization dynamic-int8 requires llm.device cpu")
    max_new_tokens = int(llm_data.get("max_new_tokens", 180))
    max_history_turns = int(llm_data.get("max_history_turns", 8))
    if not 32 <= max_new_tokens <= 1024:
        raise ValueError("llm.max_new_tokens must be between 32 and 1024")
    if not 1 <= max_history_turns <= 24:
        raise ValueError("llm.max_history_turns must be between 1 and 24")

    tts_backend = str(tts_data.get("backend", "none")).strip().lower()
    if tts_backend not in {"none", "pyttsx3", "piper-hybrid"}:
        raise ValueError("tts.backend must be none, pyttsx3, or piper-hybrid")
    if runtime_mode == "real" and tts_backend == "none":
        raise ValueError("real runtime mode requires a real TTS backend")
    if runtime_mode == "real" and audio_backend == "heuristic":
        raise ValueError("real runtime mode requires a pretrained audio-emotion backend")
    if runtime_mode == "real" and stt_backend == "none":
        raise ValueError("real runtime mode requires a real speech-to-text backend")
    if runtime_mode == "real" and vision_backend == "baseline":
        raise ValueError("real runtime mode requires a pretrained facial-emotion backend")

    resources = safety_data.get("resources") or [
        "Use a verified local crisis directory or your country's emergency service."
    ]
    return AppConfig(
        environment=str(data.get("environment", "development")),
        mock_mode=runtime_mode == "demo",
        log_level=str(os.getenv("PHANTOM_LOG_LEVEL", data.get("log_level", "INFO"))).upper(),
        max_upload_bytes=upload_mb * 1024 * 1024,
        request_timeout_seconds=float(data.get("request_timeout_seconds", 30.0)),
        session_ttl_seconds=int(data.get("session_ttl_seconds", 1800)),
        max_sessions=int(data.get("max_sessions", 1000)),
        privacy_strict=True,
        runtime=RuntimeConfig(
            mode=runtime_mode,
            preload_models=bool(runtime_data.get("preload_models", False)),
            save_audio=False,
            save_camera=False,
        ),
        audio=AudioModelConfig(
            backend=audio_backend,
            checkpoint=checkpoint,
            device=audio_device,
            n_mels=n_mels,
            max_frames=max_frames,
            model_name=str(audio_data.get("model_name", "superb/wav2vec2-base-superb-er")),
            revision=str(audio_data.get("revision", "main")),
            local_files_only=bool(audio_data.get("local_files_only", False)),
            confidence_floor=audio_confidence_floor,
            decision_margin_floor=audio_decision_margin_floor,
            angry_confidence_floor=angry_confidence_floor,
            angry_decision_margin_floor=angry_decision_margin_floor,
        ),
        stt=SpeechToTextConfig(
            backend=stt_backend,
            model_name=str(stt_data.get("model_name", "base")),
            device=stt_device,
            compute_type=str(stt_data.get("compute_type", "int8")),
            language=stt_language,
            beam_size=stt_beam_size,
        ),
        vision=VisionModelConfig(
            backend=vision_backend,
            detector=vision_detector,
            analyze_interval_seconds=analyze_interval,
            age_interval_seconds=age_interval,
            max_staleness_seconds=max_staleness,
        ),
        llm=LLMConfig(
            backend=llm_backend,
            model_name=str(llm_data.get("model_name", "Qwen/Qwen2.5-0.5B-Instruct")),
            revision=str(llm_data.get("revision", "main")),
            device=llm_device,
            quantization=llm_quantization,
            base_url=str(llm_data.get("base_url", "http://127.0.0.1:11434")).rstrip("/"),
            api_key_env=str(llm_data.get("api_key_env", "PHANTOM_LLM_API_KEY")),
            max_new_tokens=max_new_tokens,
            temperature=float(llm_data.get("temperature", 0.55)),
            max_history_turns=max_history_turns,
            timeout_seconds=float(llm_data.get("timeout_seconds", 60.0)),
        ),
        tts=TTSConfig(
            backend=tts_backend,
            voice_id=(str(tts_data["voice_id"]) if tts_data.get("voice_id") else None),
            rate=int(tts_data.get("rate", 175)),
            piper_model_path=Path(
                str(tts_data.get("piper_model_path", "models/tts/ar_JO-kareem-low.onnx"))
            ).expanduser(),
            piper_config_path=Path(
                str(tts_data.get("piper_config_path", "models/tts/ar_JO-kareem-low.onnx.json"))
            ).expanduser(),
            piper_model_sha256=str(
                tts_data.get(
                    "piper_model_sha256",
                    "2887e9d68b125965c747e1371fa21e1cef19555ea98d0795a0d5d71188b13890",
                )
            ),
            piper_config_sha256=str(
                tts_data.get(
                    "piper_config_sha256",
                    "da328e52896826135508f797c1c77b45b35117e967c71befc377d654f100f328",
                )
            ),
        ),
        fusion=FusionConfig(
            weights=normalized_weights,
            confidence_threshold=float(fusion_data.get("confidence_threshold", 0.45)),
            contradiction_threshold=float(fusion_data.get("contradiction_threshold", 0.65)),
            quality_floor=float(fusion_data.get("quality_floor", 0.2)),
        ),
        safety=SafetyConfig(
            country_code=str(safety_data.get("country_code", "GLOBAL")),
            emergency_message=str(
                safety_data.get("emergency_message", SafetyConfig().emergency_message)
            ),
            resources=tuple(str(item) for item in resources),
        ),
    )


def repository_root() -> Path:
    """Return the checkout root both in editable installs and source execution."""

    return Path(__file__).resolve().parents[2]
