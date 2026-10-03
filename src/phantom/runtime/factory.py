"""Build explicit demo or real pretrained runtime backends from configuration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from phantom.audio.inference import AudioAnalyzer, FasterWhisperSpeechToTextAdapter
from phantom.config import AppConfig, repository_root
from phantom.llm import (
    DeterministicDemoProvider,
    LLMProvider,
    OllamaProvider,
    OpenAICompatibleProvider,
    QwenTransformersProvider,
)
from phantom.schemas import BackendReadiness, BackendStatus, RuntimeMode
from phantom.service.session_manager import SessionManager
from phantom.service.turn_pipeline import AssistantRuntime
from phantom.tts import PiperArabicHybridTTS, Pyttsx3WindowsTTS, TextOnlyTTS, TTSProvider
from phantom.vision.emotion_model import DeepFaceFacialExpressionAdapter
from phantom.vision.face_detection import OpenCVHaarFaceDetector
from phantom.vision.inference import VisionAnalyzer
from phantom.vision.optional_attributes import DeepFaceApproximateAgeEstimator


def _endpoint(base_url: str, path: str) -> str:
    parsed = urlsplit(base_url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("LLM base_url must be an absolute HTTP or HTTPS URL")
    existing = parsed.path.rstrip("/")
    if existing.endswith(path):
        selected = existing
    elif existing and existing != "/" and path.startswith(existing + "/"):
        selected = path
    elif existing and existing != "/":
        selected = existing + path
    else:
        selected = path
    return urlunsplit((parsed.scheme, parsed.netloc, selected, "", ""))


@dataclass(slots=True)
class RuntimeBundle:
    config: AppConfig
    sessions: SessionManager
    runtime: AssistantRuntime
    statuses: dict[str, BackendStatus] = field(default_factory=dict)

    def _set_status(
        self,
        component: str,
        provider: str,
        model: str | None,
        revision: str | None,
        device: str,
        readiness: BackendReadiness,
        detail: str | None = None,
    ) -> None:
        self.statuses[component] = BackendStatus(
            component=component,
            provider=provider,
            model=model,
            revision=revision,
            device=device,
            mode=RuntimeMode(self.config.runtime.mode),
            readiness=readiness,
            detail=detail,
        )

    def _warm_component(
        self,
        component: str,
        provider: str,
        model: str | None,
        revision: str | None,
        device: str,
        loader: Callable[[], Any] | None,
    ) -> None:
        if loader is None:
            self._set_status(
                component,
                provider,
                model,
                revision,
                device,
                BackendReadiness.DEGRADED,
                "configured provider has no startup probe; it will be checked on first request",
            )
            return
        try:
            loader()
        except Exception:
            self._set_status(
                component,
                provider,
                model,
                revision,
                device,
                BackendReadiness.UNAVAILABLE,
                "the configured real backend could not be loaded; no demo output was substituted",
            )
            return
        self._set_status(
            component,
            provider,
            model,
            revision,
            device,
            BackendReadiness.READY,
            "loaded once and ready",
        )

    def warmup(self) -> dict[str, BackendStatus]:
        """Load configured real models once; failures remain visible and non-synthetic."""

        if self.config.runtime.mode == "demo":
            for component in (
                "speech_to_text",
                "voice_emotion",
                "facial_emotion",
                "age",
                "llm",
                "tts",
            ):
                self._set_status(
                    component,
                    "deterministic-demo",
                    None,
                    None,
                    "cpu",
                    BackendReadiness.READY,
                    "simulated demo component; not pretrained inference",
                )
            return dict(self.statuses)

        audio_model = self.runtime.audio.model
        stt = self.runtime.audio.stt_adapter
        vision_model = self.runtime.vision.emotion_model
        age_model = self.runtime.vision.optional_attribute_estimator
        llm = self.runtime.llm
        tts = self.runtime.tts
        self._warm_component(
            "voice_emotion",
            str(getattr(audio_model, "backend", type(audio_model).__name__)),
            str(getattr(audio_model, "model_name_or_path", self.config.audio.model_name)),
            self.config.audio.revision,
            self.config.audio.device,
            getattr(audio_model, "_load", None),
        )
        self._warm_component(
            "speech_to_text",
            str(getattr(stt, "backend", type(stt).__name__)),
            str(getattr(stt, "model_size_or_path", self.config.stt.model_name)),
            None,
            self.config.stt.device,
            getattr(stt, "_load", None),
        )
        self._warm_component(
            "facial_emotion",
            str(getattr(vision_model, "backend", type(vision_model).__name__)),
            str(getattr(vision_model, "model_name", "DeepFace Emotion")),
            None,
            "cpu",
            getattr(vision_model, "_load", None),
        )
        self._warm_component(
            "age",
            str(getattr(age_model, "backend", type(age_model).__name__)),
            str(getattr(age_model, "model_name", "DeepFace Age")),
            None,
            "cpu",
            getattr(age_model, "_load", None),
        )
        self._warm_component(
            "llm",
            llm.info.name,
            llm.info.model,
            llm.info.revision,
            llm.info.device or "provider",
            getattr(llm, "_load", None),
        )
        self._warm_component(
            "tts",
            tts.info.name,
            tts.info.voice,
            None,
            "cpu",
            getattr(tts, "probe", None),
        )
        return dict(self.statuses)

    def health(self) -> dict[str, Any]:
        return {
            "status": (
                "ready"
                if self.statuses
                and all(item.readiness is BackendReadiness.READY for item in self.statuses.values())
                else "degraded"
            ),
            "mode": self.config.runtime.mode,
            "components": {
                key: value.model_dump(mode="json") for key, value in self.statuses.items()
            },
            "active_sessions": self.sessions.active_count,
            "privacy": {"save_audio": False, "save_camera": False, "in_memory_only": True},
        }


def build_runtime_bundle(config: AppConfig) -> RuntimeBundle:
    """Construct exactly the configured mode; no cross-mode fallback is allowed."""

    sessions = SessionManager(config.session_ttl_seconds, config.max_sessions)
    audio = AudioAnalyzer.from_model_config(config.audio, max_bytes=config.max_upload_bytes)
    llm: LLMProvider
    tts: TTSProvider
    if config.runtime.mode == "real":
        audio.stt_adapter = FasterWhisperSpeechToTextAdapter(
            config.stt.model_name,
            device=config.stt.device if config.stt.device != "auto" else "cpu",
            compute_type=config.stt.compute_type,
            language=config.stt.language,
            beam_size=config.stt.beam_size,
            local_files_only=False,
        )
        vision = VisionAnalyzer(
            face_detector=OpenCVHaarFaceDetector(),
            emotion_model=DeepFaceFacialExpressionAdapter(),
            optional_attribute_estimator=DeepFaceApproximateAgeEstimator(),
            max_bytes=config.max_upload_bytes,
        )
        if config.llm.backend == "transformers":
            llm = QwenTransformersProvider(
                config.llm.model_name,
                revision=config.llm.revision,
                device=config.llm.device,
                quantization=config.llm.quantization,
                local_files_only=False,
                max_new_tokens=config.llm.max_new_tokens,
                temperature=config.llm.temperature,
            )
        elif config.llm.backend == "ollama":
            llm = OllamaProvider(
                config.llm.model_name,
                endpoint=_endpoint(config.llm.base_url, "/api/chat"),
                timeout_seconds=config.llm.timeout_seconds,
                max_tokens=config.llm.max_new_tokens,
                temperature=config.llm.temperature,
            )
        elif config.llm.backend == "openai-compatible":
            llm = OpenAICompatibleProvider(
                config.llm.model_name,
                endpoint=_endpoint(config.llm.base_url, "/v1/chat/completions"),
                api_key_environment_variable=config.llm.api_key_env,
                timeout_seconds=config.llm.timeout_seconds,
                max_tokens=config.llm.max_new_tokens,
                temperature=config.llm.temperature,
            )
        else:  # protected by configuration validation
            raise ValueError("real mode requires a configured real LLM provider")
        if config.tts.backend == "piper-hybrid":
            model_path = config.tts.piper_model_path
            piper_config_path = config.tts.piper_config_path
            if not model_path.is_absolute():
                model_path = repository_root() / model_path
            if not piper_config_path.is_absolute():
                piper_config_path = repository_root() / piper_config_path
            tts = PiperArabicHybridTTS(
                model_path,
                piper_config_path,
                model_sha256=config.tts.piper_model_sha256,
                config_sha256=config.tts.piper_config_sha256,
                english_voice_id=config.tts.voice_id,
                english_rate=config.tts.rate,
            )
        elif config.tts.backend == "pyttsx3":
            tts = Pyttsx3WindowsTTS(voice_id=config.tts.voice_id, rate=config.tts.rate)
        else:  # protected by configuration validation
            raise ValueError("real mode requires a configured real TTS provider")
    else:
        vision = VisionAnalyzer(max_bytes=config.max_upload_bytes)
        llm = DeterministicDemoProvider()
        tts = TextOnlyTTS("demo mode intentionally provides text-only synthetic output")

    runtime = AssistantRuntime(
        config,
        sessions,
        audio=audio,
        vision=vision,
        llm=llm,
        tts=tts,
    )
    return RuntimeBundle(config=config, sessions=sessions, runtime=runtime)
