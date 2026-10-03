from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pytest

from phantom.compat import UTC
from phantom.config import (
    AppConfig,
    AudioModelConfig,
    LLMConfig,
    RuntimeConfig,
    SpeechToTextConfig,
    TTSConfig,
    VisionModelConfig,
)
from phantom.fusion.confidence import distribution_from_label
from phantom.llm import (
    ChatMessage,
    GenerationResult,
    ProviderInfo,
    ProviderMode,
    ProviderRequestError,
)
from phantom.schemas import (
    AgeBand,
    AgreementState,
    ConsentSettings,
    EmotionLabel,
    ModalityResult,
    OptionalDemographicEstimates,
    SessionCreateRequest,
)
from phantom.service import turn_pipeline as turn_pipeline_module
from phantom.service.session_manager import SessionManager
from phantom.service.turn_pipeline import AssistantRuntime
from phantom.tts import SpeechResult, TTSInfo, TTSMode


def _result(label: EmotionLabel) -> ModalityResult:
    return ModalityResult(
        label=label,
        confidence=0.9,
        quality=0.95,
        probabilities=distribution_from_label(label, 0.9),
        metadata={"mode": "real"},
    )


class DummyAudio:
    def __init__(
        self,
        label: EmotionLabel = EmotionLabel.SAD,
        transcript: str = "I am happy today.",
    ) -> None:
        self.result = _result(label)
        self.transcript = transcript
        self.model = type("Model", (), {"backend": "test-real-ser"})()
        self.stt_adapter = type("STT", (), {"backend": "test-real-stt"})()
        self.clear_calls = 0

    def analyze_bytes(self, *args: Any) -> ModalityResult:
        del args
        return self.result

    def transcribe_bytes(self, *args: Any) -> str:
        del args
        return self.transcript

    def clear_temporal_state(self) -> None:
        self.clear_calls += 1


class StateBleedingAudio(DummyAudio):
    """Test double that repeats its prior label unless explicitly cleared."""

    def __init__(self) -> None:
        super().__init__(EmotionLabel.ANGRY)
        self.next_results = iter((_result(EmotionLabel.ANGRY), _result(EmotionLabel.SAD)))
        self.previous: ModalityResult | None = None

    def analyze_bytes(self, *args: Any) -> ModalityResult:
        del args
        if self.previous is not None:
            return self.previous
        self.previous = next(self.next_results)
        return self.previous

    def clear_temporal_state(self) -> None:
        super().clear_temporal_state()
        self.previous = None


class DummyVision:
    def __init__(self, label: EmotionLabel = EmotionLabel.NEUTRAL) -> None:
        self.result = _result(label)
        self.emotion_model = type("VisionModel", (), {"backend": "test-real-fer"})()
        self.optional_attribute_estimator = None

    def analyze_bytes(self, *args: Any) -> tuple[ModalityResult, OptionalDemographicEstimates]:
        del args
        return self.result, OptionalDemographicEstimates(
            estimated_age=24,
            age_range="20-29",
            age_band="young adult",
            confidence=0.0,
            uncertain=True,
        )


class CountingVision(DummyVision):
    def __init__(self, *, return_age: bool = True) -> None:
        super().__init__()
        self.return_age = return_age
        self.age_consent_calls: list[bool] = []
        self.clear_calls = 0

    def analyze_bytes(
        self,
        data: bytes,
        filename: str,
        content_type: str,
        consent: ConsentSettings,
    ) -> tuple[ModalityResult, OptionalDemographicEstimates]:
        del data, filename, content_type
        self.age_consent_calls.append(consent.age_band_analysis)
        if consent.age_band_analysis and self.return_age:
            return self.result, OptionalDemographicEstimates(
                estimated_age=24,
                age_range="20-29",
                age_band="young adult",
                confidence=0.0,
                uncertain=True,
            )
        if consent.age_band_analysis:
            return self.result, OptionalDemographicEstimates(
                age_band="unknown",
                confidence=0.0,
                uncertain=True,
                notice="Optional age model is unavailable.",
            )
        return self.result, OptionalDemographicEstimates()

    def clear_temporal_state(self) -> None:
        self.clear_calls += 1


class DummyLLM:
    def __init__(self) -> None:
        self.prompts: list[tuple[ChatMessage, ...]] = []
        self.calls = 0
        self._info = ProviderInfo(
            name="test-real-llm",
            mode=ProviderMode.REAL,
            model="test-model",
        )

    @property
    def info(self) -> ProviderInfo:
        return self._info

    def generate(
        self, messages: tuple[ChatMessage, ...], *, timeout_seconds: float | None = None
    ) -> GenerationResult:
        del timeout_seconds
        self.calls += 1
        self.prompts.append(tuple(messages))
        return GenerationResult(
            "I'm glad you told me. What made today feel positive?",
            self.info,
        )


class RefusingArabicLLM(DummyLLM):
    def generate(
        self, messages: tuple[ChatMessage, ...], *, timeout_seconds: float | None = None
    ) -> GenerationResult:
        del timeout_seconds
        self.calls += 1
        self.prompts.append(tuple(messages))
        return GenerationResult("آسف لا أستطيع مساعدتك", self.info)


class FailingLLM(DummyLLM):
    def generate(
        self, messages: tuple[ChatMessage, ...], *, timeout_seconds: float | None = None
    ) -> GenerationResult:
        del messages, timeout_seconds
        self.calls += 1
        raise ProviderRequestError("synthetic provider failure")


class DummyTTS:
    def __init__(self) -> None:
        self._info = TTSInfo(name="test-real-tts", mode=TTSMode.REAL)

    @property
    def info(self) -> TTSInfo:
        return self._info

    def synthesize(self, text: str) -> SpeechResult:
        assert text
        return SpeechResult(self.info, b"RIFF" + b"0" * 40, "audio/wav", True, False)

    def stop(self) -> None:
        return None


def _runtime(
    audio: DummyAudio | None = None,
    vision: DummyVision | None = None,
) -> tuple[AssistantRuntime, SessionManager, DummyLLM]:
    config = AppConfig(
        mock_mode=False,
        runtime=RuntimeConfig(mode="real"),
        audio=AudioModelConfig(backend="huggingface"),
        stt=SpeechToTextConfig(backend="faster-whisper"),
        vision=VisionModelConfig(backend="deepface"),
        llm=LLMConfig(backend="transformers", max_history_turns=4),
        tts=TTSConfig(backend="pyttsx3"),
    )
    sessions = SessionManager()
    llm = DummyLLM()
    runtime = AssistantRuntime(
        config,
        sessions,
        audio=audio or DummyAudio(),  # type: ignore[arg-type]
        vision=vision or DummyVision(),  # type: ignore[arg-type]
        llm=llm,
        tts=DummyTTS(),
    )
    return runtime, sessions, llm


def _session(sessions: SessionManager, *, mock_mode: bool = False) -> str:
    return sessions.create(
        SessionCreateRequest(
            consent=ConsentSettings(
                microphone=True,
                camera=True,
                text_analysis=True,
                age_band_analysis=True,
            ),
            mock_mode=mock_mode,
        )
    ).session_id


def test_complete_turn_prioritizes_explicit_words_and_keeps_history() -> None:
    runtime, sessions, llm = _runtime()
    session_id = _session(sessions)
    runtime.analyze_audio(session_id, b"wav", "capture.wav", "audio/wav")
    runtime.analyze_vision(session_id, b"jpg", "frame.jpg", "image/jpeg")

    first = runtime.respond(session_id, "I am happy today.")

    assert first.fusion.agreement is AgreementState.CONFLICT
    assert first.fusion.explicit_self_report is True
    assert first.transcript_source == "speech_to_text"
    assert first.tts_audio_base64 is not None
    assert first.tts_mime_type == "audio/wav"
    assert first.history_turns == 1
    assert "primary source" in llm.prompts[0][0].content

    second = runtime.respond(session_id, "It was because my test passed.")
    assert second.history_turns == 2
    assert second.tts_audio_base64 is not None
    assert second.tts_mime_type == "audio/wav"
    assert any(message.content == first.assistant_text for message in llm.prompts[1])


def test_crisis_words_bypass_llm_and_use_configured_safety_route() -> None:
    runtime, sessions, llm = _runtime()
    session_id = _session(sessions)

    response = runtime.respond(session_id, "I want to kill myself")

    assert response.safety.route.value == "crisis"
    assert response.llm_provider == "deterministic-safety-router"
    assert llm.calls == 0


def test_arabic_immediate_danger_gets_localized_deterministic_safety_response() -> None:
    runtime, sessions, llm = _runtime()
    session_id = _session(sessions)

    response = runtime.respond(session_id, "سأقتل نفسي الآن ولدي سكين بجانبي")

    assert response.safety.route.value == "crisis"
    assert response.llm_provider == "deterministic-safety-router"
    assert "ابتعد" in response.assistant_text
    assert "خدمات الطوارئ المحلية" in response.assistant_text
    assert "هل تستطيع فعل ذلك الآن" in response.assistant_text
    assert not response.assistant_text.startswith("If you")
    assert llm.calls == 0


@pytest.mark.parametrize(
    "user_text",
    [
        "I am going to hurt myself right now.",
        "سأنتحر الآن.",
        "I will kill my friend now.",
        "سأقتل صديقي الآن.",
    ],
)
def test_explicit_imminent_threats_bypass_the_llm(user_text: str) -> None:
    runtime, sessions, llm = _runtime()
    session_id = _session(sessions)

    response = runtime.respond(session_id, user_text)

    assert response.safety.route.value == "crisis"
    assert response.llm_provider == "deterministic-safety-router"
    assert llm.calls == 0


def test_arabic_distress_replaces_model_refusal_with_supportive_same_language_reply() -> None:
    user_text = "لقد تعرضت لحادث اليوم ونفسيتي تعبانة جدًا وحزين ولا أعلم ما العمل"
    runtime, sessions, _llm = _runtime(audio=DummyAudio(transcript=user_text))
    refusing = RefusingArabicLLM()
    runtime.llm = refusing
    session_id = _session(sessions)
    observation = runtime.analyze_audio(session_id, b"wav", "capture.wav", "audio/wav")

    response = runtime.respond(session_id, user_text)

    assert observation.transcript == user_text
    assert response.transcript_source == "speech_to_text"
    assert response.safety.route.value == "normal"
    assert response.llm_provider == refusing.info.name
    assert "لا أستطيع مساعدتك" not in response.assistant_text
    assert "هل أنت الآن في مكان آمن" in response.assistant_text
    assert "رعاية طبية عاجلة" in response.assistant_text


def test_arabic_passive_death_wish_replaces_incomplete_model_support() -> None:
    runtime, sessions, _llm = _runtime()
    session_id = _session(sessions)

    response = runtime.respond(session_id, "أتمنى أن أنام ولا أستيقظ غدًا.")

    assert response.safety.route.value == "support"
    assert "هل تفكر الآن في الانتحار أو إيذاء نفسك؟" in response.assistant_text
    assert "شخص تثق به" in response.assistant_text


def test_english_passive_death_wish_provider_failure_uses_support_fallback() -> None:
    runtime, sessions, _llm = _runtime()
    failing = FailingLLM()
    runtime.llm = failing
    session_id = _session(sessions)

    response = runtime.respond(session_id, "I don't want to live.")

    assert response.safety.route.value == "support"
    assert response.llm_provider.endswith("-failed-safe")
    assert (
        "Are you thinking about suicide or harming yourself right now?" in response.assistant_text
    )
    assert "someone you trust" in response.assistant_text


def test_real_and_demo_inputs_cannot_cross_modes() -> None:
    runtime, sessions, _llm = _runtime()
    real_session = _session(sessions)
    with pytest.raises(ValueError, match="demo"):
        runtime.set_demo_audio(
            real_session,
            transcript="demo",
            signal=type("Signal", (), {})(),  # type: ignore[arg-type]
        )

    demo_session = _session(sessions, mock_mode=True)
    with pytest.raises(ValueError, match="real"):
        runtime.analyze_audio(demo_session, b"wav", "capture.wav", "audio/wav")


def test_real_recordings_are_independent_across_sessions() -> None:
    audio = StateBleedingAudio()
    runtime, sessions, _llm = _runtime(audio)
    first_session = _session(sessions)
    second_session = _session(sessions)

    first = runtime.analyze_audio(first_session, b"wav", "first.wav", "audio/wav")
    second = runtime.analyze_audio(second_session, b"wav", "second.wav", "audio/wav")

    assert first.result.label is EmotionLabel.ANGRY
    assert second.result.label is EmotionLabel.SAD
    assert audio.clear_calls == 4


def test_age_model_runs_on_its_own_interval_while_expression_runs_each_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FrozenDateTime:
        current = datetime(2026, 1, 1, tzinfo=UTC)

        @classmethod
        def now(cls, tz: Any = None) -> datetime:
            del tz
            return cls.current

    monkeypatch.setattr(turn_pipeline_module, "datetime", FrozenDateTime)
    vision = CountingVision()
    runtime, sessions, _llm = _runtime(vision=vision)
    session_id = _session(sessions)

    first = runtime.analyze_vision(session_id, b"jpg", "frame.jpg", "image/jpeg")
    FrozenDateTime.current += timedelta(seconds=1)
    cached = runtime.analyze_vision(session_id, b"jpg", "frame.jpg", "image/jpeg")
    FrozenDateTime.current += timedelta(seconds=20)
    refreshed = runtime.analyze_vision(session_id, b"jpg", "frame.jpg", "image/jpeg")

    assert vision.age_consent_calls == [True, False, True]
    assert first.optional_demographic_estimates.estimated_age == 24
    assert cached.optional_demographic_estimates.estimated_age == 24
    assert refreshed.optional_demographic_estimates.estimated_age == 24


def test_unavailable_age_model_is_not_retried_on_every_expression_frame() -> None:
    vision = CountingVision(return_age=False)
    runtime, sessions, _llm = _runtime(vision=vision)
    session_id = _session(sessions)

    first = runtime.analyze_vision(session_id, b"jpg", "frame.jpg", "image/jpeg")
    second = runtime.analyze_vision(session_id, b"jpg", "frame.jpg", "image/jpeg")

    assert vision.age_consent_calls == [True, False]
    assert first.optional_demographic_estimates.age_band is AgeBand.UNKNOWN
    assert second.optional_demographic_estimates.age_band is AgeBand.UNKNOWN
    assert "unavailable" in second.optional_demographic_estimates.notice


@pytest.mark.parametrize("face_count", [0, 1, 2])
def test_interrupted_face_continuity_discards_cached_age(face_count: int) -> None:
    vision = CountingVision()
    runtime, sessions, _llm = _runtime(vision=vision)
    session_id = _session(sessions)
    runtime.analyze_vision(session_id, b"jpg", "valid.jpg", "image/jpeg")
    vision.result = ModalityResult(
        label=(EmotionLabel.UNCERTAIN if face_count > 1 else EmotionLabel.INSUFFICIENT_QUALITY),
        confidence=0.0,
        quality=0.0,
        available=False,
        temporal_consistency=0.0,
        reason="face continuity interrupted",
        metadata={"face_count": face_count},
    )
    runtime.analyze_vision(session_id, b"jpg", "interrupted.jpg", "image/jpeg")
    vision.result = _result(EmotionLabel.NEUTRAL)
    refreshed = runtime.analyze_vision(
        session_id,
        b"jpg",
        "valid-again.jpg",
        "image/jpeg",
    )

    assert vision.age_consent_calls == [True, False, True]
    assert refreshed.optional_demographic_estimates.estimated_age == 24


def test_vision_temporal_state_and_age_cache_are_isolated_by_session() -> None:
    vision = CountingVision()
    runtime, sessions, _llm = _runtime(vision=vision)
    first_session = _session(sessions)
    second_session = _session(sessions)

    runtime.analyze_vision(first_session, b"jpg", "first.jpg", "image/jpeg")
    runtime.analyze_vision(first_session, b"jpg", "first-2.jpg", "image/jpeg")
    runtime.analyze_vision(second_session, b"jpg", "second.jpg", "image/jpeg")

    assert vision.age_consent_calls == [True, False, True]
    assert vision.clear_calls == 2


def test_session_clear_removes_conversation_and_observations() -> None:
    runtime, sessions, _llm = _runtime()
    session_id = _session(sessions)
    runtime.respond(session_id, "Hello")
    state = sessions.get(session_id)
    assert state.derived_features

    runtime.clear_session_runtime(state)

    assert not state.derived_features
