"""One integrated, consent-gated multimodal conversational turn pipeline."""

from __future__ import annotations

import base64
import threading
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from phantom.audio.inference import AudioAnalyzer
from phantom.compat import UTC
from phantom.config import AppConfig
from phantom.exceptions import ConsentRequiredError, InvalidMediaError, PayloadTooLargeError
from phantom.fusion.confidence import distribution_from_label
from phantom.fusion.explanation import build_fusion_trace
from phantom.fusion.late_fusion import LateFusion
from phantom.llm import (
    AgreementLevel,
    ApproximateAgeEstimate,
    ConversationMemory,
    EmotionEstimate,
    LLMError,
    LLMProvider,
    MultimodalPromptContext,
    PromptBuilder,
    SafetyDirective,
)
from phantom.safety.crisis_router import CrisisRouter
from phantom.safety.guardrails import enforce_response, safe_fallback
from phantom.safety.privacy import require_consent
from phantom.schemas import (
    AgeBand,
    AgreementState,
    AudioObservation,
    ConsentSettings,
    EmotionLabel,
    MockSignal,
    Modality,
    ModalityResult,
    OptionalDemographicEstimates,
    RuntimeTurnResponse,
    SafetyRoute,
    VisionObservation,
)
from phantom.service.session_manager import SessionManager, SessionState
from phantom.tts import TTSError, TTSProvider
from phantom.vision.inference import VisionAnalyzer

_AUDIO_KEY = "runtime:latest_audio"
_VISION_KEY = "runtime:latest_vision"
_VISION_ATTRIBUTES_KEY = "runtime:latest_visual_attributes"
_MEMORY_KEY = "runtime:conversation_memory"


@dataclass(frozen=True, slots=True)
class _CachedVisualAttributes:
    """Short-lived, derived-only cache; no image or face representation is retained."""

    captured_at: datetime
    estimates: OptionalDemographicEstimates


def _unavailable(reason: str) -> ModalityResult:
    return ModalityResult.unavailable(reason)


def _demo_result(signal: MockSignal, modality: Modality) -> ModalityResult:
    if modality is Modality.VISION:
        face_count = signal.face_count if signal.face_count is not None else 1
        if face_count == 0:
            return _unavailable("simulated no-face condition")
        if face_count > 1:
            return ModalityResult(
                label=EmotionLabel.UNCERTAIN,
                confidence=0.0,
                quality=signal.quality,
                available=False,
                temporal_consistency=0.0,
                reason="simulated multiple-face condition; no person was analyzed",
                metadata={"mode": "demo", "synthetic": True, "face_count": face_count},
            )
    probabilities = distribution_from_label(signal.label, signal.confidence)
    return ModalityResult(
        label=signal.label,
        confidence=signal.confidence,
        quality=signal.quality,
        probabilities=probabilities,
        temporal_consistency=signal.temporal_consistency,
        reason="simulated demonstration value; not model inference",
        metadata={"mode": "demo", "synthetic": True, "face_count": 1},
    )


def _age_band(age: float) -> AgeBand:
    if age < 13:
        return AgeBand.CHILD
    if age < 18:
        return AgeBand.TEENAGER
    if age < 30:
        return AgeBand.YOUNG_ADULT
    if age < 65:
        return AgeBand.ADULT
    return AgeBand.OLDER_ADULT


def _emotion_context(source: str, result: ModalityResult) -> EmotionEstimate:
    usable_label = (
        result.label.value
        if result.label not in {EmotionLabel.UNCERTAIN, EmotionLabel.INSUFFICIENT_QUALITY}
        else None
    )
    return EmotionEstimate(
        source=source,
        label=usable_label,
        confidence=result.confidence if result.available else None,
        quality=result.quality if result.available else None,
        uncertain=not result.available or usable_label is None,
        available=result.available,
        reason=result.reason,
    )


def _agreement_context(value: AgreementState) -> AgreementLevel:
    return {
        AgreementState.STRONG_AGREEMENT: AgreementLevel.STRONG,
        AgreementState.PARTIAL_AGREEMENT: AgreementLevel.WEAK,
        AgreementState.CONFLICT: AgreementLevel.CONFLICT,
        AgreementState.INSUFFICIENT_EVIDENCE: AgreementLevel.INSUFFICIENT,
    }[value]


class AssistantRuntime:
    """Coordinate perception, fusion, guarded generation, memory, and synthesis."""

    def __init__(
        self,
        config: AppConfig,
        sessions: SessionManager,
        *,
        audio: AudioAnalyzer,
        vision: VisionAnalyzer,
        llm: LLMProvider,
        tts: TTSProvider,
        fusion: LateFusion | None = None,
        crisis: CrisisRouter | None = None,
        prompt_builder: PromptBuilder | None = None,
    ) -> None:
        self.config = config
        self.sessions = sessions
        self.audio = audio
        self.vision = vision
        self.llm = llm
        self.tts = tts
        self.fusion = fusion or LateFusion(config.fusion)
        self.crisis = crisis or CrisisRouter(config.safety)
        self.prompt_builder = prompt_builder or PromptBuilder(
            max_history_messages=config.llm.max_history_turns * 2
        )
        self._audio_lock = threading.RLock()
        self._vision_lock = threading.RLock()
        self._llm_lock = threading.RLock()
        self._tts_lock = threading.RLock()
        self._state_lock = threading.RLock()
        self._vision_temporal_owner: str | None = None

    def _require_mode(self, state: SessionState, expected: str) -> None:
        actual = "demo" if state.mock_mode else "real"
        if actual != expected or actual != self.config.runtime.mode:
            raise ValueError(f"this operation requires a {expected} session")

    def _memory(self, state: SessionState) -> ConversationMemory:
        with self._state_lock:
            memory = state.derived_features.get(_MEMORY_KEY)
            if not isinstance(memory, ConversationMemory):
                memory = ConversationMemory(max_turns=self.config.llm.max_history_turns)
                state.derived_features[_MEMORY_KEY] = memory
            return memory

    def analyze_audio(
        self,
        session_id: str,
        data: bytes,
        filename: str,
        content_type: str,
    ) -> AudioObservation:
        state = self.sessions.get(session_id)
        self._require_mode(state, "real")
        require_consent(state.consent, Modality.AUDIO)

        with self._audio_lock:
            # Each push-to-talk upload is one independent utterance.  Clear the
            # analyzer's optional temporal buffer on both sides so a prior
            # recording (including one from another session) cannot bias it.
            self.audio.clear_temporal_state()
            try:
                try:
                    acoustic = self.audio.analyze_bytes(data, filename, content_type)
                except (InvalidMediaError, PayloadTooLargeError):
                    raise
                except RuntimeError:
                    acoustic = _unavailable("the pretrained voice-emotion backend failed")
            finally:
                self.audio.clear_temporal_state()
            try:
                transcript = self.audio.transcribe_bytes(data, filename, content_type)
            except (InvalidMediaError, PayloadTooLargeError):
                raise
            except RuntimeError:
                transcript = None
        observation = AudioObservation(
            result=acoustic,
            transcript=transcript,
            language=self.config.stt.language,
        )
        with self._state_lock:
            state.derived_features[_AUDIO_KEY] = observation
        return observation

    def analyze_vision(
        self,
        session_id: str,
        data: bytes,
        filename: str,
        content_type: str,
    ) -> VisionObservation:
        state = self.sessions.get(session_id)
        self._require_mode(state, "real")
        require_consent(state.consent, Modality.VISION)
        with self._vision_lock:
            # VisionAnalyzer owns a small in-memory temporal buffer.  Never let
            # it combine observations from different sessions.
            if self._vision_temporal_owner != session_id:
                clear_temporal_state = getattr(self.vision, "clear_temporal_state", None)
                if callable(clear_temporal_state):
                    clear_temporal_state()
                self._vision_temporal_owner = session_id

            age_requested = state.consent.age_band_analysis
            with self._state_lock:
                cached = state.derived_features.get(_VISION_ATTRIBUTES_KEY)
            if not isinstance(cached, _CachedVisualAttributes):
                cached = None
            now = datetime.now(UTC)
            cache_is_fresh = (
                age_requested
                and cached is not None
                and (now - cached.captured_at).total_seconds()
                < self.config.vision.age_interval_seconds
            )
            # Apparent age changes far more slowly than expression and is not
            # emotion evidence.  Honor the configured age cadence by disabling
            # only that optional consent for intermediate model calls, then
            # return the session's last safe derived estimate when a valid face
            # is still present.  No raw frame or embedding is cached.
            analysis_consent = (
                state.consent.model_copy(update={"age_band_analysis": False})
                if cache_is_fresh
                else state.consent
            )
            try:
                result, demographics = self.vision.analyze_bytes(
                    data,
                    filename,
                    content_type,
                    analysis_consent,
                )
            except (InvalidMediaError, PayloadTooLargeError):
                raise
            except RuntimeError:
                clear_temporal_state = getattr(self.vision, "clear_temporal_state", None)
                if callable(clear_temporal_state):
                    clear_temporal_state()
                result = _unavailable("the pretrained facial-analysis backend failed")
                demographics = OptionalDemographicEstimates(
                    age_band=(
                        AgeBand.UNKNOWN if state.consent.age_band_analysis else AgeBand.DISABLED
                    ),
                    notice="Approximate age is unavailable; it is never emotion evidence.",
                )
            face_count = result.metadata.get("face_count")
            single_usable_face = result.available and face_count in {None, 1}
            with self._state_lock:
                if not age_requested or not single_usable_face:
                    state.derived_features.pop(_VISION_ATTRIBUTES_KEY, None)
                elif cache_is_fresh and cached is not None:
                    demographics = cached.estimates
                else:
                    # Cache the consented attempt even when the optional model
                    # is unavailable.  Otherwise a missing/failed age backend
                    # would be retried on every expression frame and recreate
                    # the latency this interval is meant to prevent.
                    state.derived_features[_VISION_ATTRIBUTES_KEY] = _CachedVisualAttributes(
                        captured_at=datetime.now(UTC),
                        estimates=demographics,
                    )
        observation = VisionObservation(result=result, optional_demographic_estimates=demographics)
        with self._state_lock:
            state.derived_features[_VISION_KEY] = observation
        return observation

    def set_demo_audio(
        self,
        session_id: str,
        *,
        transcript: str,
        signal: MockSignal,
    ) -> AudioObservation:
        state = self.sessions.get(session_id)
        self._require_mode(state, "demo")
        require_consent(state.consent, Modality.AUDIO)
        clean = transcript.strip()
        if not clean:
            raise ValueError("demo transcript cannot be empty")
        observation = AudioObservation(
            result=_demo_result(signal, Modality.AUDIO),
            transcript=clean,
            language="demo",
        )
        with self._state_lock:
            state.derived_features[_AUDIO_KEY] = observation
        return observation

    def set_demo_vision(
        self,
        session_id: str,
        *,
        signal: MockSignal,
        estimated_age: float | None = None,
    ) -> VisionObservation:
        state = self.sessions.get(session_id)
        self._require_mode(state, "demo")
        require_consent(state.consent, Modality.VISION)
        demographics = OptionalDemographicEstimates()
        if state.consent.age_band_analysis and estimated_age is not None:
            bounded_age = min(120.0, max(0.0, float(estimated_age)))
            lower = int(bounded_age // 10) * 10
            demographics = OptionalDemographicEstimates(
                age_band=_age_band(bounded_age),
                estimated_age=bounded_age,
                age_range=f"{lower}-{lower + 9}",
                confidence=0.0,
                uncertain=True,
                notice="Simulated approximate age for demo mode; not model inference.",
            )
        observation = VisionObservation(
            result=_demo_result(signal, Modality.VISION),
            optional_demographic_estimates=demographics,
        )
        with self._state_lock:
            state.derived_features[_VISION_KEY] = observation
        return observation

    def _latest_audio(self, state: SessionState) -> AudioObservation:
        value = state.derived_features.get(_AUDIO_KEY)
        return (
            value
            if isinstance(value, AudioObservation)
            else AudioObservation(result=_unavailable("no current voice observation"))
        )

    def _latest_vision(
        self, state: SessionState
    ) -> tuple[ModalityResult, OptionalDemographicEstimates]:
        value = state.derived_features.get(_VISION_KEY)
        if not isinstance(value, VisionObservation):
            return _unavailable("no current face observation"), OptionalDemographicEstimates()
        age_seconds = (datetime.now(UTC) - value.captured_at).total_seconds()
        if age_seconds > self.config.vision.max_staleness_seconds:
            return (
                _unavailable("the latest face observation is stale and was excluded"),
                OptionalDemographicEstimates(
                    age_band=(
                        AgeBand.UNKNOWN if state.consent.age_band_analysis else AgeBand.DISABLED
                    ),
                    notice="The latest approximate age estimate is stale and was excluded.",
                ),
            )
        return value.result, value.optional_demographic_estimates

    def _crisis_response(self, state: SessionState, user_text: str) -> str:
        is_arabic = any("\u0600" <= character <= "\u06ff" for character in user_text)
        resources = " ".join(self.config.safety.resources)
        name = state.preferences.preferred_name
        prefix = f"{name}, " if name else ""
        if is_arabic:
            emergency_contact = (
                "رقم الطوارئ الموحد 911"
                if self.config.safety.country_code.upper() == "JO"
                else "خدمات الطوارئ المحلية"
            )
            return enforce_response(
                f"{prefix}سلامتك الآن هي الأولوية. ابتعد عن أي سلاح أو وسيلة قد تسبب "
                f"الأذى، واتصل فورًا بـ{emergency_contact}، واطلب من شخص تثق به أن "
                "يبقى معك. هل تستطيع فعل ذلك الآن؟ لا يستطيع PHANTOM تقديم مساعدة "
                "طارئة أو إدارة هذا الموقف وحده.",
                user_text=user_text,
            )
        return enforce_response(
            f"{prefix}{self.config.safety.emergency_message} {resources} "
            "Move away from any weapon or means of harm and ask a trusted nearby person to "
            "stay with you. Can you do that now? I cannot provide emergency help or manage "
            "this situation alone.",
            user_text=user_text,
        )

    def respond(self, session_id: str, user_text: str) -> RuntimeTurnResponse:
        state = self.sessions.get(session_id)
        require_consent(state.consent, Modality.TEXT)
        clean_text = user_text.strip()
        if not clean_text:
            raise ValueError("user_text cannot be empty")

        with self._state_lock:
            audio_observation = self._latest_audio(state)
            vision_result, demographics = self._latest_vision(state)
        results = {
            Modality.AUDIO: audio_observation.result,
            Modality.VISION: vision_result,
        }
        fused = self.fusion.fuse(results)
        trace = build_fusion_trace(
            results,
            fused,
            user_text=clean_text,
            config=self.config.fusion,
        )
        safety = self.crisis.assess(clean_text)
        memory = self._memory(state)
        transcript_source = "typed"
        if audio_observation.transcript:
            transcript_source = (
                "speech_to_text"
                if clean_text == audio_observation.transcript.strip()
                else "user_edited"
            )

        llm_provider = self.llm.info.name
        if safety.route is SafetyRoute.CRISIS:
            assistant_text = self._crisis_response(state, clean_text)
            llm_provider = "deterministic-safety-router"
        else:
            if self.llm.info.sends_user_data_off_device and not state.consent.cloud_upload:
                raise ConsentRequiredError(
                    "explicit cloud-upload consent is required for the configured remote LLM"
                )
            age_context = None
            if demographics.estimated_age is not None:
                age_context = ApproximateAgeEstimate(
                    approximate_years=round(demographics.estimated_age),
                    age_range=demographics.age_range,
                    confidence=(demographics.confidence or None),
                    available=True,
                    uncertain=True,
                )
            context = MultimodalPromptContext(
                voice=_emotion_context("voice", audio_observation.result),
                face=_emotion_context("face", vision_result),
                age=age_context,
                agreement=_agreement_context(trace.agreement),
                safety=(
                    SafetyDirective.SUPPORT
                    if safety.route is SafetyRoute.SUPPORT
                    else SafetyDirective.NORMAL
                ),
                user_corrected_transcript=transcript_source == "user_edited",
            )
            bundle = self.prompt_builder.build(
                clean_text,
                context=context,
                history=memory.messages(),
            )
            try:
                with self._llm_lock:
                    generated = self.llm.generate(
                        bundle.messages,
                        timeout_seconds=self.config.llm.timeout_seconds,
                    )
                # Keep the user's language and situation available to the
                # deterministic output boundary.  A small local model can
                # otherwise turn clear Arabic distress into a generic English
                # refusal even when the transcript itself is correct.
                assistant_text = enforce_response(generated.text, user_text=clean_text)
                llm_provider = generated.provider.name
            except LLMError:
                assistant_text = safe_fallback(clean_text)
                llm_provider = f"{self.llm.info.name}-failed-safe"

        memory.append_turn(clean_text, assistant_text)
        tts_provider = self.tts.info.name
        tts_audio_base64: str | None = None
        tts_mime_type: str | None = None
        try:
            with self._tts_lock:
                speech = self.tts.synthesize(assistant_text)
            tts_provider = speech.provider.name
            if speech.audio_wav is not None:
                tts_audio_base64 = base64.b64encode(speech.audio_wav).decode("ascii")
                tts_mime_type = speech.mime_type
        except TTSError:
            tts_provider = f"{self.tts.info.name}-failed"

        with self._state_lock:
            state.derived_features.pop(_AUDIO_KEY, None)

        return RuntimeTurnResponse(
            session_id=session_id,
            transcript=clean_text,
            transcript_source=transcript_source,
            audio=audio_observation.result,
            vision=vision_result,
            fusion=trace,
            optional_demographic_estimates=demographics,
            assistant_text=assistant_text,
            safety=safety,
            llm_provider=llm_provider,
            tts_provider=tts_provider,
            tts_audio_base64=tts_audio_base64,
            tts_mime_type=tts_mime_type,
            history_turns=memory.turn_count,
        )

    def clear_session_runtime(self, state: SessionState) -> None:
        """Stop transient output and clear session-owned runtime state."""

        with self._tts_lock, suppress(TTSError):
            self.tts.stop()
        memory = state.derived_features.get(_MEMORY_KEY)
        if isinstance(memory, ConversationMemory):
            memory.clear()
        with self._vision_lock:
            if self._vision_temporal_owner == state.session_id:
                clear_temporal_state = getattr(self.vision, "clear_temporal_state", None)
                if callable(clear_temporal_state):
                    clear_temporal_state()
                self._vision_temporal_owner = None
        state.clear()

    @property
    def mode(self) -> str:
        return self.config.runtime.mode

    def status_summary(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "llm": self.llm.info.name,
            "llm_model": self.llm.info.model,
            "llm_device": self.llm.info.device,
            "tts": self.tts.info.name,
            "audio": getattr(
                self.audio.model,
                "provenance",
                {"backend": getattr(self.audio.model, "architecture", "unknown")},
            ),
            "stt": getattr(self.audio.stt_adapter, "provenance", None),
            "vision": getattr(
                self.vision.emotion_model,
                "provenance",
                {"backend": type(self.vision.emotion_model).__name__},
            ),
            "privacy": {"save_audio": False, "save_camera": False},
        }


def consent_for_runtime(
    *,
    microphone: bool,
    camera: bool,
    text_analysis: bool,
    age_analysis: bool,
    cloud_upload: bool = False,
) -> ConsentSettings:
    """Build the least-privilege consent object used by the real-time UI."""

    return ConsentSettings(
        microphone=microphone,
        camera=camera,
        text_analysis=text_analysis,
        age_band_analysis=age_analysis and camera,
        cloud_upload=cloud_upload,
        store_raw_data=False,
        store_optional_attributes=False,
    )
