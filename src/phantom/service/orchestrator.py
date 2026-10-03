"""Consent-gated orchestration across replaceable modality analyzers."""

from __future__ import annotations

import threading
from datetime import datetime

from phantom.audio.inference import AudioAnalyzer
from phantom.compat import UTC
from phantom.config import AppConfig
from phantom.dialogue.response_generator import DeterministicResponseGenerator
from phantom.fusion.confidence import distribution_from_label
from phantom.fusion.late_fusion import LateFusion
from phantom.fusion.temporal import TemporalSmoother
from phantom.safety.crisis_router import CrisisRouter
from phantom.safety.privacy import require_consent
from phantom.schemas import (
    AnalysisResult,
    DialogueRequest,
    DialogueResponse,
    EmotionLabel,
    MockSignal,
    Modality,
    ModalityResult,
    MultimodalAnalysisRequest,
    OptionalDemographicEstimates,
)
from phantom.service.session_manager import SessionManager, SessionState
from phantom.text.analyzer import TextAnalyzer
from phantom.vision.inference import VisionAnalyzer


class PhantomOrchestrator:
    """Coordinates perception and policy without retaining raw uploaded media."""

    def __init__(
        self,
        config: AppConfig,
        sessions: SessionManager,
        *,
        audio: AudioAnalyzer | None = None,
        vision: VisionAnalyzer | None = None,
        text: TextAnalyzer | None = None,
        fusion: LateFusion | None = None,
        crisis: CrisisRouter | None = None,
        dialogue: DeterministicResponseGenerator | None = None,
    ) -> None:
        self.config = config
        self.sessions = sessions
        self.audio = audio or AudioAnalyzer.from_model_config(
            config.audio,
            max_bytes=config.max_upload_bytes,
        )
        self.vision = vision or VisionAnalyzer()
        self.text = text or TextAnalyzer()
        self.fusion = fusion or LateFusion(config.fusion)
        self.crisis = crisis or CrisisRouter(config.safety)
        self.dialogue = dialogue or DeterministicResponseGenerator()
        # Pipeline adapters may own model objects. Locks let us reset their internal
        # smoothing safely before delegating session-scoped smoothing below.
        self._audio_lock = threading.Lock()
        self._vision_lock = threading.Lock()

    @staticmethod
    def mock_result(signal: MockSignal, modality: Modality) -> ModalityResult:
        if modality is Modality.VISION:
            if signal.face_count == 0:
                return ModalityResult(
                    label=EmotionLabel.UNCERTAIN,
                    confidence=0.0,
                    quality=signal.quality,
                    available=True,
                    reason="no face detected",
                )
            if signal.face_count is not None and signal.face_count > 1:
                return ModalityResult(
                    label=EmotionLabel.UNCERTAIN,
                    confidence=0.0,
                    quality=signal.quality,
                    available=True,
                    reason="multiple faces detected; user selection is required",
                    metadata={"face_count": signal.face_count},
                )
        probabilities = distribution_from_label(signal.label, signal.confidence)
        label = signal.label
        confidence = signal.confidence
        if signal.quality < 0.2:
            label = EmotionLabel.INSUFFICIENT_QUALITY
            confidence = 0.0
            probabilities = {}
        return ModalityResult(
            label=label,
            confidence=confidence,
            quality=signal.quality,
            probabilities=probabilities,
            temporal_consistency=signal.temporal_consistency,
            reason="deterministic mock signal",
            metadata={"mode": "mock"},
        )

    def _assemble(
        self,
        session_id: str,
        results: dict[Modality, ModalityResult],
        *,
        source_text: str | None = None,
        demographics: OptionalDemographicEstimates | None = None,
    ) -> AnalysisResult:
        available = [modality for modality, result in results.items() if result.available]
        analysis = AnalysisResult(
            session_id=session_id,
            timestamp=datetime.now(UTC),
            modalities_available=available,
            modality_results=results,
            fused_affect=self.fusion.fuse(results),
            optional_demographic_estimates=demographics or OptionalDemographicEstimates(),
            safety=self.crisis.assess(source_text),
        )
        self.sessions.set_last_analysis(session_id, analysis)
        return analysis

    @staticmethod
    def _smooth_for_session(
        state: SessionState,
        modality: Modality,
        result: ModalityResult,
    ) -> ModalityResult:
        """Smooth only within one consented session; deletion clears this state."""

        if not result.available or not result.probabilities:
            return result
        key = f"temporal:{modality.value}"
        smoother = state.derived_features.get(key)
        if not isinstance(smoother, TemporalSmoother):
            smoother = TemporalSmoother()
            state.derived_features[key] = smoother
        smoothed = smoother.update(result.probabilities)
        distance = 0.5 * sum(
            abs(smoothed[label] - result.probabilities.get(label, 0.0)) for label in smoothed
        )
        label = result.label
        if label not in {EmotionLabel.UNCERTAIN, EmotionLabel.INSUFFICIENT_QUALITY}:
            label = max(smoothed.items(), key=lambda item: item[1])[0]
        return result.model_copy(
            update={
                "label": label,
                "probabilities": smoothed,
                "temporal_consistency": max(0.0, min(1.0, 1.0 - distance)),
            }
        )

    def analyze_text(self, session_id: str, text: str) -> AnalysisResult:
        state = self.sessions.get(session_id)
        require_consent(state.consent, Modality.TEXT)
        result = self._smooth_for_session(state, Modality.TEXT, self.text.analyze(text))
        return self._assemble(session_id, {Modality.TEXT: result}, source_text=text)

    def analyze_audio(
        self,
        session_id: str,
        data: bytes,
        filename: str,
        content_type: str,
    ) -> AnalysisResult:
        state = self.sessions.get(session_id)
        if state.mock_mode:
            raise ValueError("real audio analysis is unavailable in a demo session")
        require_consent(state.consent, Modality.AUDIO)
        with self._audio_lock:
            self.audio.clear_temporal_state()
            try:
                raw_result = self.audio.analyze_bytes(data, filename, content_type)
            finally:
                self.audio.clear_temporal_state()
        # A submitted recording is a complete utterance, not another frame in
        # a continuous stream.  Do not carry its categorical scores into the
        # next recording; this avoids self-reinforcing class bias.
        return self._assemble(session_id, {Modality.AUDIO: raw_result})

    def analyze_image(
        self,
        session_id: str,
        data: bytes,
        filename: str,
        content_type: str,
    ) -> AnalysisResult:
        state = self.sessions.get(session_id)
        if state.mock_mode:
            raise ValueError("real image analysis is unavailable in a demo session")
        require_consent(state.consent, Modality.VISION)
        with self._vision_lock:
            self.vision.clear_temporal_state()
            try:
                raw_result, demographics = self.vision.analyze_bytes(
                    data, filename, content_type, state.consent
                )
            finally:
                self.vision.clear_temporal_state()
        result = self._smooth_for_session(state, Modality.VISION, raw_result)
        return self._assemble(
            session_id,
            {Modality.VISION: result},
            demographics=demographics,
        )

    def analyze_multimodal(self, request: MultimodalAnalysisRequest) -> AnalysisResult:
        state = self.sessions.get(request.session_id)
        if not state.mock_mode and (
            request.mock_audio is not None or request.mock_vision is not None
        ):
            raise ValueError("mock signals are not allowed in a real session")
        results: dict[Modality, ModalityResult] = {}
        if request.mock_audio is not None:
            require_consent(state.consent, Modality.AUDIO)
            results[Modality.AUDIO] = self._smooth_for_session(
                state,
                Modality.AUDIO,
                self.mock_result(request.mock_audio, Modality.AUDIO),
            )
        if request.mock_vision is not None:
            require_consent(state.consent, Modality.VISION)
            results[Modality.VISION] = self._smooth_for_session(
                state,
                Modality.VISION,
                self.mock_result(request.mock_vision, Modality.VISION),
            )
        if request.text is not None:
            require_consent(state.consent, Modality.TEXT)
            results[Modality.TEXT] = self._smooth_for_session(
                state, Modality.TEXT, self.text.analyze(request.text)
            )
        return self._assemble(request.session_id, results, source_text=request.text)

    def analyze_local_media(
        self,
        session_id: str,
        *,
        audio: tuple[bytes, str, str] | None = None,
        image: tuple[bytes, str, str] | None = None,
    ) -> AnalysisResult:
        """Combine consented local media without retaining either raw payload."""

        if audio is None and image is None:
            raise ValueError("provide audio, image, or both")
        results: dict[Modality, ModalityResult] = {}
        demographics = OptionalDemographicEstimates()
        if audio is not None:
            audio_analysis = self.analyze_audio(session_id, *audio)
            results[Modality.AUDIO] = audio_analysis.modality_results[Modality.AUDIO]
        if image is not None:
            image_analysis = self.analyze_image(session_id, *image)
            results[Modality.VISION] = image_analysis.modality_results[Modality.VISION]
            demographics = image_analysis.optional_demographic_estimates
        return self._assemble(session_id, results, demographics=demographics)

    def respond(self, request: DialogueRequest) -> DialogueResponse:
        state = self.sessions.get(request.session_id)
        if request.analysis is not None and request.analysis.session_id != request.session_id:
            raise ValueError("analysis session_id does not match the active session")
        analysis = request.analysis or state.last_analysis
        safety = self.crisis.assess(request.user_message)
        # Preserve an already-triggered safety route when the response request contains no text.
        if not request.user_message and analysis is not None:
            safety = analysis.safety
        return self.dialogue.generate(
            analysis=analysis,
            safety=safety,
            preferences=state.preferences,
            estimate_is_incorrect=request.estimate_is_incorrect,
            advice_permission=request.advice_permission,
        )
