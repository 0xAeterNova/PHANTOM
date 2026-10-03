from __future__ import annotations

import pytest

from phantom.config import AppConfig
from phantom.exceptions import ConsentRequiredError, SessionNotFoundError
from phantom.schemas import (
    AgeBand,
    ConsentSettings,
    DialogueRequest,
    EmotionLabel,
    GenderPresentation,
    MockSignal,
    MultimodalAnalysisRequest,
    SafetyRoute,
    SessionCreateRequest,
)
from phantom.service.orchestrator import PhantomOrchestrator
from phantom.service.session_manager import SessionManager


@pytest.fixture
def service() -> tuple[SessionManager, PhantomOrchestrator, str]:
    manager = SessionManager()
    orchestrator = PhantomOrchestrator(AppConfig(), manager)
    session = manager.create(
        SessionCreateRequest(
            consent=ConsentSettings(microphone=True, camera=True, text_analysis=True),
            mock_mode=True,
        )
    )
    return manager, orchestrator, session.session_id


def test_end_to_end_mock_analysis_and_correction(
    service: tuple[SessionManager, PhantomOrchestrator, str],
) -> None:
    manager, orchestrator, session_id = service
    analysis = orchestrator.analyze_multimodal(
        MultimodalAnalysisRequest(
            session_id=session_id,
            text="I am glad the synthetic demo is running.",
            mock_audio=MockSignal(label=EmotionLabel.HAPPY, confidence=0.95, quality=0.95),
        )
    )
    assert analysis.modalities_available
    assert analysis.safety.route is SafetyRoute.NORMAL
    assert analysis.optional_demographic_estimates.age_band is AgeBand.DISABLED
    assert (
        analysis.optional_demographic_estimates.perceived_gender_presentation
        is GenderPresentation.DISABLED
    )

    correction = orchestrator.respond(
        DialogueRequest(session_id=session_id, estimate_is_incorrect=True)
    )
    assert correction.mentioned_emotion is False
    assert "correcting" in correction.text.lower()

    manager.delete(session_id)
    with pytest.raises(SessionNotFoundError):
        orchestrator.respond(DialogueRequest(session_id=session_id))


def test_missing_modalities_are_supported(
    service: tuple[SessionManager, PhantomOrchestrator, str],
) -> None:
    _, orchestrator, session_id = service
    analysis = orchestrator.analyze_multimodal(
        MultimodalAnalysisRequest(
            session_id=session_id,
            mock_audio=MockSignal(label=EmotionLabel.NEUTRAL, confidence=0.95),
        )
    )
    assert [item.value for item in analysis.modalities_available] == ["audio"]
    assert set(analysis.modality_results) == {"audio"}


def test_contradictory_mock_modalities_abstain(
    service: tuple[SessionManager, PhantomOrchestrator, str],
) -> None:
    _, orchestrator, session_id = service
    analysis = orchestrator.analyze_multimodal(
        MultimodalAnalysisRequest(
            session_id=session_id,
            mock_audio=MockSignal(label=EmotionLabel.HAPPY, confidence=0.95),
            mock_vision=MockSignal(label=EmotionLabel.SAD, confidence=0.95, face_count=1),
        )
    )
    assert analysis.fused_affect.uncertain is True
    assert analysis.fused_affect.label is EmotionLabel.UNCERTAIN


def test_no_face_and_multiple_faces_abstain_safely(
    service: tuple[SessionManager, PhantomOrchestrator, str],
) -> None:
    _, orchestrator, session_id = service
    for face_count, expected in ((0, "no face"), (2, "multiple faces")):
        analysis = orchestrator.analyze_multimodal(
            MultimodalAnalysisRequest(
                session_id=session_id,
                mock_vision=MockSignal(face_count=face_count),
            )
        )
        vision = analysis.modality_results["vision"]
        assert vision.label is EmotionLabel.UNCERTAIN
        assert expected in (vision.reason or "")


def test_crisis_text_overrides_ordinary_dialogue(
    service: tuple[SessionManager, PhantomOrchestrator, str],
) -> None:
    _, orchestrator, session_id = service
    analysis = orchestrator.analyze_multimodal(
        MultimodalAnalysisRequest(session_id=session_id, text="I want to end my life")
    )
    assert analysis.safety.route is SafetyRoute.CRISIS
    response = orchestrator.respond(DialogueRequest(session_id=session_id))
    assert response.route is SafetyRoute.CRISIS
    assert response.mentioned_emotion is False


def test_analysis_without_modality_consent_is_denied() -> None:
    manager = SessionManager()
    orchestrator = PhantomOrchestrator(AppConfig(), manager)
    session = manager.create(SessionCreateRequest())
    with pytest.raises(ConsentRequiredError):
        orchestrator.analyze_multimodal(
            MultimodalAnalysisRequest(
                session_id=session.session_id,
                mock_audio=MockSignal(),
            )
        )


def test_temporal_smoothing_is_session_scoped_and_cleared_on_deletion() -> None:
    manager = SessionManager()
    orchestrator = PhantomOrchestrator(AppConfig(), manager)
    consent = ConsentSettings(microphone=True)
    first = manager.create(SessionCreateRequest(consent=consent))
    second = manager.create(SessionCreateRequest(consent=consent))

    for _ in range(3):
        orchestrator.analyze_multimodal(
            MultimodalAnalysisRequest(
                session_id=first.session_id,
                mock_audio=MockSignal(label=EmotionLabel.HAPPY, confidence=0.95),
            )
        )
    second_result = orchestrator.analyze_multimodal(
        MultimodalAnalysisRequest(
            session_id=second.session_id,
            mock_audio=MockSignal(label=EmotionLabel.SAD, confidence=0.95),
        )
    ).modality_results["audio"]
    assert (
        second_result.probabilities[EmotionLabel.SAD]
        > second_result.probabilities[EmotionLabel.HAPPY]
    )

    first_state = manager.get(first.session_id)
    assert first_state.derived_features
    manager.delete(first.session_id)
    assert first_state.derived_features == {}
