from __future__ import annotations

from phantom.fusion.confidence import distribution_from_label
from phantom.fusion.explanation import build_fusion_trace, has_explicit_self_report
from phantom.fusion.late_fusion import LateFusion
from phantom.schemas import AgreementState, EmotionLabel, Modality, ModalityResult


def _result(label: EmotionLabel, confidence: float = 0.9) -> ModalityResult:
    return ModalityResult(
        label=label,
        confidence=confidence,
        quality=0.95,
        probabilities=distribution_from_label(label, confidence),
    )


def test_strong_nonverbal_conflict_abstains_and_words_remain_primary() -> None:
    results = {
        Modality.AUDIO: _result(EmotionLabel.SAD),
        Modality.VISION: _result(EmotionLabel.NEUTRAL),
    }
    fused = LateFusion().fuse(results)
    trace = build_fusion_trace(results, fused, user_text="I am happy today.")

    assert trace.agreement is AgreementState.CONFLICT
    assert trace.words_have_priority is True
    assert trace.explicit_self_report is True
    assert "words" in trace.explanation.lower()


def test_arabic_explicit_self_report_is_detected() -> None:
    assert has_explicit_self_report("أنا حزين اليوم")


def test_missing_face_is_explained() -> None:
    results = {Modality.AUDIO: _result(EmotionLabel.HAPPY)}
    fused = LateFusion().fuse(results)
    trace = build_fusion_trace(results, fused, user_text="That worked.")

    assert trace.agreement is AgreementState.INSUFFICIENT_EVIDENCE
    assert trace.excluded[Modality.VISION] == "not supplied"
