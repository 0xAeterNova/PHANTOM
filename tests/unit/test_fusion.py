from __future__ import annotations

import pytest

from phantom.config import FusionConfig
from phantom.fusion.confidence import distribution_from_label
from phantom.fusion.late_fusion import LateFusion
from phantom.schemas import EmotionLabel, Modality, ModalityResult


def result(label: EmotionLabel, confidence: float = 0.9, quality: float = 1.0) -> ModalityResult:
    return ModalityResult(
        label=label,
        confidence=confidence,
        quality=quality,
        probabilities=distribution_from_label(label, confidence),
    )


def test_missing_modalities_abstain() -> None:
    fused = LateFusion().fuse({Modality.AUDIO: ModalityResult.unavailable()})
    assert fused.label is EmotionLabel.UNCERTAIN
    assert fused.uncertain is True
    assert fused.confidence == 0.0


def test_high_quality_single_modality_can_produce_caveated_result() -> None:
    fused = LateFusion().fuse({Modality.AUDIO: result(EmotionLabel.HAPPY, 0.95)})
    assert fused.label is EmotionLabel.HAPPY
    assert fused.uncertain is False
    assert "not a medical assessment" in fused.explanation.lower()
    assert fused.contributions[Modality.AUDIO] == pytest.approx(1.0)


def test_contradictory_strong_modalities_force_uncertainty() -> None:
    fused = LateFusion().fuse(
        {
            Modality.AUDIO: result(EmotionLabel.HAPPY, 0.92),
            Modality.VISION: result(EmotionLabel.SAD, 0.92),
        }
    )
    assert fused.label is EmotionLabel.UNCERTAIN
    assert fused.uncertain is True
    assert "different directions" in fused.explanation
    assert sum(fused.contributions.values()) == pytest.approx(1.0)


def test_low_quality_signal_has_no_influence() -> None:
    fusion = LateFusion(FusionConfig(quality_floor=0.3))
    fused = fusion.fuse(
        {
            Modality.AUDIO: result(EmotionLabel.ANGRY, 0.99, quality=0.1),
            Modality.TEXT: result(EmotionLabel.NEUTRAL, 0.95, quality=1.0),
        }
    )
    assert fused.label is EmotionLabel.NEUTRAL
    assert Modality.AUDIO not in fused.contributions


def test_demographic_values_are_not_accepted_by_fusion_interface() -> None:
    assert set(FusionConfig().weights) == {"audio", "vision", "text"}
