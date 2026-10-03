from __future__ import annotations

import io
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest
from PIL import Image

from phantom.exceptions import ConsentRequiredError, InvalidMediaError
from phantom.fusion.confidence import distribution_from_label
from phantom.schemas import (
    AgeBand,
    ConsentSettings,
    EmotionLabel,
    GenderPresentation,
)
from phantom.vision.face_detection import MockFaceDetector
from phantom.vision.inference import VisionAnalyzer
from phantom.vision.optional_attributes import MockOptionalAttributeEstimator
from phantom.vision.preprocessing import crop_image, decode_image, preprocess_image


def png_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array.astype(np.uint8), mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def synthetic_image() -> np.ndarray:
    image = np.full((96, 96, 3), 128, dtype=np.uint8)
    image[18:78, 18:78] = 190
    image[30:42, 30:42] = 30
    image[30:42, 55:67] = 30
    image[60:66, 34:64] = 60
    return image


class _SequenceEmotionModel:
    def __init__(self, labels: Sequence[EmotionLabel]) -> None:
        self._labels = iter(labels)

    def predict(self, face: Any) -> dict[EmotionLabel, float]:
        del face
        return distribution_from_label(next(self._labels), 0.98)


class _MissingFaceRejectingEstimator:
    experimental = True
    supported_attributes = frozenset({"age"})

    def __init__(self) -> None:
        self.calls = 0

    def estimate(self, face: Any) -> Any:
        self.calls += 1
        if face is None:
            raise AssertionError("the optional model must not receive a missing face")
        raise RuntimeError("optional model unavailable")


def test_image_decode_crop_and_preprocess() -> None:
    decoded = decode_image(png_bytes(synthetic_image()))
    assert decoded.shape == (96, 96, 3)
    cropped = crop_image(decoded, (16, 16, 64, 64), padding=0.0)
    assert cropped.shape == (64, 64, 3)
    prepared = preprocess_image(cropped, size=(32, 32))
    assert prepared.shape == (32, 32, 3)
    assert prepared.dtype == np.float32
    assert 0.0 <= float(prepared.min()) <= float(prepared.max()) <= 1.0


@pytest.mark.parametrize(
    ("face_count", "reason"),
    [(0, "no face"), (2, "multiple faces")],
)
def test_no_face_and_multiple_faces_abstain(face_count: int, reason: str) -> None:
    analyzer = VisionAnalyzer(face_detector=MockFaceDetector(face_count))
    result, attributes = analyzer.analyze_bytes(
        png_bytes(synthetic_image()),
        "synthetic.png",
        "image/png",
        ConsentSettings(camera=True),
    )
    assert result.confidence == 0.0
    assert reason in (result.reason or "")
    assert attributes.age_band is AgeBand.DISABLED
    assert attributes.perceived_gender_presentation is GenderPresentation.DISABLED


def test_vision_requires_consent() -> None:
    with pytest.raises(ConsentRequiredError):
        VisionAnalyzer(face_detector=MockFaceDetector()).analyze_bytes(
            png_bytes(synthetic_image()),
            "synthetic.png",
            "image/png",
            ConsentSettings(),
        )


def test_mislabeled_image_is_rejected() -> None:
    with pytest.raises(InvalidMediaError):
        VisionAnalyzer().analyze_bytes(
            b"not png",
            "fake.png",
            "image/png",
            ConsentSettings(camera=True),
        )


def test_optional_attributes_need_separate_consent_and_never_change_affect() -> None:
    estimator = MockOptionalAttributeEstimator(
        age_band=AgeBand.ADULT,
        perceived_gender_presentation=GenderPresentation.AMBIGUOUS,
        confidence=0.8,
    )
    analyzer = VisionAnalyzer(
        face_detector=MockFaceDetector(1), optional_attribute_estimator=estimator
    )
    disabled_result, disabled = analyzer.analyze_bytes(
        png_bytes(synthetic_image()),
        "synthetic.png",
        "image/png",
        ConsentSettings(camera=True),
    )
    enabled_result, enabled = analyzer.analyze_bytes(
        png_bytes(synthetic_image()),
        "synthetic.png",
        "image/png",
        ConsentSettings(
            camera=True,
            age_band_analysis=True,
            perceived_gender_presentation_analysis=True,
        ),
    )
    assert disabled.age_band is AgeBand.DISABLED
    assert enabled.age_band is AgeBand.ADULT
    assert enabled.perceived_gender_presentation is GenderPresentation.AMBIGUOUS
    assert disabled_result.label is enabled_result.label


def test_low_quality_face_abstains() -> None:
    black = np.zeros((96, 96, 3), dtype=np.uint8)
    result, _ = VisionAnalyzer(face_detector=MockFaceDetector(1)).analyze_bytes(
        png_bytes(black), "dark.png", "image/png", ConsentSettings(camera=True)
    )
    assert result.label is EmotionLabel.INSUFFICIENT_QUALITY
    assert result.confidence == 0.0


def test_default_temporal_window_tracks_a_real_expression_change_on_next_frame() -> None:
    model = _SequenceEmotionModel(
        [
            EmotionLabel.ANGRY,
            EmotionLabel.ANGRY,
            EmotionLabel.ANGRY,
            EmotionLabel.ANGRY,
            EmotionLabel.HAPPY,
        ]
    )
    analyzer = VisionAnalyzer(
        face_detector=MockFaceDetector(1),
        emotion_model=model,
        quality_floor=0.0,
        confidence_floor=0.0,
        temperature=1.0,
    )
    frame = png_bytes(synthetic_image())
    consent = ConsentSettings(camera=True)

    for _ in range(4):
        assert analyzer.analyze_bytes(frame, "frame.png", "image/png", consent)[0].label is (
            EmotionLabel.ANGRY
        )
    changed, _ = analyzer.analyze_bytes(frame, "frame.png", "image/png", consent)

    assert changed.label is EmotionLabel.HAPPY


def test_no_face_interrupt_clears_old_expression_history() -> None:
    detector = MockFaceDetector(1)
    analyzer = VisionAnalyzer(
        face_detector=detector,
        emotion_model=_SequenceEmotionModel(
            [
                EmotionLabel.ANGRY,
                EmotionLabel.ANGRY,
                EmotionLabel.ANGRY,
                EmotionLabel.HAPPY,
            ]
        ),
        quality_floor=0.0,
        confidence_floor=0.0,
        temperature=1.0,
        temporal_window=5,
        temporal_decay=0.75,
    )
    frame = png_bytes(synthetic_image())
    consent = ConsentSettings(camera=True)
    for _ in range(3):
        analyzer.analyze_bytes(frame, "frame.png", "image/png", consent)

    detector.face_count = 0
    missing, _ = analyzer.analyze_bytes(frame, "frame.png", "image/png", consent)
    detector.face_count = 1
    changed, _ = analyzer.analyze_bytes(frame, "frame.png", "image/png", consent)

    assert missing.available is False
    assert changed.label is EmotionLabel.HAPPY


@pytest.mark.parametrize("face_count", [0, 2])
def test_optional_model_is_not_run_without_exactly_one_face(face_count: int) -> None:
    estimator = _MissingFaceRejectingEstimator()
    analyzer = VisionAnalyzer(
        face_detector=MockFaceDetector(face_count),
        optional_attribute_estimator=estimator,
    )

    result, attributes = analyzer.analyze_bytes(
        png_bytes(synthetic_image()),
        "frame.png",
        "image/png",
        ConsentSettings(camera=True, age_band_analysis=True),
    )

    assert result.available is False
    assert estimator.calls == 0
    assert attributes.age_band is AgeBand.UNKNOWN


def test_optional_model_failure_does_not_discard_expression_result() -> None:
    estimator = _MissingFaceRejectingEstimator()
    analyzer = VisionAnalyzer(
        face_detector=MockFaceDetector(1),
        emotion_model=_SequenceEmotionModel([EmotionLabel.HAPPY]),
        optional_attribute_estimator=estimator,
        quality_floor=0.0,
        confidence_floor=0.0,
        temperature=1.0,
    )

    result, attributes = analyzer.analyze_bytes(
        png_bytes(synthetic_image()),
        "frame.png",
        "image/png",
        ConsentSettings(camera=True, age_band_analysis=True),
    )

    assert result.available is True
    assert result.label is EmotionLabel.HAPPY
    assert estimator.calls == 1
    assert attributes.age_band is AgeBand.UNKNOWN
    assert "temporarily unavailable" in attributes.notice
