"""Consent-aware facial-expression inference with safe abstention."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from phantom.exceptions import ConsentRequiredError, InvalidMediaError, PayloadTooLargeError
from phantom.fusion.confidence import distribution_from_label, temperature_scale
from phantom.fusion.temporal import TemporalSmoother
from phantom.schemas import (
    ConsentSettings,
    EmotionLabel,
    MockSignal,
    ModalityResult,
    OptionalDemographicEstimates,
)
from phantom.vision.emotion_model import BaselineFacialExpressionModel, FacialExpressionModel
from phantom.vision.face_detection import FaceDetector, OpenCVHaarFaceDetector
from phantom.vision.optional_attributes import (
    ExperimentalVisualAttributeEstimator,
    estimate_optional_attributes,
)
from phantom.vision.preprocessing import crop_image, decode_image, preprocess_image
from phantom.vision.quality import ImageQuality, assess_image_quality


class CameraFrameSource(Protocol):
    """Explicit lifecycle for camera adapters; calling ``start`` activates it."""

    def start(self, consent: ConsentSettings) -> None: ...

    def read(self) -> Any | None: ...

    def stop(self) -> None: ...


class OpenCVCameraSource:
    """Optional camera adapter that cannot start without current camera consent."""

    def __init__(self, device_index: int = 0) -> None:
        self.device_index = device_index
        self._capture: Any | None = None

    def start(self, consent: ConsentSettings) -> None:
        if not consent.camera:
            raise ConsentRequiredError("camera consent is required before activation")
        if self._capture is not None:
            return
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required for camera input") from exc
        capture = cv2.VideoCapture(self.device_index)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError("camera could not be opened")
        self._capture = capture

    def read(self) -> Any | None:
        if self._capture is None:
            raise RuntimeError("camera has not been started")
        ok, frame = self._capture.read()
        if not ok:
            return None
        import cv2

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def stop(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None


def _abstain(
    reason: str,
    *,
    quality: float = 0.0,
    face_count: int | None = None,
    detector: str | None = None,
) -> ModalityResult:
    metadata: dict[str, Any] = {}
    if face_count is not None:
        metadata["face_count"] = face_count
    if detector is not None:
        metadata["face_detector"] = detector
    label = (
        EmotionLabel.UNCERTAIN
        if face_count and face_count > 1
        else EmotionLabel.INSUFFICIENT_QUALITY
    )
    return ModalityResult(
        label=label,
        confidence=0.0,
        quality=max(0.0, min(1.0, quality)),
        available=False,
        temporal_consistency=0.0,
        reason=reason,
        metadata=metadata,
    )


class VisionAnalyzer:
    """Decode, count faces, quality-gate, classify, and optionally estimate attributes."""

    ACCEPTED_CONTENT_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})
    ACCEPTED_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})

    def __init__(
        self,
        *,
        face_detector: FaceDetector | None = None,
        emotion_model: FacialExpressionModel | None = None,
        optional_attribute_estimator: ExperimentalVisualAttributeEstimator | None = None,
        max_bytes: int = 10 * 1024 * 1024,
        max_pixels: int = 16_000_000,
        quality_floor: float = 0.25,
        confidence_floor: float = 0.28,
        temperature: float = 1.35,
        temporal_window: int = 2,
        temporal_decay: float = 0.35,
    ) -> None:
        self.face_detector = face_detector or OpenCVHaarFaceDetector()
        self.emotion_model = emotion_model or BaselineFacialExpressionModel()
        self.optional_attribute_estimator = optional_attribute_estimator
        self.max_bytes = max_bytes
        self.max_pixels = max_pixels
        self.quality_floor = quality_floor
        self.confidence_floor = confidence_floor
        self.temperature = temperature
        # Camera frames arrive close together, so a long, slowly decaying window
        # makes a genuine expression change look several frames late.  Two
        # samples still damp one-frame jitter while giving the newest frame most
        # of the weight.
        self.smoother = TemporalSmoother(
            window_size=temporal_window,
            decay=temporal_decay,
        )

    def _validate_media(self, data: bytes, filename: str, content_type: str) -> None:
        if len(data) > self.max_bytes:
            raise PayloadTooLargeError(f"image exceeds the {self.max_bytes}-byte limit")
        if not data:
            raise InvalidMediaError("image payload is empty")
        media_type = content_type.partition(";")[0].strip().lower()
        if media_type not in self.ACCEPTED_CONTENT_TYPES:
            raise InvalidMediaError("only JPEG, PNG, and WebP images are accepted")
        suffix = Path(filename).suffix.lower()
        if suffix not in self.ACCEPTED_SUFFIXES:
            raise InvalidMediaError("image filename must use .jpg, .jpeg, .png, or .webp")
        is_jpeg = data.startswith(b"\xff\xd8\xff")
        is_png = data.startswith(b"\x89PNG\r\n\x1a\n")
        is_webp = len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP"
        expected = {
            "image/jpeg": is_jpeg,
            "image/png": is_png,
            "image/webp": is_webp,
        }
        if not expected.get(media_type, False):
            raise InvalidMediaError("image content does not match its declared media type")
        if suffix in {".jpg", ".jpeg"} and not is_jpeg:
            raise InvalidMediaError("image content does not match its filename")
        if suffix == ".png" and not is_png:
            raise InvalidMediaError("image content does not match its filename")
        if suffix == ".webp" and not is_webp:
            raise InvalidMediaError("image content does not match its filename")

    def _optional(self, face: Any, consent: ConsentSettings) -> OptionalDemographicEstimates:
        # Never invoke a model with a missing face.  In particular, the
        # DeepFace age adapter expects an RGB crop and must not run on no-face or
        # multiple-face frames.
        estimator = (
            self.optional_attribute_estimator
            if face is not None
            or getattr(self.optional_attribute_estimator, "accepts_missing_face", False)
            else None
        )
        try:
            return estimate_optional_attributes(face, consent, estimator)
        except RuntimeError:
            # Optional apparent-age inference must not discard an otherwise
            # usable expression result when its separate model is unavailable.
            unavailable = estimate_optional_attributes(None, consent, None)
            return unavailable.model_copy(
                update={
                    "notice": (
                        "Optional visual-attribute analysis is temporarily unavailable; "
                        "no estimate was produced and it was not used as emotion evidence."
                    )
                }
            )

    def _mock(
        self, signal: MockSignal, consent: ConsentSettings
    ) -> tuple[ModalityResult, OptionalDemographicEstimates]:
        face_count = signal.face_count if signal.face_count is not None else 1
        if face_count == 0:
            self.clear_temporal_state()
            return (
                _abstain("no face detected", quality=signal.quality, face_count=0, detector="mock"),
                self._optional(None, consent),
            )
        if face_count > 1:
            self.clear_temporal_state()
            return (
                _abstain(
                    "multiple faces detected; select exactly one face before analysis",
                    quality=signal.quality,
                    face_count=face_count,
                    detector="mock",
                ),
                self._optional(None, consent),
            )
        probabilities = distribution_from_label(signal.label, signal.confidence)
        result = ModalityResult(
            label=signal.label,
            confidence=signal.confidence,
            quality=signal.quality,
            probabilities=probabilities if signal.label in probabilities else {},
            available=True,
            temporal_consistency=signal.temporal_consistency,
            reason="deterministic mock visual signal; not a medical assessment",
            metadata={"mock": True, "face_count": 1, "identity_recognition": False},
        )
        return result, self._optional(None, consent)

    def _infer_face(
        self,
        face: Any,
        quality: ImageQuality,
        detector_name: str,
    ) -> ModalityResult:
        if not quality.sufficient or quality.overall < self.quality_floor:
            self.clear_temporal_state()
            return _abstain(
                quality.reason or "face image quality is below the configured threshold",
                quality=quality.overall,
                face_count=1,
                detector=detector_name,
            )
        model_input = preprocess_image(face, size=(224, 224), normalize=True)
        raw = self.emotion_model.predict(model_input)
        calibrated = temperature_scale(raw, self.temperature)
        smoothed = self.smoother.update(calibrated)
        top_label, top_probability = max(smoothed.items(), key=lambda item: item[1])
        distance = 0.5 * sum(abs(smoothed[label] - calibrated[label]) for label in smoothed)
        consistency = max(0.0, min(1.0, 1.0 - distance))
        confidence = max(0.0, min(1.0, top_probability * (0.55 + 0.45 * quality.overall)))
        uncertain = confidence < self.confidence_floor
        return ModalityResult(
            label=EmotionLabel.UNCERTAIN if uncertain else top_label,
            confidence=confidence,
            quality=quality.overall,
            probabilities=smoothed,
            available=True,
            temporal_consistency=consistency,
            reason=(
                "the conservative visual baseline is uncertain"
                if uncertain
                else "facial-expression estimate; not proof of an internal state"
            ),
            metadata={
                "face_count": 1,
                "face_detector": detector_name,
                "brightness": round(quality.brightness, 4),
                "contrast": round(quality.contrast, 4),
                "sharpness": round(quality.sharpness, 4),
                "identity_recognition": False,
                "face_embeddings_stored": False,
                "optional_attributes_used_for_emotion": False,
            },
        )

    def analyze_bytes(
        self,
        data: bytes,
        filename: str,
        content_type: str,
        consent: ConsentSettings,
        mock_signal: MockSignal | None = None,
    ) -> tuple[ModalityResult, OptionalDemographicEstimates]:
        """Analyze an image after camera/image consent, or use an explicit mock."""

        if not consent.camera:
            raise ConsentRequiredError("camera/image consent is required for vision analysis")
        if mock_signal is not None:
            return self._mock(mock_signal, consent)
        self._validate_media(data, filename, content_type)
        image = decode_image(data, max_pixels=self.max_pixels)
        detection = self.face_detector.detect(image)
        if not detection.available:
            self.clear_temporal_state()
            return (
                _abstain(
                    detection.reason or "face detector is unavailable",
                    face_count=0,
                    detector=detection.detector,
                ),
                self._optional(None, consent),
            )
        if detection.face_count == 0:
            self.clear_temporal_state()
            return (
                _abstain("no face detected", face_count=0, detector=detection.detector),
                self._optional(None, consent),
            )
        if detection.face_count > 1:
            self.clear_temporal_state()
            return (
                _abstain(
                    "multiple faces detected; select exactly one face before analysis",
                    face_count=detection.face_count,
                    detector=detection.detector,
                ),
                self._optional(None, consent),
            )
        face = crop_image(image, detection.faces[0].as_tuple())
        quality = assess_image_quality(face, quality_floor=self.quality_floor)
        result = self._infer_face(face, quality, detection.detector)
        attributes = (
            self._optional(face, consent)
            if quality.sufficient
            else estimate_optional_attributes(None, consent, None)
        )
        return result, attributes

    def clear_temporal_state(self) -> None:
        self.smoother.clear()


VisionEmotionPipeline = VisionAnalyzer
