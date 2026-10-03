"""Face-counting interfaces with no identity recognition or embedding storage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class FaceBox:
    x: int
    y: int
    width: int
    height: int
    confidence: float = 1.0

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.width, self.height


@dataclass(frozen=True, slots=True)
class FaceDetectionResult:
    faces: tuple[FaceBox, ...]
    detector: str
    available: bool = True
    reason: str | None = None

    @property
    def face_count(self) -> int:
        return len(self.faces)


@runtime_checkable
class FaceDetector(Protocol):
    """Return bounding boxes only; implementations must not identify people."""

    def detect(self, image: Any) -> FaceDetectionResult: ...


class UnavailableFaceDetector:
    """Safe fallback used when no optional local detector is installed."""

    def __init__(self, reason: str = "local face detector is unavailable") -> None:
        self.reason = reason

    def detect(self, image: Any) -> FaceDetectionResult:
        del image
        return FaceDetectionResult((), "unavailable", available=False, reason=self.reason)


class MockFaceDetector:
    """Deterministic detector for synthetic tests; never analyzes identity."""

    def __init__(self, face_count: int = 1) -> None:
        if face_count < 0:
            raise ValueError("face_count cannot be negative")
        self.face_count = face_count

    def detect(self, image: Any) -> FaceDetectionResult:
        height, width = image.shape[:2]
        count = self.face_count
        if count == 0:
            return FaceDetectionResult((), "mock")
        box_width = max(1, width // (count + 2))
        box_height = max(1, min(height * 3 // 4, box_width * 4 // 3))
        top = max(0, (height - box_height) // 2)
        boxes = tuple(
            FaceBox(
                x=max(
                    0, min(width - box_width, (index + 1) * width // (count + 1) - box_width // 2)
                ),
                y=top,
                width=box_width,
                height=box_height,
                confidence=1.0,
            )
            for index in range(count)
        )
        return FaceDetectionResult(boxes, "mock")


class OpenCVHaarFaceDetector:
    """Optional CPU Haar-cascade detector bundled with OpenCV installations."""

    def __init__(self, *, min_face_size: int = 48, scale_factor: float = 1.1) -> None:
        self.min_face_size = min_face_size
        self.scale_factor = scale_factor
        self._classifier: Any | None = None

    def _load(self) -> tuple[Any, Any]:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is not installed; real face detection is optional") from exc
        if self._classifier is None:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            classifier = cv2.CascadeClassifier(cascade_path)
            if classifier.empty():
                raise RuntimeError("OpenCV's frontal-face cascade could not be loaded")
            self._classifier = classifier
        return cv2, self._classifier

    def detect(self, image: Any) -> FaceDetectionResult:
        try:
            cv2, classifier = self._load()
        except RuntimeError as exc:
            return FaceDetectionResult((), "opencv-haar", available=False, reason=str(exc))
        array = image
        gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY) if len(array.shape) == 3 else array
        raw_boxes = classifier.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=5,
            minSize=(self.min_face_size, self.min_face_size),
        )
        boxes = tuple(
            FaceBox(int(x), int(y), int(width), int(height), 1.0)
            for x, y, width, height in raw_boxes
        )
        return FaceDetectionResult(boxes, "opencv-haar")


def count_faces(image: Any, detector: FaceDetector) -> int:
    return detector.detect(image).face_count
