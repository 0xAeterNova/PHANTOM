"""Replaceable facial-expression model interfaces and local baselines."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from importlib import import_module
from pathlib import Path
from threading import Lock
from typing import Any, ClassVar, Protocol, runtime_checkable

from phantom.fusion.confidence import normalize_probabilities
from phantom.schemas import FUSION_EMOTIONS, EmotionLabel


@runtime_checkable
class FacialExpressionModel(Protocol):
    """Classify an isolated face crop without identifying the person."""

    def predict(self, face: Any) -> Mapping[EmotionLabel, float]: ...


class TrainableFacialExpressionModel(Protocol):
    """Experimental training interface requiring subject-independent evaluation."""

    def fit(self, faces: Sequence[Any], labels: Sequence[EmotionLabel]) -> None: ...

    def predict(self, face: Any) -> Mapping[EmotionLabel, float]: ...


class BaselineFacialExpressionModel:
    """Conservative deterministic image-statistics baseline.

    It is deliberately not presented as a trained expression recognizer.  It
    provides stable local behavior for integration tests and should remain low
    confidence until a documented model and evaluation are configured.
    """

    def predict(self, face: Any) -> Mapping[EmotionLabel, float]:
        try:
            np = import_module("numpy")
        except ModuleNotFoundError as exc:  # pragma: no cover - packaging guard
            raise RuntimeError("NumPy is required for baseline vision inference") from exc
        array = np.asarray(face, dtype=np.float32)
        if array.size == 0:
            return normalize_probabilities({EmotionLabel.NEUTRAL: 1.0})
        if float(np.max(array, initial=0.0)) > 1.0:
            array = array / 255.0
        brightness = float(np.mean(array))
        contrast = float(np.std(array))
        channel_means = np.mean(array[:, :, :3], axis=(0, 1))
        warmth = float(channel_means[0] - channel_means[2])
        height = array.shape[0]
        upper = float(np.mean(array[: max(1, height // 2)]))
        lower = float(np.mean(array[max(1, height // 2) :])) if height > 1 else upper
        vertical_difference = abs(upper - lower)

        scores = {
            EmotionLabel.NEUTRAL: 2.55 - 0.25 * min(1.0, contrast * 4.0),
            EmotionLabel.HAPPY: 1.0 + 0.25 * brightness + 0.25 * max(0.0, warmth),
            EmotionLabel.SAD: 1.0 + 0.25 * (1.0 - brightness),
            EmotionLabel.ANGRY: 0.85 + 0.35 * min(1.0, contrast * 4.0),
            EmotionLabel.FEARFUL: 0.80 + 0.25 * min(1.0, contrast * 4.0),
            EmotionLabel.SURPRISED: 0.80 + 0.35 * min(1.0, vertical_difference * 6.0),
            EmotionLabel.DISGUSTED: 0.80 + 0.20 * max(0.0, -warmth),
        }
        logits = np.asarray([scores[label] for label in FUSION_EMOTIONS], dtype=np.float64) / 1.65
        logits -= float(np.max(logits))
        values = np.exp(logits)
        values /= float(values.sum())
        return {label: float(values[index]) for index, label in enumerate(FUSION_EMOTIONS)}


def _prepare_deepface_rgb_face(face: Any) -> Any:
    """Validate an isolated RGB crop and return DeepFace-compatible BGR float data."""

    try:
        np = import_module("numpy")
    except ModuleNotFoundError as exc:  # pragma: no cover - packaging guard
        raise RuntimeError("NumPy is required for DeepFace vision inference") from exc

    array = np.asarray(face)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError("DeepFace expects an isolated RGB face crop with three channels")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("DeepFace face crop must not be empty")
    array = np.asarray(array[:, :, :3], dtype=np.float32)
    if not bool(np.isfinite(array).all()):
        raise ValueError("DeepFace face crop contains non-finite values")

    minimum = float(array.min())
    maximum = float(array.max())
    if minimum < 0.0 or maximum > 255.0:
        raise ValueError("DeepFace face crop values must be in the range [0, 1] or [0, 255]")
    if maximum > 1.0:
        array = array / 255.0

    if tuple(array.shape[:2]) != (224, 224):
        try:
            cv2 = import_module("cv2")
        except ModuleNotFoundError as exc:
            raise RuntimeError("OpenCV is required to resize DeepFace face crops") from exc
        interpolation = (
            cv2.INTER_AREA if array.shape[0] > 224 or array.shape[1] > 224 else cv2.INTER_LINEAR
        )
        array = cv2.resize(array, (224, 224), interpolation=interpolation)

    # The PHANTOM vision pipeline supplies RGB. DeepFace's demographic clients
    # accept BGR arrays normalized to [0, 1]. No detector or identity model is used.
    return np.ascontiguousarray(array[:, :, ::-1], dtype=np.float32)


class DeepFaceFacialExpressionAdapter:
    """Lazy pretrained DeepFace facial-expression adapter for isolated RGB crops.

    The adapter loads only DeepFace's ``Emotion`` facial-attribute client. It
    deliberately bypasses face detection and does not create identity embeddings.
    """

    _SOURCE_LABELS: ClassVar[tuple[str, ...]] = (
        "angry",
        "disgust",
        "fear",
        "happy",
        "sad",
        "surprise",
        "neutral",
    )
    _LABEL_MAP: ClassVar[dict[str, EmotionLabel]] = {
        "angry": EmotionLabel.ANGRY,
        "disgust": EmotionLabel.DISGUSTED,
        "fear": EmotionLabel.FEARFUL,
        "happy": EmotionLabel.HAPPY,
        "sad": EmotionLabel.SAD,
        "surprise": EmotionLabel.SURPRISED,
        "neutral": EmotionLabel.NEUTRAL,
    }

    backend = "deepface"
    architecture = "deepface_emotion"
    model_name = "Emotion"

    def __init__(self) -> None:
        self._model: Any | None = None
        self._load_lock = Lock()
        self.provenance: dict[str, Any] = {
            "backend": self.backend,
            "provider": "serengil/deepface",
            "model": self.model_name,
            "task": "facial_attribute",
            "weights": "facial_expression_model_weights.h5",
            "input": "isolated_rgb_face_crop",
            "identity_recognition": False,
            "face_embeddings_created": False,
        }

    def _load(self) -> Any:
        with self._load_lock:
            if self._model is None:
                try:
                    # DeepFace 0.0.95 prints emoji while downloading weights;
                    # Windows' legacy cp1252 console raises UnicodeEncodeError.
                    os.environ.setdefault("DEEPFACE_LOG_LEVEL", "30")
                    deepface_module = import_module("deepface")
                    deepface = getattr(deepface_module, "DeepFace", None)
                    if deepface is None:
                        # DeepFace 0.0.95 exposes its public facade as the
                        # ``deepface.DeepFace`` module, not on the package root.
                        deepface = import_module("deepface.DeepFace")
                except (AttributeError, ImportError, ModuleNotFoundError) as exc:
                    raise RuntimeError(
                        "DeepFace facial-expression inference requires the optional deepface, "
                        "TensorFlow, and OpenCV dependencies"
                    ) from exc
                self._model = deepface.build_model(
                    task="facial_attribute",
                    model_name=self.model_name,
                )
                version = getattr(deepface_module, "__version__", None)
                if version is not None:
                    self.provenance["backend_version"] = str(version)
            return self._model

    def predict(self, face: Any) -> Mapping[EmotionLabel, float]:
        model_input = _prepare_deepface_rgb_face(face)
        model = self._load()
        try:
            np = import_module("numpy")
        except ModuleNotFoundError as exc:  # pragma: no cover - packaging guard
            raise RuntimeError("NumPy is required for DeepFace vision inference") from exc

        raw = np.asarray(model.predict(model_input), dtype=np.float64).reshape(-1)
        if raw.size != len(self._SOURCE_LABELS):
            raise RuntimeError(
                "DeepFace Emotion returned an unexpected number of class probabilities"
            )
        if not bool(np.isfinite(raw).all()) or bool((raw < 0.0).any()):
            raise RuntimeError("DeepFace Emotion returned invalid class probabilities")

        mapped = dict.fromkeys(FUSION_EMOTIONS, 0.0)
        for source_label, probability in zip(self._SOURCE_LABELS, raw, strict=True):
            canonical = self._LABEL_MAP.get(source_label)
            if canonical is not None:
                mapped[canonical] += float(probability)
        if sum(mapped.values()) <= 0.0:
            raise RuntimeError("DeepFace Emotion returned no supported emotion probability")
        return normalize_probabilities(mapped)


class HuggingFaceFacialExpressionAdapter:
    """Optional local-only Hugging Face image-classification adapter."""

    _ALIASES: ClassVar[dict[str, EmotionLabel]] = {
        "neutral": EmotionLabel.NEUTRAL,
        "happy": EmotionLabel.HAPPY,
        "happiness": EmotionLabel.HAPPY,
        "sad": EmotionLabel.SAD,
        "sadness": EmotionLabel.SAD,
        "angry": EmotionLabel.ANGRY,
        "anger": EmotionLabel.ANGRY,
        "fear": EmotionLabel.FEARFUL,
        "fearful": EmotionLabel.FEARFUL,
        "surprise": EmotionLabel.SURPRISED,
        "surprised": EmotionLabel.SURPRISED,
        "disgust": EmotionLabel.DISGUSTED,
        "disgusted": EmotionLabel.DISGUSTED,
    }

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        revision: str,
        device: str = "cpu",
        local_files_only: bool = True,
    ) -> None:
        self.model_name_or_path = str(model_name_or_path)
        self.revision = revision
        self.device = device
        self.local_files_only = local_files_only
        self._processor: Any | None = None
        self._model: Any | None = None

    def _load(self) -> tuple[Any, Any]:
        if self._processor is not None and self._model is not None:
            return self._processor, self._model
        try:
            from transformers import AutoImageProcessor, AutoModelForImageClassification
        except ImportError as exc:
            raise RuntimeError(
                "The pretrained vision adapter requires optional transformers and PyTorch"
            ) from exc
        self._processor = AutoImageProcessor.from_pretrained(
            self.model_name_or_path,
            revision=self.revision,
            local_files_only=self.local_files_only,
        )
        self._model = AutoModelForImageClassification.from_pretrained(
            self.model_name_or_path,
            revision=self.revision,
            local_files_only=self.local_files_only,
        )
        self._model.to(self.device)
        self._model.eval()
        return self._processor, self._model

    def predict(self, face: Any) -> Mapping[EmotionLabel, float]:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("The pretrained vision adapter requires PyTorch") from exc
        processor, model = self._load()
        inputs = processor(images=face, return_tensors="pt")
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        with torch.inference_mode():
            raw = torch.softmax(model(**inputs).logits[0], dim=-1).detach().cpu().tolist()
        id_to_label = getattr(model.config, "id2label", {})
        mapped = dict.fromkeys(FUSION_EMOTIONS, 0.0)
        for index, probability in enumerate(raw):
            source = str(id_to_label.get(index, id_to_label.get(str(index), ""))).strip().lower()
            canonical = self._ALIASES.get(source)
            if canonical is not None:
                mapped[canonical] += float(probability)
        if sum(mapped.values()) <= 0.0:
            raise RuntimeError("pretrained model labels do not map to PHANTOM emotion labels")
        return normalize_probabilities(mapped)


VisionEmotionModel = FacialExpressionModel
PretrainedVisionEmotionAdapter = HuggingFaceFacialExpressionAdapter
DeepFaceEmotionAdapter = DeepFaceFacialExpressionAdapter
