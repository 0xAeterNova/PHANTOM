"""Explicitly experimental, consent-gated visual presentation estimates.

These attributes are never inputs to affect inference, fusion, access control,
or safety decisions.  No identity or face embedding is created or retained.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from importlib import import_module
from threading import Lock
from typing import Any, ClassVar, Protocol, runtime_checkable

from phantom.schemas import (
    AgeBand,
    ConsentSettings,
    GenderPresentation,
    OptionalDemographicEstimates,
)
from phantom.vision.emotion_model import _prepare_deepface_rgb_face

_AGE_ATTRIBUTE = "age"
_GENDER_PRESENTATION_ATTRIBUTE = "perceived_gender_presentation"


@runtime_checkable
class ExperimentalVisualAttributeEstimator(Protocol):
    """Opt-in research interface; implementations require independent review."""

    experimental: bool

    def estimate(self, face: Any) -> OptionalDemographicEstimates: ...


OptionalAttributeEstimator = ExperimentalVisualAttributeEstimator


class DisabledOptionalAttributeEstimator:
    """Privacy-safe default that performs no sensitive visual analysis."""

    experimental = True
    supported_attributes: frozenset[str] = frozenset()

    def estimate(self, face: Any) -> OptionalDemographicEstimates:
        del face
        return OptionalDemographicEstimates()


@dataclass(slots=True)
class MockOptionalAttributeEstimator:
    """Deterministic opt-in test adapter; not an appearance inference model."""

    age_band: AgeBand = AgeBand.UNKNOWN
    perceived_gender_presentation: GenderPresentation = GenderPresentation.UNKNOWN
    confidence: float = 0.0
    experimental: bool = True
    accepts_missing_face: ClassVar[bool] = True
    supported_attributes: ClassVar[frozenset[str]] = frozenset(
        {_AGE_ATTRIBUTE, _GENDER_PRESENTATION_ATTRIBUTE}
    )

    def estimate(self, face: Any) -> OptionalDemographicEstimates:
        del face
        confidence = max(0.0, min(1.0, float(self.confidence)))
        return OptionalDemographicEstimates(
            age_band=self.age_band,
            perceived_gender_presentation=self.perceived_gender_presentation,
            confidence=confidence,
            uncertain=confidence < 0.65,
            notice=(
                "Experimental mock presentation estimates; not identity, biological sex, "
                "or emotion evidence."
            ),
        )


def _age_band_for_estimate(age: float) -> AgeBand:
    if age < 13.0:
        return AgeBand.CHILD
    if age < 18.0:
        return AgeBand.TEENAGER
    if age < 30.0:
        return AgeBand.YOUNG_ADULT
    if age < 65.0:
        return AgeBand.ADULT
    return AgeBand.OLDER_ADULT


def _decade_age_range(age: float) -> str:
    lower = int(age // 10.0) * 10
    upper = min(120, lower + 9)
    return f"{lower}-{upper}"


class DeepFaceApproximateAgeEstimator:
    """Consent-gated apparent-age adapter using only DeepFace's age model.

    This adapter does not load gender, race, face-recognition, or identity models.
    DeepFace does not expose a calibrated confidence for apparent age, so the
    returned estimate is always explicitly uncertain with confidence ``0.0``.
    """

    experimental = True
    supported_attributes: frozenset[str] = frozenset({_AGE_ATTRIBUTE})
    backend = "deepface"
    model_name = "Age"

    def __init__(self) -> None:
        self._model: Any | None = None
        self._load_lock = Lock()
        self.provenance: dict[str, Any] = {
            "backend": self.backend,
            "provider": "serengil/deepface",
            "model": self.model_name,
            "task": "facial_attribute",
            "weights": "age_model_weights.h5",
            "input": "isolated_rgb_face_crop",
            "calibrated_confidence_available": False,
            "identity_recognition": False,
            "face_embeddings_created": False,
            "gender_model_loaded": False,
            "race_model_loaded": False,
        }

    def _load(self) -> Any:
        with self._load_lock:
            if self._model is None:
                try:
                    os.environ.setdefault("DEEPFACE_LOG_LEVEL", "30")
                    deepface_module = import_module("deepface")
                    deepface = getattr(deepface_module, "DeepFace", None)
                    if deepface is None:
                        # DeepFace 0.0.95 keeps the facade in this submodule.
                        deepface = import_module("deepface.DeepFace")
                except (AttributeError, ImportError, ModuleNotFoundError) as exc:
                    raise RuntimeError(
                        "DeepFace age estimation requires the optional deepface, TensorFlow, "
                        "and OpenCV dependencies"
                    ) from exc
                self._model = deepface.build_model(
                    task="facial_attribute",
                    model_name=self.model_name,
                )
                version = getattr(deepface_module, "__version__", None)
                if version is not None:
                    self.provenance["backend_version"] = str(version)
            return self._model

    def estimate(self, face: Any) -> OptionalDemographicEstimates:
        model_input = _prepare_deepface_rgb_face(face)
        apparent_age = float(self._load().predict(model_input))
        if not math.isfinite(apparent_age) or not 0.0 <= apparent_age <= 120.0:
            raise RuntimeError("DeepFace Age returned an invalid apparent-age estimate")
        estimated_age = round(apparent_age, 1)
        return OptionalDemographicEstimates(
            age_band=_age_band_for_estimate(apparent_age),
            estimated_age=estimated_age,
            age_range=_decade_age_range(apparent_age),
            perceived_gender_presentation=GenderPresentation.DISABLED,
            confidence=0.0,
            uncertain=True,
            notice=(
                "Experimental DeepFace apparent-age estimate; approximate and uncalibrated. "
                "No gender, race, identity, or face-recognition analysis was performed, and "
                "the estimate is not emotion evidence."
            ),
        )


def estimate_optional_attributes(
    face: Any,
    consent: ConsentSettings,
    estimator: ExperimentalVisualAttributeEstimator | None = None,
) -> OptionalDemographicEstimates:
    """Enforce separate consent and mask every non-consented output."""

    age_enabled = consent.camera and consent.age_band_analysis
    gender_enabled = consent.camera and consent.perceived_gender_presentation_analysis
    if not age_enabled and not gender_enabled:
        return OptionalDemographicEstimates()
    if estimator is None:
        return OptionalDemographicEstimates(
            age_band=AgeBand.UNKNOWN if age_enabled else AgeBand.DISABLED,
            perceived_gender_presentation=(
                GenderPresentation.UNKNOWN if gender_enabled else GenderPresentation.DISABLED
            ),
            confidence=0.0,
            uncertain=True,
            notice=(
                "Optional analysis was consented to, but no experimental attribute model is "
                "configured; attributes are not emotion evidence."
            ),
        )
    if not getattr(estimator, "experimental", False):
        raise ValueError(
            "optional visual attribute adapters must be explicitly marked experimental"
        )
    supported = frozenset(
        getattr(
            estimator,
            "supported_attributes",
            {_AGE_ATTRIBUTE, _GENDER_PRESENTATION_ATTRIBUTE},
        )
    )
    age_supported = age_enabled and _AGE_ATTRIBUTE in supported
    gender_supported = gender_enabled and _GENDER_PRESENTATION_ATTRIBUTE in supported
    if not age_supported and not gender_supported:
        return OptionalDemographicEstimates(
            age_band=AgeBand.UNKNOWN if age_enabled else AgeBand.DISABLED,
            perceived_gender_presentation=(
                GenderPresentation.UNKNOWN if gender_enabled else GenderPresentation.DISABLED
            ),
            confidence=0.0,
            uncertain=True,
            notice=(
                "Optional analysis was consented to, but the configured experimental model "
                "does not support that attribute; no estimate was produced."
            ),
        )
    estimate = estimator.estimate(face)
    return OptionalDemographicEstimates(
        age_band=estimate.age_band if age_supported else AgeBand.DISABLED,
        estimated_age=estimate.estimated_age if age_supported else None,
        age_range=estimate.age_range if age_supported else None,
        perceived_gender_presentation=(
            estimate.perceived_gender_presentation
            if gender_supported
            else GenderPresentation.DISABLED
        ),
        confidence=estimate.confidence,
        uncertain=estimate.uncertain,
        notice=estimate.notice,
    )


disabled_optional_attributes = OptionalDemographicEstimates

# Short aliases for configuration code while preserving descriptive public names.
DeepFaceAgeEstimator = DeepFaceApproximateAgeEstimator
