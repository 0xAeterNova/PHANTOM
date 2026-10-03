from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from phantom.schemas import (
    FUSION_EMOTIONS,
    AgeBand,
    ConsentSettings,
    EmotionLabel,
    GenderPresentation,
)
from phantom.vision import emotion_model as emotion_module
from phantom.vision import optional_attributes as attributes_module
from phantom.vision.emotion_model import DeepFaceFacialExpressionAdapter
from phantom.vision.optional_attributes import (
    DeepFaceApproximateAgeEstimator,
    estimate_optional_attributes,
)


class _FakeAttributeModel:
    def __init__(self, output: Any) -> None:
        self.output = output
        self.inputs: list[np.ndarray] = []

    def predict(self, face: Any) -> Any:
        self.inputs.append(np.asarray(face).copy())
        return self.output


class _FakeDeepFaceFacade:
    def __init__(self, model: _FakeAttributeModel) -> None:
        self.model = model
        self.build_calls: list[tuple[str, str]] = []

    def build_model(self, *, task: str, model_name: str) -> _FakeAttributeModel:
        self.build_calls.append((task, model_name))
        return self.model


def _patch_deepface_import(
    monkeypatch: pytest.MonkeyPatch,
    module: Any,
    model: _FakeAttributeModel,
) -> _FakeDeepFaceFacade:
    facade = _FakeDeepFaceFacade(model)
    real_import = module.import_module

    def fake_import(name: str) -> Any:
        if name == "deepface":
            return SimpleNamespace(DeepFace=facade, __version__="test-version")
        return real_import(name)

    monkeypatch.setattr(module, "import_module", fake_import)
    return facade


def _red_rgb_crop() -> np.ndarray:
    crop = np.zeros((224, 224, 3), dtype=np.uint8)
    crop[:, :, 0] = 255
    return crop


def test_deepface_emotion_adapter_loads_once_and_maps_real_model_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _FakeAttributeModel(
        np.asarray([0.06, 0.04, 0.10, 0.55, 0.08, 0.12, 0.05], dtype=np.float32)
    )
    facade = _patch_deepface_import(monkeypatch, emotion_module, model)
    adapter = DeepFaceFacialExpressionAdapter()

    first = adapter.predict(_red_rgb_crop())
    second = adapter.predict(_red_rgb_crop())

    assert facade.build_calls == [("facial_attribute", "Emotion")]
    assert len(model.inputs) == 2
    assert model.inputs[0].shape == (224, 224, 3)
    assert model.inputs[0].dtype == np.float32
    assert model.inputs[0][0, 0].tolist() == pytest.approx([0.0, 0.0, 1.0])
    assert set(first) == set(FUSION_EMOTIONS)
    assert sum(first.values()) == pytest.approx(1.0)
    assert first[EmotionLabel.HAPPY] == pytest.approx(0.55)
    assert first[EmotionLabel.FEARFUL] == pytest.approx(0.10)
    assert second == pytest.approx(first)
    assert adapter.provenance == {
        "backend": "deepface",
        "provider": "serengil/deepface",
        "model": "Emotion",
        "task": "facial_attribute",
        "weights": "facial_expression_model_weights.h5",
        "input": "isolated_rgb_face_crop",
        "identity_recognition": False,
        "face_embeddings_created": False,
        "backend_version": "test-version",
    }


def test_deepface_emotion_adapter_rejects_unexpected_model_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _FakeAttributeModel(np.asarray([0.5, 0.5], dtype=np.float32))
    _patch_deepface_import(monkeypatch, emotion_module, model)

    with pytest.raises(RuntimeError, match="unexpected number"):
        DeepFaceFacialExpressionAdapter().predict(_red_rgb_crop())


def test_deepface_0095_submodule_facade_and_windows_logging_are_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _FakeAttributeModel(
        np.asarray([0.06, 0.04, 0.10, 0.55, 0.08, 0.12, 0.05], dtype=np.float32)
    )
    facade = _FakeDeepFaceFacade(model)
    imports: list[str] = []
    real_import = emotion_module.import_module

    def fake_import(name: str) -> Any:
        imports.append(name)
        if name == "deepface":
            return SimpleNamespace(__version__="0.0.95")
        if name == "deepface.DeepFace":
            return facade
        return real_import(name)

    monkeypatch.delenv("DEEPFACE_LOG_LEVEL", raising=False)
    monkeypatch.setattr(emotion_module, "import_module", fake_import)

    result = DeepFaceFacialExpressionAdapter().predict(_red_rgb_crop())

    assert [name for name in imports if name.startswith("deepface")][:2] == [
        "deepface",
        "deepface.DeepFace",
    ]
    assert facade.build_calls == [("facial_attribute", "Emotion")]
    assert sum(result.values()) == pytest.approx(1.0)
    assert emotion_module.os.environ["DEEPFACE_LOG_LEVEL"] == "30"


def test_deepface_age_estimator_is_lazy_uncertain_and_age_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _FakeAttributeModel(np.float64(27.4))
    facade = _patch_deepface_import(monkeypatch, attributes_module, model)
    estimator = DeepFaceApproximateAgeEstimator()

    direct = estimator.estimate(_red_rgb_crop())
    consented = estimate_optional_attributes(
        _red_rgb_crop(),
        ConsentSettings(camera=True, age_band_analysis=True),
        estimator,
    )

    assert facade.build_calls == [("facial_attribute", "Age")]
    assert direct.estimated_age == pytest.approx(27.4)
    assert direct.age_range == "20-29"
    assert direct.age_band is AgeBand.YOUNG_ADULT
    assert direct.confidence == 0.0
    assert direct.uncertain is True
    assert direct.perceived_gender_presentation is GenderPresentation.DISABLED
    assert consented.estimated_age == pytest.approx(27.4)
    assert consented.age_range == "20-29"
    assert consented.age_band is AgeBand.YOUNG_ADULT
    assert consented.confidence == 0.0
    assert consented.uncertain is True
    assert model.inputs[0][0, 0].tolist() == pytest.approx([0.0, 0.0, 1.0])
    assert estimator.provenance["model"] == "Age"
    assert estimator.provenance["gender_model_loaded"] is False
    assert estimator.provenance["race_model_loaded"] is False
    assert estimator.provenance["identity_recognition"] is False


def test_age_estimator_is_not_called_without_separate_age_consent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _FakeAttributeModel(np.float64(42.0))
    facade = _patch_deepface_import(monkeypatch, attributes_module, model)
    estimator = DeepFaceApproximateAgeEstimator()

    result = estimate_optional_attributes(
        _red_rgb_crop(),
        ConsentSettings(
            camera=True,
            perceived_gender_presentation_analysis=True,
        ),
        estimator,
    )

    assert facade.build_calls == []
    assert model.inputs == []
    assert result.age_band is AgeBand.DISABLED
    assert result.estimated_age is None
    assert result.age_range is None
    assert result.perceived_gender_presentation is GenderPresentation.UNKNOWN
    assert result.confidence == 0.0
    assert result.uncertain is True


@pytest.mark.parametrize(
    ("age", "band", "age_range"),
    [
        (8.5, AgeBand.CHILD, "0-9"),
        (13.0, AgeBand.TEENAGER, "10-19"),
        (18.0, AgeBand.YOUNG_ADULT, "10-19"),
        (30.0, AgeBand.ADULT, "30-39"),
        (65.0, AgeBand.OLDER_ADULT, "60-69"),
    ],
)
def test_age_band_and_decade_range_boundaries(
    age: float,
    band: AgeBand,
    age_range: str,
) -> None:
    assert attributes_module._age_band_for_estimate(age) is band
    assert attributes_module._decade_age_range(age) == age_range
