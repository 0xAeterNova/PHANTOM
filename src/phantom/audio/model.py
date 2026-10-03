"""Replaceable acoustic-emotion model interfaces and conservative baselines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable

from phantom.fusion.confidence import normalize_probabilities, temperature_scale
from phantom.schemas import FUSION_EMOTIONS, EmotionLabel


@runtime_checkable
class AcousticEmotionModel(Protocol):
    """Feature-based inference interface used by the lightweight baseline."""

    def predict(self, features: Any) -> Mapping[EmotionLabel, float]:
        """Return uncalibrated probabilities over canonical affect labels."""
        ...


@runtime_checkable
class WaveformEmotionModel(Protocol):
    """Interface for pretrained speech encoders that consume raw waveforms."""

    def predict_waveform(self, samples: Any, sample_rate: int) -> Mapping[EmotionLabel, float]: ...


class TrainableAcousticModel(Protocol):
    """Experimental training surface; implementations must document data splits."""

    def fit(self, features: Sequence[Any], labels: Sequence[EmotionLabel]) -> None: ...

    def predict(self, features: Any) -> Mapping[EmotionLabel, float]: ...


@dataclass(frozen=True, slots=True)
class TemperatureCalibrator:
    """Probability-space temperature scaling for a held-out calibrated model."""

    temperature: float = 1.35

    def calibrate(self, probabilities: Mapping[EmotionLabel, float]) -> dict[EmotionLabel, float]:
        return temperature_scale(probabilities, self.temperature)


class BaselineAcousticEmotionModel:
    """Small deterministic heuristic baseline over log-Mel features.

    This baseline is deliberately conservative and exists to exercise the full
    local pipeline.  It is not trained, clinically validated, or suitable for
    claims about a person's internal state.  Replace it with a documented,
    evaluated adapter for research experiments.
    """

    def predict(self, features: Any) -> Mapping[EmotionLabel, float]:
        try:
            np = import_module("numpy")
        except ModuleNotFoundError as exc:  # pragma: no cover - packaging guard
            raise RuntimeError("NumPy is required for baseline audio inference") from exc
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2 or values.size == 0:
            return normalize_probabilities({EmotionLabel.NEUTRAL: 1.0})

        mean_energy = float(np.mean(values))
        dynamics = float(np.std(np.mean(values, axis=1))) if values.shape[0] > 1 else 0.0
        per_band = np.maximum(np.mean(np.exp(np.clip(values, -30.0, 20.0)), axis=0), 1e-12)
        positions = np.linspace(0.0, 1.0, per_band.size)
        brightness = float(np.sum(positions * per_band) / np.sum(per_band))
        energy = 1.0 / (1.0 + np.exp(-(mean_energy + 9.0) / 2.5))
        changing = 1.0 / (1.0 + np.exp(-(dynamics - 0.7) * 2.0))

        # Scores encode only coarse acoustic activation; softmax below prevents
        # this untrained baseline from becoming overconfident.
        scores = {
            EmotionLabel.NEUTRAL: 2.4 - 0.25 * changing,
            EmotionLabel.HAPPY: 1.0 + 0.45 * energy + 0.35 * brightness,
            EmotionLabel.SAD: 1.0 + 0.45 * (1.0 - energy) + 0.2 * (1.0 - brightness),
            EmotionLabel.ANGRY: 0.9 + 0.55 * energy + 0.35 * changing,
            EmotionLabel.FEARFUL: 0.8 + 0.35 * brightness + 0.35 * changing,
            EmotionLabel.SURPRISED: 0.8 + 0.25 * energy + 0.55 * changing,
            EmotionLabel.DISGUSTED: 0.8 + 0.25 * (1.0 - brightness),
        }
        logits = np.asarray([scores[label] for label in FUSION_EMOTIONS], dtype=np.float64) / 1.65
        logits -= float(np.max(logits))
        probabilities = np.exp(logits)
        probabilities /= float(probabilities.sum())
        return {label: float(probabilities[index]) for index, label in enumerate(FUSION_EMOTIONS)}


class HuggingFaceSpeechEmotionAdapter:
    """Optional local-only Hugging Face audio-classification adapter.

    ``transformers`` and ``torch`` are optional dependencies.  By default the
    adapter only opens model files already present on the device and never
    downloads weights.  The caller must supply a model whose labels can be
    mapped to PHANTOM's canonical non-clinical affect labels.
    """

    backend = "huggingface-audio-classification"

    _ALIASES: ClassVar[dict[str, EmotionLabel]] = {
        "neutral": EmotionLabel.NEUTRAL,
        "neu": EmotionLabel.NEUTRAL,
        "happy": EmotionLabel.HAPPY,
        "happiness": EmotionLabel.HAPPY,
        "hap": EmotionLabel.HAPPY,
        "sad": EmotionLabel.SAD,
        "sadness": EmotionLabel.SAD,
        "angry": EmotionLabel.ANGRY,
        "anger": EmotionLabel.ANGRY,
        "ang": EmotionLabel.ANGRY,
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
        self._extractor: Any | None = None
        self._model: Any | None = None

    @property
    def provenance(self) -> dict[str, str | bool]:
        """Return the exact pretrained backend and immutable model selection."""

        provenance: dict[str, str | bool] = {
            "backend": self.backend,
            "model_name_or_path": self.model_name_or_path,
            "revision": self.revision,
            "device": self.device,
            "local_files_only": self.local_files_only,
        }
        if self.model_name_or_path.lower().rstrip("/") == "superb/wav2vec2-base-superb-er":
            provenance.update(
                {
                    "evaluation_dataset": "IEMOCAP four-class protocol",
                    "source_language": "English",
                    "source_labels": "neutral,happy,angry,sad",
                    "arabic_validation_documented": False,
                }
            )
        return provenance

    def _load(self) -> tuple[Any, Any]:
        if self._extractor is not None and self._model is not None:
            return self._extractor, self._model
        try:
            from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
        except ImportError as exc:
            raise RuntimeError(
                "The pretrained audio adapter requires the optional 'transformers' and 'torch' packages"
            ) from exc
        self._extractor = AutoFeatureExtractor.from_pretrained(
            self.model_name_or_path,
            revision=self.revision,
            local_files_only=self.local_files_only,
        )
        self._model = AutoModelForAudioClassification.from_pretrained(
            self.model_name_or_path,
            revision=self.revision,
            local_files_only=self.local_files_only,
        )
        self._model.to(self.device)
        self._model.eval()
        return self._extractor, self._model

    def predict_waveform(self, samples: Any, sample_rate: int) -> Mapping[EmotionLabel, float]:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("The pretrained audio adapter requires PyTorch") from exc
        extractor, model = self._load()
        inputs = extractor(samples, sampling_rate=sample_rate, return_tensors="pt")
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


# Explicit name used by research configuration files.
PretrainedSpeechEmotionAdapter = HuggingFaceSpeechEmotionAdapter
