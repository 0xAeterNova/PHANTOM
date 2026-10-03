"""Small, inspectable confidence-calibration utilities."""

from __future__ import annotations

import math
from collections.abc import Mapping

from phantom.schemas import FUSION_EMOTIONS, EmotionLabel


def normalize_probabilities(
    values: Mapping[EmotionLabel, float],
) -> dict[EmotionLabel, float]:
    """Clamp and normalize emotion probabilities over the canonical label set."""

    cleaned = {label: max(0.0, float(values.get(label, 0.0))) for label in FUSION_EMOTIONS}
    total = sum(cleaned.values())
    if total <= 0.0:
        uniform = 1.0 / len(FUSION_EMOTIONS)
        return dict.fromkeys(FUSION_EMOTIONS, uniform)
    return {label: value / total for label, value in cleaned.items()}


def temperature_scale(
    probabilities: Mapping[EmotionLabel, float], temperature: float = 1.0
) -> dict[EmotionLabel, float]:
    """Apply probability-space temperature scaling without requiring model logits."""

    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    normalized = normalize_probabilities(probabilities)
    adjusted = {
        label: math.exp(math.log(max(value, 1e-12)) / temperature)
        for label, value in normalized.items()
    }
    return normalize_probabilities(adjusted)


def normalized_entropy(probabilities: Mapping[EmotionLabel, float]) -> float:
    """Return Shannon entropy in [0, 1], with 1 representing maximum ambiguity."""

    normalized = normalize_probabilities(probabilities)
    entropy = -sum(value * math.log(max(value, 1e-12)) for value in normalized.values())
    return min(1.0, max(0.0, entropy / math.log(len(FUSION_EMOTIONS))))


def distribution_from_label(label: EmotionLabel, confidence: float) -> dict[EmotionLabel, float]:
    """Create a conservative distribution when an adapter only exposes top-1 output."""

    if label not in FUSION_EMOTIONS:
        return normalize_probabilities({})
    bounded = min(1.0, max(1.0 / len(FUSION_EMOTIONS), confidence))
    remainder = (1.0 - bounded) / (len(FUSION_EMOTIONS) - 1)
    return {
        candidate: bounded if candidate == label else remainder for candidate in FUSION_EMOTIONS
    }
