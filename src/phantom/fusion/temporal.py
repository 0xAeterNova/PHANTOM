"""Short-lived temporal smoothing; no observations are persisted to disk."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping

from phantom.fusion.confidence import normalize_probabilities
from phantom.schemas import FUSION_EMOTIONS, EmotionLabel


class TemporalSmoother:
    """Exponentially weight a bounded in-memory window of distributions."""

    def __init__(self, window_size: int = 5, decay: float = 0.75) -> None:
        if window_size < 1:
            raise ValueError("window_size must be positive")
        if not 0.0 < decay <= 1.0:
            raise ValueError("decay must be in (0, 1]")
        self._values: deque[dict[EmotionLabel, float]] = deque(maxlen=window_size)
        self._decay = decay

    def update(self, probabilities: Mapping[EmotionLabel, float]) -> dict[EmotionLabel, float]:
        self._values.append(normalize_probabilities(probabilities))
        accum = dict.fromkeys(FUSION_EMOTIONS, 0.0)
        weight_total = 0.0
        for age, distribution in enumerate(reversed(self._values)):
            weight = self._decay**age
            weight_total += weight
            for label, value in distribution.items():
                accum[label] += weight * value
        return {label: value / weight_total for label, value in accum.items()}

    def clear(self) -> None:
        self._values.clear()
