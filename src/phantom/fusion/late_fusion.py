"""Confidence-aware late fusion with explicit missing-data and conflict handling."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from phantom.config import FusionConfig
from phantom.fusion.confidence import (
    distribution_from_label,
    normalize_probabilities,
    normalized_entropy,
)
from phantom.schemas import FUSION_EMOTIONS, EmotionLabel, FusedAffect, Modality, ModalityResult


class LearnedFusionAdapter(Protocol):
    """Experimental interface for a future validated learned-fusion model."""

    def predict(self, results: Mapping[Modality, ModalityResult]) -> FusedAffect:
        """Fuse modality outputs without using optional demographic estimates."""
        ...


class LateFusion:
    """Transparent weighted late fusion suitable as a conservative baseline."""

    def __init__(self, config: FusionConfig | None = None) -> None:
        self.config = config or FusionConfig()

    def fuse(self, results: Mapping[Modality, ModalityResult]) -> FusedAffect:
        usable: list[tuple[Modality, ModalityResult, float]] = []
        for modality, result in results.items():
            if not result.available or result.quality < self.config.quality_floor:
                continue
            base_weight = float(self.config.weights.get(modality.value, 0.0))
            evidence_weight = (
                base_weight * result.quality * result.confidence * result.temporal_consistency
            )
            if evidence_weight > 0.0:
                usable.append((modality, result, evidence_weight))

        if not usable:
            return FusedAffect(
                label=EmotionLabel.UNCERTAIN,
                confidence=0.0,
                uncertain=True,
                explanation=(
                    "The available signals were missing or below the configured quality threshold; "
                    "this is not a medical assessment."
                ),
            )

        weight_total = sum(item[2] for item in usable)
        aggregate = dict.fromkeys(FUSION_EMOTIONS, 0.0)
        contributions: dict[Modality, float] = {}
        top_labels: list[tuple[EmotionLabel, float]] = []
        for modality, result, evidence_weight in usable:
            distribution = (
                normalize_probabilities(result.probabilities)
                if result.probabilities
                else distribution_from_label(result.label, result.confidence)
            )
            share = evidence_weight / weight_total
            contributions[modality] = share
            for label, value in distribution.items():
                aggregate[label] += share * value
            if result.label in FUSION_EMOTIONS:
                top_labels.append((result.label, result.confidence))

        aggregate = normalize_probabilities(aggregate)
        ranked = sorted(aggregate.items(), key=lambda item: item[1], reverse=True)
        top_label, top_probability = ranked[0]
        margin = top_probability - ranked[1][1]
        entropy = normalized_entropy(aggregate)
        strong_labels = {label for label, confidence in top_labels if confidence >= 0.65}
        contradiction = len(strong_labels) > 1

        evidence_strength = min(1.0, weight_total / max(1.0, len(usable)))
        confidence = max(
            0.0, min(1.0, top_probability * (1.0 - 0.35 * entropy) * evidence_strength)
        )
        uncertain = (
            confidence < self.config.confidence_threshold
            or margin < 0.10
            or (
                contradiction
                and max(item[1] for item in top_labels) >= self.config.contradiction_threshold
            )
        )
        label = EmotionLabel.UNCERTAIN if uncertain else top_label
        if contradiction:
            explanation = (
                "The available modalities point in different directions, so the model is uncertain. "
                "No signal is treated as proof of an internal state; this is not a medical assessment."
            )
        elif uncertain:
            explanation = (
                "The available signals are weak or ambiguous, so the model abstained. "
                "This is not a medical assessment."
            )
        else:
            explanation = (
                f"The available signals may indicate {top_label.value}, with uncertainty. "
                "This is not a medical assessment."
            )
        return FusedAffect(
            label=label,
            confidence=confidence,
            uncertain=uncertain,
            probabilities=aggregate,
            contributions=contributions,
            explanation=explanation,
        )
