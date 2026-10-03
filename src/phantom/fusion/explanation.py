"""Transparent nonverbal fusion traces with an explicit words-first policy."""

from __future__ import annotations

import re
from collections.abc import Mapping

from phantom.config import FusionConfig
from phantom.schemas import (
    AgreementState,
    EmotionLabel,
    FusedAffect,
    FusionTrace,
    Modality,
    ModalityResult,
)

_SELF_REPORT = re.compile(
    r"\b(?:i\s+am|i'm|i\s+feel|i'm\s+feeling)\s+"
    r"(?:happy|sad|angry|mad|afraid|scared|fearful|surprised|disgusted|neutral)\b"
    r"|(?:أنا|انا|إني|اني)\s+(?:أشعر\s+بأنني\s+|اشعر\s+بانني\s+|)"
    r"(?:سعيد|سعيدة|حزين|حزينة|غاضب|غاضبة|خائف|خائفة|متفاجئ|متفاجئة|مشمئز|مشمئزة)",
    re.IGNORECASE,
)


def has_explicit_self_report(text: str) -> bool:
    """Return whether the user directly describes their own current feeling."""

    return bool(_SELF_REPORT.search(text.strip()))


def build_fusion_trace(
    results: Mapping[Modality, ModalityResult],
    fused: FusedAffect,
    *,
    user_text: str,
    config: FusionConfig | None = None,
) -> FusionTrace:
    """Explain voice/face agreement without treating text or age as sensors."""

    selected = config or FusionConfig()
    nonverbal = {
        modality: result
        for modality, result in results.items()
        if modality in {Modality.AUDIO, Modality.VISION}
    }
    excluded: dict[Modality, str] = {}
    strong: list[tuple[Modality, EmotionLabel]] = []
    usable: list[tuple[Modality, EmotionLabel]] = []
    for modality in (Modality.AUDIO, Modality.VISION):
        result = nonverbal.get(modality)
        if result is None:
            excluded[modality] = "not supplied"
            continue
        if not result.available:
            excluded[modality] = result.reason or "unavailable"
            continue
        if result.quality < selected.quality_floor:
            excluded[modality] = "below quality threshold"
            continue
        if result.label not in {
            EmotionLabel.NEUTRAL,
            EmotionLabel.HAPPY,
            EmotionLabel.SAD,
            EmotionLabel.ANGRY,
            EmotionLabel.FEARFUL,
            EmotionLabel.SURPRISED,
            EmotionLabel.DISGUSTED,
        }:
            excluded[modality] = result.reason or "no supported top label"
            continue
        usable.append((modality, result.label))
        if result.confidence >= selected.contradiction_threshold:
            strong.append((modality, result.label))

    if len(strong) == 2 and strong[0][1] == strong[1][1]:
        agreement = AgreementState.STRONG_AGREEMENT
    elif len(strong) == 2 and strong[0][1] != strong[1][1]:
        agreement = AgreementState.CONFLICT
    elif len(usable) == 2 and usable[0][1] == usable[1][1]:
        agreement = AgreementState.PARTIAL_AGREEMENT
    else:
        agreement = AgreementState.INSUFFICIENT_EVIDENCE

    explicit = has_explicit_self_report(user_text)
    if agreement is AgreementState.CONFLICT:
        explanation = (
            "Voice and facial-expression estimates conflict, so the nonverbal estimate abstains. "
            "The user's words remain the primary source for the response."
        )
    elif agreement is AgreementState.STRONG_AGREEMENT:
        explanation = (
            "Voice and facial-expression estimates agree, but they remain probabilistic style "
            "context; the user's words still take priority."
        )
    elif agreement is AgreementState.PARTIAL_AGREEMENT:
        explanation = (
            "Voice and facial-expression estimates point in the same direction with limited "
            "strength. The user's words take priority."
        )
    else:
        explanation = (
            "There is not enough reliable nonverbal evidence to assess agreement. The user's "
            "words take priority."
        )
    if explicit:
        explanation += " An explicit self-report was detected and overrides conflicting sensors."

    return FusionTrace(
        agreement=agreement,
        label=fused.label,
        confidence=fused.confidence,
        uncertain=fused.uncertain,
        contributions=fused.contributions,
        excluded=excluded,
        explanation=explanation,
        words_have_priority=True,
        explicit_self_report=explicit,
    )
