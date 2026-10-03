"""Deterministic response generator and optional provider interface."""

from __future__ import annotations

from typing import Protocol

from phantom.dialogue.policy import DialoguePolicy
from phantom.dialogue.templates import (
    CORRECTION,
    EMOTION_ACKNOWLEDGEMENTS,
    GROUNDING_PERMISSION_GRANTED,
    HUMAN_SUPPORT,
    NON_MEDICAL_NOTICE,
    UNCERTAINTY,
)
from phantom.safety.guardrails import enforce_response
from phantom.schemas import (
    AnalysisResult,
    DialogueResponse,
    SafetyResult,
    SafetyRoute,
    UserPreferences,
)


class LLMProvider(Protocol):
    """Optional provider boundary; implementations must still pass policy guardrails."""

    def generate(self, system_prompt: str, user_prompt: str) -> str: ...


class DeterministicResponseGenerator:
    def __init__(self, policy: DialoguePolicy | None = None) -> None:
        self.policy = policy or DialoguePolicy()

    def generate(
        self,
        *,
        analysis: AnalysisResult | None,
        safety: SafetyResult,
        preferences: UserPreferences,
        estimate_is_incorrect: bool = False,
        advice_permission: bool = False,
    ) -> DialogueResponse:
        decision = self.policy.decide(
            analysis,
            safety,
            preferences,
            estimate_is_incorrect=estimate_is_incorrect,
            advice_permission=advice_permission,
        )
        if decision.route is SafetyRoute.CRISIS:
            parts = [
                safety.message
                or "If you may be in immediate danger, contact local emergency services now.",
                *safety.resources,
                "I cannot provide emergency help or manage this situation alone.",
            ]
        elif decision.acknowledge_correction:
            parts = [CORRECTION]
        elif decision.mention_emotion and analysis is not None:
            acknowledgement = EMOTION_ACKNOWLEDGEMENTS.get(
                analysis.fused_affect.label.value, UNCERTAINTY
            )
            parts = [
                acknowledgement,
                "I could be wrong. Would you like to tell me more?",
                NON_MEDICAL_NOTICE,
            ]
        else:
            parts = [UNCERTAINTY]

        if decision.suggest_grounding:
            parts.append(GROUNDING_PERMISSION_GRANTED)
        if decision.recommend_human_support and decision.route is not SafetyRoute.CRISIS:
            parts.append(HUMAN_SUPPORT)
        text = enforce_response(" ".join(part.strip() for part in parts if part))
        return DialogueResponse(
            text=text,
            route=decision.route,
            mentioned_emotion=decision.mention_emotion,
            asks_clarifying_question=decision.ask_clarifying_question,
            suggested_grounding=decision.suggest_grounding,
            response_speed=0.8 if decision.slow_pacing else 1.0,
            pause_seconds=1.0 if decision.slow_pacing else 0.25,
        )
