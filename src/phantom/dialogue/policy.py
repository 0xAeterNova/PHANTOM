"""Policy decisions made before any deterministic or generative response."""

from __future__ import annotations

from dataclasses import dataclass

from phantom.schemas import (
    AgeBand,
    AnalysisResult,
    EmotionLabel,
    SafetyResult,
    SafetyRoute,
    UserPreferences,
)


@dataclass(frozen=True, slots=True)
class DialogueDecision:
    route: SafetyRoute
    mention_emotion: bool
    ask_clarifying_question: bool
    use_short_response: bool
    slow_pacing: bool
    suggest_grounding: bool
    recommend_human_support: bool
    acknowledge_correction: bool


class DialoguePolicy:
    """Deterministic policy separating safety/product boundaries from wording."""

    def decide(
        self,
        analysis: AnalysisResult | None,
        safety: SafetyResult,
        preferences: UserPreferences,
        *,
        estimate_is_incorrect: bool = False,
        advice_permission: bool = False,
    ) -> DialogueDecision:
        if safety.route is SafetyRoute.CRISIS:
            return DialogueDecision(
                route=SafetyRoute.CRISIS,
                mention_emotion=False,
                ask_clarifying_question=False,
                use_short_response=True,
                slow_pacing=True,
                suggest_grounding=False,
                recommend_human_support=True,
                acknowledge_correction=False,
            )

        uncertain = analysis is None or analysis.fused_affect.uncertain
        label = analysis.fused_affect.label if analysis else EmotionLabel.UNCERTAIN
        potentially_underage = preferences.self_reported_age_band in {
            AgeBand.CHILD,
            AgeBand.TEENAGER,
        }
        grounding_labels = {EmotionLabel.SAD, EmotionLabel.FEARFUL, EmotionLabel.ANGRY}
        return DialogueDecision(
            route=safety.route,
            mention_emotion=not uncertain and not estimate_is_incorrect,
            ask_clarifying_question=uncertain or estimate_is_incorrect,
            use_short_response=potentially_underage or preferences.reading_level == "simple",
            slow_pacing=preferences.response_pace == "slow" or label in grounding_labels,
            suggest_grounding=bool(
                advice_permission and not uncertain and label in grounding_labels
            ),
            recommend_human_support=safety.human_support_recommended,
            acknowledge_correction=estimate_is_incorrect,
        )
