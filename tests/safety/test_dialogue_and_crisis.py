from __future__ import annotations

import pytest

from phantom.config import SafetyConfig
from phantom.dialogue.response_generator import DeterministicResponseGenerator
from phantom.safety.crisis_router import CrisisRouter
from phantom.safety.guardrails import enforce_response, safe_fallback, validate_response
from phantom.schemas import (
    AnalysisResult,
    EmotionLabel,
    FusedAffect,
    SafetyResult,
    SafetyRoute,
    UserPreferences,
)


def analysis(label: EmotionLabel = EmotionLabel.SAD, uncertain: bool = False) -> AnalysisResult:
    return AnalysisResult(
        session_id="test_session_123456789",
        modalities_available=[],
        modality_results={},
        fused_affect=FusedAffect(
            label=EmotionLabel.UNCERTAIN if uncertain else label,
            confidence=0.8 if not uncertain else 0.2,
            uncertain=uncertain,
            explanation="Signals may indicate something; not a medical assessment.",
        ),
    )


def test_crisis_language_activates_human_support_route() -> None:
    router = CrisisRouter(SafetyConfig(resources=("Call the verified local emergency number.",)))
    result = router.assess("I want to kill myself and I am not safe right now")
    assert result.route is SafetyRoute.CRISIS
    assert result.human_support_recommended is True
    assert result.resources


def test_third_person_research_context_is_not_silently_treated_as_first_person() -> None:
    result = CrisisRouter().assess("A research article discusses suicide prevention.")
    assert result.route is SafetyRoute.SUPPORT
    assert result.human_support_recommended is True


def test_bare_suicide_reference_is_support_not_emergency() -> None:
    result = CrisisRouter().assess("Suicide is a difficult subject to discuss.")
    assert result.route is SafetyRoute.SUPPORT


def test_past_self_harm_reference_is_support_not_emergency() -> None:
    result = CrisisRouter().assess("I used to hurt myself years ago, but I am safe now.")
    assert result.route is SafetyRoute.SUPPORT


def test_passive_death_wish_is_support_not_imminent_emergency() -> None:
    result = CrisisRouter().assess("I don't want to live.")
    assert result.route is SafetyRoute.SUPPORT


def test_ordinary_english_distress_does_not_route_to_emergency() -> None:
    result = CrisisRouter().assess("I had an accident today and feel deeply sad and overwhelmed.")
    assert result.route is SafetyRoute.NORMAL


@pytest.mark.parametrize(
    "text",
    [
        "I am going to hurt myself right now.",
        "سأنتحر الآن.",
        "سوف أنتحر الآن.",
        "I will kill my friend now.",
        "سأقتل صديقي الآن.",
    ],
)
def test_explicit_imminent_self_or_other_harm_routes_to_crisis(text: str) -> None:
    assert CrisisRouter().assess(text).route is SafetyRoute.CRISIS


@pytest.mark.parametrize(
    "text",
    [
        "I am angry with my friend, but I will not hurt him.",
        'A study quotes someone saying "I am going to hurt myself right now."',
        "I was going to hurt myself years ago, but I am safe now.",
        "في رواية، سأقتل صديقي الآن.",
    ],
)
def test_anger_fiction_research_and_historical_mentions_do_not_route_to_crisis(
    text: str,
) -> None:
    assert CrisisRouter().assess(text).route is not SafetyRoute.CRISIS


def test_crisis_response_stops_ordinary_emotion_coaching() -> None:
    safety = CrisisRouter().assess("I want to end my life")
    response = DeterministicResponseGenerator().generate(
        analysis=analysis(), safety=safety, preferences=UserPreferences()
    )
    assert response.route is SafetyRoute.CRISIS
    assert response.mentioned_emotion is False
    assert response.suggested_grounding is False
    assert "emergency" in response.text.lower()


def test_grounding_requires_permission() -> None:
    generator = DeterministicResponseGenerator()
    without = generator.generate(
        analysis=analysis(), safety=SafetyResult(), preferences=UserPreferences()
    )
    with_permission = generator.generate(
        analysis=analysis(),
        safety=SafetyResult(),
        preferences=UserPreferences(),
        advice_permission=True,
    )
    assert without.suggested_grounding is False
    assert with_permission.suggested_grounding is True


def test_user_correction_takes_priority() -> None:
    response = DeterministicResponseGenerator().generate(
        analysis=analysis(),
        safety=SafetyResult(),
        preferences=UserPreferences(),
        estimate_is_incorrect=True,
    )
    assert "thank you for correcting" in response.text.lower()
    assert response.mentioned_emotion is False


def test_prohibited_diagnostic_output_is_blocked() -> None:
    unsafe = "You definitely have depression. As your therapist, stop your medication."
    valid, violations = validate_response(unsafe)
    assert valid is False
    assert violations
    replacement = enforce_response(unsafe)
    assert "depression" not in replacement.lower()
    assert "here with you" in replacement.lower()


def test_terse_english_refusal_becomes_supportive_fallback() -> None:
    replacement = enforce_response(
        "I'm sorry, I can't help with that.",
        user_text="I feel emotionally exhausted after an accident.",
    )
    assert "can't help" not in replacement.lower()
    assert "here with you" in replacement.lower()
    assert "urgent medical attention" in replacement.lower()


def test_terse_arabic_refusal_becomes_supportive_arabic_fallback() -> None:
    user_text = "لقد تعرضت لحادث اليوم ونفسيتي تعبانة جدًا وحزين ولا أعلم ما العمل"
    replacement = enforce_response("آسف لا أستطيع مساعدتك", user_text=user_text)
    assert "لا أستطيع مساعدتك" not in replacement
    assert "أنا معك" in replacement
    assert "هل أنت الآن في مكان آمن" in replacement
    assert "رعاية طبية عاجلة" in replacement


def test_arabic_fallback_keeps_non_clinical_boundaries() -> None:
    replacement = safe_fallback("أنا حزين ومتعب نفسيًا")
    assert "تشخيص" not in replacement
    assert "دواء" not in replacement
    assert "معالجك" not in replacement


def test_generic_arabic_model_answer_is_replaced_with_situation_aware_support() -> None:
    user_text = "لقد تعرضت لحادث اليوم ونفسيتي تعبانة جدًا وحزين ولا أعلم ما العمل"

    replacement = enforce_response(
        "أنا هنا للمساعدة! كيف يمكنني مساعدتك اليوم؟",
        user_text=user_text,
    )

    assert "كيف يمكنني مساعدتك اليوم" not in replacement
    assert "هل أنت الآن في مكان آمن" in replacement
    assert "رعاية طبية عاجلة" in replacement


def test_wrong_language_model_answer_uses_the_user_language() -> None:
    user_text = "أنا حزين ومتعب نفسيًا"

    replacement = enforce_response(
        "I am sorry this feels difficult. What feels hardest right now?",
        user_text=user_text,
    )

    assert "أنا معك" in replacement
    assert not replacement.startswith("I am")


def test_arabic_project_word_does_not_trigger_accident_fallback() -> None:
    replacement = safe_fallback("ما هو المشروع؟")

    assert "رعاية طبية عاجلة" not in replacement
    assert "مكان آمن" not in replacement


def test_arabic_passive_death_wish_requires_direct_check_and_human_connection() -> None:
    user_text = "أتمنى أن أنام ولا أستيقظ غدًا."
    replacement = enforce_response(
        "أنا معك، وما تمر به ثقيل. خذ نفسًا ببطء.",
        user_text=user_text,
    )
    assert "هل تفكر الآن في الانتحار أو إيذاء نفسك؟" in replacement
    assert "هل أنت بأمان الآن؟" in replacement
    assert "شخص تثق به" in replacement
    assert "لن أحكم عليك" in replacement


def test_english_passive_death_wish_requires_direct_check_and_human_connection() -> None:
    user_text = "I wish I would not wake up tomorrow."
    replacement = enforce_response(
        "I hear you, and that sounds heavy. Try one slow breath.",
        user_text=user_text,
    )
    assert "Are you thinking about suicide or harming yourself right now?" in replacement
    assert "Are you safe now?" in replacement
    assert "someone you trust" in replacement
    assert "will not judge you" in replacement


def test_third_person_passive_death_wish_does_not_trigger_direct_support_contract() -> None:
    user_text = 'A research article quotes someone saying "I wish I would not wake up."'
    answer = "That research subject is difficult. What would you like to understand about it?"
    assert enforce_response(answer, user_text=user_text) == answer
