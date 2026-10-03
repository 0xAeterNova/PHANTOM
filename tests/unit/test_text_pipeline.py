from __future__ import annotations

from phantom.schemas import EmotionLabel, SafetyRoute
from phantom.text.analyzer import ConversationalIntent, TextAnalyzer
from phantom.text.safety_language import SafetyLanguageAnalyzer


def test_text_affect_sentiment_and_intent_remain_inspectable() -> None:
    analyzer = TextAnalyzer()
    details = analyzer.analyze_detailed("I am very glad! Thank you.")
    result = analyzer.analyze("I am very glad! Thank you.")
    assert details.intent is ConversationalIntent.GRATITUDE
    assert result.probabilities[EmotionLabel.HAPPY] > result.probabilities[EmotionLabel.SAD]
    assert result.metadata["crisis_language_analyzed"] is False


def test_text_without_affect_terms_is_uncertain() -> None:
    result = TextAnalyzer().analyze("The package is on the table.")
    assert result.label is EmotionLabel.UNCERTAIN
    assert "ambiguous" in (result.reason or "")


def test_safety_language_is_a_separate_signal() -> None:
    analyzer = SafetyLanguageAnalyzer()
    signal = analyzer.analyze("I want to end my life")
    assert signal.route is SafetyRoute.CRISIS
    assert signal.human_support_recommended is True
    lexical = TextAnalyzer().analyze("I want to end my life")
    assert lexical.metadata["crisis_language_analyzed"] is False
