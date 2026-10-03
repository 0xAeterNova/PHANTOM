"""Separate, conservative language-only safety routing signal.

This module is a safety backstop, not a diagnosis or clinical classifier.  It
does not produce emotion probabilities and is intentionally separate from the
text affect analyzer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from phantom.compat import StrEnum
from phantom.config import SafetyConfig
from phantom.schemas import SafetyResult, SafetyRoute


class SafetyLanguageCategory(StrEnum):
    SELF_HARM = "self_harm_or_suicide"
    IMMEDIATE_DANGER = "immediate_danger"
    HARM_TO_OTHERS = "harm_to_others"
    ABUSE_OR_COERCION = "abuse_or_coercion"


@dataclass(frozen=True, slots=True)
class SafetyLanguageResult:
    route: SafetyRoute
    confidence: float
    categories: tuple[SafetyLanguageCategory, ...]
    first_person: bool
    contextual_or_quoted: bool
    reason: str

    @property
    def human_support_recommended(self) -> bool:
        return self.route in {SafetyRoute.SUPPORT, SafetyRoute.CRISIS}


_PATTERNS: dict[SafetyLanguageCategory, tuple[re.Pattern[str], ...]] = {
    SafetyLanguageCategory.SELF_HARM: (
        re.compile(r"\b(?:kill|hurt)\s+myself\b", re.IGNORECASE),
        re.compile(r"\bend\s+(?:my\s+)?life\b", re.IGNORECASE),
        re.compile(r"\bsuicid(?:e|al)\b", re.IGNORECASE),
        re.compile(r"\b(?:do\s+not|don't)\s+want\s+to\s+(?:live|be\s+alive)\b", re.IGNORECASE),
    ),
    SafetyLanguageCategory.IMMEDIATE_DANGER: (
        re.compile(r"\b(?:in\s+immediate\s+danger|not\s+safe\s+right\s+now)\b", re.IGNORECASE),
        re.compile(r"\bsomeone\s+is\s+(?:trying\s+to\s+)?hurt(?:ing)?\s+me\b", re.IGNORECASE),
    ),
    SafetyLanguageCategory.HARM_TO_OTHERS: (
        re.compile(
            r"\b(?:i\s+(?:am\s+going\s+to|will|want\s+to))\s+(?:kill|hurt)\s+(?:them|him|her|someone)\b",
            re.IGNORECASE,
        ),
    ),
    SafetyLanguageCategory.ABUSE_OR_COERCION: (
        re.compile(
            r"\b(?:being\s+abused|they\s+threatened\s+me|forced\s+me|controlling\s+me)\b",
            re.IGNORECASE,
        ),
    ),
}
_FIRST_PERSON = re.compile(r"\b(?:i|i'm|me|my|myself)\b", re.IGNORECASE)
_CONTEXT = re.compile(
    r"\b(?:article|book|movie|story|example|research|assignment|quote|quoted|fiction|someone\s+said)\b",
    re.IGNORECASE,
)


class SafetyLanguageAnalyzer:
    """Route explicit safety language while preserving contextual uncertainty."""

    def __init__(self, config: SafetyConfig | None = None) -> None:
        self.config = config or SafetyConfig()

    def analyze(self, text: str | None) -> SafetyLanguageResult:
        clean = (text or "").strip()
        if not clean:
            return SafetyLanguageResult(
                route=SafetyRoute.NORMAL,
                confidence=1.0,
                categories=(),
                first_person=False,
                contextual_or_quoted=False,
                reason="no safety-related language supplied",
            )
        categories = tuple(
            category
            for category, patterns in _PATTERNS.items()
            if any(pattern.search(clean) for pattern in patterns)
        )
        first_person = bool(_FIRST_PERSON.search(clean))
        contextual = bool(_CONTEXT.search(clean))
        if not categories:
            return SafetyLanguageResult(
                route=SafetyRoute.NORMAL,
                confidence=0.95,
                categories=(),
                first_person=first_person,
                contextual_or_quoted=contextual,
                reason="no configured safety phrase matched",
            )
        urgent = bool(
            set(categories)
            & {
                SafetyLanguageCategory.SELF_HARM,
                SafetyLanguageCategory.IMMEDIATE_DANGER,
                SafetyLanguageCategory.HARM_TO_OTHERS,
            }
        )
        if urgent and first_person and not contextual:
            return SafetyLanguageResult(
                route=SafetyRoute.CRISIS,
                confidence=0.90,
                categories=categories,
                first_person=True,
                contextual_or_quoted=False,
                reason="explicit first-person immediate-safety language matched",
            )
        return SafetyLanguageResult(
            route=SafetyRoute.SUPPORT,
            confidence=0.60 if contextual else 0.75,
            categories=categories,
            first_person=first_person,
            contextual_or_quoted=contextual,
            reason="safety-related language matched with contextual uncertainty",
        )

    def to_safety_result(self, result: SafetyLanguageResult) -> SafetyResult:
        if result.route == SafetyRoute.NORMAL:
            return SafetyResult()
        if result.route == SafetyRoute.CRISIS:
            return SafetyResult(
                route=SafetyRoute.CRISIS,
                human_support_recommended=True,
                message=self.config.emergency_message,
                resources=list(self.config.resources),
            )
        return SafetyResult(
            route=SafetyRoute.SUPPORT,
            human_support_recommended=True,
            message=(
                "This language can concern safety. If it describes an immediate situation, "
                "contact local emergency services or a trusted nearby person now."
            ),
            resources=list(self.config.resources),
        )

    def assess(self, text: str | None) -> SafetyResult:
        """Convenience adapter matching the project's crisis-router interface."""

        return self.to_safety_result(self.analyze(text))


def analyze_safety_language(text: str | None) -> SafetyLanguageResult:
    return SafetyLanguageAnalyzer().analyze(text)
