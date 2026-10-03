"""Conservative language-based crisis routing.

This is a safety backstop, not a validated clinical classifier.  Ordinary
affect labels never activate the crisis route; only user-provided text can.
"""

from __future__ import annotations

import re

from phantom.config import SafetyConfig
from phantom.schemas import SafetyResult, SafetyRoute

_DIRECT_IMMEDIATE_PATTERNS = (
    re.compile(
        r"^\s*i\s+(?:will\s+|(?:want|plan|intend|am\s+going)\s+to\s+)"
        r"(?:hurt|harm|injure)\s+myself\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*i\s+(?:will\s+|(?:plan|intend|am\s+going)\s+to\s+)"
        r"(?:kill|murder|hurt|harm)\s+(?!myself\b).{1,100}?"
        r"(?:\bright\s+now\b|\bnow\b|\btonight\b)",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:أنا\s+|انا\s+)?(?:سأنتحر|سانتحر|سوف\s+(?:أنتحر|انتحر))"
        r"(?:\s|[.!؟،]|$)"
    ),
    re.compile(r"^\s*(?:أنا\s+|انا\s+)?(?:أريد|اريد|بدي)\s+(?:أن\s+)?(?:أنتحر|انتحر)"),
    re.compile(
        r"^\s*(?:أنا\s+|انا\s+)?(?:س(?:أقتل|اقتل)|سوف\s+(?:أقتل|اقتل))\s+"
        r"(?!نفسي|حالي).{1,120}?(?:الآن|حالًا|حالا)"
    ),
)

_IMMEDIATE_PATTERNS = (
    re.compile(r"\bend\s+my\s+life\b", re.IGNORECASE),
    re.compile(
        r"\bi\s+(?:will|plan|intend|want|am\s+going)\s+to\s+"
        r"(?:commit\s+suicide|kill\s+myself|end\s+my\s+life)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:in\s+immediate\s+danger|not\s+safe\s+right\s+now)\b", re.IGNORECASE),
    re.compile(r"(?:أريد|اريد|بدي)\s+(?:أن\s+)?(?:أقتل|اقتل)\s+(?:نفسي|حالي)"),
    re.compile(r"(?:س(?:أقتل|اقتل)|سوف\s+(?:أقتل|اقتل))\s+(?:نفسي|حالي)"),
    re.compile(
        r"(?:س(?:أقتل|اقتل)|سوف\s+(?:أقتل|اقتل))\s+"
        r"(?!نفسي|حالي).{1,120}?(?:الآن|حالًا|حالا)"
    ),
    re.compile(
        r"(?:س(?:أقتل|اقتل)|سوف\s+(?:أقتل|اقتل))\s+"
        r"(?!نفسي|حالي).{1,160}?(?:السلاح|سلاح|سكين|مسدس)"
        r".{0,40}?(?:معي|بجانبي|عندي)"
    ),
    re.compile(r"(?:أنهي|انهي)\s+حياتي"),
    re.compile(r"(?:أنا|انا|إني|اني)\s+(?:لست|مش)\s+(?:بأمان|بامان|آمن|امن)"),
)
_SAFETY_MENTION_PATTERNS = (
    re.compile(r"\b(?:kill|hurt)\s+myself\b", re.IGNORECASE),
    re.compile(r"\b(?:do not|don't)\s+want\s+to\s+(?:be\s+alive|live)\b", re.IGNORECASE),
    re.compile(r"\bsuicid(?:e|al)\b", re.IGNORECASE),
    re.compile(r"(?:أنتحر|انتحر|الانتحار|انتحار)"),
    re.compile(r"(?:لا|مش)\s+(?:أريد|اريد|عايز|بدي)\s+(?:أن\s+)?(?:أعيش|اعيش|الحياة)"),
    re.compile(
        r"(?:أتمنى|اتمنى)\s+(?:أن\s+)?(?:أنام|انام)\s+"
        r"(?:و)?لا\s+(?:أستيقظ|استيقظ)"
    ),
    re.compile(r"(?:أتمنى|اتمنى)\s+(?:ألا|الا)\s+(?:أستيقظ|استيقظ)"),
)
_THIRD_PERSON_OR_QUOTED = re.compile(
    r"\b(?:article|book|movie|example|research|someone|friend|they|he|she)\b"
    r"|(?:مقال|كتاب|فيلم|مثال|بحث|رواية|قصة|لعبة|مسلسل|شخص\s+آخر|صديق(?:ي)?|هو|هي)",
    re.IGNORECASE,
)


class CrisisRouter:
    def __init__(self, config: SafetyConfig | None = None) -> None:
        self.config = config or SafetyConfig()

    def assess(self, text: str | None) -> SafetyResult:
        if not text or not text.strip():
            return SafetyResult()
        # Anchoring direct future/intent statements keeps prefixed fiction,
        # research, quotations, and historical descriptions out of the crisis
        # override while still treating an explicit current threat as urgent.
        direct_immediate = any(pattern.search(text) for pattern in _DIRECT_IMMEDIATE_PATTERNS)
        immediate = direct_immediate or any(pattern.search(text) for pattern in _IMMEDIATE_PATTERNS)
        matched = immediate or any(pattern.search(text) for pattern in _SAFETY_MENTION_PATTERNS)
        contextual = bool(_THIRD_PERSON_OR_QUOTED.search(text))
        # Context words reduce certainty but do not silently dismiss explicit first-person danger.
        first_person = bool(
            re.search(
                r"\b(?:i|me|my|myself)\b|(?:أنا|انا|إني|اني|نفسي|حالي)",
                text,
                re.IGNORECASE,
            )
        )
        # A bare reference to suicide is concerning, but it is not enough to
        # infer imminent intent. Reserve the emergency route for explicit
        # immediate-danger or first-person action language.
        if direct_immediate or (immediate and (first_person or not contextual)):
            return SafetyResult(
                route=SafetyRoute.CRISIS,
                human_support_recommended=True,
                message=self.config.emergency_message,
                resources=list(self.config.resources),
            )
        if matched:
            return SafetyResult(
                route=SafetyRoute.SUPPORT,
                human_support_recommended=True,
                message=(
                    "This language can concern immediate safety. If it describes you or someone "
                    "nearby, contact local emergency services or a trusted person now."
                ),
                resources=list(self.config.resources),
            )
        return SafetyResult()
