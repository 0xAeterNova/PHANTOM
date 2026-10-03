"""Output boundary checks for deterministic or optional generated responses."""

from __future__ import annotations

import re

_PROHIBITED = (
    re.compile(
        r"\byou (?:definitely )?(?:have|suffer from)\s+(?:depression|anxiety|bipolar|ptsd)\b", re.I
    ),
    re.compile(r"\bas your (?:psychiatrist|therapist|doctor)\b", re.I),
    re.compile(r"\b(?:start|stop|change) (?:your )?medication\b", re.I),
    re.compile(r"\byour face proves\b", re.I),
    re.compile(r"\bi am all you need\b", re.I),
    re.compile(
        r"(?:أنت|انت)\s+(?:بالتأكيد\s+)?(?:مصاب|تعاني)\s+(?:من\s+)?(?:الاكتئاب|القلق|ثنائي\s+القطب|اضطراب\s+ما\s+بعد\s+الصدمة)"
    ),
    re.compile(r"(?:بصفتي|كـ)\s*(?:طبيبك|معالجك|طبيبك\s+النفسي)"),
    re.compile(r"(?:ابدأ|أوقف|اوقف|غيّر|غير)\s+(?:دواءك|أدويتك|ادويتك)"),
)

_ARABIC_LETTER = re.compile(r"[\u0600-\u06ff]")
_ACCIDENT_OR_INJURY = re.compile(
    r"\b(?:accident|crash|injur(?:y|ed))\b|(?:حادث|مصاب|إصابة|اصابة|جرح|ألم)",
    re.I,
)
_UNHELPFUL_REFUSALS = (
    re.compile(
        r"^\s*(?:i(?:'m| am)\s+sorry[,.]?\s*)?(?:but\s+)?"
        r"i\s+(?:cannot|can't|am\s+unable\s+to)\s+(?:help|assist)"
        r"(?:\s+you|\s+with\s+(?:that|this))?[.!?]?\s*$",
        re.I,
    ),
    re.compile(
        r"^\s*(?:(?:أنا\s+)?آسف(?:ة)?|عذر(?:\u0627|ًا))[،,.:\s]*"
        r"(?:لكن(?:ني)?\s+)?لا\s+(?:أستطيع|استطيع|يمكنني)\s+(?:أن\s+)?"
        r"(?:مساعدتك|المساعدة(?:\s+في\s+(?:ذلك|هذا))?)[.!؟]?\s*$"
    ),
)

_ENGLISH_DISTRESS = re.compile(
    r"\b(?:sad|anxious|anxiety|afraid|scared|panic|grief|grieving|lonely|"
    r"overwhelmed|distressed|depressed|stress(?:ed)?|emotionally exhausted|"
    r"cannot cope|can't cope|accident|crash|injur(?:y|ed)|trauma|insomnia)\b",
    re.I,
)
_ARABIC_DISTRESS_TERMS = (
    "حزين",
    "حزينة",
    "حزن",
    "قلق",
    "خائف",
    "خائفة",
    "متوتر",
    "متوترة",
    "مرهق",
    "مرهقة",
    "نفسيتي",
    "تعبان نفسي",
    "تعبانة نفسي",
    "مكتئب",
    "مكتئبة",
    "وحيد",
    "وحيدة",
    "مضغوط",
    "مضغوطة",
    "لا أعلم ما العمل",
    "لا أعرف ماذا أفعل",
    "صدمة",
    "حادث",
    "فقدت",
    "وفاة",
    "أرق",
    "هلع",
)
_ENGLISH_ACKNOWLEDGEMENT = re.compile(
    r"\b(?:i hear you|i'm here with you|i am here with you|i'm sorry|i am sorry|"
    r"that sounds|it sounds|what happened sounds|what you're going through|"
    r"what you are going through|difficult|painful|heavy|overwhelming)\b",
    re.I,
)
_ARABIC_ACKNOWLEDGEMENT_TERMS = (
    "أنا معك",
    "أسمعك",
    "أفهم",
    "يؤسفني",
    "آسف",
    "آسفة",
    "يبدو صعب",
    "يبدو مؤلم",
    "يبدو ثقيل",
    "ما حدث",
    "ما تمر به",
    "ما تمرين به",
    "سلامتك",
)
_PASSIVE_SELF_HARM = (
    re.compile(
        r"\b(?:i\s+)?(?:wish|hope)\s+(?:i\s+)?(?:would(?:n't| not)|didn't|did not)\s+"
        r"(?:wake\s+up|exist)\b",
        re.I,
    ),
    re.compile(r"\bi\s+(?:do not|don't)\s+want\s+to\s+(?:live|be\s+alive)\b", re.I),
    re.compile(
        r"(?:أتمنى|اتمنى)\s+(?:أن\s+)?(?:أنام|انام)\s+(?:و)?لا\s+"
        r"(?:أستيقظ|استيقظ)"
    ),
    re.compile(r"(?:أتمنى|اتمنى)\s+(?:ألا|الا)\s+(?:أستيقظ|استيقظ|أكون|اكون)"),
    re.compile(r"(?:أنا|انا)\s+(?:لا|مش)\s+(?:أريد|اريد|عايز|بدي)\s+(?:أن\s+)?(?:أعيش|اعيش)"),
)
_PASSIVE_SELF_HARM_CONTEXT = re.compile(
    r"\b(?:article|book|movie|example|research|someone|friend|they|he|she)\b"
    r"|(?:مقال|كتاب|فيلم|مثال|بحث|رواية|قصة|لعبة|مسلسل|شخص\s+آخر|صديق(?:ي)?|هو|هي)",
    re.I,
)
_ENGLISH_DIRECT_SAFETY_QUESTION = re.compile(
    r"(?:are|do)\s+you\s+(?:safe|thinking\s+(?:about|of)\s+"
    r"(?:suicide|killing\s+yourself|harming\s+yourself|self[- ]harm))",
    re.I,
)
_ARABIC_DIRECT_SAFETY_QUESTION = re.compile(
    r"(?:هل\s+أنت|هل\s+انت)\s+(?:بأمان|بامان|آمن|امن)"
    r"|هل\s+(?:تفكر|تفكرين)\s+(?:في\s+)?(?:الانتحار|إيذاء\s+نفسك|ايذاء\s+نفسك|قتل\s+نفسك)"
)
_ENGLISH_TRUSTED_CONNECTION = re.compile(
    r"(?:trusted|trust)\s+(?:person|someone|friend|family|nearby)|"
    r"(?:contact|call|tell|reach\s+out\s+to|stay\s+with)\s+someone\s+you\s+trust",
    re.I,
)
_ARABIC_TRUSTED_CONNECTION = re.compile(
    r"(?:شخص|أحد|احد)\s+تثق\s+به|(?:صديق|قريب|فرد\s+من\s+العائلة)\s+تثق\s+به"
)


def validate_response(text: str) -> tuple[bool, list[str]]:
    violations = [pattern.pattern for pattern in _PROHIBITED if pattern.search(text)]
    compact = " ".join(text.split())
    if len(compact) <= 180:
        violations.extend(
            pattern.pattern for pattern in _UNHELPFUL_REFUSALS if pattern.search(compact)
        )
    return not violations, violations


def _expresses_distress(text: str) -> bool:
    return bool(_ENGLISH_DISTRESS.search(text)) or any(
        term in text for term in _ARABIC_DISTRESS_TERMS
    )


def _acknowledges_distress(text: str, *, arabic: bool) -> bool:
    if arabic:
        return any(term in text for term in _ARABIC_ACKNOWLEDGEMENT_TERMS)
    return bool(_ENGLISH_ACKNOWLEDGEMENT.search(text))


def _is_first_person_passive_self_harm(text: str) -> bool:
    """Identify a narrow passive death-wish signal without treating quotations as self-report."""

    return not _PASSIVE_SELF_HARM_CONTEXT.search(text) and any(
        pattern.search(text) for pattern in _PASSIVE_SELF_HARM
    )


def _meets_passive_self_harm_support_contract(text: str, *, arabic: bool) -> bool:
    if arabic:
        return bool(_ARABIC_DIRECT_SAFETY_QUESTION.search(text)) and bool(
            _ARABIC_TRUSTED_CONNECTION.search(text)
        )
    return bool(_ENGLISH_DIRECT_SAFETY_QUESTION.search(text)) and bool(
        _ENGLISH_TRUSTED_CONNECTION.search(text)
    )


def safe_fallback(context_text: str | None = None) -> str:
    """Return a brief supportive fallback in the user's apparent language.

    The fallback offers low-risk grounding and connection without diagnosis,
    medication guidance, professional-status claims, or promises of treatment.
    """

    if context_text and _ARABIC_LETTER.search(context_text):
        if _is_first_person_passive_self_harm(context_text):
            return (
                "أنا سعيد لأنك أخبرتني، ولن أحكم عليك. هل تفكر الآن في الانتحار أو إيذاء نفسك؟ "
                "هل أنت بأمان الآن؟ تواصل فورًا مع شخص تثق به واطلب منه البقاء معك."
            )
        if _ACCIDENT_OR_INJURY.search(context_text):
            return (
                "أنا معك، وما حدث يبدو مؤلمًا ومربكًا. أولًا، هل أنت الآن في مكان آمن "
                "وهل تحتاج إلى رعاية طبية عاجلة؟ إن كنت آمنًا، خذ نفسًا ببطء وتواصل مع شخص تثق به."
            )
        return (
            "أنا معك، وما تمر به يبدو ثقيلًا. خذ نفسًا ببطء إن كان ذلك مريحًا، "
            "وحاول البقاء قرب شخص تثق به. ما أكثر شيء تحتاجه الآن؟"
        )
    if context_text and _is_first_person_passive_self_harm(context_text):
        return (
            "I am glad you told me, and I will not judge you. Are you thinking about suicide "
            "or harming yourself right now? Are you safe now? Please contact someone you "
            "trust and ask them to stay with you."
        )
    if context_text and _ACCIDENT_OR_INJURY.search(context_text):
        return (
            "I am here with you, and what happened sounds painful and overwhelming. First, are "
            "you somewhere safe, and do you need urgent medical attention? If safe, take one "
            "slow breath and contact someone you trust."
        )
    return (
        "I am here with you, and what you are going through sounds heavy. If it feels "
        "comfortable, take one slow breath and stay near someone you trust. What would help "
        "most right now?"
    )


def enforce_response(text: str, *, user_text: str | None = None) -> str:
    safe, _ = validate_response(text)
    if user_text:
        user_is_arabic = bool(_ARABIC_LETTER.search(user_text))
        response_is_arabic = bool(_ARABIC_LETTER.search(text))
        # Language matching is part of usability, not a cosmetic preference:
        # an English refusal is not a meaningful answer to clear Arabic words.
        if user_is_arabic != response_is_arabic:
            safe = False
        if _expresses_distress(user_text) and not _acknowledges_distress(
            text, arabic=user_is_arabic
        ):
            safe = False
        if _is_first_person_passive_self_harm(user_text) and not (
            _meets_passive_self_harm_support_contract(text, arabic=user_is_arabic)
        ):
            safe = False
    return text if safe else safe_fallback(user_text or text)
