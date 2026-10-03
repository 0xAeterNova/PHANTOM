"""Auditable wording used by the local deterministic response policy."""

NON_MEDICAL_NOTICE = "This is not a medical assessment."

UNCERTAINTY = (
    "The available signals are unclear, and I do not want to guess how you feel. "
    "Would you like to tell me what you are noticing? " + NON_MEDICAL_NOTICE
)

CORRECTION = (
    "Thank you for correcting the estimate. Your description takes priority, and I will not treat "
    "the earlier signal as fact. What would you like me to understand instead?"
)

GROUNDING_PERMISSION_GRANTED = (
    "If it feels comfortable, you could pause and notice one thing you can see, one sound you can "
    "hear, and the support beneath your feet. You can stop at any time."
)

HUMAN_SUPPORT = "Consider contacting a trusted person or qualified professional for human support."

EMOTION_ACKNOWLEDGEMENTS = {
    "neutral": "The available signals may indicate a fairly neutral tone.",
    "happy": "The available signals may indicate a positive or upbeat tone.",
    "sad": "The available signals may indicate sadness or low energy.",
    "angry": "The available signals may indicate tension or frustration.",
    "fearful": "The available signals may indicate worry or unease.",
    "surprised": "The available signals may indicate surprise or heightened attention.",
    "disgusted": "The available signals may indicate discomfort or aversion.",
}
