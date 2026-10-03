"""Shared, validation-first domain schemas.

The schemas deliberately distinguish observations from conclusions.  Every
perception result carries quality and confidence, and every fused result can
abstain.  Optional demographic presentation estimates are never fusion inputs.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from phantom.compat import UTC, StrEnum

Probability = Annotated[float, Field(ge=0.0, le=1.0)]
SessionId = Annotated[
    str, StringConstraints(min_length=16, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
]


class StrictModel(BaseModel):
    """Base schema that rejects unexpected input instead of silently ignoring it."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class EmotionLabel(StrEnum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEARFUL = "fearful"
    SURPRISED = "surprised"
    DISGUSTED = "disgusted"
    UNCERTAIN = "uncertain"
    INSUFFICIENT_QUALITY = "insufficient_quality"


FUSION_EMOTIONS: tuple[EmotionLabel, ...] = (
    EmotionLabel.NEUTRAL,
    EmotionLabel.HAPPY,
    EmotionLabel.SAD,
    EmotionLabel.ANGRY,
    EmotionLabel.FEARFUL,
    EmotionLabel.SURPRISED,
    EmotionLabel.DISGUSTED,
)


class Modality(StrEnum):
    AUDIO = "audio"
    VISION = "vision"
    TEXT = "text"


class SafetyRoute(StrEnum):
    NORMAL = "normal"
    SUPPORT = "support"
    CRISIS = "crisis"


class AgeBand(StrEnum):
    CHILD = "child"
    TEENAGER = "teenager"
    YOUNG_ADULT = "young adult"
    ADULT = "adult"
    OLDER_ADULT = "older adult"
    UNKNOWN = "unknown"
    DISABLED = "analysis-disabled"


class GenderPresentation(StrEnum):
    MASCULINE = "masculine-presenting"
    FEMININE = "feminine-presenting"
    AMBIGUOUS = "androgynous or ambiguous"
    UNKNOWN = "unknown"
    DISABLED = "analysis-disabled"


class ConsentSettings(StrictModel):
    """Explicit, revocable permissions for one short-lived session."""

    microphone: bool = False
    camera: bool = False
    text_analysis: bool = False
    age_band_analysis: bool = False
    perceived_gender_presentation_analysis: bool = False
    store_raw_data: bool = False
    cloud_upload: bool = False
    store_optional_attributes: bool = False

    @model_validator(mode="after")
    def validate_sensitive_permissions(self) -> ConsentSettings:
        if self.age_band_analysis and not self.camera:
            raise ValueError("age-band analysis requires camera/image consent")
        if self.perceived_gender_presentation_analysis and not self.camera:
            raise ValueError("gender-presentation analysis requires camera/image consent")
        if self.store_optional_attributes and not (
            self.age_band_analysis or self.perceived_gender_presentation_analysis
        ):
            raise ValueError("optional-attribute storage requires optional analysis consent")
        return self


class UserPreferences(StrictModel):
    preferred_name: str | None = Field(default=None, max_length=80)
    preferred_form_of_address: str | None = Field(default=None, max_length=80)
    self_reported_age_band: AgeBand | None = None
    reading_level: Annotated[str, StringConstraints(pattern=r"^(simple|standard|detailed)$")] = (
        "standard"
    )
    response_pace: Annotated[str, StringConstraints(pattern=r"^(slow|normal|fast)$")] = "normal"

    @field_validator("self_reported_age_band")
    @classmethod
    def self_report_must_be_meaningful(cls, value: AgeBand | None) -> AgeBand | None:
        if value in {AgeBand.DISABLED, AgeBand.UNKNOWN}:
            return None
        return value


class SessionCreateRequest(StrictModel):
    consent: ConsentSettings = Field(default_factory=ConsentSettings)
    preferences: UserPreferences = Field(default_factory=UserPreferences)
    mock_mode: bool = True


class SessionInfo(StrictModel):
    session_id: SessionId
    created_at: datetime
    expires_at: datetime
    consent: ConsentSettings
    preferences: UserPreferences
    mock_mode: bool


class PrivacySettings(StrictModel):
    store_raw_data: bool = False
    store_derived_features: bool = False
    store_optional_attributes: bool = False
    cloud_upload: bool = False
    session_ttl_seconds: int = Field(default=1800, ge=60, le=86_400)

    @model_validator(mode="after")
    def enforce_local_only(self) -> PrivacySettings:
        if self.cloud_upload:
            raise ValueError("cloud upload is disabled by this local-only build")
        return self


class ModalityResult(StrictModel):
    label: EmotionLabel
    confidence: Probability
    quality: Probability = 1.0
    probabilities: dict[EmotionLabel, Probability] = Field(default_factory=dict)
    available: bool = True
    temporal_consistency: Probability = 1.0
    reason: str | None = Field(default=None, max_length=240)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("probabilities")
    @classmethod
    def probabilities_are_normalized(
        cls, values: dict[EmotionLabel, float]
    ) -> dict[EmotionLabel, float]:
        if not values:
            return values
        total = sum(values.values())
        if not 0.98 <= total <= 1.02:
            raise ValueError("modality probabilities must sum to approximately 1")
        return values

    @classmethod
    def unavailable(cls, reason: str = "modality not provided") -> ModalityResult:
        return cls(
            label=EmotionLabel.UNCERTAIN,
            confidence=0.0,
            quality=0.0,
            available=False,
            temporal_consistency=0.0,
            reason=reason,
        )


class OptionalDemographicEstimates(StrictModel):
    age_band: AgeBand = AgeBand.DISABLED
    estimated_age: float | None = Field(default=None, ge=0.0, le=120.0)
    age_range: str | None = Field(default=None, max_length=32)
    perceived_gender_presentation: GenderPresentation = GenderPresentation.DISABLED
    confidence: Probability = 0.0
    uncertain: bool = True
    notice: str = "Optional visual presentation estimates are disabled by default and are not emotion evidence."


class RuntimeMode(StrEnum):
    DEMO = "demo"
    REAL = "real"


class AgreementState(StrEnum):
    STRONG_AGREEMENT = "strong_agreement"
    PARTIAL_AGREEMENT = "partial_agreement"
    CONFLICT = "conflict"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class BackendReadiness(StrEnum):
    READY = "ready"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


class BackendStatus(StrictModel):
    component: str = Field(min_length=1, max_length=80)
    provider: str = Field(min_length=1, max_length=120)
    model: str | None = Field(default=None, max_length=240)
    revision: str | None = Field(default=None, max_length=120)
    device: str = Field(default="cpu", max_length=40)
    mode: RuntimeMode
    readiness: BackendReadiness
    detail: str | None = Field(default=None, max_length=500)


class FusionTrace(StrictModel):
    agreement: AgreementState
    label: EmotionLabel
    confidence: Probability
    uncertain: bool = True
    contributions: dict[Modality, Probability] = Field(default_factory=dict)
    excluded: dict[Modality, str] = Field(default_factory=dict)
    explanation: str = Field(max_length=700)
    words_have_priority: bool = True
    explicit_self_report: bool = False


class VisionObservation(StrictModel):
    captured_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    result: ModalityResult
    optional_demographic_estimates: OptionalDemographicEstimates = Field(
        default_factory=OptionalDemographicEstimates
    )
    stale: bool = False


class AudioObservation(StrictModel):
    captured_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    result: ModalityResult
    transcript: str | None = Field(default=None, max_length=10_000)
    language: str | None = Field(default=None, max_length=24)


class FusedAffect(StrictModel):
    label: EmotionLabel
    confidence: Probability
    uncertain: bool = True
    probabilities: dict[EmotionLabel, Probability] = Field(default_factory=dict)
    contributions: dict[Modality, Probability] = Field(default_factory=dict)
    explanation: str = Field(max_length=500)


class SafetyResult(StrictModel):
    route: SafetyRoute = SafetyRoute.NORMAL
    human_support_recommended: bool = False
    message: str | None = Field(default=None, max_length=1000)
    resources: list[str] = Field(default_factory=list, max_length=10)


class RuntimeTurnResponse(StrictModel):
    session_id: SessionId
    transcript: str = Field(min_length=1, max_length=10_000)
    transcript_source: str = Field(max_length=40)
    audio: ModalityResult
    vision: ModalityResult
    fusion: FusionTrace
    optional_demographic_estimates: OptionalDemographicEstimates = Field(
        default_factory=OptionalDemographicEstimates
    )
    assistant_text: str = Field(min_length=1, max_length=10_000)
    safety: SafetyResult
    llm_provider: str = Field(max_length=120)
    tts_provider: str = Field(max_length=120)
    tts_audio_base64: str | None = None
    tts_mime_type: str | None = Field(default=None, max_length=80)
    history_turns: int = Field(ge=0, le=100)


class AnalysisResult(StrictModel):
    session_id: SessionId
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    modalities_available: list[Modality]
    modality_results: dict[Modality, ModalityResult]
    fused_affect: FusedAffect
    optional_demographic_estimates: OptionalDemographicEstimates = Field(
        default_factory=OptionalDemographicEstimates
    )
    safety: SafetyResult = Field(default_factory=SafetyResult)


class TextAnalysisRequest(StrictModel):
    session_id: SessionId
    text: str = Field(min_length=1, max_length=10_000)


class MockSignal(StrictModel):
    label: EmotionLabel = EmotionLabel.NEUTRAL
    confidence: Probability = 0.8
    quality: Probability = 0.9
    temporal_consistency: Probability = 1.0
    face_count: int | None = Field(default=None, ge=0, le=20)


class MultimodalAnalysisRequest(StrictModel):
    session_id: SessionId
    text: str | None = Field(default=None, max_length=10_000)
    mock_audio: MockSignal | None = None
    mock_vision: MockSignal | None = None

    @model_validator(mode="after")
    def require_a_modality(self) -> MultimodalAnalysisRequest:
        if self.text is None and self.mock_audio is None and self.mock_vision is None:
            raise ValueError("provide at least one modality")
        return self


class DialogueRequest(StrictModel):
    session_id: SessionId
    analysis: AnalysisResult | None = None
    user_message: str = Field(default="", max_length=10_000)
    estimate_is_incorrect: bool = False
    advice_permission: bool = False


class DialogueResponse(StrictModel):
    text: str
    route: SafetyRoute
    mentioned_emotion: bool
    asks_clarifying_question: bool
    suggested_grounding: bool
    response_speed: Annotated[float, Field(ge=0.5, le=2.0)] = 1.0
    pause_seconds: Annotated[float, Field(ge=0.0, le=10.0)] = 0.0


class ErrorDetail(StrictModel):
    code: str
    message: str
    request_id: str | None = None


class ErrorResponse(StrictModel):
    error: ErrorDetail
