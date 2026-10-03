"""Deterministic lexical affect, sentiment, and conversational-intent analysis.

Crisis and immediate-safety language is intentionally not handled here.  Use
``phantom.text.safety_language`` as a separate safety signal so affect labels
cannot accidentally trigger or suppress crisis routing.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable

from phantom.compat import StrEnum
from phantom.fusion.confidence import normalize_probabilities, temperature_scale
from phantom.schemas import FUSION_EMOTIONS, EmotionLabel, ModalityResult


class SentimentLabel(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"
    UNCERTAIN = "uncertain"


class ConversationalIntent(StrEnum):
    GREETING = "greeting"
    FAREWELL = "farewell"
    QUESTION = "question"
    SUPPORT_REQUEST = "support_request"
    CORRECTION = "correction"
    GRATITUDE = "gratitude"
    STATEMENT = "statement"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True, slots=True)
class TextSignals:
    probabilities: dict[EmotionLabel, float]
    sentiment: SentimentLabel
    sentiment_score: float
    intent: ConversationalIntent
    uncertainty: float
    affect_term_count: int
    token_count: int


@runtime_checkable
class TextAffectModel(Protocol):
    """Replaceable text-only affect model; no safety-route responsibility."""

    def predict(self, text: str) -> Mapping[EmotionLabel, float]: ...


class LexiconTextAffectModel:
    """Transparent English lexicon baseline with basic negation handling."""

    LEXICONS: ClassVar[dict[EmotionLabel, frozenset[str]]] = {
        EmotionLabel.HAPPY: frozenset(
            {
                "happy",
                "glad",
                "joy",
                "joyful",
                "excited",
                "delighted",
                "pleased",
                "cheerful",
                "سعيد",
                "سعيدة",
                "مبسوط",
                "مبسوطة",
                "فرحان",
                "فرحانة",
                "فرح",
            }
        ),
        EmotionLabel.SAD: frozenset(
            {
                "sad",
                "down",
                "unhappy",
                "lonely",
                "crying",
                "grief",
                "miserable",
                "heartbroken",
                "حزين",
                "حزينة",
                "زعلان",
                "زعلانة",
                "وحيد",
                "وحيدة",
                "ببكي",
            }
        ),
        EmotionLabel.ANGRY: frozenset(
            {
                "angry",
                "mad",
                "furious",
                "annoyed",
                "irritated",
                "outraged",
                "frustrated",
                "غاضب",
                "غاضبة",
                "معصب",
                "معصبة",
                "منزعج",
                "منزعجة",
                "محبط",
                "محبطة",
            }
        ),
        EmotionLabel.FEARFUL: frozenset(
            {
                "afraid",
                "scared",
                "fearful",
                "worried",
                "nervous",
                "anxious",
                "terrified",
                "uneasy",
                "خائف",
                "خائفة",
                "خايف",
                "خايفة",
                "قلق",
                "قلقة",
                "متوتر",
                "متوترة",
            }
        ),
        EmotionLabel.SURPRISED: frozenset(
            {
                "surprised",
                "surprise",
                "astonished",
                "amazed",
                "unexpected",
                "shocked",
                "wow",
                "متفاجئ",
                "متفاجئة",
                "مندهش",
                "مندهشة",
                "مصدوم",
                "مصدومة",
            }
        ),
        EmotionLabel.DISGUSTED: frozenset(
            {
                "disgusted",
                "disgusting",
                "gross",
                "repulsed",
                "revolting",
                "sickened",
                "مشمئز",
                "مشمئزة",
                "مقرف",
                "مقرفة",
                "قرفان",
                "قرفانة",
            }
        ),
    }
    NEGATIONS = frozenset(
        {
            "not",
            "no",
            "never",
            "hardly",
            "isn't",
            "wasn't",
            "don't",
            "didn't",
            "لا",
            "مش",
            "مو",
            "لست",
        }
    )
    INTENSIFIERS = frozenset(
        {"very", "really", "extremely", "so", "deeply", "incredibly", "جدا", "جداً", "كثير", "كتير"}
    )

    @staticmethod
    def tokenize(text: str) -> list[str]:
        normalized = text.lower().replace("\N{RIGHT SINGLE QUOTATION MARK}", "'")
        return re.findall(r"[a-z]+(?:'[a-z]+)?|[\u0600-\u06ff]+", normalized)

    def score(self, text: str) -> tuple[dict[EmotionLabel, float], int]:
        tokens = self.tokenize(text)
        scores = dict.fromkeys(FUSION_EMOTIONS, 0.35)
        scores[EmotionLabel.NEUTRAL] = 1.1
        matches = 0
        for index, token in enumerate(tokens):
            for label, lexicon in self.LEXICONS.items():
                if token not in lexicon:
                    continue
                matches += 1
                context = tokens[max(0, index - 2) : index]
                negated = any(word in self.NEGATIONS for word in context)
                intensity = 1.35 if context and context[-1] in self.INTENSIFIERS else 1.0
                if negated:
                    scores[label] += 0.15
                    scores[EmotionLabel.NEUTRAL] += 0.45
                else:
                    scores[label] += 2.0 * intensity
        if "!" in text and matches:
            for label in FUSION_EMOTIONS:
                if label != EmotionLabel.NEUTRAL and scores[label] > 0.35:
                    scores[label] += min(0.4, text.count("!") * 0.1)
        # Softmax gives a complete distribution while preserving transparent scores.
        maximum = max(scores.values())
        exponentials = {
            label: math.exp((value - maximum) / 1.25) for label, value in scores.items()
        }
        total = sum(exponentials.values())
        return ({label: value / total for label, value in exponentials.items()}, matches)

    def predict(self, text: str) -> Mapping[EmotionLabel, float]:
        return self.score(text)[0]


class HuggingFaceTextAffectAdapter:
    """Optional local-only sequence-classification adapter.

    The configured checkpoint must expose labels that map to PHANTOM's affect
    vocabulary.  It is never loaded by the default analyzer and never performs
    safety routing.
    """

    _ALIASES: ClassVar[dict[str, EmotionLabel]] = {
        "neutral": EmotionLabel.NEUTRAL,
        "happy": EmotionLabel.HAPPY,
        "happiness": EmotionLabel.HAPPY,
        "joy": EmotionLabel.HAPPY,
        "sad": EmotionLabel.SAD,
        "sadness": EmotionLabel.SAD,
        "angry": EmotionLabel.ANGRY,
        "anger": EmotionLabel.ANGRY,
        "fear": EmotionLabel.FEARFUL,
        "fearful": EmotionLabel.FEARFUL,
        "surprise": EmotionLabel.SURPRISED,
        "surprised": EmotionLabel.SURPRISED,
        "disgust": EmotionLabel.DISGUSTED,
        "disgusted": EmotionLabel.DISGUSTED,
    }

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        revision: str,
        device: str = "cpu",
        local_files_only: bool = True,
    ) -> None:
        self.model_name_or_path = str(model_name_or_path)
        self.revision = revision
        self.device = device
        self.local_files_only = local_files_only
        self._tokenizer: Any | None = None
        self._model: Any | None = None

    def _load(self) -> tuple[Any, Any]:
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("optional text adapter requires transformers and PyTorch") from exc
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            revision=self.revision,
            local_files_only=self.local_files_only,
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name_or_path,
            revision=self.revision,
            local_files_only=self.local_files_only,
        )
        self._model.to(self.device)
        self._model.eval()
        return self._tokenizer, self._model

    def predict(self, text: str) -> Mapping[EmotionLabel, float]:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("optional text adapter requires PyTorch") from exc
        tokenizer, model = self._load()
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        with torch.inference_mode():
            raw = torch.softmax(model(**inputs).logits[0], dim=-1).detach().cpu().tolist()
        id_to_label = getattr(model.config, "id2label", {})
        mapped = dict.fromkeys(FUSION_EMOTIONS, 0.0)
        for index, probability in enumerate(raw):
            source = str(id_to_label.get(index, id_to_label.get(str(index), ""))).strip().lower()
            canonical = self._ALIASES.get(source)
            if canonical is not None:
                mapped[canonical] += float(probability)
        if sum(mapped.values()) <= 0.0:
            raise RuntimeError("pretrained model labels do not map to PHANTOM emotion labels")
        return normalize_probabilities(mapped)


class TextAnalyzer:
    """Text-only affect analyzer with separately exposed sentiment and intent."""

    _UNCERTAINTY = re.compile(
        r"\b(?:maybe|perhaps|possibly|probably|might|unsure|uncertain|i\s+think|i\s+guess)\b",
        re.IGNORECASE,
    )
    _SUPPORT = re.compile(
        r"\b(?:help|support|listen|talk|advice|what\s+should\s+i\s+do)\b", re.IGNORECASE
    )
    _CORRECTION = re.compile(
        r"\b(?:that(?:'s|\s+is)\s+(?:wrong|incorrect)|not\s+how\s+i\s+feel|estimate\s+is\s+wrong)\b",
        re.IGNORECASE,
    )

    def __init__(
        self,
        *,
        model: TextAffectModel | None = None,
        temperature: float = 1.20,
        confidence_floor: float = 0.30,
    ) -> None:
        self.model = model or LexiconTextAffectModel()
        self.temperature = temperature
        self.confidence_floor = confidence_floor

    @staticmethod
    def _intent(text: str) -> ConversationalIntent:
        lowered = text.strip().lower()
        if TextAnalyzer._CORRECTION.search(text):
            return ConversationalIntent.CORRECTION
        if TextAnalyzer._SUPPORT.search(text):
            return ConversationalIntent.SUPPORT_REQUEST
        if re.match(r"^(?:hi|hello|hey|good\s+(?:morning|afternoon|evening))\b", lowered):
            return ConversationalIntent.GREETING
        if re.search(r"\b(?:bye|goodbye|see\s+you)\b", lowered):
            return ConversationalIntent.FAREWELL
        if re.search(r"\b(?:thanks|thank\s+you|grateful)\b", lowered):
            return ConversationalIntent.GRATITUDE
        if "?" in text or re.match(
            r"^(?:who|what|when|where|why|how|can|could|would|is|are)\b", lowered
        ):
            return ConversationalIntent.QUESTION
        return ConversationalIntent.STATEMENT

    @staticmethod
    def _sentiment(probabilities: Mapping[EmotionLabel, float]) -> tuple[SentimentLabel, float]:
        positive = float(probabilities.get(EmotionLabel.HAPPY, 0.0))
        negative = sum(
            float(probabilities.get(label, 0.0))
            for label in (
                EmotionLabel.SAD,
                EmotionLabel.ANGRY,
                EmotionLabel.FEARFUL,
                EmotionLabel.DISGUSTED,
            )
        )
        neutral = float(probabilities.get(EmotionLabel.NEUTRAL, 0.0))
        signed = max(-1.0, min(1.0, positive - negative))
        if positive > 0.28 and negative > 0.35 and abs(signed) < 0.15:
            return SentimentLabel.MIXED, signed
        if positive > max(negative, neutral) and signed > 0.10:
            return SentimentLabel.POSITIVE, signed
        if negative > max(positive, neutral) and signed < -0.10:
            return SentimentLabel.NEGATIVE, signed
        return SentimentLabel.NEUTRAL, signed

    def analyze_detailed(self, text: str) -> TextSignals:
        clean = text.strip()
        tokens = LexiconTextAffectModel.tokenize(clean)
        if not clean or not tokens:
            return TextSignals(
                probabilities=normalize_probabilities({}),
                sentiment=SentimentLabel.UNCERTAIN,
                sentiment_score=0.0,
                intent=ConversationalIntent.UNCERTAIN,
                uncertainty=1.0,
                affect_term_count=0,
                token_count=0,
            )
        probabilities = temperature_scale(self.model.predict(clean), self.temperature)
        if isinstance(self.model, LexiconTextAffectModel):
            _unscaled, affect_count = self.model.score(clean)
        else:
            affect_count = 1
        ranked = sorted(probabilities.values(), reverse=True)
        margin = ranked[0] - ranked[1]
        marker_count = len(self._UNCERTAINTY.findall(clean))
        marker_uncertainty = min(0.65, marker_count * 0.20)
        ambiguity = max(0.0, 1.0 - margin * 3.5)
        if affect_count == 0:
            ambiguity = max(ambiguity, 0.75)
        uncertainty = max(marker_uncertainty, min(1.0, ambiguity))
        sentiment, score = self._sentiment(probabilities)
        return TextSignals(
            probabilities=dict(probabilities),
            sentiment=sentiment,
            sentiment_score=score,
            intent=self._intent(clean),
            uncertainty=uncertainty,
            affect_term_count=affect_count,
            token_count=len(tokens),
        )

    def analyze(self, text: str) -> ModalityResult:
        """Return schema-compatible lexical affect without crisis classification."""

        details = self.analyze_detailed(text)
        if details.token_count == 0:
            return ModalityResult.unavailable("text contains no analyzable words")
        top_label, top_probability = max(details.probabilities.items(), key=lambda item: item[1])
        quality = min(1.0, 0.45 + 0.08 * details.token_count)
        evidence = min(1.0, 0.45 + 0.25 * details.affect_term_count)
        confidence = top_probability * quality * evidence * (1.0 - 0.35 * details.uncertainty)
        uncertain = confidence < self.confidence_floor or details.uncertainty > 0.82
        return ModalityResult(
            label=EmotionLabel.UNCERTAIN if uncertain else top_label,
            confidence=max(0.0, min(1.0, confidence)),
            quality=quality,
            probabilities=details.probabilities,
            available=True,
            temporal_consistency=1.0,
            reason=(
                "lexical affect is weak or ambiguous"
                if uncertain
                else "text-only lexical estimate; not a medical assessment"
            ),
            metadata={
                "sentiment": details.sentiment.value,
                "sentiment_score": round(details.sentiment_score, 4),
                "intent": details.intent.value,
                "uncertainty": round(details.uncertainty, 4),
                "affect_term_count": details.affect_term_count,
                "crisis_language_analyzed": False,
            },
        )


LexicalTextAnalyzer = TextAnalyzer
