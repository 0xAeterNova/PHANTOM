"""Privacy-preserving orchestration for acoustic inference and optional STT."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol

from phantom.audio.features import extract_log_mel
from phantom.audio.model import (
    AcousticEmotionModel,
    BaselineAcousticEmotionModel,
    HuggingFaceSpeechEmotionAdapter,
    TemperatureCalibrator,
    WaveformEmotionModel,
)
from phantom.audio.preprocessing import (
    AudioBuffer,
    detect_voice_activity,
    load_wav,
    normalize_audio,
    resample_audio,
)
from phantom.audio.quality import AudioQuality, estimate_audio_quality
from phantom.config import AudioModelConfig, repository_root
from phantom.exceptions import ConsentRequiredError, InvalidMediaError, PayloadTooLargeError
from phantom.fusion.confidence import distribution_from_label
from phantom.fusion.temporal import TemporalSmoother
from phantom.schemas import ConsentSettings, EmotionLabel, MockSignal, ModalityResult


class MicrophoneSource(Protocol):
    """Explicit-lifecycle microphone boundary; ``start`` must enforce consent."""

    def start(self, consent: ConsentSettings) -> None: ...

    def read_wav(self) -> bytes | None: ...

    def stop(self) -> None: ...


class MockMicrophoneSource:
    """Deterministic source for integration tests; it never opens real hardware."""

    def __init__(self, wav_data: bytes) -> None:
        self._wav_data = bytes(wav_data)
        self._started = False

    def start(self, consent: ConsentSettings) -> None:
        if not consent.microphone:
            raise ConsentRequiredError("microphone consent is required before activation")
        self._started = True

    def read_wav(self) -> bytes | None:
        if not self._started:
            raise RuntimeError("microphone source has not been started")
        return self._wav_data

    def stop(self) -> None:
        self._started = False


class SpeechToTextAdapter(Protocol):
    """Optional transcription interface; transcripts remain separate text input."""

    def transcribe(self, audio: AudioBuffer) -> str | None: ...


class NullSpeechToTextAdapter:
    """Default adapter that performs no lexical analysis or cloud transfer."""

    def transcribe(self, audio: AudioBuffer) -> str | None:
        del audio
        return None


class HuggingFaceSpeechToTextAdapter:
    """Optional local-only speech-to-text adapter for compatible HF models."""

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
        self._processor: Any | None = None
        self._model: Any | None = None

    def _load(self) -> tuple[Any, Any]:
        if self._processor is not None and self._model is not None:
            return self._processor, self._model
        try:
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("optional STT requires transformers and PyTorch") from exc
        self._processor = AutoProcessor.from_pretrained(
            self.model_name_or_path,
            revision=self.revision,
            local_files_only=self.local_files_only,
        )
        self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_name_or_path,
            revision=self.revision,
            local_files_only=self.local_files_only,
        )
        self._model.to(self.device)
        self._model.eval()
        return self._processor, self._model

    def transcribe(self, audio: AudioBuffer) -> str | None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("optional STT requires PyTorch") from exc
        processor, model = self._load()
        inputs = processor(audio.samples, sampling_rate=audio.sample_rate, return_tensors="pt")
        input_features = inputs.get("input_features", inputs.get("input_values"))
        if input_features is None:
            raise RuntimeError("STT processor produced no audio input tensor")
        with torch.inference_mode():
            generated = model.generate(input_features.to(self.device))
        text = str(processor.batch_decode(generated, skip_special_tokens=True)[0]).strip()
        return text or None


class FasterWhisperSpeechToTextAdapter:
    """Local multilingual transcription through ``faster-whisper``.

    The default ``base`` checkpoint is multilingual (unlike ``base.en``), and
    CPU ``int8`` inference is suitable for machines without CUDA.  Model
    construction is lazy and protected by a lock so one adapter instance loads
    its checkpoint at most once, even when concurrent requests arrive.
    """

    backend = "faster-whisper"
    target_sample_rate = 16_000

    def __init__(
        self,
        model_size_or_path: str | Path = "base",
        *,
        device: str = "cpu",
        compute_type: str = "int8",
        language: str | None = None,
        beam_size: int = 5,
        download_root: str | Path | None = None,
        local_files_only: bool = False,
    ) -> None:
        if not str(model_size_or_path).strip():
            raise ValueError("faster-whisper model size or path cannot be empty")
        if not device.strip():
            raise ValueError("faster-whisper device cannot be empty")
        if not compute_type.strip():
            raise ValueError("faster-whisper compute type cannot be empty")
        if beam_size < 1:
            raise ValueError("faster-whisper beam size must be positive")
        normalized_language = language.strip().lower() if language is not None else None
        self.model_size_or_path = str(model_size_or_path)
        self.device = device.strip().lower()
        self.compute_type = compute_type.strip().lower()
        self.language = normalized_language or None
        self.beam_size = beam_size
        self.download_root = str(download_root) if download_root is not None else None
        self.local_files_only = local_files_only
        self._model: Any | None = None
        self._load_lock = threading.Lock()

    @property
    def provenance(self) -> dict[str, str | bool | None]:
        """Return inspectable backend/model settings without loading weights."""

        return {
            "backend": self.backend,
            "model_name_or_path": self.model_size_or_path,
            "device": self.device,
            "compute_type": self.compute_type,
            "language": self.language,
            "local_files_only": self.local_files_only,
        }

    def _load(self) -> Any:
        with self._load_lock:
            if self._model is not None:
                return self._model
            try:
                whisper_model = import_module("faster_whisper").WhisperModel
            except ImportError as exc:
                raise RuntimeError(
                    "FasterWhisperSpeechToTextAdapter requires the optional "
                    "'faster-whisper' package"
                ) from exc
            options: dict[str, Any] = {
                "device": self.device,
                "compute_type": self.compute_type,
                "local_files_only": self.local_files_only,
            }
            if self.download_root is not None:
                options["download_root"] = self.download_root
            try:
                self._model = whisper_model(self.model_size_or_path, **options)
            except Exception as exc:
                raise RuntimeError(
                    "faster-whisper could not load the configured model "
                    f"{self.model_size_or_path!r} on {self.device} with {self.compute_type}"
                ) from exc
            return self._model

    def transcribe(self, audio: AudioBuffer) -> str | None:
        if audio.sample_rate <= 0:
            raise ValueError("audio sample rate must be positive")
        samples = audio.samples
        if audio.sample_rate != self.target_sample_rate:
            samples = resample_audio(samples, audio.sample_rate, self.target_sample_rate)
        model = self._load()
        try:
            segments, _info = model.transcribe(
                samples,
                language=self.language,
                beam_size=self.beam_size,
                vad_filter=True,
            )
            parts = [str(segment.text).strip() for segment in segments]
        except Exception as exc:
            raise RuntimeError("faster-whisper transcription failed") from exc
        text = " ".join(part for part in parts if part).strip()
        return text or None


@dataclass(frozen=True, slots=True)
class AudioAnalysis:
    """Detailed result; transcript is opt-in and never mixed with acoustics."""

    acoustic_result: ModalityResult
    transcript: str | None = None


def _mock_result(signal: MockSignal) -> ModalityResult:
    probabilities = distribution_from_label(signal.label, signal.confidence)
    return ModalityResult(
        label=signal.label,
        confidence=signal.confidence,
        quality=signal.quality,
        probabilities=probabilities if signal.label in probabilities else {},
        available=True,
        temporal_consistency=signal.temporal_consistency,
        reason="deterministic mock acoustic signal",
        metadata={"mock": True},
    )


class AudioAnalyzer:
    """Decode, quality-gate, featurize, calibrate, and smooth speech acoustics."""

    ACCEPTED_CONTENT_TYPES = frozenset({"audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave"})
    ACCEPTED_SUFFIXES = frozenset({".wav", ".wave"})

    def __init__(
        self,
        *,
        model: AcousticEmotionModel | WaveformEmotionModel | None = None,
        stt_adapter: SpeechToTextAdapter | None = None,
        target_sample_rate: int = 16_000,
        max_bytes: int = 10 * 1024 * 1024,
        quality_floor: float = 0.25,
        confidence_floor: float = 0.36,
        decision_margin_floor: float = 0.07,
        angry_confidence_floor: float = 0.62,
        angry_decision_margin_floor: float = 0.22,
        calibrator: TemperatureCalibrator | None = None,
        temporal_window: int = 1,
        max_feature_frames: int = 3_000,
    ) -> None:
        if max_feature_frames < 1:
            raise ValueError("max_feature_frames must be positive")
        decision_settings = {
            "quality_floor": quality_floor,
            "confidence_floor": confidence_floor,
            "decision_margin_floor": decision_margin_floor,
            "angry_confidence_floor": angry_confidence_floor,
            "angry_decision_margin_floor": angry_decision_margin_floor,
        }
        for setting, value in decision_settings.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{setting} must be between 0 and 1")
        if angry_confidence_floor < confidence_floor:
            raise ValueError("angry_confidence_floor cannot be lower than confidence_floor")
        if angry_decision_margin_floor < decision_margin_floor:
            raise ValueError(
                "angry_decision_margin_floor cannot be lower than decision_margin_floor"
            )
        self.model = model or BaselineAcousticEmotionModel()
        self.stt_adapter = stt_adapter
        self.target_sample_rate = target_sample_rate
        self.max_bytes = max_bytes
        self.quality_floor = quality_floor
        self.confidence_floor = confidence_floor
        self.decision_margin_floor = decision_margin_floor
        self.angry_confidence_floor = angry_confidence_floor
        self.angry_decision_margin_floor = angry_decision_margin_floor
        if calibrator is None:
            model_temperature = getattr(self.model, "temperature", None)
            if model_temperature is None:
                calibrator = TemperatureCalibrator()
            else:
                converted_temperature = float(model_temperature)
                if not math.isfinite(converted_temperature) or converted_temperature <= 0.0:
                    raise ValueError("model temperature must be a positive finite number")
                calibrator = TemperatureCalibrator(converted_temperature)
        self.calibrator = calibrator
        self.smoother = TemporalSmoother(window_size=temporal_window)
        self.max_feature_frames = max_feature_frames

    @classmethod
    def from_model_config(
        cls,
        config: AudioModelConfig,
        *,
        max_bytes: int = 10 * 1024 * 1024,
    ) -> AudioAnalyzer:
        """Build the configured local backend without ever enabling random weights."""

        if config.backend == "heuristic":
            return cls(
                max_bytes=max_bytes,
                max_feature_frames=config.max_frames,
                confidence_floor=config.confidence_floor,
                decision_margin_floor=config.decision_margin_floor,
                angry_confidence_floor=config.angry_confidence_floor,
                angry_decision_margin_floor=config.angry_decision_margin_floor,
            )
        if config.backend == "huggingface":
            pretrained_model = HuggingFaceSpeechEmotionAdapter(
                config.model_name,
                revision=config.revision,
                device=config.device if config.device != "auto" else "cpu",
                local_files_only=config.local_files_only,
            )
            return cls(
                model=pretrained_model,
                max_bytes=max_bytes,
                max_feature_frames=config.max_frames,
                confidence_floor=config.confidence_floor,
                decision_margin_floor=config.decision_margin_floor,
                angry_confidence_floor=config.angry_confidence_floor,
                angry_decision_margin_floor=config.angry_decision_margin_floor,
            )
        if config.checkpoint is None:  # guarded by ``load_config``; protects direct construction
            raise ValueError("a checkpoint is required for a CNN or CRNN audio backend")
        checkpoint = config.checkpoint
        if not checkpoint.is_absolute():
            checkpoint = repository_root() / checkpoint
        from phantom.audio.neural import SpectrogramEmotionAdapter

        neural_model = SpectrogramEmotionAdapter(
            checkpoint,
            device=config.device,
            expected_architecture=config.backend,
            max_frames=config.max_frames,
        )
        if neural_model.n_mels != config.n_mels:
            raise ValueError(
                "configured audio.n_mels does not match the checkpoint: "
                f"{config.n_mels} != {neural_model.n_mels}"
            )
        return cls(
            model=neural_model,
            max_bytes=max_bytes,
            calibrator=TemperatureCalibrator(neural_model.temperature),
            max_feature_frames=neural_model.max_frames,
            confidence_floor=config.confidence_floor,
            decision_margin_floor=config.decision_margin_floor,
            angry_confidence_floor=config.angry_confidence_floor,
            angry_decision_margin_floor=config.angry_decision_margin_floor,
        )

    def _validate_media(self, data: bytes, filename: str, content_type: str) -> None:
        if len(data) > self.max_bytes:
            raise PayloadTooLargeError(f"audio exceeds the {self.max_bytes}-byte limit")
        if not data:
            raise InvalidMediaError("audio payload is empty")
        media_type = content_type.partition(";")[0].strip().lower()
        if media_type not in self.ACCEPTED_CONTENT_TYPES:
            raise InvalidMediaError("only PCM WAV audio is accepted")
        if Path(filename).suffix.lower() not in self.ACCEPTED_SUFFIXES:
            raise InvalidMediaError("audio filename must use a .wav or .wave extension")
        if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
            raise InvalidMediaError("audio content does not match its WAV media type")

    def _prepare(self, data: bytes) -> tuple[AudioBuffer, AudioQuality]:
        decoded = load_wav(data)
        samples = resample_audio(decoded.samples, decoded.sample_rate, self.target_sample_rate)
        activity = detect_voice_activity(samples, self.target_sample_rate)
        quality = estimate_audio_quality(samples, self.target_sample_rate, vad=activity)
        normalized = normalize_audio(samples)
        return AudioBuffer(normalized, self.target_sample_rate), quality

    def _infer(self, audio: AudioBuffer, quality: AudioQuality) -> ModalityResult:
        if not quality.sufficient or quality.overall < self.quality_floor:
            return ModalityResult(
                label=EmotionLabel.INSUFFICIENT_QUALITY,
                confidence=0.0,
                quality=quality.overall,
                available=False,
                temporal_consistency=0.0,
                reason=quality.reason or "audio quality is below the configured threshold",
                metadata={
                    "duration_seconds": round(quality.duration_seconds, 4),
                    "speech_ratio": round(quality.speech_ratio, 4),
                    "snr_db": round(quality.snr_db, 2),
                    "clipping_fraction": round(quality.clipping_fraction, 6),
                },
            )
        if isinstance(self.model, WaveformEmotionModel):
            raw = self.model.predict_waveform(audio.samples, audio.sample_rate)
            feature_name = "pretrained-waveform"
            feature_frames = None
            feature_truncated = False
        else:
            n_mels = int(getattr(self.model, "n_mels", 40))
            features = extract_log_mel(audio.samples, audio.sample_rate, n_mels=n_mels)
            feature_truncated = bool(features.shape[0] > self.max_feature_frames)
            if feature_truncated:
                crop_start = (int(features.shape[0]) - self.max_feature_frames) // 2
                features = features[crop_start : crop_start + self.max_feature_frames]
            else:
                crop_start = 0
            raw = self.model.predict(features)
            feature_name = str(getattr(self.model, "feature_name", "log-mel"))
            feature_frames = int(features.shape[0])
        calibrated = self.calibrator.calibrate(raw)
        # The shipped SUPERB checkpoint supports four labels while PHANTOM's
        # canonical schema supports seven.  Track the model's effective label
        # support so a 0.36 four-way score is not presented as though it came
        # from a seven-way classifier.  Temperature scaling turns exact zeros
        # into tiny positive values, so support is measured before calibration.
        raw_supported_label_count = sum(
            1
            for probability in raw.values()
            if math.isfinite(float(probability)) and float(probability) > 1e-8
        )
        supported_label_count = raw_supported_label_count or len(calibrated)
        chance_probability = 1.0 / supported_label_count
        initial_ranked = sorted(calibrated.items(), key=lambda item: item[1], reverse=True)
        initial_label, initial_probability = initial_ranked[0]
        initial_margin = initial_probability - initial_ranked[1][1]
        # Confidence describes the calibrated model distribution.  Recording
        # quality is exposed independently, has its own sufficiency gate, and
        # is multiplied into the modality weight during fusion.  Folding it
        # into confidence here penalized ordinary microphone audio twice and
        # caused otherwise decisive recordings to abstain pathologically.
        initial_confidence = max(0.0, min(1.0, initial_probability))
        initial_required_confidence = (
            self.angry_confidence_floor
            if initial_label is EmotionLabel.ANGRY
            else self.confidence_floor
        )
        initial_required_margin = (
            self.angry_decision_margin_floor
            if initial_label is EmotionLabel.ANGRY
            else self.decision_margin_floor
        )

        # A weak candidate must not enter temporal state.  This is especially
        # important for a cross-language checkpoint: otherwise one ambiguous
        # "angry" result can bias later, independent recordings.
        initial_reasons: list[str] = []
        if initial_confidence < initial_required_confidence:
            initial_reasons.append("top score below the decision threshold")
        if initial_margin < initial_required_margin:
            initial_reasons.append("winning margin too small")
        if initial_reasons:
            return self._abstained_result(
                quality=quality,
                feature_name=feature_name,
                feature_frames=feature_frames,
                feature_truncated=feature_truncated,
                crop_start=crop_start if feature_frames is not None else None,
                calibrated=calibrated,
                candidate_label=initial_label,
                candidate_probability=initial_probability,
                candidate_confidence=initial_confidence,
                candidate_margin=initial_margin,
                required_confidence=initial_required_confidence,
                required_margin=initial_required_margin,
                abstention_reasons=initial_reasons,
                supported_label_count=supported_label_count,
                chance_probability=chance_probability,
            )

        smoothed = self.smoother.update(calibrated)
        ranked = sorted(smoothed.items(), key=lambda item: item[1], reverse=True)
        top_label, top_probability = ranked[0]
        margin = top_probability - ranked[1][1]
        distance = 0.5 * sum(abs(smoothed[label] - calibrated[label]) for label in smoothed)
        temporal_consistency = max(0.0, min(1.0, 1.0 - distance))
        confidence = max(0.0, min(1.0, top_probability))
        required_confidence = (
            self.angry_confidence_floor
            if top_label is EmotionLabel.ANGRY
            else self.confidence_floor
        )
        required_margin = (
            self.angry_decision_margin_floor
            if top_label is EmotionLabel.ANGRY
            else self.decision_margin_floor
        )
        abstention_reasons: list[str] = []
        if confidence < required_confidence:
            abstention_reasons.append("top score below the decision threshold")
        if margin < required_margin:
            abstention_reasons.append("winning margin too small")
        if abstention_reasons:
            return self._abstained_result(
                quality=quality,
                feature_name=feature_name,
                feature_frames=feature_frames,
                feature_truncated=feature_truncated,
                crop_start=crop_start if feature_frames is not None else None,
                calibrated=smoothed,
                candidate_label=top_label,
                candidate_probability=top_probability,
                candidate_confidence=confidence,
                candidate_margin=margin,
                required_confidence=required_confidence,
                required_margin=required_margin,
                abstention_reasons=abstention_reasons,
                temporal_consistency=temporal_consistency,
                supported_label_count=supported_label_count,
                chance_probability=chance_probability,
            )
        model_backend = str(
            getattr(self.model, "backend", getattr(self.model, "architecture", "heuristic"))
        )
        model_provenance = getattr(self.model, "provenance", None)
        metadata: dict[str, Any] = {
            "duration_seconds": round(quality.duration_seconds, 4),
            "sample_rate": audio.sample_rate,
            "speech_ratio": round(quality.speech_ratio, 4),
            "snr_db": round(quality.snr_db, 2),
            "clipping_fraction": round(quality.clipping_fraction, 6),
            "features": feature_name,
            "feature_frames": feature_frames,
            "feature_truncated": feature_truncated,
            "feature_crop_start": crop_start if feature_frames is not None else None,
            "model_backend": model_backend,
            "transcript_analyzed": False,
            "decision_policy": "quality-separated-margin-gated-v2",
            "candidate_label": top_label.value,
            "candidate_probability": round(top_probability, 6),
            "candidate_confidence": round(confidence, 6),
            "decision_margin": round(margin, 6),
            "required_confidence": required_confidence,
            "required_margin": required_margin,
            "supported_label_count": supported_label_count,
            "chance_probability": round(chance_probability, 6),
            "quality_weighted_confidence": round(confidence * quality.overall, 6),
        }
        if isinstance(model_provenance, dict):
            metadata["model_provenance"] = dict(model_provenance)
        return ModalityResult(
            label=top_label,
            confidence=confidence,
            quality=quality.overall,
            probabilities=smoothed,
            available=True,
            temporal_consistency=temporal_consistency,
            reason="acoustic estimate; not a medical assessment",
            metadata=metadata,
        )

    def _abstained_result(
        self,
        *,
        quality: AudioQuality,
        feature_name: str,
        feature_frames: int | None,
        feature_truncated: bool,
        crop_start: int | None,
        calibrated: dict[EmotionLabel, float],
        candidate_label: EmotionLabel,
        candidate_probability: float,
        candidate_confidence: float,
        candidate_margin: float,
        required_confidence: float,
        required_margin: float,
        abstention_reasons: list[str],
        supported_label_count: int,
        chance_probability: float,
        temporal_consistency: float = 0.0,
    ) -> ModalityResult:
        """Return an explicit abstention without feeding a guess into fusion."""

        model_backend = str(
            getattr(self.model, "backend", getattr(self.model, "architecture", "heuristic"))
        )
        model_provenance = getattr(self.model, "provenance", None)
        metadata: dict[str, Any] = {
            "duration_seconds": round(quality.duration_seconds, 4),
            "sample_rate": self.target_sample_rate,
            "speech_ratio": round(quality.speech_ratio, 4),
            "snr_db": round(quality.snr_db, 2),
            "clipping_fraction": round(quality.clipping_fraction, 6),
            "features": feature_name,
            "feature_frames": feature_frames,
            "feature_truncated": feature_truncated,
            "feature_crop_start": crop_start,
            "model_backend": model_backend,
            "transcript_analyzed": False,
            "decision_policy": "quality-separated-margin-gated-v2",
            "candidate_label": candidate_label.value,
            "candidate_probability": round(candidate_probability, 6),
            "candidate_confidence": round(candidate_confidence, 6),
            "runner_up_probability": round(sorted(calibrated.values(), reverse=True)[1], 6),
            "decision_margin": round(candidate_margin, 6),
            "required_confidence": required_confidence,
            "required_margin": required_margin,
            "supported_label_count": supported_label_count,
            "chance_probability": round(chance_probability, 6),
            "quality_weighted_confidence": round(candidate_confidence * quality.overall, 6),
            "abstention_reasons": list(abstention_reasons),
            "candidate_probabilities": {
                label.value: round(probability, 6) for label, probability in calibrated.items()
            },
            "withheld_from_fusion": True,
        }
        if isinstance(model_provenance, dict):
            metadata["model_provenance"] = dict(model_provenance)
        return ModalityResult(
            label=EmotionLabel.UNCERTAIN,
            confidence=0.0,
            quality=quality.overall,
            probabilities={},
            available=True,
            temporal_consistency=temporal_consistency,
            reason="acoustic estimate abstained: " + " and ".join(abstention_reasons),
            metadata=metadata,
        )

    def analyze_bytes(
        self,
        data: bytes,
        filename: str,
        content_type: str,
        mock_signal: MockSignal | None = None,
    ) -> ModalityResult:
        """Analyze a WAV payload or return an explicit deterministic mock signal."""

        if mock_signal is not None:
            return _mock_result(mock_signal)
        self._validate_media(data, filename, content_type)
        audio, quality = self._prepare(data)
        return self._infer(audio, quality)

    def analyze_detailed(
        self,
        data: bytes,
        filename: str,
        content_type: str,
        *,
        include_transcript: bool = False,
        mock_signal: MockSignal | None = None,
    ) -> AudioAnalysis:
        """Return acoustics and, only when opted in, a separate transcript."""

        result = self.analyze_bytes(data, filename, content_type, mock_signal)
        if mock_signal is not None or not include_transcript or self.stt_adapter is None:
            return AudioAnalysis(result, None)
        audio, _quality = self._prepare(data)
        transcript = self.stt_adapter.transcribe(audio)
        return AudioAnalysis(result, transcript)

    def transcribe_bytes(self, data: bytes, filename: str, content_type: str) -> str | None:
        """Transcribe a validated WAV independently from acoustic-emotion inference."""

        if self.stt_adapter is None:
            return None
        self._validate_media(data, filename, content_type)
        audio, _quality = self._prepare(data)
        return self.stt_adapter.transcribe(audio)

    def clear_temporal_state(self) -> None:
        self.smoother.clear()


# Older integration name retained as a descriptive alias.
AudioEmotionPipeline = AudioAnalyzer
WhisperSpeechToTextAdapter = HuggingFaceSpeechToTextAdapter
