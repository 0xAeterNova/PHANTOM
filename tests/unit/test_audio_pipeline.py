from __future__ import annotations

import io
import math
import wave

import numpy as np
import pytest

from phantom.audio.features import extract_log_mel, extract_mfcc
from phantom.audio.inference import AudioAnalyzer, MockMicrophoneSource
from phantom.audio.model import TemperatureCalibrator
from phantom.audio.preprocessing import (
    detect_voice_activity,
    load_wav,
    normalize_audio,
    resample_audio,
)
from phantom.audio.quality import estimate_audio_quality
from phantom.exceptions import ConsentRequiredError, InvalidMediaError
from phantom.fusion.late_fusion import LateFusion
from phantom.schemas import ConsentSettings, EmotionLabel, Modality


class SequenceWaveformModel:
    backend = "test-waveform-sequence"

    def __init__(self, outputs: list[dict[EmotionLabel, float]]) -> None:
        self.outputs = iter(outputs)

    def predict_waveform(self, samples: np.ndarray, sample_rate: int) -> dict[EmotionLabel, float]:
        del samples, sample_rate
        return next(self.outputs)


def wav_bytes(samples: np.ndarray, sample_rate: int = 16_000) -> bytes:
    buffer = io.BytesIO()
    pcm = np.clip(samples, -1.0, 1.0)
    encoded = (pcm * 32767.0).astype("<i2").tobytes()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(encoded)
    return buffer.getvalue()


def wav_24bit_bytes(samples: np.ndarray, sample_rate: int = 48_000) -> bytes:
    buffer = io.BytesIO()
    integers = np.round(np.clip(samples, -1.0, 1.0) * 8_388_607.0).astype(np.int32)
    packed = np.empty((integers.size, 3), dtype=np.uint8)
    packed[:, 0] = integers & 0xFF
    packed[:, 1] = (integers >> 8) & 0xFF
    packed[:, 2] = (integers >> 16) & 0xFF
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(3)
        output.setframerate(sample_rate)
        output.writeframes(packed.tobytes())
    return buffer.getvalue()


def tone(duration: float = 0.6, sample_rate: int = 16_000) -> np.ndarray:
    time = np.arange(round(duration * sample_rate), dtype=np.float32) / sample_rate
    # Amplitude modulation makes the synthetic sample exercise the energy VAD.
    envelope = 0.55 + 0.35 * np.sin(2 * math.pi * 3 * time)
    return (0.30 * envelope * np.sin(2 * math.pi * 220 * time)).astype(np.float32)


def test_wav_loading_resampling_normalization_and_vad() -> None:
    decoded = load_wav(wav_bytes(tone(sample_rate=8_000), sample_rate=8_000))
    assert decoded.sample_rate == 8_000
    assert decoded.duration_seconds == pytest.approx(0.6, rel=0.02)
    resampled = resample_audio(decoded.samples, 8_000, 16_000)
    assert len(resampled) == pytest.approx(len(decoded.samples) * 2, abs=1)
    normalized = normalize_audio(resampled)
    assert np.max(np.abs(normalized)) == pytest.approx(0.95, rel=0.01)
    activity = detect_voice_activity(normalized, 16_000)
    assert activity.has_speech


def test_recommended_24bit_mono_pcm_wav_decodes() -> None:
    expected = np.array([-0.75, -0.1, 0.0, 0.1, 0.75], dtype=np.float32)

    decoded = load_wav(wav_24bit_bytes(expected))

    assert decoded.sample_rate == 48_000
    assert decoded.samples == pytest.approx(expected, abs=2e-7)


def test_downsampling_filters_frequencies_above_target_nyquist() -> None:
    source_rate = 48_000
    target_rate = 16_000
    time = np.arange(source_rate, dtype=np.float64) / source_rate
    passband = np.sin(2 * math.pi * 1_000 * time).astype(np.float32)
    stopband = np.sin(2 * math.pi * 12_000 * time).astype(np.float32)

    passband_output = resample_audio(passband, source_rate, target_rate)
    stopband_output = resample_audio(stopband, source_rate, target_rate)
    input_rms = float(np.sqrt(np.mean(np.square(passband, dtype=np.float64))))
    passband_rms = float(np.sqrt(np.mean(np.square(passband_output, dtype=np.float64))))
    stopband_rms = float(np.sqrt(np.mean(np.square(stopband_output, dtype=np.float64))))

    assert len(passband_output) == target_rate
    assert passband_rms == pytest.approx(input_rms, rel=0.02)
    assert stopband_rms < passband_rms * 0.02


def test_upsampling_preserves_linear_interpolation_behavior() -> None:
    samples = np.array([0.0, 1.0, 0.0, -1.0], dtype=np.float32)

    output = resample_audio(samples, 4, 8)

    assert output == pytest.approx([0.0, 0.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.0])


def test_log_mel_and_mfcc_have_finite_shapes() -> None:
    sample = tone()
    mel = extract_log_mel(sample, 16_000)
    mfcc = extract_mfcc(sample, 16_000)
    assert mel.ndim == 2 and mel.size > 0
    assert mfcc.ndim == 2 and mfcc.size > 0
    assert np.isfinite(mel).all()
    assert np.isfinite(mfcc).all()


def test_low_quality_silent_audio_abstains() -> None:
    silent = np.zeros(16_000, dtype=np.float32)
    quality = estimate_audio_quality(silent, 16_000)
    assert quality.sufficient is False
    result = AudioAnalyzer().analyze_bytes(wav_bytes(silent), "silence.wav", "audio/wav")
    assert result.label is EmotionLabel.INSUFFICIENT_QUALITY
    assert result.confidence == 0.0


def test_mislabeled_audio_is_rejected_by_magic_bytes() -> None:
    with pytest.raises(InvalidMediaError):
        AudioAnalyzer().analyze_bytes(b"not a wav", "fake.wav", "audio/wav")


def test_usable_audio_returns_calibrated_distribution() -> None:
    result = AudioAnalyzer().analyze_bytes(wav_bytes(tone()), "tone.wav", "audio/wav")
    assert 0.0 <= result.quality <= 1.0
    assert 0.0 <= result.confidence <= 1.0
    if result.probabilities:
        assert sum(result.probabilities.values()) == pytest.approx(1.0)


def test_ambiguous_angry_candidate_abstains_and_is_withheld_from_fusion() -> None:
    model = SequenceWaveformModel(
        [
            {
                EmotionLabel.ANGRY: 0.48,
                EmotionLabel.NEUTRAL: 0.43,
                EmotionLabel.SAD: 0.05,
                EmotionLabel.HAPPY: 0.04,
            }
        ]
    )
    analyzer = AudioAnalyzer(
        model=model,
        calibrator=TemperatureCalibrator(1.0),
        confidence_floor=0.20,
        decision_margin_floor=0.04,
        angry_confidence_floor=0.30,
        angry_decision_margin_floor=0.18,
    )

    result = analyzer.analyze_bytes(wav_bytes(tone()), "tone.wav", "audio/wav")

    assert result.label is EmotionLabel.UNCERTAIN
    assert result.confidence == 0.0
    assert result.probabilities == {}
    assert result.metadata["candidate_label"] == "angry"
    assert result.metadata["decision_margin"] == pytest.approx(0.05)
    assert result.metadata["withheld_from_fusion"] is True
    assert "winning margin too small" in result.metadata["abstention_reasons"]
    fused = LateFusion().fuse({Modality.AUDIO: result})
    assert fused.label is EmotionLabel.UNCERTAIN
    assert fused.contributions == {}


def test_clear_four_label_candidate_is_not_double_penalized_by_audio_quality() -> None:
    model = SequenceWaveformModel(
        [
            {
                EmotionLabel.SAD: 0.39,
                EmotionLabel.NEUTRAL: 0.30,
                EmotionLabel.HAPPY: 0.20,
                EmotionLabel.ANGRY: 0.11,
            }
        ]
    )
    analyzer = AudioAnalyzer(
        model=model,
        calibrator=TemperatureCalibrator(1.0),
        confidence_floor=0.38,
        decision_margin_floor=0.08,
    )

    result = analyzer.analyze_bytes(wav_bytes(tone()), "tone.wav", "audio/wav")

    assert result.label is EmotionLabel.SAD
    assert result.confidence == pytest.approx(0.39)
    assert result.quality < 1.0
    assert result.metadata["quality_weighted_confidence"] < 0.38
    assert result.metadata["supported_label_count"] == 4
    assert result.metadata["chance_probability"] == pytest.approx(0.25)
    assert result.metadata["decision_policy"] == "quality-separated-margin-gated-v2"


def test_weak_four_label_near_tie_still_abstains() -> None:
    model = SequenceWaveformModel(
        [
            {
                EmotionLabel.NEUTRAL: 0.34,
                EmotionLabel.SAD: 0.32,
                EmotionLabel.HAPPY: 0.18,
                EmotionLabel.ANGRY: 0.16,
            }
        ]
    )
    analyzer = AudioAnalyzer(
        model=model,
        calibrator=TemperatureCalibrator(1.0),
        confidence_floor=0.36,
        decision_margin_floor=0.07,
    )

    result = analyzer.analyze_bytes(wav_bytes(tone()), "tone.wav", "audio/wav")

    assert result.label is EmotionLabel.UNCERTAIN
    assert result.confidence == 0.0
    assert result.metadata["candidate_label"] == "neutral"
    assert result.metadata["candidate_confidence"] == pytest.approx(0.34)
    assert result.metadata["decision_margin"] == pytest.approx(0.02)
    assert result.metadata["supported_label_count"] == 4
    assert result.metadata["withheld_from_fusion"] is True
    assert result.metadata["abstention_reasons"] == [
        "top score below the decision threshold",
        "winning margin too small",
    ]


def test_angry_candidate_keeps_the_stricter_confidence_gate() -> None:
    model = SequenceWaveformModel(
        [
            {
                EmotionLabel.ANGRY: 0.61,
                EmotionLabel.NEUTRAL: 0.19,
                EmotionLabel.SAD: 0.10,
                EmotionLabel.HAPPY: 0.10,
            }
        ]
    )
    analyzer = AudioAnalyzer(
        model=model,
        calibrator=TemperatureCalibrator(1.0),
        confidence_floor=0.36,
        decision_margin_floor=0.07,
        angry_confidence_floor=0.62,
        angry_decision_margin_floor=0.22,
    )

    result = analyzer.analyze_bytes(wav_bytes(tone()), "tone.wav", "audio/wav")

    assert result.label is EmotionLabel.UNCERTAIN
    assert result.metadata["candidate_label"] == "angry"
    assert result.metadata["candidate_confidence"] == pytest.approx(0.61)
    assert result.metadata["decision_margin"] == pytest.approx(0.42)
    assert result.metadata["abstention_reasons"] == ["top score below the decision threshold"]


def test_independent_recordings_do_not_inherit_a_prior_angry_prediction() -> None:
    model = SequenceWaveformModel(
        [
            {EmotionLabel.ANGRY: 0.92, EmotionLabel.NEUTRAL: 0.08},
            {EmotionLabel.SAD: 0.92, EmotionLabel.NEUTRAL: 0.08},
        ]
    )
    analyzer = AudioAnalyzer(
        model=model,
        calibrator=TemperatureCalibrator(1.0),
    )

    first = analyzer.analyze_bytes(wav_bytes(tone()), "first.wav", "audio/wav")
    second = analyzer.analyze_bytes(wav_bytes(tone()), "second.wav", "audio/wav")

    assert first.label is EmotionLabel.ANGRY
    assert second.label is EmotionLabel.SAD
    assert second.probabilities[EmotionLabel.SAD] > second.probabilities[EmotionLabel.ANGRY]


def test_microphone_abstraction_enforces_consent_and_stop() -> None:
    source = MockMicrophoneSource(wav_bytes(tone()))
    with pytest.raises(ConsentRequiredError):
        source.start(ConsentSettings())
    source.start(ConsentSettings(microphone=True))
    assert source.read_wav() is not None
    source.stop()
    with pytest.raises(RuntimeError):
        source.read_wav()
