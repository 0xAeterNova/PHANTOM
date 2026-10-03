"""Inspectable audio quality estimates used for abstention and weighting."""

from __future__ import annotations

import math
from dataclasses import dataclass
from importlib import import_module
from typing import Any

from phantom.audio.preprocessing import VADResult, detect_voice_activity


@dataclass(frozen=True, slots=True)
class AudioQuality:
    overall: float
    snr_db: float
    clipping_fraction: float
    speech_ratio: float
    rms: float
    duration_seconds: float
    sufficient: bool
    reason: str | None = None


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def estimate_audio_quality(
    samples: Any,
    sample_rate: int,
    *,
    vad: VADResult | None = None,
    minimum_duration_seconds: float = 0.25,
) -> AudioQuality:
    """Estimate audibility, speech coverage, clipping, duration, and SNR.

    SNR uses quiet frames as a noise proxy and is therefore an estimate, not a
    calibrated measurement of recording hardware.
    """

    try:
        np = import_module("numpy")
    except ModuleNotFoundError as exc:  # pragma: no cover - packaging guard
        raise RuntimeError("NumPy is required for audio quality analysis") from exc
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    waveform = np.asarray(samples, dtype=np.float32).reshape(-1)
    duration = waveform.size / float(sample_rate)
    if waveform.size == 0:
        return AudioQuality(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, "empty audio")
    waveform = np.nan_to_num(waveform)
    rms = float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64)) + 1e-12))
    clipping = float(np.mean(np.abs(waveform) >= 0.985))
    activity = vad or detect_voice_activity(waveform, sample_rate)

    frame_length = activity.frame_length
    hop_length = activity.hop_length
    frame_rms: list[float] = []
    for start in range(0, max(1, waveform.size - frame_length + 1), hop_length):
        frame = waveform[start : start + frame_length]
        if frame.size:
            frame_rms.append(float(np.sqrt(np.mean(np.square(frame, dtype=np.float64)) + 1e-12)))
    if frame_rms:
        noise = max(float(np.percentile(frame_rms, 20)), 1e-6)
        signal = max(float(np.percentile(frame_rms, 80)), noise)
        snr_db = 20.0 * math.log10(signal / noise)
    else:
        snr_db = 0.0

    audibility_score = _clamp((rms - 0.002) / 0.035)
    snr_score = _clamp((snr_db - 3.0) / 22.0)
    speech_score = _clamp(activity.speech_ratio / 0.35)
    clipping_score = _clamp(1.0 - clipping / 0.05)
    duration_score = _clamp(duration / max(minimum_duration_seconds, 1e-6))
    overall = (
        0.24 * audibility_score
        + 0.24 * snr_score
        + 0.26 * speech_score
        + 0.16 * clipping_score
        + 0.10 * duration_score
    )
    reason: str | None = None
    if duration < minimum_duration_seconds:
        reason = "audio is too short"
    elif rms < 0.002:
        reason = "audio level is too low"
    elif activity.speech_ratio <= 0.0:
        reason = "no voice activity detected"
    elif clipping > 0.05:
        reason = "audio is heavily clipped"
    elif overall < 0.25:
        reason = "audio quality is insufficient"
    sufficient = reason is None
    return AudioQuality(
        overall=_clamp(overall),
        snr_db=float(snr_db),
        clipping_fraction=clipping,
        speech_ratio=activity.speech_ratio,
        rms=rms,
        duration_seconds=duration,
        sufficient=sufficient,
        reason=reason,
    )


# Backwards-friendly name for callers that treat quality as a single operation.
assess_audio_quality = estimate_audio_quality
