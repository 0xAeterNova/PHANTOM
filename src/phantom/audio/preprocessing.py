"""Local, dependency-light audio decoding and preprocessing.

Only uncompressed PCM WAV is accepted by the baseline.  The module imports
NumPy inside functions so importing :mod:`phantom.audio` stays inexpensive and
optional pretrained adapters do not become runtime requirements.
"""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, BinaryIO

from phantom.exceptions import InvalidMediaError


@dataclass(frozen=True, slots=True)
class AudioBuffer:
    """A mono floating-point waveform in the range ``[-1, 1]``."""

    samples: Any
    sample_rate: int

    @property
    def duration_seconds(self) -> float:
        return float(len(self.samples)) / float(self.sample_rate) if self.sample_rate else 0.0


@dataclass(frozen=True, slots=True)
class VADResult:
    """Frame-level energy VAD output; this is not a speaker detector."""

    frame_mask: Any
    speech_ratio: float
    voiced_samples: Any
    frame_length: int
    hop_length: int

    @property
    def has_speech(self) -> bool:
        return bool(self.speech_ratio > 0.0 and len(self.voiced_samples) > 0)


def _numpy() -> Any:
    try:
        return import_module("numpy")
    except ModuleNotFoundError as exc:  # pragma: no cover - packaging guard
        raise RuntimeError(
            "NumPy is required for audio processing; install project-phantom"
        ) from exc


def _open_wave(source: bytes | bytearray | memoryview | str | Path | BinaryIO) -> wave.Wave_read:
    try:
        if isinstance(source, (bytes, bytearray, memoryview)):
            return wave.open(io.BytesIO(bytes(source)), "rb")
        if isinstance(source, (str, Path)):
            return wave.open(str(source), "rb")
        return wave.open(source, "rb")
    except (EOFError, OSError, wave.Error) as exc:
        raise InvalidMediaError("audio must be a valid uncompressed PCM WAV file") from exc


def _decode_pcm(raw: bytes, sample_width: int) -> Any:
    np = _numpy()
    if sample_width == 1:
        return (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    if sample_width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if sample_width == 3:
        packed = np.frombuffer(raw, dtype=np.uint8)
        if packed.size % 3:
            raise InvalidMediaError("24-bit WAV payload is truncated")
        triples = packed.reshape(-1, 3).astype(np.int32)
        values = triples[:, 0] | (triples[:, 1] << 8) | (triples[:, 2] << 16)
        values = np.where(values & 0x800000, values - 0x1000000, values)
        return values.astype(np.float32) / 8388608.0
    if sample_width == 4:
        return np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    raise InvalidMediaError(f"unsupported PCM sample width: {sample_width} bytes")


def load_wav(source: bytes | bytearray | memoryview | str | Path | BinaryIO) -> AudioBuffer:
    """Decode an uncompressed integer-PCM WAV source to a mono waveform.

    Channels are averaged rather than discarded.  Floating-point and compressed
    WAV variants are rejected so media handling has a small, auditable surface.
    """

    np = _numpy()
    reader = _open_wave(source)
    try:
        channels = reader.getnchannels()
        sample_rate = reader.getframerate()
        sample_width = reader.getsampwidth()
        frame_count = reader.getnframes()
        compression = reader.getcomptype()
        if compression != "NONE":
            raise InvalidMediaError("compressed WAV audio is not supported")
        if not 1 <= channels <= 8:
            raise InvalidMediaError("WAV must contain between 1 and 8 channels")
        if not 8_000 <= sample_rate <= 192_000:
            raise InvalidMediaError("WAV sample rate must be between 8 kHz and 192 kHz")
        if frame_count <= 0:
            raise InvalidMediaError("WAV contains no audio frames")
        raw = reader.readframes(frame_count)
    except (EOFError, OSError, wave.Error) as exc:
        raise InvalidMediaError("WAV payload is malformed or truncated") from exc
    finally:
        reader.close()

    samples = _decode_pcm(raw, sample_width)
    expected = frame_count * channels
    if samples.size != expected:
        raise InvalidMediaError("WAV sample count does not match its header")
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    samples = np.nan_to_num(samples.astype(np.float32, copy=False), copy=False)
    return AudioBuffer(samples=np.clip(samples, -1.0, 1.0), sample_rate=sample_rate)


# Friendly alias used by integrations that do not need format dispatch yet.
load_audio = load_wav


def normalize_audio(samples: Any, peak: float = 0.95, remove_dc: bool = True) -> Any:
    """Return a finite float waveform with optional DC removal and peak scaling."""

    if not 0.0 < peak <= 1.0:
        raise ValueError("peak must be in (0, 1]")
    np = _numpy()
    output = np.asarray(samples, dtype=np.float32).reshape(-1).copy()
    if output.size == 0:
        return output
    output = np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
    if remove_dc:
        output -= float(output.mean())
    maximum = float(np.max(np.abs(output)))
    if maximum > 1e-8:
        output *= peak / maximum
    return np.clip(output, -1.0, 1.0).astype(np.float32, copy=False)


def resample_audio(samples: Any, source_rate: int, target_rate: int) -> Any:
    """Deterministically resample a mono waveform with linear interpolation.

    Downsampling first applies a windowed-sinc low-pass filter so frequencies
    above the target Nyquist limit do not fold into the retained spectrum.
    """

    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rates must be positive")
    np = _numpy()
    waveform = np.asarray(samples, dtype=np.float32).reshape(-1)
    if waveform.size == 0 or source_rate == target_rate:
        return waveform.copy()
    if target_rate < source_rate:
        # Scale the kernel with the reduction ratio so its transition band
        # remains useful for both gentle and aggressive downsampling.  A
        # Blackman window gives strong stop-band attenuation without adding a
        # signal-processing dependency to the baseline installation.
        half_width = max(16, int(np.ceil(16.0 * source_rate / target_rate)))
        offsets = np.arange(-half_width, half_width + 1, dtype=np.float64)
        cutoff = 0.95 * 0.5 * target_rate / source_rate
        kernel = 2.0 * cutoff * np.sinc(2.0 * cutoff * offsets)
        kernel *= np.blackman(kernel.size)
        kernel /= kernel.sum()

        pad_mode = "reflect" if waveform.size > 1 else "edge"
        padded = np.pad(waveform, (half_width, half_width), mode=pad_mode)
        waveform = np.convolve(padded, kernel, mode="valid").astype(np.float32)
    output_length = max(1, round(waveform.size * target_rate / source_rate))
    old_positions = np.arange(waveform.size, dtype=np.float64)
    new_positions = np.arange(output_length, dtype=np.float64) * source_rate / target_rate
    new_positions = np.minimum(new_positions, waveform.size - 1)
    return np.interp(new_positions, old_positions, waveform).astype(np.float32)


def _frames(samples: Any, frame_length: int, hop_length: int) -> Any:
    np = _numpy()
    waveform = np.asarray(samples, dtype=np.float32).reshape(-1)
    if waveform.size == 0:
        return np.empty((0, frame_length), dtype=np.float32)
    if waveform.size < frame_length:
        waveform = np.pad(waveform, (0, frame_length - waveform.size))
    count = 1 + int(np.ceil((waveform.size - frame_length) / hop_length))
    padded_length = (count - 1) * hop_length + frame_length
    if waveform.size < padded_length:
        waveform = np.pad(waveform, (0, padded_length - waveform.size))
    indices = np.arange(frame_length)[None, :] + hop_length * np.arange(count)[:, None]
    return waveform[indices]


def detect_voice_activity(
    samples: Any,
    sample_rate: int,
    *,
    frame_ms: float = 30.0,
    hop_ms: float = 10.0,
    relative_threshold_db: float = 12.0,
    absolute_rms_floor: float = 0.003,
) -> VADResult:
    """Apply a conservative energy VAD and return voiced frames and samples.

    The detector is intended for input gating and quality weighting, not for
    biometric identification or clinical interpretation.
    """

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    np = _numpy()
    waveform = np.asarray(samples, dtype=np.float32).reshape(-1)
    frame_length = max(1, round(sample_rate * frame_ms / 1000.0))
    hop_length = max(1, round(sample_rate * hop_ms / 1000.0))
    framed = _frames(waveform, frame_length, hop_length)
    if framed.size == 0:
        return VADResult(
            frame_mask=np.zeros(0, dtype=bool),
            speech_ratio=0.0,
            voiced_samples=np.zeros(0, dtype=np.float32),
            frame_length=frame_length,
            hop_length=hop_length,
        )
    rms = np.sqrt(np.mean(np.square(framed, dtype=np.float64), axis=1) + 1e-12)
    # The 20th percentile is a stable local noise proxy for short clips.
    noise_rms = float(np.percentile(rms, 20))
    relative_floor = noise_rms * (10.0 ** (relative_threshold_db / 20.0))
    energetic_floor = float(np.percentile(rms, 80)) * 0.60
    # Continuous voiced segments may contain no quiet calibration frame.  Cap
    # the adaptive threshold so an energetic, steady segment is not rejected.
    threshold = max(absolute_rms_floor, min(relative_floor, energetic_floor))
    mask = rms >= threshold
    if mask.any():
        voiced = framed[mask].reshape(-1).astype(np.float32, copy=False)
    else:
        voiced = np.zeros(0, dtype=np.float32)
    return VADResult(
        frame_mask=mask,
        speech_ratio=float(mask.mean()),
        voiced_samples=voiced,
        frame_length=frame_length,
        hop_length=hop_length,
    )


def preprocess_audio(
    audio: AudioBuffer,
    *,
    target_sample_rate: int = 16_000,
    normalize: bool = True,
) -> AudioBuffer:
    """Convert a decoded waveform to the baseline model's mono sample rate."""

    samples = resample_audio(audio.samples, audio.sample_rate, target_sample_rate)
    if normalize:
        samples = normalize_audio(samples)
    return AudioBuffer(samples=samples, sample_rate=target_sample_rate)


def decode_and_preprocess_wav(
    source: bytes | bytearray | memoryview | str | Path | BinaryIO,
    *,
    target_sample_rate: int = 16_000,
) -> AudioBuffer:
    return preprocess_audio(load_wav(source), target_sample_rate=target_sample_rate)
