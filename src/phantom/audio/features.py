"""Deterministic log-Mel and MFCC feature extraction without librosa."""

from __future__ import annotations

from importlib import import_module
from typing import Any


def _numpy() -> Any:
    try:
        return import_module("numpy")
    except ModuleNotFoundError as exc:  # pragma: no cover - packaging guard
        raise RuntimeError("NumPy is required for acoustic feature extraction") from exc


def _hz_to_mel(value: Any) -> Any:
    np = _numpy()
    return 2595.0 * np.log10(1.0 + np.asarray(value) / 700.0)


def _mel_to_hz(value: Any) -> Any:
    np = _numpy()
    return 700.0 * (np.power(10.0, np.asarray(value) / 2595.0) - 1.0)


def _frame(samples: Any, frame_length: int, hop_length: int) -> Any:
    np = _numpy()
    waveform = np.asarray(samples, dtype=np.float32).reshape(-1)
    if waveform.size == 0:
        return np.empty((0, frame_length), dtype=np.float32)
    if waveform.size < frame_length:
        waveform = np.pad(waveform, (0, frame_length - waveform.size))
    frame_count = 1 + int(np.ceil((waveform.size - frame_length) / hop_length))
    padded = (frame_count - 1) * hop_length + frame_length
    if waveform.size < padded:
        waveform = np.pad(waveform, (0, padded - waveform.size))
    positions = np.arange(frame_length)[None, :] + hop_length * np.arange(frame_count)[:, None]
    return waveform[positions]


def mel_filterbank(
    sample_rate: int,
    *,
    n_fft: int = 512,
    n_mels: int = 40,
    f_min: float = 20.0,
    f_max: float | None = None,
) -> Any:
    """Create triangular Slaney-style frequency bins (without area normalization)."""

    if sample_rate <= 0 or n_fft < 2 or n_mels < 2:
        raise ValueError("sample_rate, n_fft, and n_mels must be positive")
    np = _numpy()
    upper = float(f_max if f_max is not None else sample_rate / 2.0)
    if not 0.0 <= f_min < upper <= sample_rate / 2.0:
        raise ValueError("Mel frequency range must lie within the Nyquist interval")
    mel_points = np.linspace(_hz_to_mel(f_min), _hz_to_mel(upper), n_mels + 2)
    bins = np.floor((n_fft + 1) * _mel_to_hz(mel_points) / sample_rate).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)
    bank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for index in range(n_mels):
        left, center, right = int(bins[index]), int(bins[index + 1]), int(bins[index + 2])
        center = max(center, left + 1)
        right = max(right, center + 1)
        right = min(right, n_fft // 2 + 1)
        for frequency_bin in range(left, min(center, bank.shape[1])):
            bank[index, frequency_bin] = (frequency_bin - left) / max(1, center - left)
        for frequency_bin in range(center, right):
            bank[index, frequency_bin] = (right - frequency_bin) / max(1, right - center)
    return bank


def extract_log_mel(
    samples: Any,
    sample_rate: int,
    *,
    n_mels: int = 40,
    n_fft: int = 512,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    f_min: float = 20.0,
    f_max: float | None = None,
) -> Any:
    """Return a ``[frames, mel_bins]`` natural-log power spectrogram."""

    np = _numpy()
    frame_length = max(2, round(sample_rate * frame_ms / 1000.0))
    hop_length = max(1, round(sample_rate * hop_ms / 1000.0))
    fft_size = max(n_fft, 1 << (frame_length - 1).bit_length())
    framed = _frame(samples, frame_length, hop_length)
    if framed.shape[0] == 0:
        return np.empty((0, n_mels), dtype=np.float32)
    windowed = framed * np.hanning(frame_length).astype(np.float32)
    spectrum = np.fft.rfft(windowed, n=fft_size, axis=1)
    power = (np.abs(spectrum) ** 2) / float(fft_size)
    bank = mel_filterbank(
        sample_rate,
        n_fft=fft_size,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
    )
    mel_power = power @ bank.T
    return np.log(np.maximum(mel_power, 1e-10)).astype(np.float32)


def extract_mfcc(
    samples: Any,
    sample_rate: int,
    *,
    n_mfcc: int = 13,
    n_mels: int = 40,
    **log_mel_options: Any,
) -> Any:
    """Return DCT-II coefficients over the log-Mel representation."""

    if not 1 <= n_mfcc <= n_mels:
        raise ValueError("n_mfcc must be between 1 and n_mels")
    np = _numpy()
    log_mel = extract_log_mel(samples, sample_rate, n_mels=n_mels, **log_mel_options)
    if log_mel.shape[0] == 0:
        return np.empty((0, n_mfcc), dtype=np.float32)
    mel_indices = np.arange(n_mels, dtype=np.float64) + 0.5
    coefficient_indices = np.arange(n_mfcc, dtype=np.float64)[:, None]
    basis = np.cos(np.pi / n_mels * coefficient_indices * mel_indices[None, :])
    basis[0] *= 1.0 / np.sqrt(2.0)
    basis *= np.sqrt(2.0 / n_mels)
    return (log_mel @ basis.T).astype(np.float32)


# Concise aliases commonly used in feature-pipeline configurations.
log_mel_spectrogram = extract_log_mel
mfcc = extract_mfcc
