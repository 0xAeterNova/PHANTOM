"""Optional PyTorch CNN and CRNN architectures for log-Mel spectrograms.

This module intentionally is not imported by :mod:`phantom.audio`.  PyTorch is
an optional dependency, so lightweight PHANTOM installations remain usable
without importing it.  Inputs use ``[batch, 1, mel_bins, frames]`` ordering;
``lengths`` always refers to the unpadded number of input frames.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

try:
    import torch
    from torch import Tensor, nn
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
except ImportError as exc:  # pragma: no cover - depends on optional ML extras
    raise RuntimeError(
        "Spectrogram neural models require the optional PyTorch dependency; "
        "install Project PHANTOM with the 'audio-ml' extra"
    ) from exc


SUPPORTED_ARCHITECTURES = frozenset({"cnn", "crnn"})


def pooled_frame_lengths(lengths: Tensor) -> Tensor:
    """Return frame lengths after the encoder's two ceil-mode time pools."""

    return torch.div(lengths + 3, 4, rounding_mode="floor")


def _validate_positive(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_dropout(name: str, value: float) -> float:
    converted = float(value)
    if not 0.0 <= converted < 1.0:
        raise ValueError(f"{name} must be in [0, 1)")
    return converted


def _validate_channels(channels: Sequence[int]) -> tuple[int, int, int]:
    converted = tuple(channels)
    if len(converted) != 3:
        raise ValueError("channels must contain exactly three values")
    for index, value in enumerate(converted):
        _validate_positive(f"channels[{index}]", value)
    return converted


def _time_mask(lengths: Tensor, frames: int) -> Tensor:
    positions = torch.arange(frames, device=lengths.device)
    return positions.unsqueeze(0) < lengths.unsqueeze(1)


def _validated_lengths(spectrograms: Tensor, lengths: Tensor | None) -> Tensor:
    batch_size = spectrograms.shape[0]
    frame_count = spectrograms.shape[-1]
    if lengths is None:
        return torch.full((batch_size,), frame_count, dtype=torch.long, device=spectrograms.device)
    if lengths.ndim != 1 or lengths.shape[0] != batch_size:
        raise ValueError("lengths must have shape [batch]")
    checked = lengths.to(device=spectrograms.device, dtype=torch.long)
    if bool(torch.any(checked <= 0)) or bool(torch.any(checked > frame_count)):
        raise ValueError("every length must be between 1 and the padded frame count")
    return checked


class _ChannelLayerNorm(nn.Module):  # type: ignore[misc]
    """Normalize channels independently at each time-frequency location."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(channels)

    def forward(self, values: Tensor) -> Tensor:
        normalized = self.normalization(values.permute(0, 2, 3, 1))
        return normalized.permute(0, 3, 1, 2).contiguous()


class _ConvolutionBlock(nn.Module):  # type: ignore[misc]
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        *,
        pool: tuple[int, int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.convolution = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.normalization = _ChannelLayerNorm(output_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout2d(dropout)
        self.pool = nn.MaxPool2d(kernel_size=pool, stride=pool, ceil_mode=True)

    def forward(self, values: Tensor, valid_frames: Tensor) -> Tensor:
        values = self.convolution(values)
        values = self.normalization(values)
        values = self.activation(values)
        values = self.dropout(values)
        mask = _time_mask(valid_frames, values.shape[-1])
        values = values.masked_fill(~mask[:, None, None, :], torch.finfo(values.dtype).min)
        return self.pool(values)


class _SpectrogramEncoder(nn.Module):  # type: ignore[misc]
    """Convert a padded spectrogram batch into a masked frame sequence."""

    def __init__(
        self,
        *,
        n_mels: int,
        channels: tuple[int, int, int],
        projection_size: int,
        conv_dropout: float,
    ) -> None:
        super().__init__()
        self.n_mels = _validate_positive("n_mels", n_mels)
        if self.n_mels < 8:
            raise ValueError("n_mels must be at least 8 for the three-stage CNN encoder")
        self.channels = channels
        self.projection_size = _validate_positive("projection_size", projection_size)
        self.blocks = nn.ModuleList(
            (
                _ConvolutionBlock(1, channels[0], pool=(2, 2), dropout=conv_dropout),
                _ConvolutionBlock(channels[0], channels[1], pool=(2, 2), dropout=conv_dropout),
                _ConvolutionBlock(channels[1], channels[2], pool=(2, 1), dropout=conv_dropout),
            )
        )
        frequency_bins = self.n_mels
        for _ in range(3):
            frequency_bins = (frequency_bins + 1) // 2
        self.frequency_bins = frequency_bins
        self.projection = nn.Sequential(
            nn.Linear(channels[-1] * frequency_bins, self.projection_size),
            nn.LayerNorm(self.projection_size),
            nn.GELU(),
            nn.Dropout(conv_dropout),
        )

    def forward(self, spectrograms: Tensor, lengths: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if spectrograms.ndim != 4:
            raise ValueError("spectrograms must have shape [batch, 1, mel_bins, frames]")
        if spectrograms.shape[1] != 1:
            raise ValueError("spectrograms must contain exactly one input channel")
        if spectrograms.shape[2] != self.n_mels:
            raise ValueError(f"expected {self.n_mels} mel bins, received {spectrograms.shape[2]}")
        if spectrograms.shape[0] <= 0 or spectrograms.shape[-1] <= 0:
            raise ValueError("spectrogram batches and frame sequences cannot be empty")
        checked_lengths = _validated_lengths(spectrograms, lengths)
        mask = _time_mask(checked_lengths, spectrograms.shape[-1])
        values = spectrograms.masked_fill(~mask[:, None, None, :], 0.0)

        for index, block in enumerate(self.blocks):
            values = block(values, checked_lengths)
            if index < 2:
                checked_lengths = torch.div(checked_lengths + 1, 2, rounding_mode="floor")
            mask = _time_mask(checked_lengths, values.shape[-1])
            values = values.masked_fill(~mask[:, None, None, :], 0.0)

        batch_size, channels, frequency_bins, frame_count = values.shape
        sequence = values.permute(0, 3, 1, 2).reshape(
            batch_size, frame_count, channels * frequency_bins
        )
        sequence = self.projection(sequence)
        sequence = sequence.masked_fill(~mask.unsqueeze(-1), 0.0)
        return sequence, checked_lengths


class _MaskedAttentionPooling(nn.Module):  # type: ignore[misc]
    def __init__(self, input_size: int, attention_size: int) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(input_size, attention_size),
            nn.Tanh(),
            nn.Linear(attention_size, 1),
        )

    def forward(self, sequence: Tensor, lengths: Tensor) -> Tensor:
        mask = _time_mask(lengths, sequence.shape[1])
        scores = self.scorer(sequence).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        return torch.bmm(weights.unsqueeze(1), sequence).squeeze(1)


def _classification_head(
    input_size: int, hidden_size: int, num_classes: int, dropout: float
) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(input_size),
        nn.Linear(input_size, hidden_size),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_size, num_classes),
    )


class SpectrogramCNN(nn.Module):  # type: ignore[misc]
    """Compact CNN-only spectrogram classifier for controlled ablations."""

    architecture = "cnn"

    def __init__(
        self,
        *,
        n_mels: int = 40,
        num_classes: int = 7,
        channels: Sequence[int] = (32, 64, 128),
        projection_size: int = 128,
        attention_size: int = 128,
        conv_dropout: float = 0.10,
        classifier_dropout: float = 0.35,
    ) -> None:
        super().__init__()
        checked_channels = _validate_channels(channels)
        checked_projection = _validate_positive("projection_size", projection_size)
        checked_attention = _validate_positive("attention_size", attention_size)
        checked_classes = _validate_positive("num_classes", num_classes)
        checked_conv_dropout = _validate_dropout("conv_dropout", conv_dropout)
        checked_classifier_dropout = _validate_dropout("classifier_dropout", classifier_dropout)
        self.n_mels = _validate_positive("n_mels", n_mels)
        self.num_classes = checked_classes
        self.model_config: dict[str, Any] = {
            "channels": list(checked_channels),
            "projection_size": checked_projection,
            "attention_size": checked_attention,
            "conv_dropout": checked_conv_dropout,
            "classifier_dropout": checked_classifier_dropout,
        }
        self.encoder = _SpectrogramEncoder(
            n_mels=self.n_mels,
            channels=checked_channels,
            projection_size=checked_projection,
            conv_dropout=checked_conv_dropout,
        )
        self.pooling = _MaskedAttentionPooling(checked_projection, checked_attention)
        self.classifier = _classification_head(
            checked_projection,
            checked_projection,
            checked_classes,
            checked_classifier_dropout,
        )

    def forward(self, spectrograms: Tensor, lengths: Tensor | None = None) -> Tensor:
        sequence, reduced_lengths = self.encoder(spectrograms, lengths)
        embedding = self.pooling(sequence, reduced_lengths)
        return self.classifier(embedding)


class SpectrogramCRNN(nn.Module):  # type: ignore[misc]
    """CNN plus bidirectional GRU/LSTM spectrogram emotion classifier."""

    architecture = "crnn"

    def __init__(
        self,
        *,
        n_mels: int = 40,
        num_classes: int = 7,
        channels: Sequence[int] = (32, 64, 128),
        projection_size: int = 128,
        attention_size: int = 128,
        conv_dropout: float = 0.10,
        classifier_dropout: float = 0.35,
        recurrent_type: str = "gru",
        recurrent_hidden_size: int = 128,
        recurrent_layers: int = 2,
        recurrent_dropout: float = 0.25,
    ) -> None:
        super().__init__()
        checked_channels = _validate_channels(channels)
        checked_projection = _validate_positive("projection_size", projection_size)
        checked_attention = _validate_positive("attention_size", attention_size)
        checked_classes = _validate_positive("num_classes", num_classes)
        checked_hidden = _validate_positive("recurrent_hidden_size", recurrent_hidden_size)
        checked_layers = _validate_positive("recurrent_layers", recurrent_layers)
        checked_conv_dropout = _validate_dropout("conv_dropout", conv_dropout)
        checked_classifier_dropout = _validate_dropout("classifier_dropout", classifier_dropout)
        checked_recurrent_dropout = _validate_dropout("recurrent_dropout", recurrent_dropout)
        normalized_recurrent = recurrent_type.strip().lower()
        if normalized_recurrent not in {"gru", "lstm"}:
            raise ValueError("recurrent_type must be 'gru' or 'lstm'")

        self.n_mels = _validate_positive("n_mels", n_mels)
        self.num_classes = checked_classes
        self.recurrent_type = normalized_recurrent
        self.model_config: dict[str, Any] = {
            "channels": list(checked_channels),
            "projection_size": checked_projection,
            "attention_size": checked_attention,
            "conv_dropout": checked_conv_dropout,
            "classifier_dropout": checked_classifier_dropout,
            "recurrent_type": normalized_recurrent,
            "recurrent_hidden_size": checked_hidden,
            "recurrent_layers": checked_layers,
            "recurrent_dropout": checked_recurrent_dropout,
        }
        self.encoder = _SpectrogramEncoder(
            n_mels=self.n_mels,
            channels=checked_channels,
            projection_size=checked_projection,
            conv_dropout=checked_conv_dropout,
        )
        recurrent_class = nn.GRU if normalized_recurrent == "gru" else nn.LSTM
        self.recurrent = recurrent_class(
            input_size=checked_projection,
            hidden_size=checked_hidden,
            num_layers=checked_layers,
            batch_first=True,
            bidirectional=True,
            dropout=checked_recurrent_dropout if checked_layers > 1 else 0.0,
        )
        recurrent_output_size = checked_hidden * 2
        self.pooling = _MaskedAttentionPooling(recurrent_output_size, checked_attention)
        self.classifier = _classification_head(
            recurrent_output_size,
            checked_projection,
            checked_classes,
            checked_classifier_dropout,
        )

    def forward(self, spectrograms: Tensor, lengths: Tensor | None = None) -> Tensor:
        sequence, reduced_lengths = self.encoder(spectrograms, lengths)
        packed = pack_padded_sequence(
            sequence,
            reduced_lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _state = self.recurrent(packed)
        recurrent_output, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=sequence.shape[1],
        )
        embedding = self.pooling(recurrent_output, reduced_lengths)
        return self.classifier(embedding)


def build_spectrogram_model(
    architecture: str,
    *,
    n_mels: int = 40,
    num_classes: int = 7,
    **model_config: Any,
) -> nn.Module:
    """Construct a CNN or CNN-RNN architecture without creating random fallbacks."""

    normalized = architecture.strip().lower()
    if normalized == "cnn":
        return SpectrogramCNN(n_mels=n_mels, num_classes=num_classes, **model_config)
    if normalized == "crnn":
        return SpectrogramCRNN(n_mels=n_mels, num_classes=num_classes, **model_config)
    choices = ", ".join(sorted(SUPPORTED_ARCHITECTURES))
    raise ValueError(f"unsupported spectrogram architecture {architecture!r}; choose {choices}")


__all__ = [
    "SUPPORTED_ARCHITECTURES",
    "SpectrogramCNN",
    "SpectrogramCRNN",
    "build_spectrogram_model",
    "pooled_frame_lengths",
]
