"""Safe, lazy adapters for locally trained spectrogram neural checkpoints.

Importing this module does not import PyTorch.  A neural checkpoint is loaded
only after an explicit adapter construction, with PyTorch's ``weights_only``
mode and strict validation of architecture, labels, preprocessing, and tensor
shapes.  There is deliberately no random-weight or download fallback.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

from phantom.schemas import FUSION_EMOTIONS, EmotionLabel

CHECKPOINT_FORMAT_VERSION = 1
MAX_CHECKPOINT_BYTES = 256 * 1024 * 1024
MAX_STATE_DICT_BYTES = 256 * 1024 * 1024
MAX_NORMALIZATION_MAGNITUDE = 1_000_000.0
MIN_NORMALIZATION_STD = 1e-6
SUPPORTED_ARCHITECTURES = frozenset({"cnn", "crnn"})

_CHECKPOINT_REQUIRED_KEYS = frozenset(
    {
        "format_version",
        "architecture",
        "model_config",
        "feature_config",
        "labels",
        "normalization",
        "temperature",
        "state_dict",
    }
)
_CHECKPOINT_OPTIONAL_KEYS = frozenset({"training"})
_COMMON_MODEL_CONFIG_KEYS = frozenset(
    {
        "channels",
        "projection_size",
        "attention_size",
        "conv_dropout",
        "classifier_dropout",
    }
)
_CRNN_MODEL_CONFIG_KEYS = _COMMON_MODEL_CONFIG_KEYS | {
    "recurrent_type",
    "recurrent_hidden_size",
    "recurrent_layers",
    "recurrent_dropout",
}


def _require_torch() -> Any:
    try:
        return import_module("torch")
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError(
            "Spectrogram neural inference requires PyTorch; install Project PHANTOM "
            "with the 'audio-ml' extra"
        ) from exc


def resolve_torch_device(device: str = "cpu") -> str:
    """Resolve ``auto`` safely while rejecting unavailable explicit accelerators."""

    torch = _require_torch()
    requested = str(device).strip().lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and bool(mps.is_available()):
            return "mps"
        return "cpu"
    try:
        selected = torch.device(requested)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"invalid PyTorch device: {device!r}") from exc
    if selected.type not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be cpu, cuda, mps, or auto")
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if selected.type == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not bool(mps.is_available()):
            raise RuntimeError("MPS was requested but is not available")
    return str(selected)


def _require_exact_keys(
    values: Mapping[str, Any], required: frozenset[str], optional: frozenset[str], context: str
) -> None:
    keys = set(values)
    if missing := required - keys:
        raise ValueError(f"{context} is missing required keys: {sorted(missing)}")
    if unexpected := keys - required - optional:
        raise ValueError(f"{context} contains unexpected keys: {sorted(unexpected)}")


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be a finite number")
    return converted


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return int(value)


@dataclass(frozen=True, slots=True)
class SpectrogramFeatureConfig:
    """Feature settings supported by the current ``extract_log_mel`` call site."""

    sample_rate: int = 16_000
    n_mels: int = 40
    n_fft: int = 512
    frame_ms: float = 25.0
    hop_ms: float = 10.0
    f_min: float = 20.0
    f_max: float | None = None
    max_frames: int | None = None

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> SpectrogramFeatureConfig:
        required_keys = frozenset(
            {"sample_rate", "n_mels", "n_fft", "frame_ms", "hop_ms", "f_min", "f_max"}
        )
        _require_exact_keys(values, required_keys, frozenset({"max_frames"}), "feature_config")
        sample_rate = _integer(values["sample_rate"], "feature_config.sample_rate")
        n_mels = _integer(values["n_mels"], "feature_config.n_mels")
        n_fft = _integer(values["n_fft"], "feature_config.n_fft")
        frame_ms = _number(values["frame_ms"], "feature_config.frame_ms")
        hop_ms = _number(values["hop_ms"], "feature_config.hop_ms")
        f_min = _number(values["f_min"], "feature_config.f_min")
        raw_f_max = values["f_max"]
        f_max = None if raw_f_max is None else _number(raw_f_max, "feature_config.f_max")
        max_frames = (
            _integer(values["max_frames"], "feature_config.max_frames")
            if "max_frames" in values
            else None
        )

        expected = cls()
        exact_values = {
            "sample_rate": (sample_rate, expected.sample_rate),
            "n_fft": (n_fft, expected.n_fft),
            "frame_ms": (frame_ms, expected.frame_ms),
            "hop_ms": (hop_ms, expected.hop_ms),
            "f_min": (f_min, expected.f_min),
        }
        for name, (actual, supported) in exact_values.items():
            if not math.isclose(float(actual), float(supported), rel_tol=0.0, abs_tol=1e-9):
                raise ValueError(
                    f"unsupported feature_config.{name}: expected {supported}, received {actual}"
                )
        if not 8 <= n_mels <= 256:
            raise ValueError("feature_config.n_mels must be between 8 and 256")
        if f_max is not None and not math.isclose(
            f_max, expected.sample_rate / 2.0, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError("feature_config.f_max must be null or the 8000 Hz Nyquist limit")
        if max_frames is not None and max_frames <= 0:
            raise ValueError("feature_config.max_frames must be a positive integer")
        return cls(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            frame_ms=frame_ms,
            hop_ms=hop_ms,
            f_min=f_min,
            f_max=f_max,
            max_frames=max_frames,
        )

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        if self.max_frames is None:
            values.pop("max_frames")
        return values


def _validated_model_config(architecture: str, values: Any) -> dict[str, Any]:
    if not isinstance(values, Mapping):
        raise ValueError("model_config must be a mapping")
    if any(not isinstance(key, str) for key in values):
        raise ValueError("model_config keys must be strings")
    allowed = _COMMON_MODEL_CONFIG_KEYS if architecture == "cnn" else _CRNN_MODEL_CONFIG_KEYS
    if unexpected := set(values) - allowed:
        raise ValueError(f"model_config contains unexpected keys: {sorted(unexpected)}")
    validated: dict[str, Any] = {}
    if "channels" in values:
        channels = values["channels"]
        if isinstance(channels, str | bytes) or not isinstance(channels, Sequence):
            raise ValueError("model_config.channels must be a sequence of three integers")
        if len(channels) != 3:
            raise ValueError("model_config.channels must contain exactly three integers")
        checked_channels: list[int] = []
        for index, value in enumerate(channels):
            channel_count = _integer(value, f"model_config.channels[{index}]")
            if not 1 <= channel_count <= 256:
                raise ValueError("model_config channel counts must be between 1 and 256")
            checked_channels.append(channel_count)
        validated["channels"] = checked_channels

    integer_limits = {
        "projection_size": 512,
        "attention_size": 512,
        "recurrent_hidden_size": 512,
        "recurrent_layers": 4,
    }
    for name, upper_bound in integer_limits.items():
        if name not in values:
            continue
        converted = _integer(values[name], f"model_config.{name}")
        if not 1 <= converted <= upper_bound:
            raise ValueError(f"model_config.{name} must be between 1 and {upper_bound}")
        validated[name] = converted

    for name in ("conv_dropout", "classifier_dropout", "recurrent_dropout"):
        if name not in values:
            continue
        dropout_value = _number(values[name], f"model_config.{name}")
        if not 0.0 <= dropout_value < 1.0:
            raise ValueError(f"model_config.{name} must be in [0, 1)")
        validated[name] = dropout_value

    if "recurrent_type" in values:
        recurrent_type = values["recurrent_type"]
        if not isinstance(recurrent_type, str):
            raise ValueError("model_config.recurrent_type must be 'gru' or 'lstm'")
        recurrent_type = recurrent_type.strip().lower()
        if recurrent_type not in {"gru", "lstm"}:
            raise ValueError("model_config.recurrent_type must be 'gru' or 'lstm'")
        validated["recurrent_type"] = recurrent_type
    return validated


def _validated_labels(values: Any) -> tuple[EmotionLabel, ...]:
    if isinstance(values, str | bytes) or not isinstance(values, Sequence):
        raise ValueError("labels must be an ordered sequence")
    expected = tuple(FUSION_EMOTIONS)
    received = tuple(str(value) for value in values)
    canonical = tuple(label.value for label in expected)
    if received != canonical:
        raise ValueError(
            f"checkpoint labels must exactly match PHANTOM's canonical order: {list(canonical)}"
        )
    return expected


def _validated_normalization(
    values: Any, n_mels: int
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if not isinstance(values, Mapping):
        raise ValueError("normalization must be a mapping")
    _require_exact_keys(values, frozenset({"mean", "std"}), frozenset(), "normalization")
    mean_values = values["mean"]
    std_values = values["std"]
    if (
        isinstance(mean_values, str | bytes)
        or not isinstance(mean_values, Sequence)
        or isinstance(std_values, str | bytes)
        or not isinstance(std_values, Sequence)
    ):
        raise ValueError("normalization mean and std must be sequences")
    if len(mean_values) != n_mels or len(std_values) != n_mels:
        raise ValueError(f"normalization mean and std must each contain {n_mels} values")
    mean = tuple(
        _number(value, f"normalization.mean[{index}]") for index, value in enumerate(mean_values)
    )
    std = tuple(
        _number(value, f"normalization.std[{index}]") for index, value in enumerate(std_values)
    )
    if any(abs(value) > MAX_NORMALIZATION_MAGNITUDE for value in mean):
        raise ValueError("normalization means exceed the supported numeric range")
    if any(not MIN_NORMALIZATION_STD <= value <= MAX_NORMALIZATION_MAGNITUDE for value in std):
        raise ValueError(
            "normalization standard deviations must be finite and between "
            f"{MIN_NORMALIZATION_STD} and {MAX_NORMALIZATION_MAGNITUDE}"
        )
    return mean, std


def _load_checkpoint(path: Path) -> tuple[Mapping[str, Any], Any]:
    torch = _require_torch()
    try:
        loaded = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:  # pragma: no cover - guarded by the optional dependency floor
        raise RuntimeError(
            "Project PHANTOM requires PyTorch 2.4+ for safe checkpoint loading"
        ) from exc
    except Exception as exc:
        raise ValueError(f"could not safely load spectrogram checkpoint: {path}") from exc
    if not isinstance(loaded, Mapping):
        raise ValueError("checkpoint root must be a mapping, not a raw model object")
    if any(not isinstance(key, str) for key in loaded):
        raise ValueError("checkpoint keys must be strings")
    return loaded, torch


class SpectrogramEmotionAdapter:
    """Load a validated local CNN/CRNN checkpoint and predict from ``[T, M]`` features."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str = "cpu",
        expected_architecture: str | None = None,
        max_frames: int = 3_000,
    ) -> None:
        selected_path = Path(checkpoint).expanduser()
        if not selected_path.is_file():
            raise FileNotFoundError(f"spectrogram checkpoint does not exist: {selected_path}")
        if selected_path.suffix.lower() not in {".pt", ".pth"}:
            raise ValueError("spectrogram checkpoints must use a .pt or .pth extension")
        if selected_path.stat().st_size > MAX_CHECKPOINT_BYTES:
            raise ValueError(
                f"spectrogram checkpoint exceeds the {MAX_CHECKPOINT_BYTES}-byte safety limit"
            )
        if isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames <= 0:
            raise ValueError("max_frames must be a positive integer")

        checkpoint_data, torch = _load_checkpoint(selected_path)
        _require_exact_keys(
            checkpoint_data,
            _CHECKPOINT_REQUIRED_KEYS,
            _CHECKPOINT_OPTIONAL_KEYS,
            "checkpoint",
        )
        version = _integer(checkpoint_data["format_version"], "format_version")
        if version != CHECKPOINT_FORMAT_VERSION:
            raise ValueError(
                f"unsupported checkpoint format {version}; expected {CHECKPOINT_FORMAT_VERSION}"
            )
        raw_architecture = checkpoint_data["architecture"]
        if not isinstance(raw_architecture, str):
            raise ValueError("architecture must be a string")
        architecture = raw_architecture.strip().lower()
        if architecture not in SUPPORTED_ARCHITECTURES:
            raise ValueError(f"unsupported checkpoint architecture: {raw_architecture!r}")
        if expected_architecture is not None:
            expected = expected_architecture.strip().lower()
            if expected not in SUPPORTED_ARCHITECTURES:
                raise ValueError(f"unsupported expected architecture: {expected_architecture!r}")
            if architecture != expected:
                raise ValueError(
                    f"checkpoint architecture is {architecture!r}, expected {expected!r}"
                )

        raw_feature_config = checkpoint_data["feature_config"]
        if not isinstance(raw_feature_config, Mapping):
            raise ValueError("feature_config must be a mapping")
        feature_settings = SpectrogramFeatureConfig.from_mapping(raw_feature_config)
        labels = _validated_labels(checkpoint_data["labels"])
        model_config = _validated_model_config(architecture, checkpoint_data["model_config"])
        mean, std = _validated_normalization(
            checkpoint_data["normalization"], feature_settings.n_mels
        )
        temperature = _number(checkpoint_data["temperature"], "temperature")
        if not 0.05 <= temperature <= 20.0:
            raise ValueError("temperature must be between 0.05 and 20")
        if "training" in checkpoint_data and not isinstance(checkpoint_data["training"], Mapping):
            raise ValueError("optional training metadata must be a mapping")

        state_dict = checkpoint_data["state_dict"]
        if not isinstance(state_dict, Mapping) or not state_dict:
            raise ValueError("state_dict must be a non-empty mapping")
        if any(not isinstance(key, str) for key in state_dict):
            raise ValueError("state_dict keys must be strings")
        total_state_bytes = 0
        for name, value in state_dict.items():
            if not torch.is_tensor(value):
                raise ValueError("state_dict values must all be tensors")
            if not bool(value.is_floating_point()) or value.numel() <= 0:
                raise ValueError(f"state_dict tensor {name!r} must be a non-empty floating tensor")
            total_state_bytes += int(value.numel()) * int(value.element_size())
            if total_state_bytes > MAX_STATE_DICT_BYTES:
                raise ValueError(f"state_dict exceeds the {MAX_STATE_DICT_BYTES}-byte tensor limit")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"state_dict tensor {name!r} contains non-finite values")

        resolved_device = resolve_torch_device(device)
        torch_models = import_module("phantom.audio.torch_models")
        try:
            model = torch_models.build_spectrogram_model(
                architecture,
                n_mels=feature_settings.n_mels,
                num_classes=len(labels),
                **model_config,
            )
            expected_state = model.state_dict()
            if set(state_dict) != set(expected_state):
                raise ValueError("state_dict keys do not match the declared architecture")
            for name, value in state_dict.items():
                if value.dtype != expected_state[name].dtype:
                    raise ValueError(
                        f"state_dict tensor {name!r} has dtype {value.dtype}; "
                        f"expected {expected_state[name].dtype}"
                    )
            model.load_state_dict(dict(state_dict), strict=True)
            if any(
                tensor.is_floating_point() and not bool(torch.isfinite(tensor).all())
                for tensor in model.state_dict().values()
            ):
                raise ValueError("loaded model state contains non-finite values")
        except (RuntimeError, TypeError, ValueError) as exc:
            raise ValueError(
                f"checkpoint state does not match its declared model configuration: {exc}"
            ) from exc
        model.to(resolved_device)
        model.eval()

        self.checkpoint_path = selected_path.resolve()
        self.architecture = architecture
        self.n_mels = feature_settings.n_mels
        self.temperature = temperature
        self.feature_name = f"log-mel-{architecture}"
        self.feature_settings = feature_settings
        self.feature_config = feature_settings.to_dict()
        self.model_config = dict(model.model_config)
        self.labels = labels
        self.normalization_mean = mean
        self.normalization_std = std
        self.device = resolved_device
        self.trained_max_frames = feature_settings.max_frames
        self.max_frames = (
            max_frames
            if self.trained_max_frames is None
            else min(max_frames, self.trained_max_frames)
        )
        self.model = model
        self._torch = torch

    def predict(self, features: Any) -> Mapping[EmotionLabel, float]:
        """Return uncalibrated probabilities; ``AudioAnalyzer`` applies temperature once."""

        try:
            np = import_module("numpy")
        except ModuleNotFoundError as exc:  # pragma: no cover - core packaging guard
            raise RuntimeError("NumPy is required for spectrogram neural inference") from exc
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError("spectrogram features must have shape [frames, mel_bins]")
        frame_count, mel_bins = values.shape
        if frame_count <= 0:
            raise ValueError("spectrogram features cannot be empty")
        if frame_count > self.max_frames:
            raise ValueError(
                f"spectrogram contains {frame_count} frames; maximum is {self.max_frames}"
            )
        if mel_bins != self.n_mels:
            raise ValueError(f"expected {self.n_mels} mel bins, received {mel_bins}")
        if not bool(np.isfinite(values).all()):
            raise ValueError("spectrogram features must contain only finite values")

        torch = self._torch
        contiguous = np.ascontiguousarray(values.T)
        tensor = torch.from_numpy(contiguous).unsqueeze(0).unsqueeze(0).to(self.device)
        mean = torch.tensor(self.normalization_mean, dtype=tensor.dtype, device=self.device).view(
            1, 1, self.n_mels, 1
        )
        std = torch.tensor(self.normalization_std, dtype=tensor.dtype, device=self.device).view(
            1, 1, self.n_mels, 1
        )
        normalized = (tensor - mean) / std
        lengths = torch.tensor([frame_count], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            logits = self.model(normalized, lengths)
            if logits.shape != (1, len(self.labels)) or not bool(torch.isfinite(logits).all()):
                raise RuntimeError("spectrogram model produced invalid logits")
            probabilities = torch.softmax(logits[0], dim=-1).detach().cpu().tolist()
        return {label: float(probabilities[index]) for index, label in enumerate(self.labels)}


def load_spectrogram_model(
    checkpoint: str | Path,
    *,
    device: str = "cpu",
    expected_architecture: str | None = None,
    max_frames: int = 3_000,
) -> SpectrogramEmotionAdapter:
    """Explicit factory used by application configuration; never creates random weights."""

    return SpectrogramEmotionAdapter(
        checkpoint,
        device=device,
        expected_architecture=expected_architecture,
        max_frames=max_frames,
    )


__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "MAX_CHECKPOINT_BYTES",
    "SUPPORTED_ARCHITECTURES",
    "SpectrogramEmotionAdapter",
    "SpectrogramFeatureConfig",
    "load_spectrogram_model",
    "resolve_torch_device",
]
