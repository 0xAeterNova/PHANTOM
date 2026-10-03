"""Dataset and optimization helpers for optional spectrogram neural models.

This module is deliberately not imported by :mod:`phantom.audio`; importing it
requires the optional ``ml`` dependency set.
"""

from __future__ import annotations

import csv
import hashlib
import math
import os
import random
from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from phantom.audio.features import extract_log_mel
from phantom.audio.preprocessing import load_wav, normalize_audio, resample_audio
from phantom.schemas import FUSION_EMOTIONS

np: Any = import_module("numpy")


@dataclass(frozen=True, slots=True)
class AudioManifestEntry:
    """One licensed, locally available recording and its grouping metadata."""

    sample_id: str
    path: Path
    manifest_path: str
    label: str
    speaker_id: str


@dataclass(frozen=True, slots=True)
class LogMelConfig:
    """Feature contract stored beside every compatible neural checkpoint."""

    sample_rate: int = 16_000
    n_mels: int = 40
    n_fft: int = 512
    frame_ms: float = 25.0
    hop_ms: float = 10.0
    f_min: float = 20.0
    f_max: float | None = None

    def as_checkpoint_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    train: tuple[AudioManifestEntry, ...]
    validation: tuple[AudioManifestEntry, ...]
    test: tuple[AudioManifestEntry, ...]


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    loss: float
    accuracy: float
    macro_f1: float
    logits: torch.Tensor
    targets: torch.Tensor
    sample_ids: tuple[str, ...]
    speaker_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TrainingResult:
    best_epoch: int
    epochs_completed: int
    best_validation_loss: float
    best_validation_macro_f1: float
    state_dict: dict[str, Any]
    history: tuple[dict[str, float | int], ...]


def read_audio_manifest(path: Path) -> tuple[AudioManifestEntry, ...]:
    """Read and strictly validate ``path,label,speaker_id`` CSV rows."""

    manifest = path.resolve()
    if not manifest.is_file():
        raise ValueError(f"manifest does not exist: {manifest}")
    entries: list[AudioManifestEntry] = []
    seen_ids: set[str] = set()
    seen_paths: set[Path] = set()
    allowed_labels = {label.value for label in FUSION_EMOTIONS}
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        required = {"path", "label", "speaker_id"}
        if missing := required - columns:
            raise ValueError(f"manifest is missing required columns: {sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            raw_path = str(row.get("path") or "").strip()
            label = str(row.get("label") or "").strip().lower()
            speaker_id = str(row.get("speaker_id") or "").strip()
            sample_id = str(row.get("sample_id") or f"sample-{row_number - 1}").strip()
            if not raw_path or not speaker_id or not sample_id:
                raise ValueError(f"manifest row {row_number} contains an empty required value")
            if label not in allowed_labels:
                raise ValueError(
                    f"manifest row {row_number} has unsupported label {label!r}; "
                    f"expected one of {sorted(allowed_labels)}"
                )
            audio_path = Path(raw_path)
            if not audio_path.is_absolute():
                audio_path = manifest.parent / audio_path
            audio_path = audio_path.resolve()
            if not audio_path.is_file():
                raise ValueError(f"audio for sample {sample_id!r} does not exist")
            if audio_path.suffix.lower() not in {".wav", ".wave"}:
                raise ValueError(f"audio for sample {sample_id!r} must be PCM WAV")
            if sample_id in seen_ids:
                raise ValueError(f"duplicate sample_id in manifest: {sample_id!r}")
            if audio_path in seen_paths:
                raise ValueError(f"duplicate audio path in manifest for sample {sample_id!r}")
            seen_ids.add(sample_id)
            seen_paths.add(audio_path)
            entries.append(AudioManifestEntry(sample_id, audio_path, raw_path, label, speaker_id))
    if not entries:
        raise ValueError("manifest contains no audio rows")
    observed = {entry.label for entry in entries}
    missing_labels = allowed_labels - observed
    if missing_labels:
        raise ValueError(
            "the neural checkpoint uses PHANTOM's complete canonical label space; "
            f"manifest is missing labels: {sorted(missing_labels)}"
        )
    if len({entry.speaker_id for entry in entries}) < 4:
        raise ValueError(
            "at least four independent speakers are required for train/val/test splits"
        )
    return tuple(entries)


def speaker_independent_split(entries: tuple[AudioManifestEntry, ...], seed: int) -> DatasetSplit:
    """Choose a deterministic 70/15/15 split with no speaker leakage."""

    speakers = sorted({entry.speaker_id for entry in entries})
    if len(speakers) < 4:
        raise ValueError("at least four independent speakers are required")
    train_speaker_count = min(len(speakers) - 2, max(1, round(len(speakers) * 0.70)))
    remaining_count = len(speakers) - train_speaker_count
    validation_speaker_count = min(
        remaining_count - 1,
        max(1, round(len(speakers) * 0.15)),
    )
    required_labels = {label.value for label in FUSION_EMOTIONS}
    candidates: list[tuple[tuple[int, int, float], DatasetSplit]] = []
    for attempt in range(256):
        salt = f"{seed + attempt}:"
        shuffled = sorted(
            speakers,
            key=lambda speaker: hashlib.sha256(f"{salt}{speaker}".encode()).digest(),
        )
        train_speakers = set(shuffled[:train_speaker_count])
        validation_end = train_speaker_count + validation_speaker_count
        validation_speakers = set(shuffled[train_speaker_count:validation_end])
        test_speakers = set(shuffled[validation_end:])
        train_entries = tuple(entry for entry in entries if entry.speaker_id in train_speakers)
        if {entry.label for entry in train_entries} != required_labels:
            continue
        split = DatasetSplit(
            train_entries,
            tuple(entry for entry in entries if entry.speaker_id in validation_speakers),
            tuple(entry for entry in entries if entry.speaker_id in test_speakers),
        )
        speaker_sets = [
            {entry.speaker_id for entry in part}
            for part in (split.train, split.validation, split.test)
        ]
        if (
            speaker_sets[0] & speaker_sets[1]
            or speaker_sets[0] & speaker_sets[2]
            or speaker_sets[1] & speaker_sets[2]
        ):
            raise RuntimeError("speaker-independent split invariant failed")
        validation_coverage = len({entry.label for entry in split.validation})
        test_coverage = len({entry.label for entry in split.test})
        fraction_error = abs(len(split.train) / len(entries) - 0.70)
        candidates.append(((validation_coverage, test_coverage, -fraction_error), split))
    if not candidates:
        raise ValueError(
            "could not create a speaker-independent split containing every canonical label in "
            "training; add speakers per class or provide a curated split manifest"
        )
    return max(candidates, key=lambda item: item[0])[1]


def write_split_manifest(path: Path, split: DatasetSplit) -> None:
    """Persist the immutable local split assignment without copying any audio."""

    path.parent.mkdir(parents=True, exist_ok=True)
    output_directory = path.parent.resolve()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["sample_id", "path", "label", "speaker_id", "split"]
        )
        writer.writeheader()
        for split_name, entries in (
            ("train", split.train),
            ("validation", split.validation),
            ("test", split.test),
        ):
            for entry in entries:
                try:
                    recorded_path = Path(os.path.relpath(entry.path, output_directory)).as_posix()
                except ValueError:  # Windows paths can reside on different drives.
                    recorded_path = str(entry.path)
                writer.writerow(
                    {
                        "sample_id": entry.sample_id,
                        "path": recorded_path,
                        "label": entry.label,
                        "speaker_id": entry.speaker_id,
                        "split": split_name,
                    }
                )


def configure_determinism(seed: int) -> torch.Generator:
    """Seed CPU/GPU execution and return the DataLoader shuffle generator."""

    if seed < 0:
        raise ValueError("seed must be non-negative")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    return torch.device(requested)


def _spectrogram(
    entry: AudioManifestEntry,
    feature_config: LogMelConfig,
    max_frames: int,
    *,
    random_crop: bool,
    crop_seed: int | None = None,
) -> Any:
    try:
        decoded = load_wav(entry.path)
        samples = resample_audio(decoded.samples, decoded.sample_rate, feature_config.sample_rate)
        samples = normalize_audio(samples)
        values = extract_log_mel(
            samples,
            feature_config.sample_rate,
            n_mels=feature_config.n_mels,
            n_fft=feature_config.n_fft,
            frame_ms=feature_config.frame_ms,
            hop_ms=feature_config.hop_ms,
            f_min=feature_config.f_min,
            f_max=feature_config.f_max,
        )
    except Exception as exc:
        raise RuntimeError(f"failed to preprocess audio sample {entry.sample_id!r}") from exc
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] != feature_config.n_mels:
        raise RuntimeError(f"invalid log-Mel shape for audio sample {entry.sample_id!r}")
    if not np.isfinite(array).all():
        raise RuntimeError(f"non-finite log-Mel values for audio sample {entry.sample_id!r}")
    if array.shape[0] > max_frames:
        available = array.shape[0] - max_frames
        if random_crop:
            if crop_seed is None:
                raise ValueError("a deterministic crop seed is required for training crops")
            start = crop_seed % (available + 1)
        else:
            start = available // 2
        array = array[start : start + max_frames]
    return np.ascontiguousarray(array)


def estimate_normalization(
    entries: tuple[AudioManifestEntry, ...],
    feature_config: LogMelConfig,
    max_frames: int,
) -> tuple[Any, Any]:
    """Compute per-Mel statistics from training speakers only."""

    sums = np.zeros(feature_config.n_mels, dtype=np.float64)
    squared_sums = np.zeros(feature_config.n_mels, dtype=np.float64)
    frame_count = 0
    for entry in entries:
        values = _spectrogram(entry, feature_config, max_frames, random_crop=False)
        sums += values.sum(axis=0, dtype=np.float64)
        squared_sums += np.square(values, dtype=np.float64).sum(axis=0)
        frame_count += int(values.shape[0])
    if frame_count < 1:
        raise ValueError("training split contains no spectrogram frames")
    mean = sums / frame_count
    variance = np.maximum(squared_sums / frame_count - np.square(mean), 1e-6)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


class SpectrogramDataset:
    """Lazy WAV-to-log-Mel dataset with bounded variable-length examples."""

    def __init__(
        self,
        entries: tuple[AudioManifestEntry, ...],
        feature_config: LogMelConfig,
        max_frames: int,
        mean: Any,
        std: Any,
        *,
        training: bool,
        seed: int,
    ) -> None:
        if seed < 0:
            raise ValueError("dataset seed must be non-negative")
        self.entries = entries
        self.feature_config = feature_config
        self.max_frames = max_frames
        self.mean = mean
        self.std = std
        self.training = training
        self.seed = seed
        self.label_to_index = {label.value: index for index, label in enumerate(FUSION_EMOTIONS)}

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int, str, str]:
        entry = self.entries[index]
        crop_seed = int.from_bytes(
            hashlib.sha256(f"{self.seed}:{entry.sample_id}".encode()).digest()[:8],
            byteorder="big",
        )
        values = _spectrogram(
            entry,
            self.feature_config,
            self.max_frames,
            random_crop=self.training,
            crop_seed=crop_seed,
        )
        normalized = np.ascontiguousarray((values - self.mean) / self.std, dtype=np.float32)
        return (
            torch.from_numpy(normalized),
            int(normalized.shape[0]),
            self.label_to_index[entry.label],
            entry.sample_id,
            entry.speaker_id,
        )


def collate_spectrograms(
    batch: list[tuple[torch.Tensor, int, int, str, str]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[str, ...], tuple[str, ...]]:
    """Pad log-Mel sequences into canonical ``[B,1,M,T]`` tensors."""

    if not batch:
        raise ValueError("cannot collate an empty batch")
    maximum = max(item[1] for item in batch)
    mel_bins = int(batch[0][0].shape[1])
    inputs = torch.zeros((len(batch), 1, mel_bins, maximum), dtype=torch.float32)
    lengths = torch.empty(len(batch), dtype=torch.long)
    targets = torch.empty(len(batch), dtype=torch.long)
    sample_ids: list[str] = []
    speaker_ids: list[str] = []
    for index, (features, length, target, sample_id, speaker_id) in enumerate(batch):
        inputs[index, 0, :, :length] = features.transpose(0, 1)
        lengths[index] = length
        targets[index] = target
        sample_ids.append(sample_id)
        speaker_ids.append(speaker_id)
    return inputs, lengths, targets, tuple(sample_ids), tuple(speaker_ids)


def make_data_loader(
    dataset: SpectrogramDataset,
    *,
    batch_size: int,
    shuffle: bool,
    workers: int,
    generator: torch.Generator,
    pin_memory: bool,
) -> DataLoader[Any]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate_spectrograms,
        generator=generator,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
    )


def balanced_class_weights(entries: tuple[AudioManifestEntry, ...]) -> torch.Tensor:
    counts = np.zeros(len(FUSION_EMOTIONS), dtype=np.int64)
    index = {label.value: offset for offset, label in enumerate(FUSION_EMOTIONS)}
    for entry in entries:
        counts[index[entry.label]] += 1
    if np.any(counts == 0):
        missing = [
            FUSION_EMOTIONS[offset].value
            for offset, count in enumerate(counts.tolist())
            if count == 0
        ]
        raise ValueError(f"training split is missing canonical labels: {missing}")
    weights = len(entries) / (len(FUSION_EMOTIONS) * counts.astype(np.float64))
    return torch.tensor(weights, dtype=torch.float32)


def _macro_f1(targets: torch.Tensor, predicted: torch.Tensor) -> float:
    values: list[float] = []
    for class_index in range(len(FUSION_EMOTIONS)):
        truth = targets == class_index
        chosen = predicted == class_index
        true_positive = int((truth & chosen).sum().item())
        false_positive = int((~truth & chosen).sum().item())
        false_negative = int((truth & ~chosen).sum().item())
        denominator = 2 * true_positive + false_positive + false_negative
        values.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return float(sum(values) / len(values))


def evaluate_model(
    model: Any,
    loader: DataLoader[Any],
    device: torch.device,
    criterion: Any,
) -> EvaluationResult:
    model.eval()
    loss_total = 0.0
    example_count = 0
    logits_batches: list[torch.Tensor] = []
    target_batches: list[torch.Tensor] = []
    sample_ids: list[str] = []
    speaker_ids: list[str] = []
    with torch.inference_mode():
        for inputs, lengths, targets, batch_sample_ids, batch_speaker_ids in loader:
            inputs = inputs.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(inputs, lengths)
            loss = criterion(logits, targets)
            batch_size = int(targets.shape[0])
            loss_total += float(loss.item()) * batch_size
            example_count += batch_size
            logits_batches.append(logits.detach().cpu())
            target_batches.append(targets.detach().cpu())
            sample_ids.extend(batch_sample_ids)
            speaker_ids.extend(batch_speaker_ids)
    if example_count == 0:
        raise ValueError("evaluation split is empty")
    all_logits = torch.cat(logits_batches)
    all_targets = torch.cat(target_batches)
    predicted = all_logits.argmax(dim=1)
    accuracy = float((predicted == all_targets).float().mean().item())
    return EvaluationResult(
        loss=loss_total / example_count,
        accuracy=accuracy,
        macro_f1=_macro_f1(all_targets, predicted),
        logits=all_logits,
        targets=all_targets,
        sample_ids=tuple(sample_ids),
        speaker_ids=tuple(speaker_ids),
    )


def train_spectrogram_model(
    model: Any,
    train_loader: DataLoader[Any],
    validation_loader: DataLoader[Any],
    *,
    device: torch.device,
    class_weights: torch.Tensor,
    epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip: float,
    min_delta: float,
) -> TrainingResult:
    """Train with weighted loss and validation-loss early stopping."""

    model.to(device)
    training_criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    validation_criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(1, patience // 3)
    )
    best_loss = math.inf
    best_f1 = 0.0
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        training_loss = 0.0
        training_examples = 0
        for inputs, lengths, targets, _sample_ids, _speaker_ids in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs, lengths)
            loss = training_criterion(logits, targets)
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("training loss became non-finite")
            loss.backward()
            clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            batch_size = int(targets.shape[0])
            training_loss += float(loss.item()) * batch_size
            training_examples += batch_size
        validation = evaluate_model(model, validation_loader, device, validation_criterion)
        average_training_loss = training_loss / max(1, training_examples)
        learning_rate_now = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": average_training_loss,
                "validation_loss": validation.loss,
                "validation_accuracy": validation.accuracy,
                "validation_macro_f1": validation.macro_f1,
                "learning_rate": learning_rate_now,
            }
        )
        print(
            f"epoch={epoch:03d} train_loss={average_training_loss:.4f} "
            f"val_loss={validation.loss:.4f} val_macro_f1={validation.macro_f1:.4f}"
        )
        scheduler.step(validation.loss)
        if validation.loss < best_loss - min_delta:
            best_loss = validation.loss
            best_f1 = validation.macro_f1
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break
    if best_state is None:
        raise RuntimeError("training completed without a finite validation checkpoint")
    model.load_state_dict(best_state, strict=True)
    return TrainingResult(
        best_epoch=best_epoch,
        epochs_completed=len(history),
        best_validation_loss=best_loss,
        best_validation_macro_f1=best_f1,
        state_dict=best_state,
        history=tuple(history),
    )


def fit_temperature(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Select a deterministic validation-set temperature by grid search."""

    if logits.ndim != 2 or targets.ndim != 1 or logits.shape[0] != targets.shape[0]:
        raise ValueError("invalid validation logits or targets for calibration")
    criterion = torch.nn.CrossEntropyLoss()
    temperatures = torch.logspace(math.log10(0.5), math.log10(5.0), 101)
    losses = torch.stack([criterion(logits / value, targets) for value in temperatures])
    return float(temperatures[int(losses.argmin().item())].item())


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    """Write a tensor/primitive-only checkpoint and atomically replace its target."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
