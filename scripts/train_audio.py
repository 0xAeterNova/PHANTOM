"""Train a speaker-independent CNN or CRNN from a local PCM-WAV manifest.

The input CSV must contain ``path,label,speaker_id`` columns and may contain a
stable ``sample_id`` column.  Labels use PHANTOM's seven canonical affect
labels.  No dataset or pretrained weight is downloaded by this script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from phantom.compat import UTC


def _positive_int(value: str) -> int:
    converted = int(value)
    if converted < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return converted


def _non_negative_int(value: str) -> int:
    converted = int(value)
    if converted < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return converted


def _positive_float(value: str) -> float:
    converted = float(value)
    if converted <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return converted


def _non_negative_float(value: str) -> float:
    converted = float(value)
    if converted < 0.0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return converted


def _probability(value: str) -> float:
    converted = float(value)
    if not 0.0 <= converted <= 1.0:
        raise argparse.ArgumentTypeError("value must be between 0 and 1")
    return converted


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_metadata(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    complete = {
        "created_at": datetime.now(UTC).isoformat(),
        "claim_status": "local training run; metrics require separate held-out evaluation",
        **payload,
    }
    path.write_text(json.dumps(complete, indent=2, sort_keys=True), encoding="utf-8")


def _write_predictions(
    path: Path,
    evaluation: Any,
    *,
    temperature: float,
    labels: list[str],
    abstention_threshold: float,
) -> None:
    import torch

    probabilities = torch.softmax(evaluation.logits / temperature, dim=1)
    confidence, predicted = probabilities.max(dim=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id",
        "speaker_id",
        "true_label",
        "raw_predicted_label",
        "predicted_label",
        "confidence",
        "abstained",
        "abstention_threshold",
        *[f"probability_{label}" for label in labels],
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, sample_id in enumerate(evaluation.sample_ids):
            confidence_value = float(confidence[index].item())
            raw_label = labels[int(predicted[index].item())]
            abstained = confidence_value < abstention_threshold
            row: dict[str, str | float | bool] = {
                "sample_id": sample_id,
                "speaker_id": evaluation.speaker_ids[index],
                "true_label": labels[int(evaluation.targets[index].item())],
                "raw_predicted_label": raw_label,
                "predicted_label": "uncertain" if abstained else raw_label,
                "confidence": confidence_value,
                "abstained": abstained,
                "abstention_threshold": abstention_threshold,
            }
            row.update(
                {
                    f"probability_{label}": float(probabilities[index, label_index].item())
                    for label_index, label in enumerate(labels)
                }
            )
            writer.writerow(row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest_csv",
        type=Path,
        help="CSV with path,label,speaker_id and optional sample_id columns",
    )
    parser.add_argument(
        "--architecture",
        choices=("cnn", "crnn"),
        default="crnn",
        help="CRNN combines the CNN encoder with a bidirectional GRU/LSTM",
    )
    parser.add_argument(
        "--recurrent-type",
        choices=("gru", "lstm"),
        default="gru",
        help="recurrent unit used by CRNN; ignored by the CNN-only ablation",
    )
    parser.add_argument("--output", type=Path, default=Path("models/audio_spectrogram_crnn.pt"))
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument("--predictions-output", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=_positive_int, default=60)
    parser.add_argument("--patience", type=_positive_int, default=10)
    parser.add_argument("--batch-size", type=_positive_int, default=16)
    parser.add_argument("--workers", type=_non_negative_int, default=0)
    parser.add_argument("--learning-rate", type=_positive_float, default=1e-3)
    parser.add_argument("--weight-decay", type=_non_negative_float, default=1e-4)
    parser.add_argument("--gradient-clip", type=_positive_float, default=5.0)
    parser.add_argument("--min-delta", type=_non_negative_float, default=1e-4)
    parser.add_argument(
        "--abstention-threshold",
        type=_probability,
        default=0.45,
        help=(
            "evaluation-only calibrated probability threshold; runtime also applies "
            "signal-quality and temporal checks"
        ),
    )
    parser.add_argument("--max-seconds", type=_positive_float, default=8.0)
    parser.add_argument("--n-mels", type=_positive_int, default=40)
    parser.add_argument("--seed", type=_non_negative_int, default=42)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.n_mels < 8:
        raise SystemExit("--n-mels must be at least 8 for the spectrogram CNN encoder")
    if args.output.suffix.lower() not in {".pt", ".pth"}:
        raise SystemExit("--output must use a .pt or .pth checkpoint suffix")

    try:
        import torch

        from phantom.audio.torch_models import build_spectrogram_model
        from phantom.audio.training import (
            LogMelConfig,
            SpectrogramDataset,
            atomic_torch_save,
            balanced_class_weights,
            configure_determinism,
            estimate_normalization,
            evaluate_model,
            fit_temperature,
            make_data_loader,
            read_audio_manifest,
            resolve_device,
            speaker_independent_split,
            train_spectrogram_model,
            write_split_manifest,
        )
        from phantom.schemas import FUSION_EMOTIONS
    except (ImportError, RuntimeError) as exc:  # pragma: no cover - optional dependency guard
        raise SystemExit(
            "Neural audio training requires the optional ML dependencies. "
            "Install with: pip install -e '.[audio-ml]'"
        ) from exc

    generator = configure_determinism(args.seed)
    device = resolve_device(args.device)
    entries = read_audio_manifest(args.manifest_csv)
    split = speaker_independent_split(entries, args.seed)
    split_manifest = args.split_manifest or args.output.with_suffix(".splits.csv")
    predictions_output = args.predictions_output or args.output.with_suffix(".test_predictions.csv")
    write_split_manifest(split_manifest, split)

    feature_config = LogMelConfig(n_mels=args.n_mels)
    maximum_frames = max(1, round(args.max_seconds * 1000.0 / feature_config.hop_ms))
    print(f"Computing train-only log-Mel normalization from {len(split.train)} recordings...")
    mean, std = estimate_normalization(split.train, feature_config, maximum_frames)

    train_dataset = SpectrogramDataset(
        split.train,
        feature_config,
        maximum_frames,
        mean,
        std,
        training=True,
        seed=args.seed,
    )
    validation_dataset = SpectrogramDataset(
        split.validation,
        feature_config,
        maximum_frames,
        mean,
        std,
        training=False,
        seed=args.seed,
    )
    test_dataset = SpectrogramDataset(
        split.test,
        feature_config,
        maximum_frames,
        mean,
        std,
        training=False,
        seed=args.seed,
    )
    loader_options = {
        "batch_size": args.batch_size,
        "workers": args.workers,
        "generator": generator,
        "pin_memory": device.type == "cuda",
    }
    train_loader = make_data_loader(train_dataset, shuffle=True, **loader_options)
    validation_loader = make_data_loader(validation_dataset, shuffle=False, **loader_options)
    test_loader = make_data_loader(test_dataset, shuffle=False, **loader_options)

    requested_model_config = (
        {"recurrent_type": args.recurrent_type} if args.architecture == "crnn" else {}
    )
    labels = [label.value for label in FUSION_EMOTIONS]
    model = build_spectrogram_model(
        args.architecture,
        n_mels=feature_config.n_mels,
        num_classes=len(labels),
        **requested_model_config,
    )
    model_config = dict(getattr(model, "model_config", requested_model_config))
    class_weights = balanced_class_weights(split.train)
    print(
        f"Training {args.architecture.upper()} on {device} with "
        f"{sum(parameter.numel() for parameter in model.parameters()):,} parameters..."
    )
    result = train_spectrogram_model(
        model,
        train_loader,
        validation_loader,
        device=device,
        class_weights=class_weights,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        min_delta=args.min_delta,
    )

    unweighted_criterion = torch.nn.CrossEntropyLoss()
    validation = evaluate_model(model, validation_loader, device, unweighted_criterion)
    temperature = fit_temperature(validation.logits, validation.targets)
    test = evaluate_model(model, test_loader, device, unweighted_criterion)
    _write_predictions(
        predictions_output,
        test,
        temperature=temperature,
        labels=labels,
        abstention_threshold=args.abstention_threshold,
    )

    class_counts = Counter(entry.label for entry in split.train)
    row_counts = {
        "train": len(split.train),
        "validation": len(split.validation),
        "test": len(split.test),
    }
    speaker_counts = {
        "train": len({entry.speaker_id for entry in split.train}),
        "validation": len({entry.speaker_id for entry in split.validation}),
        "test": len({entry.speaker_id for entry in split.test}),
    }
    total_rows = sum(row_counts.values())
    total_speakers = sum(speaker_counts.values())
    checkpoint_training = {
        "seed": args.seed,
        "speaker_independent": True,
        "target_split_fractions": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "actual_row_fractions": {name: count / total_rows for name, count in row_counts.items()},
        "actual_speaker_fractions": {
            name: count / total_speakers for name, count in speaker_counts.items()
        },
        "rows": row_counts,
        "speakers": speaker_counts,
        "class_counts": dict(sorted(class_counts.items())),
        "class_weights": class_weights.tolist(),
        "optimizer": "AdamW",
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "gradient_clip": args.gradient_clip,
        "batch_size": args.batch_size,
        "max_seconds": args.max_seconds,
        "training_crop_policy": "deterministic seed-plus-sample-id hash",
        "evaluation_abstention_threshold": args.abstention_threshold,
        "early_stopping": {
            "monitor": "validation_loss",
            "patience": args.patience,
            "min_delta": args.min_delta,
            "best_epoch": result.best_epoch,
            "epochs_completed": result.epochs_completed,
            "best_validation_loss": result.best_validation_loss,
            "best_validation_macro_f1": result.best_validation_macro_f1,
        },
        "history": list(result.history),
    }
    checkpoint: dict[str, Any] = {
        "format_version": 1,
        "architecture": args.architecture,
        "model_config": model_config,
        "feature_config": {
            **feature_config.as_checkpoint_dict(),
            "max_frames": maximum_frames,
        },
        "labels": labels,
        "normalization": {"mean": mean.tolist(), "std": std.tolist()},
        "temperature": temperature,
        "state_dict": result.state_dict,
        "training": checkpoint_training,
    }
    atomic_torch_save(checkpoint, args.output)

    # Prove that the artifact can be opened in weights-only mode and reconstructed.
    loaded = torch.load(args.output, map_location="cpu", weights_only=True)
    verification_model = build_spectrogram_model(
        str(loaded["architecture"]),
        n_mels=int(loaded["feature_config"]["n_mels"]),
        num_classes=len(loaded["labels"]),
        **dict(loaded["model_config"]),
    )
    verification_model.load_state_dict(loaded["state_dict"], strict=True)

    device_metadata: dict[str, str | bool] = {
        "requested": args.device,
        "resolved": str(device),
        "torch_version": str(torch.__version__),
        "cuda_available": torch.cuda.is_available(),
    }
    if device.type == "cuda":
        device_metadata["cuda_device"] = torch.cuda.get_device_name(device)
    _write_metadata(
        args.output.with_suffix(".metadata.json"),
        {
            "task": "spectrogram_acoustic_emotion_experiment",
            "architecture": args.architecture,
            "recurrent_type": (args.recurrent_type if args.architecture == "crnn" else None),
            "seed": args.seed,
            "speaker_independent": True,
            "source_manifest_sha256": _sha256(args.manifest_csv),
            "split_manifest": str(split_manifest),
            "split_manifest_sha256": _sha256(split_manifest),
            "checkpoint": str(args.output),
            "checkpoint_sha256": _sha256(args.output),
            "test_predictions": str(predictions_output),
            "test_predictions_sha256": _sha256(predictions_output),
            "rows": checkpoint_training["rows"],
            "speakers": checkpoint_training["speakers"],
            "best_epoch": result.best_epoch,
            "validation_temperature": temperature,
            "evaluation_abstention_threshold": args.abstention_threshold,
            "evaluation_scope": (
                "calibrated acoustic classifier plus probability-threshold abstention; "
                "runtime signal-quality gating and temporal smoothing are not represented"
            ),
            "device": device_metadata,
            "held_out_metrics_reported": False,
            "notice": (
                "The test prediction file is an input to scripts/evaluate.py; training success "
                "is not evidence of emotional, clinical, fairness, or deployment validity."
            ),
        },
    )
    print(f"Wrote best unvalidated checkpoint to {args.output}")
    print(f"Wrote frozen split assignment to {split_manifest}")
    print(f"Wrote held-out predictions to {predictions_output}; run scripts/evaluate.py next.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
