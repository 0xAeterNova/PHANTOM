"""Train an optional subject-independent classical expression baseline from extracted features."""

from __future__ import annotations

import argparse
from pathlib import Path

from _training_common import (
    feature_columns,
    grouped_split,
    require_ml_stack,
    seed_everything,
    write_metadata,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features_csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path("models/vision_baseline.joblib"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    pd, _ = require_ml_stack()
    from joblib import dump
    from sklearn.linear_model import SGDClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    seed_everything(args.seed)
    frame = pd.read_csv(args.features_csv)
    required = {"label", "subject_id"}
    if missing := required - set(frame.columns):
        raise SystemExit(f"missing required columns: {sorted(missing)}")
    features = feature_columns(frame, required | {"sample_id"})
    train, validation, test = grouped_split(frame, "subject_id", args.seed)
    model = make_pipeline(
        StandardScaler(),
        SGDClassifier(
            loss="log_loss",
            class_weight="balanced",
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=8,
            max_iter=1000,
            random_state=args.seed,
        ),
    )
    model.fit(train[features], train["label"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dump(
        {"model": model, "features": features, "labels": sorted(frame["label"].unique())},
        args.output,
    )
    write_metadata(
        args.output.with_suffix(".metadata.json"),
        {
            "task": "facial_expression_baseline",
            "seed": args.seed,
            "subject_independent": True,
            "rows": {"train": len(train), "validation": len(validation), "test": len(test)},
            "checkpoint": str(args.output),
            "held_out_metrics_reported": False,
        },
    )
    print(f"Wrote unvalidated baseline checkpoint to {args.output}; run scripts/evaluate.py next.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
