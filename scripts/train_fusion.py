"""Train an optional late-feature fusion experiment without demographic attributes."""

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
    parser.add_argument("--output", type=Path, default=Path("models/fusion_experimental.joblib"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    pd, _ = require_ml_stack()
    from joblib import dump
    from sklearn.linear_model import LogisticRegression

    seed_everything(args.seed)
    frame = pd.read_csv(args.features_csv)
    prohibited = {
        column for column in frame.columns if "age" in column.lower() or "gender" in column.lower()
    }
    if prohibited:
        raise SystemExit(f"demographic columns cannot be fusion inputs: {sorted(prohibited)}")
    required = {"label", "subject_id"}
    if missing := required - set(frame.columns):
        raise SystemExit(f"missing required columns: {sorted(missing)}")
    features = feature_columns(frame, required | {"sample_id"})
    train, validation, test = grouped_split(frame, "subject_id", args.seed)
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=args.seed)
    model.fit(train[features], train["label"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dump({"model": model, "features": features, "experimental": True}, args.output)
    write_metadata(
        args.output.with_suffix(".metadata.json"),
        {
            "task": "learned_multimodal_fusion_experiment",
            "seed": args.seed,
            "demographics_excluded": True,
            "rows": {"train": len(train), "validation": len(validation), "test": len(test)},
            "held_out_metrics_reported": False,
        },
    )
    print(f"Wrote experimental fusion checkpoint to {args.output}; it is not enabled by default.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
