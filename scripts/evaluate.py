"""Evaluate canonical PHANTOM predictions, calibration, and abstention."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from phantom.schemas import FUSION_EMOTIONS


def expected_calibration_error(
    confidence: list[float], correct: list[bool], bins: int = 10
) -> float:
    if bins < 1 or len(confidence) != len(correct):
        raise ValueError("calibration inputs must have equal lengths and at least one bin")
    total = max(1, len(confidence))
    value = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        members = [
            offset
            for offset, score in enumerate(confidence)
            if score >= lower and (score <= upper if index == bins - 1 else score < upper)
        ]
        if members:
            accuracy = sum(1.0 for offset in members if correct[offset]) / len(members)
            average_confidence = sum(confidence[offset] for offset in members) / len(members)
            value += len(members) / total * abs(accuracy - average_confidence)
    return value


def _read_predictions(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise SystemExit(f"predictions file does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"true_label", "predicted_label", "confidence"}
        if missing := required - set(reader.fieldnames or ()):
            raise SystemExit(f"missing required columns: {sorted(missing)}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise SystemExit("predictions file contains no rows")
    return rows


def _class_metrics(
    rows: list[dict[str, Any]], labels: list[str]
) -> tuple[dict[str, dict[str, float | int]], list[list[int]], float, float]:
    matrix = [[0 for _ in labels] for _ in labels]
    label_index = {label: index for index, label in enumerate(labels)}
    for row in rows:
        matrix[label_index[row["true_label"]]][label_index[row["predicted_label"]]] += 1

    report: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    recall_values: list[float] = []
    for index, label in enumerate(labels):
        true_positive = matrix[index][index]
        support = sum(matrix[index])
        predicted_count = sum(row[index] for row in matrix)
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        report[label] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "support": support,
        }
        f1_values.append(f1)
        recall_values.append(recall)
    return (
        report,
        matrix,
        sum(f1_values) / len(labels),
        sum(recall_values) / len(labels),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions_csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/evaluation.json"))
    args = parser.parse_args()

    rows = _read_predictions(args.predictions_csv)
    labels = [label.value for label in FUSION_EMOTIONS]
    canonical = set(labels)
    true_labels = {str(row.get("true_label") or "").strip().lower() for row in rows}
    if unexpected_truth := true_labels - canonical:
        raise SystemExit(f"unsupported true labels: {sorted(unexpected_truth)}")
    abstention_labels = {"uncertain", "insufficient_quality"}
    predicted_labels = {str(row.get("predicted_label") or "").strip().lower() for row in rows}
    if unexpected_predictions := predicted_labels - canonical - abstention_labels:
        raise SystemExit(f"unsupported predicted labels: {sorted(unexpected_predictions)}")

    normalized_rows: list[dict[str, Any]] = []
    for row_number, row in enumerate(rows, start=2):
        try:
            confidence = float(row["confidence"])
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"invalid confidence on row {row_number}") from exc
        if not 0.0 <= confidence <= 1.0:
            raise SystemExit(f"confidence must be between 0 and 1 on row {row_number}")
        normalized_rows.append(
            {
                **row,
                "true_label": str(row.get("true_label") or "").strip().lower(),
                "predicted_label": str(row.get("predicted_label") or "").strip().lower(),
                "confidence": confidence,
            }
        )

    decided = [row for row in normalized_rows if row["predicted_label"] not in abstention_labels]
    if decided:
        per_class, confusion_values, macro_f1, unweighted_recall = _class_metrics(decided, labels)
        calibration_error = expected_calibration_error(
            [float(row["confidence"]) for row in decided],
            [row["true_label"] == row["predicted_label"] for row in decided],
        )
    else:
        per_class = {}
        confusion_values = [[0 for _ in labels] for _ in labels]
        macro_f1 = None
        unweighted_recall = None
        calibration_error = None

    classifier_only = {"raw_predicted_label", "abstention_threshold"} <= set(rows[0])
    if classifier_only:
        try:
            abstention_thresholds = sorted({float(row["abstention_threshold"]) for row in rows})
        except (TypeError, ValueError) as exc:
            raise SystemExit("invalid abstention_threshold value") from exc
    else:
        abstention_thresholds = None
    coverage = len(decided) / len(normalized_rows)
    report_payload = {
        "sample_count": len(normalized_rows),
        "canonical_labels": labels,
        "absent_true_labels": sorted(canonical - true_labels),
        "coverage": coverage,
        "abstention_rate": 1.0 - coverage,
        "all_abstained": not decided,
        "abstention_thresholds": abstention_thresholds,
        "macro_f1_decided": macro_f1,
        "unweighted_average_recall_decided": unweighted_recall,
        "calibration_error_decided": calibration_error,
        "per_class": per_class,
        "confusion_matrix": {"labels": labels, "values": confusion_values},
        "evaluation_scope": (
            "calibrated classifier outputs with probability-threshold abstention; excludes "
            "runtime signal-quality gating and temporal smoothing"
            if classifier_only
            else "scope is defined by the supplied prediction file"
        ),
        "limitations": (
            "Report modality ablations, quality slices, fairness slices, latency, false "
            "reassurance, and false escalation separately. Zero-support canonical classes "
            "remain in macro metrics with zero recall/F1."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    print(f"Wrote measured evaluation results to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
