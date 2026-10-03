from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

from phantom.schemas import FUSION_EMOTIONS
from scripts import evaluate


def _write_predictions(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _evaluate(
    predictions: Path, output: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, object]:
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate.py", str(predictions), "--output", str(output)],
    )
    assert evaluate.main() == 0
    return json.loads(output.read_text(encoding="utf-8"))


def test_evaluation_emits_a_valid_report_when_every_row_abstains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    predictions = tmp_path / "predictions.csv"
    output = tmp_path / "report.json"
    _write_predictions(
        predictions,
        [
            {
                "true_label": "neutral",
                "raw_predicted_label": "neutral",
                "predicted_label": "uncertain",
                "confidence": "0.20",
                "abstention_threshold": "0.45",
            },
            {
                "true_label": "happy",
                "raw_predicted_label": "sad",
                "predicted_label": "uncertain",
                "confidence": "0.30",
                "abstention_threshold": "0.45",
            },
        ],
    )

    report = _evaluate(predictions, output, monkeypatch)

    assert report["coverage"] == 0.0
    assert report["abstention_rate"] == 1.0
    assert report["all_abstained"] is True
    assert report["macro_f1_decided"] is None
    assert report["abstention_thresholds"] == [0.45]


def test_macro_metrics_retain_zero_support_canonical_classes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    predictions = tmp_path / "predictions.csv"
    output = tmp_path / "report.json"
    _write_predictions(
        predictions,
        [
            {
                "true_label": "neutral",
                "predicted_label": "neutral",
                "confidence": "0.90",
            }
        ],
    )

    report = _evaluate(predictions, output, monkeypatch)

    expected = 1.0 / len(FUSION_EMOTIONS)
    assert report["macro_f1_decided"] == pytest.approx(expected)
    assert report["unweighted_average_recall_decided"] == pytest.approx(expected)
    assert report["canonical_labels"] == [label.value for label in FUSION_EMOTIONS]
    assert len(report["absent_true_labels"]) == len(FUSION_EMOTIONS) - 1


def test_evaluation_rejects_noncanonical_truth_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    predictions = tmp_path / "predictions.csv"
    _write_predictions(
        predictions,
        [
            {
                "true_label": "diagnosis",
                "predicted_label": "neutral",
                "confidence": "0.50",
            }
        ],
    )
    monkeypatch.setattr(sys, "argv", ["evaluate.py", str(predictions)])

    with pytest.raises(SystemExit, match="unsupported true labels"):
        evaluate.main()
