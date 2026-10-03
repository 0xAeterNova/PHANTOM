"""Shared, reproducible helpers for the optional classical-ML training scripts."""

from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

from phantom.compat import UTC


def require_ml_stack() -> tuple[Any, Any]:
    try:
        import pandas as pd
        import sklearn
    except ImportError as exc:  # pragma: no cover - depends on optional extras
        raise SystemExit(
            "Training requires the optional ML dependencies. Install with: pip install -e '.[ml]'"
        ) from exc
    return pd, sklearn


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def grouped_split(frame: Any, group_column: str, seed: int) -> tuple[Any, Any, Any]:
    """Create 70/15/15 group-independent splits and assert no identity leakage."""

    from sklearn.model_selection import GroupShuffleSplit

    if group_column not in frame:
        raise ValueError(f"required group column is missing: {group_column}")
    if frame[group_column].nunique() < 3:
        raise ValueError("at least three independent speakers/subjects are required")
    first = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=seed)
    train_idx, remainder_idx = next(first.split(frame, groups=frame[group_column]))
    train = frame.iloc[train_idx].copy()
    remainder = frame.iloc[remainder_idx].copy()
    second = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=seed + 1)
    val_idx, test_idx = next(second.split(remainder, groups=remainder[group_column]))
    validation = remainder.iloc[val_idx].copy()
    test = remainder.iloc[test_idx].copy()
    groups = [set(part[group_column].astype(str)) for part in (train, validation, test)]
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise RuntimeError("group-independent split invariant failed")
    return train, validation, test


def feature_columns(frame: Any, excluded: set[str]) -> list[str]:
    columns = [
        str(column)
        for column in frame.columns
        if str(column) not in excluded and str(frame[column].dtype) != "object"
    ]
    if not columns:
        raise ValueError("no numeric feature columns were found")
    return columns


def write_metadata(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    complete = {
        "created_at": datetime.now(UTC).isoformat(),
        "claim_status": "local training run; metrics require separate held-out evaluation",
        **payload,
    }
    path.write_text(json.dumps(complete, indent=2, sort_keys=True), encoding="utf-8")
