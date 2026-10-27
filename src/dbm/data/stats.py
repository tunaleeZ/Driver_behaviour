"""Train-only feature normalization statistics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def fit_feature_stats(sequences: list[np.ndarray]) -> dict[str, list[float]]:
    """Fit per-feature population mean/std using training sequences only."""
    if not sequences:
        raise ValueError("At least one training sequence is required")
    arrays = [np.asarray(x, dtype=np.float64) for x in sequences]
    if any(x.ndim != 2 for x in arrays):
        raise ValueError("Each sequence must have shape (time, features)")
    dims = {x.shape[1] for x in arrays}
    if len(dims) != 1:
        raise ValueError("Training sequences have inconsistent feature dimensions")
    merged = np.concatenate(arrays, axis=0)
    mean = merged.mean(axis=0)
    std = merged.std(axis=0)
    std[std < 1e-8] = 1.0
    return {"mean": mean.tolist(), "std": std.tolist()}


def normalize_features(values: np.ndarray, stats: dict) -> np.ndarray:
    """Apply previously fitted train-set statistics."""
    x = np.asarray(values, dtype=np.float32)
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    if x.shape[-1] != len(mean) or mean.shape != std.shape:
        raise ValueError("Feature dimension does not match normalization statistics")
    return (x - mean) / std


def save_stats(stats: dict, path: str | Path) -> None:
    """Write normalization statistics as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

