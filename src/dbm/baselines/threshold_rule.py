"""Interpretable face-only threshold baseline; cannot identify phone/drinking."""

from __future__ import annotations

import numpy as np


def window_rule_features(geo: np.ndarray, ear_threshold: float = 0.2, pitch_threshold: float = -10.0) -> dict:
    """Summarize PERCLOS, longest eye closure, pitch, nod recovery, and MAR."""
    values = np.asarray(geo)
    ear = values[:, 2]; mar = values[:, 3]; pitch = values[:, 5]
    closed = ear < ear_threshold
    longest = run = 0
    for is_closed in closed:
        run = run + 1 if is_closed else 0
        longest = max(longest, run)
    nod_recovery = bool(np.any((pitch[:-1] < pitch_threshold) & (pitch[1:] >= pitch_threshold))) if len(pitch) > 1 else False
    return {"perclos": float(closed.mean()) if len(closed) else 0.0,
            "max_closed_frames": int(longest), "mean_pitch": float(pitch.mean()) if len(pitch) else 0.0,
            "nod_recovery": nod_recovery, "mean_mar": float(mar.mean()) if len(mar) else 0.0}


def predict_window(geo: np.ndarray, thresholds: dict) -> int:
    """Apply supplied thresholds; label indices follow configs/labels.yaml."""
    f = window_rule_features(geo, float(thresholds.get("ear", 0.2)), float(thresholds.get("pitch", -10.0)))
    if f["nod_recovery"]:
        return 1
    if f["perclos"] >= float(thresholds.get("perclos", 0.4)) or f["max_closed_frames"] >= int(thresholds.get("closed_frames", 12)):
        return 2
    return 0


def grid_search_thresholds(train_windows, train_labels, candidates: dict) -> dict:
    """Select rule thresholds on the supplied training data only."""
    from itertools import product
    from dbm.metrics import classification_metrics
    keys = list(candidates)
    best = None
    for values in product(*(candidates[key] for key in keys)):
        cfg = dict(zip(keys, values))
        pred = [predict_window(window, cfg) for window in train_windows]
        score = classification_metrics(train_labels, pred, 5)["macro_f1"]
        if best is None or score > best[0]:
            best = (score, cfg)
    if best is None:
        raise ValueError("Threshold grid must not be empty")
    return best[1]

