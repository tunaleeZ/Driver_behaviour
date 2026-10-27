"""Classification metrics and confusion matrix helpers without sklearn dependency."""

from __future__ import annotations

import numpy as np


def classification_metrics(y_true, y_pred, n_classes: int) -> dict:
    """Return per-class precision/recall/F1, accuracy, macro-F1, and confusion."""
    truth = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for actual, guess in zip(truth, pred):
        if 0 <= actual < n_classes and 0 <= guess < n_classes:
            cm[actual, guess] += 1
    rows = []
    for i in range(n_classes):
        tp = int(cm[i, i]); fp = int(cm[:, i].sum() - tp); fn = int(cm[i, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"precision": precision, "recall": recall, "f1": f1, "support": int(cm[i].sum())})
    total = int(cm.sum())
    normalized = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    return {"accuracy": float(np.trace(cm) / total) if total else 0.0,
            "macro_f1": float(np.mean([row["f1"] for row in rows])),
            "per_class": rows, "confusion_matrix": cm.tolist(),
            "confusion_matrix_normalized": normalized.tolist(), "n_samples": total}


def aggregate_segment_probabilities(segment_ids, probabilities, labels):
    """Average window probabilities for each segment before segment scoring."""
    grouped: dict[str, list[np.ndarray]] = {}
    target: dict[str, int] = {}
    for sid, prob, label in zip(segment_ids, probabilities, labels):
        grouped.setdefault(str(sid), []).append(np.asarray(prob))
        target[str(sid)] = int(label)
    ids = sorted(grouped)
    return ids, np.stack([np.mean(grouped[sid], axis=0) for sid in ids]), np.asarray([target[sid] for sid in ids])

