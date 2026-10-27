"""Torch datasets for per-segment landmark feature sequences."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from dbm.data.stats import normalize_features
from dbm.data.windows import Window, make_windows
from dbm.labels import label_to_index


FEATURE_COLUMNS = {"ear": (0,), "mar": (1,), "yaw": (4,), "pitch": (5,), "roll": (6,), "d_yaw": (7,), "d_pitch": (8,)}


def compose_features(data: dict, config: dict) -> np.ndarray:
    """Compose configured feature groups in temporal order."""
    geo = np.asarray(data["geo"], dtype=np.float32)
    # Stored geometry: ear_left, ear_right, ear, mar, yaw, pitch, roll.
    groups = {
        "ear": geo[:, 2:3], "mar": geo[:, 3:4], "yaw": geo[:, 4:5],
        "pitch": geo[:, 5:6], "roll": geo[:, 6:7],
        "d_yaw": _delta(geo[:, 4:5]), "d_pitch": _delta(geo[:, 5:6]),
    }
    lm = np.asarray(data["lm_sel"], dtype=np.float32)
    if "landmarks_xy" in config:
        lm = _normalize_landmarks(lm, geo, config)
        groups["landmarks_xy"] = lm.reshape(len(lm), -1)
    selected = []
    for name in config.get("groups", []):
        if name not in groups:
            raise ValueError(f"Unknown or unavailable feature group: {name}")
        selected.append(groups[name])
    if not selected:
        raise ValueError("At least one feature group is required")
    return np.concatenate(selected, axis=1).astype(np.float32)


def _delta(values: np.ndarray) -> np.ndarray:
    result = np.zeros_like(values)
    result[1:] = values[1:] - values[:-1]
    return result


def _normalize_landmarks(lm: np.ndarray, geo: np.ndarray, config: dict) -> np.ndarray:
    """Translate, scale, and optionally roll-align selected landmark coordinates."""
    points = lm.copy()
    if points.shape[-1] < 2:
        raise ValueError("lm_sel must contain x/y coordinates")
    indices = list(config.get("selected_landmarks", []))
    def selected_index(landmark_id: int) -> int | None:
        try:
            return indices.index(landmark_id)
        except ValueError:
            return None
    origin_mode = config.get("normalization_origin", "nose")
    if origin_mode == "midpoint_eyes":
        left = selected_index(263); right = selected_index(33)
        if left is not None and right is not None:
            origin = (points[:, left, :2] + points[:, right, :2]) / 2.0
        else:
            origin = np.zeros((len(points), 2), dtype=points.dtype)
    else:
        nose = selected_index(1)
        origin = points[:, nose, :2] if nose is not None else np.zeros((len(points), 2), dtype=points.dtype)
    xy = points[:, :, :2] - origin[:, None, :]
    scale_mode = config.get("normalization_scale", "inter_ocular")
    if scale_mode == "inter_ocular":
        left = selected_index(263); right = selected_index(33)
        if left is not None and right is not None:
            scale = np.linalg.norm(points[:, left, :2] - points[:, right, :2], axis=1)
        else:
            scale = np.ones(len(points), dtype=points.dtype)
        xy /= np.maximum(scale, 1e-4)[:, None, None]
    elif scale_mode != "none":
        raise ValueError(f"Unknown landmark normalization scale: {scale_mode}")
    if config.get("normalize_roll", False):
        angle = np.deg2rad(geo[:, 6])
        cosine = np.cos(angle); sine = np.sin(angle)
        x = xy[:, :, 0].copy(); y = xy[:, :, 1].copy()
        xy[:, :, 0] = cosine[:, None] * x + sine[:, None] * y
        xy[:, :, 1] = -sine[:, None] * x + cosine[:, None] * y
    return xy


class GeoSequenceDataset(Dataset):
    """Build split-filtered windows from feature NPZ files."""

    def __init__(self, metadata: pd.DataFrame, splits: pd.DataFrame, feature_root: str | Path,
                 split: str, feature_config: dict, window_size: int = 30, stride: int = 15,
                 short_segment_policy: str = "pad", stats: dict | None = None):
        self.feature_root = Path(feature_root)
        self.stats = stats
        subject_splits = dict(zip(splits["subject_id"].astype(str), splits["split"].astype(str)))
        rows = metadata.copy()
        rows = rows[rows["quality"].astype(str).str.lower() != "bad"]
        rows = rows[rows["subject_id"].astype(str).map(subject_splits) == split]
        self.windows: list[Window] = []
        for row in rows.to_dict("records"):
            path = self.feature_root / f"{row['segment_id']}.npz"
            if not path.is_file():
                raise FileNotFoundError(f"Missing landmark feature file: {path}")
            with np.load(path) as archive:
                data = {key: archive[key] for key in archive.files}
            values = compose_features(data, feature_config)
            if stats is not None:
                values = normalize_features(values, stats)
            windows = make_windows(values, str(row["segment_id"]), label_to_index(str(row["label"])),
                                   window_size, stride, short_segment_policy)
            if "valid" in data:
                valid = np.asarray(data["valid"], dtype=bool)
                min_valid_ratio = float(feature_config.get("min_valid_ratio", 0.8))
                windows = [window for window in windows
                           if valid[window.start:min(window.start + window_size, len(valid))].mean() >= min_valid_ratio]
            self.windows.extend(windows)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int):
        item = self.windows[index]
        return (torch.from_numpy(item.values.copy()), torch.tensor(item.label, dtype=torch.long),
                item.segment_id, item.padded)

