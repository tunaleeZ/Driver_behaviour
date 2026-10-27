"""Torch datasets for per-segment landmark feature sequences."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import cv2
from torch.utils.data import Dataset

from dbm.data.stats import normalize_features
from dbm.data.transforms import resize_frame
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
    if "landmarks_xy" in config.get("groups", []):
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


class FrameClipDataset(Dataset):
    """Build temporally consistent RGB clips from extracted segment frames."""

    def __init__(self, metadata: pd.DataFrame, splits: pd.DataFrame, frame_root: str | Path,
                 split: str, window_size: int = 16, stride: int = 8,
                 short_segment_policy: str = "pad", input_size: int = 112,
                 training: bool = False, augment: bool = True,
                 image_mean: tuple[float, float, float] | None = None,
                 image_std: tuple[float, float, float] | None = None):
        self.frame_root = Path(frame_root)
        self.input_size = int(input_size)
        self.window_size = int(window_size)
        self.training = bool(training)
        self.augment = bool(augment and training)
        self.image_mean = np.asarray(image_mean if image_mean is not None else (0.0, 0.0, 0.0), dtype=np.float32)
        self.image_std = np.asarray(image_std if image_std is not None else (1.0, 1.0, 1.0), dtype=np.float32)
        if self.input_size <= 0 or window_size <= 0 or stride <= 0:
            raise ValueError("input_size, window_size, and stride must be positive")
        subject_splits = dict(zip(splits.subject_id.astype(str), splits.split.astype(str)))
        rows = metadata[(metadata.quality.astype(str).str.lower() != "bad") &
                        (metadata.subject_id.astype(str).map(subject_splits) == split)]
        self.samples: list[tuple[str, int, list[Path], bool]] = []
        for row in rows.to_dict("records"):
            sid = str(row["segment_id"])
            directory = self.frame_root / sid
            frames = sorted(directory.glob("*.jpg"))
            if not frames:
                raise FileNotFoundError(f"No extracted JPG frames found for {sid}: {directory}")
            n = len(frames)
            if n < window_size:
                if short_segment_policy == "drop":
                    continue
                if short_segment_policy != "pad":
                    raise ValueError("short_segment_policy must be 'pad' or 'drop'")
                self.samples.append((sid, label_to_index(str(row["label"])), frames, True))
            else:
                for start in range(0, n - window_size + 1, stride):
                    self.samples.append((sid, label_to_index(str(row["label"])),
                                         frames[start:start + window_size], False))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sid, label, paths, padded = self.samples[index]
        images = []
        for path in paths:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Could not decode frame: {path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = resize_frame(image, (128, 128), "letterbox")
            images.append(image)
        clip = np.stack(images)
        if padded:
            clip = np.concatenate((clip, np.repeat(clip[-1:],
                                   max(0, self.window_size - len(paths)), axis=0)), axis=0)
        if self.augment:
            clip = self._augment_clip(clip)
        else:
            size = self.input_size
            top = (clip.shape[1] - size) // 2
            left = (clip.shape[2] - size) // 2
            clip = clip[:, top:top + size, left:left + size]
        tensor = torch.from_numpy(clip.copy()).permute(0, 3, 1, 2).float().div_(255.0)
        mean = torch.as_tensor(self.image_mean)[None, :, None, None]
        std = torch.as_tensor(self.image_std)[None, :, None, None]
        tensor = (tensor - mean) / std.clamp_min(1e-6)
        return tensor, torch.tensor(label, dtype=torch.long), sid, padded

    def _augment_clip(self, clip: np.ndarray) -> np.ndarray:
        """Apply one crop, rotation, brightness, and contrast choice to all frames."""
        h, w = clip.shape[1:3]
        size = min(h, w)
        crop = int(np.random.randint(self.input_size, size + 1))
        top = int(np.random.randint(0, h - crop + 1))
        left = int(np.random.randint(0, w - crop + 1))
        angle = float(np.random.uniform(-5.0, 5.0))
        brightness = float(np.random.uniform(0.9, 1.1))
        contrast = float(np.random.uniform(0.9, 1.1))
        result = []
        for frame in clip:
            frame = frame[top:top + crop, left:left + crop]
            frame = cv2.resize(frame, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
            center = self.input_size / 2
            matrix = cv2.getRotationMatrix2D((center, center), angle, 1.0)
            frame = cv2.warpAffine(frame, matrix, (self.input_size, self.input_size), borderMode=cv2.BORDER_REFLECT_101)
            frame = np.clip((frame.astype(np.float32) - 127.5) * contrast + 127.5, 0, 255)
            frame = np.clip(frame * brightness, 0, 255).astype(np.uint8)
            result.append(frame)
        return np.stack(result)

