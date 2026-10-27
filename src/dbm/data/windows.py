"""Create temporal windows without crossing segment boundaries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Window:
    """A temporal window and its source segment metadata."""

    values: np.ndarray
    segment_id: str
    label: int
    start: int
    padded: bool = False


def make_windows(
    values: np.ndarray,
    segment_id: str,
    label: int,
    window_size: int,
    stride: int,
    short_segment_policy: str = "pad",
) -> list[Window]:
    """Split one segment into windows; pad short clips with their last frame."""
    data = np.asarray(values)
    if data.ndim < 2:
        raise ValueError("values must have a time axis and at least one feature axis")
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")
    if short_segment_policy not in {"pad", "drop"}:
        raise ValueError("short_segment_policy must be 'pad' or 'drop'")
    if len(data) == 0:
        return []
    if len(data) < window_size:
        if short_segment_policy == "drop":
            return []
        pad = np.repeat(data[-1:], window_size - len(data), axis=0)
        return [Window(np.concatenate([data, pad], axis=0), segment_id, label, 0, True)]
    starts = list(range(0, len(data) - window_size + 1, stride))
    if starts[-1] + window_size < len(data):
        starts.append(len(data) - window_size)
    return [Window(data[start : start + window_size], segment_id, label, start) for start in starts]

