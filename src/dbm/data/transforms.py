"""
src/dbm/data/transforms.py — Image resizing and transform utilities.

Includes letterbox, squash, and center_crop resizing to maintain aspect ratio
and prevent geometric distortion.
"""

from __future__ import annotations

import cv2
import numpy as np


def resize_letterbox(
    image: np.ndarray,
    target_size: tuple[int, int],
    fill_color: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Resize image to target_size (height, width) with letterbox padding.

    Preserves aspect ratio, pads remaining margins with fill_color.
    """
    target_h, target_w = target_size
    orig_h, orig_w = image.shape[:2]

    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    out = np.full((target_h, target_w, image.shape[2] if image.ndim == 3 else 1), fill_color, dtype=image.dtype)
    if out.ndim == 3 and image.ndim == 2:
        out = np.squeeze(out, axis=-1)

    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    out[top : top + new_h, left : left + new_w] = resized

    return out


def resize_center_crop(
    image: np.ndarray,
    target_size: tuple[int, int],
) -> np.ndarray:
    """Resize shorter side to target_size, then crop the center."""
    target_h, target_w = target_size
    orig_h, orig_w = image.shape[:2]

    scale = max(target_w / orig_w, target_h / orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    start_x = (new_w - target_w) // 2
    start_y = (new_h - target_h) // 2
    return resized[start_y : start_y + target_h, start_x : start_x + target_w]


def resize_squash(
    image: np.ndarray,
    target_size: tuple[int, int],
) -> np.ndarray:
    """Directly resize (squash/stretch) image to target_size (height, width)."""
    target_h, target_w = target_size
    return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def resize_frame(
    image: np.ndarray,
    target_size: tuple[int, int],
    mode: str = "letterbox",
) -> np.ndarray:
    """Resize image according to mode: 'letterbox', 'squash', or 'center_crop'."""
    mode = mode.lower()
    if mode == "letterbox":
        return resize_letterbox(image, target_size)
    elif mode == "center_crop":
        return resize_center_crop(image, target_size)
    elif mode == "squash":
        return resize_squash(image, target_size)
    else:
        raise ValueError(f"Unknown resize mode: {mode!r}. Valid: 'letterbox', 'center_crop', 'squash'")

