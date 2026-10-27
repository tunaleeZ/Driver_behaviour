"""
src/dbm/features/geometry.py — Geometric feature computation: EAR and MAR.

Formula from report:
  EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
  MAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)  (for mouth inner lip points)

Calculated on pixel coordinates (x * W, y * H) to prevent aspect ratio distortion.
All indices are loaded from configs/features.yaml, not hard-coded.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def compute_aspect_ratio_6pt(points: np.ndarray) -> float:
    """Compute aspect ratio for a 6-point contour [p1, p2, p3, p4, p5, p6].

    Parameters
    ----------
    points:
        Array of shape (6, 2) representing (x, y) coordinates.
        p1: leftmost corner
        p2: top left
        p3: top right
        p4: rightmost corner
        p5: bottom right
        p6: bottom left

    Returns
    -------
    float
        Aspect ratio. If horizontal distance is 0, returns 0.0.
    """
    if points.shape[0] != 6 or points.shape[1] < 2:
        raise ValueError(f"Expected points shape (6, 2), got {points.shape}")

    p1, p2, p3, p4, p5, p6 = points[:6, :2]

    # Vertical distances
    dist_v1 = float(np.linalg.norm(p2 - p6))
    dist_v2 = float(np.linalg.norm(p3 - p5))

    # Horizontal distance
    dist_h = float(np.linalg.norm(p1 - p4))

    if dist_h <= 1e-8:
        return 0.0

    return (dist_v1 + dist_v2) / (2.0 * dist_h)


def compute_ear(
    landmarks_norm: np.ndarray,
    image_shape: tuple[int, int],
    right_eye_indices: Sequence[int],
    left_eye_indices: Sequence[int],
) -> tuple[float, float, float]:
    """Calculate EAR for right eye, left eye, and average EAR.

    Parameters
    ----------
    landmarks_norm:
        Array of shape (478, 2 or 3) normalized in [0, 1].
    image_shape:
        (height, width) of the original frame.
    right_eye_indices:
        6 indices for the right eye [p1..p6].
    left_eye_indices:
        6 indices for the left eye [p1..p6].

    Returns
    -------
    (ear_right, ear_left, ear_avg)
    """
    h, w = image_shape[:2]

    # Convert normalized (x, y) to pixel coordinates
    pts_px = landmarks_norm[:, :2].copy()
    pts_px[:, 0] *= w
    pts_px[:, 1] *= h

    pts_right = pts_px[list(right_eye_indices)]
    pts_left = pts_px[list(left_eye_indices)]

    ear_r = compute_aspect_ratio_6pt(pts_right)
    ear_l = compute_aspect_ratio_6pt(pts_left)
    ear_avg = (ear_r + ear_l) / 2.0

    return ear_r, ear_l, ear_avg


def compute_mar(
    landmarks_norm: np.ndarray,
    image_shape: tuple[int, int],
    mouth_indices: Sequence[int],
) -> float:
    """Calculate MAR (Mouth Aspect Ratio).

    Parameters
    ----------
    landmarks_norm:
        Array of shape (478, 2 or 3) normalized in [0, 1].
    image_shape:
        (height, width) of the original frame.
    mouth_indices:
        6 indices for mouth [p1..p6].

    Returns
    -------
    float
        MAR value.
    """
    h, w = image_shape[:2]

    pts_px = landmarks_norm[:, :2].copy()
    pts_px[:, 0] *= w
    pts_px[:, 1] *= h

    pts_mouth = pts_px[list(mouth_indices)]
    return compute_aspect_ratio_6pt(pts_mouth)

