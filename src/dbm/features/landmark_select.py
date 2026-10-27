"""
src/dbm/features/landmark_select.py — Selection and normalization of K face landmarks.

For Landmark-LSTM (Variant B):
  - Extracts K landmarks specified in configs/features.yaml.
  - Normalization:
      1. Origin translation (e.g. nose tip or eye midpoint).
      2. Scale normalization by inter-ocular distance.
      3. Optional roll compensation rotation.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def select_landmarks(
    landmarks: np.ndarray,
    indices: Sequence[int],
) -> np.ndarray:
    """Select K landmarks from 478 full face mesh landmarks.

    Parameters
    ----------
    landmarks:
        Array of shape (478, 2) or (478, 3) or sequence (N, 478, 2/3).
    indices:
        Sequence of K integer indices.

    Returns
    -------
    Array of shape (K, 2) or (N, K, 2) with (x, y) coordinates.
    """
    indices_list = list(indices)
    if landmarks.ndim == 2:
        return landmarks[indices_list, :2].copy()
    elif landmarks.ndim == 3:
        return landmarks[:, indices_list, :2].copy()
    else:
        raise ValueError(f"Unexpected landmarks dimension: {landmarks.ndim}")


def normalize_landmarks_2d(
    landmarks_2d: np.ndarray,
    origin_point: np.ndarray,
    inter_ocular_dist: float,
    roll_angle_deg: float = 0.0,
    compensate_roll: bool = False,
) -> np.ndarray:
    """Normalize 2D landmarks by translating to origin and scaling by inter-ocular distance.

    Parameters
    ----------
    landmarks_2d:
        Array of shape (K, 2) representing (x, y).
    origin_point:
        Array of shape (2,) representing the reference origin (e.g. nose tip).
    inter_ocular_dist:
        Distance between eyes in the same coordinate space.
    roll_angle_deg:
        Estimated roll angle in degrees.
    compensate_roll:
        Whether to rotate coordinates to cancel roll.

    Returns
    -------
    Normalized coordinates of shape (K, 2).
    """
    pts = landmarks_2d - origin_point[:2]

    scale = inter_ocular_dist if inter_ocular_dist > 1e-6 else 1.0
    pts = pts / scale

    if compensate_roll and abs(roll_angle_deg) > 1e-3:
        rad = -np.radians(roll_angle_deg)
        c, s = np.cos(rad), np.sin(rad)
        rot_mat = np.array([[c, -s], [s, c]], dtype=pts.dtype)
        pts = pts @ rot_mat.T

    return pts


def process_frame_landmarks(
    full_landmarks: np.ndarray,
    selected_indices: Sequence[int],
    origin_type: str = "nose",
    compensate_roll: bool = False,
    roll_deg: float = 0.0,
) -> np.ndarray:
    """Convenience helper to extract and normalize K landmarks for one frame.

    Parameters
    ----------
    full_landmarks:
        Array of shape (478, 2 or 3) normalized in [0, 1].
    selected_indices:
        K landmark indices.
    origin_type:
        "nose" (index 1) or "midpoint_eyes" (between 33 and 263).
    compensate_roll:
        Whether to rotate to compensate for head roll.
    roll_deg:
        Estimated roll angle in degrees.

    Returns
    -------
    Normalized landmarks of shape (K, 2).
    """
    # Reference points
    nose_pt = full_landmarks[1, :2]
    right_eye_outer = full_landmarks[33, :2]
    left_eye_outer = full_landmarks[263, :2]

    # Origin
    if origin_type == "midpoint_eyes":
        origin = (right_eye_outer + left_eye_outer) / 2.0
    else:
        origin = nose_pt

    # Scale: inter-ocular distance
    iod = float(np.linalg.norm(right_eye_outer - left_eye_outer))

    pts_k = select_landmarks(full_landmarks, selected_indices)
    return normalize_landmarks_2d(
        landmarks_2d=pts_k,
        origin_point=origin,
        inter_ocular_dist=iod,
        roll_angle_deg=roll_deg,
        compensate_roll=compensate_roll,
    )

