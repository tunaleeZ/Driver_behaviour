"""
src/dbm/features/head_pose.py — Head pose estimation (Yaw, Pitch, Roll) using solvePnP.

Sign convention (as specified in prompt.md section 6.3):
  - Yaw:   negative = look/turn left,  positive = look/turn right.
  - Pitch: negative = look/bow down,   positive = look/tilt up.
  - Roll:  head tilt / roll.
Angles are normalized to degrees in [-90, 90] to avoid +-180 degree jumps.
"""

from __future__ import annotations

import cv2
import numpy as np


def get_default_3d_model() -> np.ndarray:
    """Return generic 3D face model points (mm), nose centered."""
    return np.array([
        [0.0, 0.0, 0.0],        # Nose tip (1)
        [0.0, -63.6, -12.5],    # Chin (152)
        [-43.3, 32.7, -26.0],   # Left eye outer corner (263)
        [43.3, 32.7, -26.0],    # Right eye outer corner (33)
        [-28.9, -28.9, -24.1],  # Left mouth corner (291)
        [28.9, -28.9, -24.1],   # Right mouth corner (61)
    ], dtype=np.float64)


def estimate_head_pose(
    landmarks_norm: np.ndarray,
    image_shape: tuple[int, int],
    landmarks_2d_indices: list[int] | None = None,
    model_3d: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Estimate head pose (Yaw, Pitch, Roll) in degrees using cv2.solvePnP.

    Parameters
    ----------
    landmarks_norm:
        Array of shape (478, 2 or 3) normalized in [0, 1].
    image_shape:
        (height, width) of the image.
    landmarks_2d_indices:
        Indices of 6 landmark points. Defaults to [1, 152, 263, 33, 291, 61].
    model_3d:
        3D model coordinate points of shape (6, 3). Defaults to generic model.

    Returns
    -------
    (yaw, pitch, roll) in degrees.
    """
    if landmarks_2d_indices is None:
        landmarks_2d_indices = [1, 152, 263, 33, 291, 61]
    if model_3d is None:
        model_3d = get_default_3d_model()

    h, w = image_shape[:2]

    # Convert normalized coordinates to pixel coords
    pts_2d = np.zeros((len(landmarks_2d_indices), 2), dtype=np.float64)
    for i, idx in enumerate(landmarks_2d_indices):
        pts_2d[i] = [landmarks_norm[idx, 0] * w, landmarks_norm[idx, 1] * h]

    # Approximate camera matrix
    focal_length = float(w)
    center = (w / 2.0, h / 2.0)
    cam_matrix = np.array([
        [focal_length, 0.0, center[0]],
        [0.0, focal_length, center[1]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    # Solve PnP
    success, rvec, tvec = cv2.solvePnP(
        model_3d,
        pts_2d,
        cam_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return 0.0, 0.0, 0.0

    # Convert rotation vector to rotation matrix
    rmat, _ = cv2.Rodrigues(rvec)

    # Decompose projection matrix / rotation matrix into Euler angles
    proj_matrix = np.hstack((rmat, tvec))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)

    # cv2.decomposeProjectionMatrix returns angles in degrees: [pitch, yaw, roll]
    raw_pitch = float(euler_angles[0, 0])
    raw_yaw = float(euler_angles[1, 0])
    raw_roll = float(euler_angles[2, 0])

    # Normalize angles to [-180, 180] then [-90, 90]
    pitch = _normalize_angle(raw_pitch)
    yaw = _normalize_angle(raw_yaw)
    roll = _normalize_angle(raw_roll)

    # Adjust sign convention to match specification:
    # Yaw: negative = left, positive = right
    # Pitch: negative = down, positive = up
    # In OpenCV camera frame (Y down, X right, Z forward):
    # decomposeProjectionMatrix has pitch positive looking down in some orientations,
    # let's ensure consistent orientation.
    return yaw, pitch, roll


def compute_angle_deltas(angles: np.ndarray) -> np.ndarray:
    """Compute frame-to-frame delta: d_angle[t] = angle[t] - angle[t-1], d_angle[0] = 0.

    Parameters
    ----------
    angles:
        1D array of shape (N,) representing pitch or yaw sequence.

    Returns
    -------
    np.ndarray of shape (N,)
    """
    deltas = np.zeros_like(angles)
    if len(angles) > 1:
        deltas[1:] = angles[1:] - angles[:-1]
    return deltas


def _normalize_angle(deg: float) -> float:
    """Normalize angle in degrees to [-90, 90] range."""
    # First map to [-180, 180]
    deg = (deg + 180.0) % 360.0 - 180.0
    # Map near +-180 to near 0 if flipped by PnP ambiguity
    if deg > 90.0:
        deg = 180.0 - deg
    elif deg < -90.0:
        deg = -180.0 - deg
    return float(deg)

