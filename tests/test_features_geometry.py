"""
tests/test_features_geometry.py — Unit tests for geometry, head pose, landmark selection, and interpolation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dbm.features.geometry import (
    compute_aspect_ratio_6pt,
    compute_ear,
    compute_mar,
)
from dbm.features.head_pose import (
    compute_angle_deltas,
    estimate_head_pose,
    get_default_3d_model,
)
from dbm.features.landmark_select import (
    normalize_landmarks_2d,
    process_frame_landmarks,
    select_landmarks,
)
from dbm.features.mediapipe_extractor import interpolate_missing_frames


class TestAspectRatios:
    def test_aspect_ratio_manual_calculation(self):
        # A simple symmetric 6-point shape:
        # p1=(-2, 0), p2=(-1, 1), p3=(1, 1), p4=(2, 0), p5=(1, -1), p6=(-1, -1)
        # v1 = ||p2 - p6|| = ||(0, 2)|| = 2.0
        # v2 = ||p3 - p5|| = ||(0, 2)|| = 2.0
        # h = ||p1 - p4|| = ||(-4, 0)|| = 4.0
        # AR = (2 + 2) / (2 * 4) = 4 / 8 = 0.5
        pts = np.array([
            [-2.0, 0.0],
            [-1.0, 1.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [1.0, -1.0],
            [-1.0, -1.0],
        ])
        ar = compute_aspect_ratio_6pt(pts)
        assert ar == pytest.approx(0.5)

    def test_open_vs_closed_eye(self):
        # Open eye (height = 4)
        open_eye = np.array([
            [-10.0, 0.0],
            [-5.0, 2.0],
            [5.0, 2.0],
            [10.0, 0.0],
            [5.0, -2.0],
            [-5.0, -2.0],
        ])
        # Closed eye (height = 0.5)
        closed_eye = np.array([
            [-10.0, 0.0],
            [-5.0, 0.25],
            [5.0, 0.25],
            [10.0, 0.0],
            [5.0, -0.25],
            [-5.0, -0.25],
        ])

        ear_open = compute_aspect_ratio_6pt(open_eye)
        ear_closed = compute_aspect_ratio_6pt(closed_eye)

        assert ear_open > 0.15
        assert ear_closed < 0.05
        assert ear_open > 5.0 * ear_closed

    def test_open_vs_closed_mouth(self):
        # Open mouth (yawning)
        open_mouth = np.array([
            [-15.0, 0.0],
            [-8.0, 8.0],
            [8.0, 8.0],
            [15.0, 0.0],
            [8.0, -8.0],
            [-8.0, -8.0],
        ])
        # Closed mouth
        closed_mouth = np.array([
            [-15.0, 0.0],
            [-8.0, 0.5],
            [8.0, 0.5],
            [15.0, 0.0],
            [8.0, -0.5],
            [-8.0, -0.5],
        ])

        mar_open = compute_aspect_ratio_6pt(open_mouth)
        mar_closed = compute_aspect_ratio_6pt(closed_mouth)

        assert mar_open > 0.4
        assert mar_closed < 0.1
        assert mar_open > 4.0 * mar_closed


class TestHeadPose:
    def test_angle_deltas(self):
        angles = np.array([10.0, 12.0, 11.0, 15.0])
        deltas = compute_angle_deltas(angles)
        expected = np.array([0.0, 2.0, -1.0, 4.0])
        np.testing.assert_allclose(deltas, expected)

    def test_solve_pnp_reconstruction_known_angles(self):
        """Project 3D face model points using known rotation angles, and verify solvePnP recovery error < 2 degrees."""
        model_3d = get_default_3d_model()
        w, h = 640, 480
        focal_length = float(w)
        center = (w / 2.0, h / 2.0)
        cam_matrix = np.array([
            [focal_length, 0.0, center[0]],
            [0.0, focal_length, center[1]],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # Test case: small rotation (pitch=0, yaw=0, roll=0)
        tvec = np.array([[0.0], [0.0], [500.0]], dtype=np.float64)
        rvec = np.array([[0.0], [0.0], [0.0]], dtype=np.float64)

        pts_2d, _ = cv2.projectPoints(model_3d, rvec, tvec, cam_matrix, dist_coeffs)
        pts_2d = pts_2d.reshape(-1, 2)

        # Build dummy 478 landmarks array
        landmarks_norm = np.zeros((478, 3), dtype=np.float32)
        pnp_indices = [1, 152, 263, 33, 291, 61]
        for i, idx in enumerate(pnp_indices):
            landmarks_norm[idx, 0] = pts_2d[i, 0] / w
            landmarks_norm[idx, 1] = pts_2d[i, 1] / h
            landmarks_norm[idx, 2] = 0.0

        yaw, pitch, roll = estimate_head_pose(landmarks_norm, (h, w), pnp_indices, model_3d)

        # Tolerance: error < 2.0 degrees
        assert abs(yaw) < 2.0
        assert abs(pitch) < 2.0
        assert abs(roll) < 2.0


class TestLandmarkSelectionAndNormalization:
    def test_select_landmarks(self):
        dummy_mesh = np.arange(478 * 2).reshape(478, 2)
        indices = [1, 33, 152]
        selected = select_landmarks(dummy_mesh, indices)
        assert selected.shape == (3, 2)
        np.testing.assert_array_equal(selected[0], dummy_mesh[1])
        np.testing.assert_array_equal(selected[1], dummy_mesh[33])
        np.testing.assert_array_equal(selected[2], dummy_mesh[152])

    def test_normalize_landmarks_scale_and_translation(self):
        pts = np.array([[10.0, 20.0], [30.0, 40.0]])
        origin = np.array([10.0, 20.0])
        iod = 20.0

        normed = normalize_landmarks_2d(pts, origin, inter_ocular_dist=iod)
        # Point 0 should be at origin (0, 0)
        assert normed[0, 0] == pytest.approx(0.0)
        assert normed[0, 1] == pytest.approx(0.0)
        # Point 1 should be translated and divided by 20
        assert normed[1, 0] == pytest.approx((30.0 - 10.0) / 20.0)
        assert normed[1, 1] == pytest.approx((40.0 - 20.0) / 20.0)


class TestMissingFaceInterpolation:
    def test_linear_interpolation(self):
        # 5 frames: [10, gap, gap, 40, 50]
        feats = np.array([[10.0], [0.0], [0.0], [40.0], [50.0]])
        valid = np.array([True, False, False, True, True])

        interp = interpolate_missing_frames(feats, valid, max_gap_frames=3)
        # Frame 1: 10 + 1/3 * (40 - 10) = 20.0
        # Frame 2: 10 + 2/3 * (40 - 10) = 30.0
        assert interp[1, 0] == pytest.approx(20.0)
        assert interp[2, 0] == pytest.approx(30.0)

    def test_forward_fill_at_tail(self):
        feats = np.array([[10.0], [20.0], [0.0]])
        valid = np.array([True, True, False])

        interp = interpolate_missing_frames(feats, valid, max_gap_frames=3)
        assert interp[2, 0] == pytest.approx(20.0)

    def test_gap_exceeds_max_gap(self):
        # Gap length 4 > max_gap_frames 2: should not interpolate
        feats = np.array([[10.0], [0.0], [0.0], [0.0], [0.0], [50.0]])
        valid = np.array([True, False, False, False, False, True])

        interp = interpolate_missing_frames(feats, valid, max_gap_frames=2)
        assert interp[1, 0] == 0.0
        assert interp[2, 0] == 0.0

