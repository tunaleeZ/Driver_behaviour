"""
src/dbm/features/mediapipe_extractor.py — MediaPipe FaceLandmarker feature extractor.

Features extracted per frame:
  - 478 face mesh 3D landmarks
  - Geometric features: [ear_left, ear_right, ear_avg, mar, yaw, pitch, roll]
  - Selected K normalized landmarks (x, y)
  - Missing face interpolation and validity masking
"""

from __future__ import annotations

import logging
import os
import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from dbm.config import load_config
from dbm.features.geometry import compute_ear, compute_mar
from dbm.features.head_pose import estimate_head_pose
from dbm.features.landmark_select import process_frame_landmarks
from dbm.paths import DatasetPaths

logger = logging.getLogger(__name__)


def ensure_model(model_path: Optional[Path | str] = None, model_url: Optional[str] = None) -> Path:
    """Ensure the MediaPipe face_landmarker.task model file is available on disk.

    Priority:
      1. Provided model_path (or env var DBM_FACE_LANDMARKER_PATH)
      2. Download from official model_url to models/face_landmarker.task
      3. Friendly error message if offline with instructions for manual placement.
    """
    paths = DatasetPaths()
    target_path = Path(model_path) if model_path else paths.face_landmarker

    if target_path.exists() and target_path.stat().st_size > 1000:
        return target_path

    if model_url is None:
        cfg_file = Path(__file__).resolve().parents[3] / "configs" / "mediapipe.yaml"
        if cfg_file.exists():
            cfg = load_config(cfg_file)
            model_url = cfg.get("model_url")

    if not model_url:
        model_url = (
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/1/face_landmarker.task"
        )

    logger.info("Downloading MediaPipe face landmarker model from: %s", model_url)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Download with timeout
        urllib.request.urlretrieve(model_url, target_path)
        logger.info("Successfully downloaded face_landmarker.task to %s", target_path)
        return target_path
    except Exception as exc:
        if target_path.exists():
            target_path.unlink()  # remove partial download
        msg = (
            f"\n{'='*70}\n"
            f"ERROR: Could not download face_landmarker.task model: {exc}\n\n"
            f"If running in an offline environment (such as Kaggle without Internet):\n"
            f"1. Download face_landmarker.task manually from:\n"
            f"   {model_url}\n"
            f"2. Place it at: {target_path}\n"
            f"   OR set the environment variable: DBM_FACE_LANDMARKER_PATH=/path/to/face_landmarker.task\n"
            f"{'='*70}\n"
        )
        logger.error(msg)
        raise RuntimeError(f"Missing MediaPipe model file at {target_path}. See instructions above.") from exc


def interpolate_missing_frames(
    features: np.ndarray,
    valid_mask: np.ndarray,
    max_gap_frames: int = 5,
) -> np.ndarray:
    """Interpolate features for missing frames up to max_gap_frames.

    Parameters
    ----------
    features:
        Array of shape (N, ...) with extracted features.
    valid_mask:
        Boolean array of shape (N,) indicating whether each frame was detected.
    max_gap_frames:
        Maximum consecutive missing frames to interpolate.

    Returns
    -------
    Interpolated features array of the same shape.
    """
    n = len(valid_mask)
    if n == 0 or np.all(valid_mask):
        return features

    out = features.copy()
    orig_shape = out.shape
    out_2d = out.reshape(n, -1)

    i = 0
    while i < n:
        if not valid_mask[i]:
            gap_start = i
            while i < n and not valid_mask[i]:
                i += 1
            gap_end = i
            gap_len = gap_end - gap_start

            if gap_len <= max_gap_frames:
                # Can interpolate if bounded on at least one side
                left_idx = gap_start - 1 if gap_start > 0 else None
                right_idx = gap_end if gap_end < n else None

                if left_idx is not None and right_idx is not None:
                    # Linear interpolation between left and right
                    left_val = out_2d[left_idx]
                    right_val = out_2d[right_idx]
                    for step, idx in enumerate(range(gap_start, gap_end), start=1):
                        alpha = step / (gap_len + 1)
                        out_2d[idx] = (1.0 - alpha) * left_val + alpha * right_val
                elif left_idx is not None:
                    # Forward fill
                    out_2d[gap_start:gap_end] = out_2d[left_idx]
                elif right_idx is not None:
                    # Backward fill
                    out_2d[gap_start:gap_end] = out_2d[right_idx]
        else:
            i += 1

    return out_2d.reshape(orig_shape)


class MediaPipeFaceExtractor:
    """Extracts face landmarks and geometric features from video segments."""

    def __init__(
        self,
        model_path: Optional[Path | str] = None,
        config: Optional[dict] = None,
    ) -> None:
        self.model_path = ensure_model(model_path)
        self.config = config or {}

        # Load configs
        repo_root = Path(__file__).resolve().parents[3]
        feat_cfg_file = repo_root / "configs" / "features.yaml"
        mp_cfg_file = repo_root / "configs" / "mediapipe.yaml"

        self.features_cfg = load_config(feat_cfg_file) if feat_cfg_file.exists() else {}
        self.mediapipe_cfg = load_config(mp_cfg_file) if mp_cfg_file.exists() else {}

        # Indices
        self.right_eye_indices = self.features_cfg.get("ear", {}).get(
            "right_eye", [33, 160, 158, 133, 153, 144]
        )
        self.left_eye_indices = self.features_cfg.get("ear", {}).get(
            "left_eye", [362, 385, 387, 263, 373, 380]
        )
        self.mouth_indices = self.features_cfg.get("mar", {}).get(
            "mouth", [78, 82, 312, 308, 317, 87]
        )
        self.selected_indices = self.features_cfg.get(
            "selected_landmarks", [33, 133, 362, 263, 1, 152]
        )

        self.min_detection_conf = float(self.mediapipe_cfg.get("min_face_detection_confidence", 0.5))
        self.min_presence_conf = float(self.mediapipe_cfg.get("min_face_presence_confidence", 0.5))
        self.min_tracking_conf = float(self.mediapipe_cfg.get("min_tracking_confidence", 0.5))
        self.max_gap_frames = int(self.mediapipe_cfg.get("max_gap_frames", 5))

    def _create_landmarker(self):
        """Create a fresh FaceLandmarker instance in VIDEO mode."""
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        base_options = python.BaseOptions(model_asset_path=str(self.model_path))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=self.min_detection_conf,
            min_face_presence_confidence=self.min_presence_conf,
            min_tracking_confidence=self.min_tracking_conf,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        return vision.FaceLandmarker.create_from_options(options)

    def extract_from_video(
        self,
        video_path: Path | str,
        target_fps: Optional[float] = 30.0,
        save_full: bool = False,
    ) -> dict:
        """Extract landmarks and geometric features from a video file.

        Returns dict:
          - timestamps_ms: np.ndarray (N,)
          - valid: np.ndarray (N,) bool
          - fps: float
          - geo: np.ndarray (N, 7) [ear_left, ear_right, ear_avg, mar, yaw, pitch, roll]
          - lm_sel: np.ndarray (N, K, 2)
          - full_landmarks: np.ndarray (N, 478, 3) float16 (if save_full)
        """
        import mediapipe as mp

        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {video_path}")

        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fps = float(target_fps if target_fps and target_fps > 0 else (orig_fps if orig_fps > 0 else 30.0))

        frames_rgb: list[np.ndarray] = []
        frame_timestamps_ms: list[int] = []

        cur_idx = 0
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames_rgb.append(frame_rgb)
            ts_ms = int(round((cur_idx / (orig_fps if orig_fps > 0 else 30.0)) * 1000.0))
            frame_timestamps_ms.append(ts_ms)
            cur_idx += 1

        cap.release()

        n_frames = len(frames_rgb)
        if n_frames == 0:
            raise RuntimeError(f"No frames could be read from {video_path}")

        k_landmarks = len(self.selected_indices)
        valid_arr = np.zeros(n_frames, dtype=bool)
        geo_arr = np.zeros((n_frames, 7), dtype=np.float32)
        lm_sel_arr = np.zeros((n_frames, k_landmarks, 2), dtype=np.float32)
        full_lm_list: list[np.ndarray] = [] if save_full else None

        landmarker = self._create_landmarker()

        try:
            # Monotonically increasing timestamp tracking
            last_ts_ms = -1

            for i in range(n_frames):
                ts = frame_timestamps_ms[i]
                if ts <= last_ts_ms:
                    ts = last_ts_ms + 1
                last_ts_ms = ts

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frames_rgb[i])
                detection_result = landmarker.detect_for_video(mp_image, ts)

                if detection_result.face_landmarks and len(detection_result.face_landmarks) > 0:
                    face = detection_result.face_landmarks[0]
                    # Convert to normalized array (478, 3)
                    lm_norm = np.array([[p.x, p.y, p.z] for p in face], dtype=np.float32)

                    valid_arr[i] = True

                    # Geometric features
                    ear_r, ear_l, ear_avg = compute_ear(
                        lm_norm, (height, width), self.right_eye_indices, self.left_eye_indices
                    )
                    mar = compute_mar(lm_norm, (height, width), self.mouth_indices)
                    yaw, pitch, roll = estimate_head_pose(lm_norm, (height, width))

                    # geo: [ear_left, ear_right, ear_avg, mar, yaw, pitch, roll]
                    geo_arr[i] = [ear_l, ear_r, ear_avg, mar, yaw, pitch, roll]

                    # Selected normalized landmarks
                    k_pts = process_frame_landmarks(
                        lm_norm,
                        self.selected_indices,
                        origin_type="nose",
                        compensate_roll=False,
                    )
                    lm_sel_arr[i] = k_pts

                    if save_full:
                        full_lm_list.append(lm_norm)
                else:
                    valid_arr[i] = False
                    if save_full:
                        full_lm_list.append(np.zeros((478, 3), dtype=np.float32))

        finally:
            landmarker.close()

        # Interpolate missing frames
        if not np.all(valid_arr):
            geo_arr = interpolate_missing_frames(geo_arr, valid_arr, max_gap_frames=self.max_gap_frames)
            lm_sel_arr = interpolate_missing_frames(lm_sel_arr, valid_arr, max_gap_frames=self.max_gap_frames)

        result = {
            "timestamps_ms": np.array(frame_timestamps_ms, dtype=np.int64),
            "valid": valid_arr,
            "fps": fps,
            "geo": geo_arr,
            "lm_sel": lm_sel_arr,
        }
        if save_full:
            result["full_landmarks"] = np.array(full_lm_list, dtype=np.float16)

        return result

