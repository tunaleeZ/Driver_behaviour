"""
tests/test_mediapipe_extraction.py — Tests for MediaPipe FaceLandmarker extractor integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dbm.features.mediapipe_extractor import (
    MediaPipeFaceExtractor,
    ensure_model,
)


class TestMediaPipeExtractor:
    def test_ensure_model_finds_downloaded_model(self):
        model_path = ensure_model()
        assert model_path.exists()
        assert model_path.stat().st_size > 1000000  # ~3.7MB

    def test_extract_from_synthetic_video(self, tmp_path):
        """Verify extractor runs end-to-end on video without face, properly marking valid=False."""
        vid_file = tmp_path / "blank_video.mp4"
        fps = 30
        w, h = 160, 120
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(vid_file), fourcc, fps, (w, h))
        for _ in range(15):  # 0.5 second
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            out.write(frame)
        out.release()

        extractor = MediaPipeFaceExtractor()
        data = extractor.extract_from_video(vid_file, target_fps=30.0, save_full=True)

        assert "timestamps_ms" in data
        assert "valid" in data
        assert "fps" in data
        assert "geo" in data
        assert "lm_sel" in data
        assert "full_landmarks" in data

        assert len(data["timestamps_ms"]) == 15
        assert len(data["valid"]) == 15
        assert np.all(data["valid"] == False)  # blank video has no faces
        assert data["geo"].shape == (15, 7)
        assert data["lm_sel"].shape[0] == 15
        assert data["lm_sel"].shape[2] == 2
        assert data["full_landmarks"].shape == (15, 478, 3)

