"""
tests/test_extract_frames.py — Unit tests for frame resizing and extraction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "driver_behavior_dataset_v01" / "scripts"))

from dbm.data.transforms import (
    resize_center_crop,
    resize_frame,
    resize_letterbox,
    resize_squash,
)
from extract_frames import extract_frames_from_video


class TestTransforms:
    def test_resize_letterbox_preserves_aspect_ratio(self):
        # 100x200 image -> resize to 128x128
        img = np.ones((100, 200, 3), dtype=np.uint8) * 255
        target = (128, 128)
        out = resize_letterbox(img, target, fill_color=(0, 0, 0))

        assert out.shape == (128, 128, 3)
        # Top and bottom rows should be letterbox black (0)
        assert np.all(out[0, :, :] == 0)
        assert np.all(out[-1, :, :] == 0)
        # Center should be non-zero (white image content)
        assert np.any(out[64, 64, :] > 0)

    def test_resize_squash(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        out = resize_squash(img, (128, 128))
        assert out.shape == (128, 128, 3)

    def test_resize_center_crop(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        out = resize_center_crop(img, (128, 128))
        assert out.shape == (128, 128, 3)

    def test_resize_frame_invalid_mode(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            resize_frame(img, (128, 128), mode="invalid_mode")


class TestExtractFramesPipeline:
    def test_extract_frames_execution_and_pack(self, tmp_path):
        # Generate synthetic 1.0s 30fps video
        vid_file = tmp_path / "test_seg.mp4"
        fps = 30
        w, h = 120, 90
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(vid_file), fourcc, fps, (w, h))
        for i in range(30):
            frame = np.full((h, w, 3), fill_value=i * 5, dtype=np.uint8)
            out.write(frame)
        out.release()

        frames_out_dir = tmp_path / "processed" / "frames" / "SEG000001"

        saved_count, size_str = extract_frames_from_video(
            video_path=vid_file,
            out_dir=frames_out_dir,
            target_fps=10.0,
            store_size=(128, 128),
            resize_mode="letterbox",
            pack_npy=True,
        )

        assert saved_count == 10  # 1.0s at 10 fps
        assert size_str == "128x128"

        jpgs = list(frames_out_dir.glob("*.jpg"))
        assert len(jpgs) == 10

        # Check packed npy
        npy_file = tmp_path / "processed" / "frames" / "SEG000001.npy"
        assert npy_file.exists()
        arr = np.load(npy_file)
        assert arr.shape == (10, 128, 128, 3)
        assert arr.dtype == np.uint8

