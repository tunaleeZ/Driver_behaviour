"""
tests/test_cut_segments.py — Unit tests for video segment cutting.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "driver_behavior_dataset_v01" / "scripts"))

from cut_segments import (
    build_ffmpeg_cmd,
    get_ffmpeg_binary,
    process_segment,
    verify_cut_video,
)


class TestBuildFFmpegCmd:
    def test_cmd_structure(self):
        cmd = build_ffmpeg_cmd(
            ffmpeg_bin="ffmpeg",
            src_path="raw/video.mp4",
            dst_path="segments/normal/SEG000001.mp4",
            start_time="00:00:05.000",
            end_time="00:00:10.000",
        )
        assert cmd[0] == "ffmpeg"
        assert "-y" in cmd
        assert "-ss" in cmd
        assert "00:00:05.000" in cmd
        assert "-to" in cmd
        assert "00:00:10.000" in cmd
        assert "-an" in cmd
        assert "-c:v" in cmd
        assert "libx264" in cmd
        assert "-avoid_negative_ts" in cmd
        assert "make_zero" in cmd
        assert cmd[-1] == "segments/normal/SEG000001.mp4"

    def test_cmd_with_custom_fps(self):
        cmd = build_ffmpeg_cmd(
            ffmpeg_bin="ffmpeg",
            src_path="raw/video.mp4",
            dst_path="segments/normal/SEG000001.mp4",
            start_time=1.5,
            end_time=3.5,
            fps=15.0,
        )
        assert "-r" in cmd
        idx = cmd.index("-r")
        assert cmd[idx + 1] == "15.0"


class TestVerifyCutVideo:
    def test_nonexistent_video(self, tmp_path):
        valid, dur, err = verify_cut_video(tmp_path / "missing.mp4", expected_duration=5.0)
        assert not valid
        assert "does not exist" in err

    def test_synthetic_video_duration(self, tmp_path):
        # Create a synthetic 2.0s 30fps video
        vid_path = tmp_path / "test_synth.mp4"
        fps = 30
        w, h = 64, 64
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(vid_path), fourcc, fps, (w, h))
        for _ in range(60):  # 60 frames = 2.0s
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            out.write(frame)
        out.release()

        # Check exact expected
        valid, dur, err = verify_cut_video(vid_path, expected_duration=2.0)
        assert valid
        assert dur == pytest.approx(2.0, abs=0.1)

        # Check mismatched expected
        valid, dur, err = verify_cut_video(vid_path, expected_duration=5.0)
        assert not valid
        assert "Duration mismatch" in err


class TestProcessSegmentSynthetic:
    @pytest.fixture
    def synthetic_raw_video(self, tmp_path):
        raw_dir = tmp_path / "raw" / "public"
        raw_dir.mkdir(parents=True, exist_ok=True)
        vid_file = raw_dir / "test_source.mp4"

        fps = 30
        w, h = 64, 64
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(vid_file), fourcc, fps, (w, h))
        for i in range(90):  # 3 seconds
            frame = np.full((h, w, 3), fill_value=(i % 255), dtype=np.uint8)
            out.write(frame)
        out.release()
        return tmp_path, vid_file

    def test_process_segment_execution(self, synthetic_raw_video):
        tmp_path, vid_file = synthetic_raw_video
        raw_root = tmp_path / "raw"
        work_root = tmp_path / "work"

        try:
            ffmpeg_bin = get_ffmpeg_binary()
        except RuntimeError:
            pytest.skip("FFmpeg not available on this machine")

        row = {
            "segment_id": "SEG000001",
            "source_video": "public/test_source.mp4",
            "label": "drowsy",
            "start_time": "00:00:00.500",
            "end_time": "00:00:02.000",  # 1.5s
        }

        # Dry run test first
        rep_dry = process_segment(
            row=row,
            raw_root=raw_root,
            work_root=work_root,
            ffmpeg_bin=ffmpeg_bin,
            dry_run=True,
        )
        assert rep_dry["status"] == "DRY_RUN"

        # Actual cut test
        rep = process_segment(
            row=row,
            raw_root=raw_root,
            work_root=work_root,
            ffmpeg_bin=ffmpeg_bin,
            dry_run=False,
        )
        assert rep["status"] == "SUCCESS"
        out_file = work_root / "segments" / "drowsy" / "SEG000001.mp4"
        assert out_file.exists()
        assert rep["actual_duration"] == pytest.approx(1.5, abs=0.2)

        # Idempotent skip existing test
        rep_skip = process_segment(
            row=row,
            raw_root=raw_root,
            work_root=work_root,
            ffmpeg_bin=ffmpeg_bin,
            overwrite=False,
        )
        assert rep_skip["status"] == "SKIPPED_EXISTING"

