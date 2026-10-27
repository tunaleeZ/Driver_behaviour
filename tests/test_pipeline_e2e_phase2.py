"""
tests/test_pipeline_e2e_phase2.py — End-to-end pipeline test for Phase 2.

Tests the full sequence:
  1. init_dataset creates directory structure.
  2. Synthetic video created in raw/public/vid_001.mp4.
  3. inventory.py probes raw videos and generates annotations/inventory.csv.
  4. make_metadata_template.py generates metadata.xlsx template.
  5. User fills metadata with valid segments.
  6. validate_metadata.py validates metadata, checks inventory, and exports metadata.csv.
  7. cut_segments.py cuts video segments using FFmpeg and writes cut_report.csv.
  8. make_splits.py divides subjects into train/val/test and writes splits.csv.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "driver_behavior_dataset_v01" / "scripts"))

from cut_segments import get_ffmpeg_binary
from init_dataset import init_dataset


def create_synthetic_video(path: Path, duration_s: float = 4.0, fps: int = 30) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = 64, 64
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    num_frames = int(duration_s * fps)
    for i in range(num_frames):
        frame = np.full((h, w, 3), fill_value=(i * 2) % 255, dtype=np.uint8)
        out.write(frame)
    out.release()


def test_phase2_end_to_end_pipeline(tmp_path):
    dataset_root = tmp_path / "dataset"
    init_dataset(dataset_root)

    # 1. Create 2 synthetic raw videos for 2 different subjects
    raw_v1 = dataset_root / "raw" / "public" / "video_s1.mp4"
    raw_v2 = dataset_root / "raw" / "public" / "video_s2.mp4"
    create_synthetic_video(raw_v1, duration_s=4.0, fps=30)
    create_synthetic_video(raw_v2, duration_s=4.0, fps=30)

    # 2. Run inventory.py
    cmd_inv = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "driver_behavior_dataset_v01" / "scripts" / "inventory.py"),
        "--root", str(dataset_root),
        "--hash",
    ]
    res_inv = subprocess.run(cmd_inv, capture_output=True, text=True, check=False)
    assert res_inv.returncode == 0, f"inventory.py failed: {res_inv.stderr}"
    inv_csv = dataset_root / "annotations" / "inventory.csv"
    assert inv_csv.exists()
    df_inv = pd.read_csv(inv_csv)
    assert len(df_inv) == 2

    # 3. Create metadata with 3 valid segments
    # Note: 2 segments from video_s1 (S001), 1 segment from video_s2 (S002)
    meta_rows = [
        {
            "segment_id": "SEG000001",
            "source": "public",
            "source_video": "public/video_s1.mp4",
            "subject_id": "S001",
            "start_time": "00:00:00.500",
            "end_time": "00:00:02.000",
            "duration": "1.500",
            "label": "normal",
            "camera_view": "front",
            "lighting": "day",
            "glasses": "none",
            "occlusion": "none",
            "quality": "good",
            "notes": "test segment 1",
        },
        {
            "segment_id": "SEG000002",
            "source": "public",
            "source_video": "public/video_s1.mp4",
            "subject_id": "S001",
            "start_time": "00:00:02.200",
            "end_time": "00:00:03.700",
            "duration": "1.500",
            "label": "drowsy",
            "camera_view": "front",
            "lighting": "day",
            "glasses": "none",
            "occlusion": "none",
            "quality": "good",
            "notes": "test segment 2",
        },
        {
            "segment_id": "SEG000003",
            "source": "public",
            "source_video": "public/video_s2.mp4",
            "subject_id": "S002",
            "start_time": "00:00:01.000",
            "end_time": "00:00:03.000",
            "duration": "2.000",
            "label": "drinking",
            "camera_view": "front",
            "lighting": "day",
            "glasses": "none",
            "occlusion": "none",
            "quality": "good",
            "notes": "test segment 3",
        },
    ]
    meta_df = pd.DataFrame(meta_rows)
    meta_raw_csv = dataset_root / "annotations" / "metadata_raw.csv"
    meta_df.to_csv(meta_raw_csv, index=False)

    # 4. Run validate_metadata.py
    meta_out_csv = dataset_root / "annotations" / "metadata.csv"
    cmd_val = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "driver_behavior_dataset_v01" / "scripts" / "validate_metadata.py"),
        "--root", str(dataset_root),
        "--input", str(meta_raw_csv),
        "--output", str(meta_out_csv),
    ]
    res_val = subprocess.run(cmd_val, capture_output=True, text=True, check=False)
    assert res_val.returncode == 0, f"validate_metadata.py failed: {res_val.stderr}"
    assert meta_out_csv.exists()

    # 5. Run cut_segments.py
    cmd_cut = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "driver_behavior_dataset_v01" / "scripts" / "cut_segments.py"),
        "--root", str(dataset_root),
        "--metadata", str(meta_out_csv),
    ]
    res_cut = subprocess.run(cmd_cut, capture_output=True, text=True, check=False)
    assert res_cut.returncode == 0, f"cut_segments.py failed: {res_cut.stderr}"

    # Verify cut segments exist
    seg1 = dataset_root / "segments" / "normal" / "SEG000001.mp4"
    seg2 = dataset_root / "segments" / "drowsy" / "SEG000002.mp4"
    seg3 = dataset_root / "segments" / "drinking" / "SEG000003.mp4"
    assert seg1.exists(), "SEG000001.mp4 missing"
    assert seg2.exists(), "SEG000002.mp4 missing"
    assert seg3.exists(), "SEG000003.mp4 missing"

    # Verify cut report
    cut_rep = dataset_root / "annotations" / "cut_report.csv"
    assert cut_rep.exists()
    df_rep = pd.read_csv(cut_rep)
    assert (df_rep["status"] == "SUCCESS").all()

    # 6. Run make_splits.py
    splits_csv = dataset_root / "annotations" / "splits.csv"
    cmd_split = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "driver_behavior_dataset_v01" / "scripts" / "make_splits.py"),
        "--root", str(dataset_root),
        "--metadata", str(meta_out_csv),
        "--output", str(splits_csv),
    ]
    res_split = subprocess.run(cmd_split, capture_output=True, text=True, check=False)
    assert res_split.returncode == 0, f"make_splits.py failed: {res_split.stderr}"
    assert splits_csv.exists()

    df_splits = pd.read_csv(splits_csv)
    assert len(df_splits) == 2
    assert set(df_splits["subject_id"]) == {"S001", "S002"}
