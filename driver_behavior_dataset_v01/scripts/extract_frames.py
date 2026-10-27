#!/usr/bin/env python3
"""
extract_frames.py — Trích xuất frame hình ảnh từ video segments phục vụ CNN-LSTM.

Trích frame theo target_fps (mặc định 10), resize (letterbox/squash/center_crop)
về store_size (mặc định 128x128), lưu JPG chất lượng 95 vào:
  processed/frames/<segment_id>/000001.jpg
Ghi processed/frames_index.csv.
Tùy chọn --pack lưu thêm file .npy uint8 mỗi segment.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from dbm.config import load_config
from dbm.data.transforms import resize_frame
from dbm.logging_utils import setup_logging
from dbm.paths import DatasetPaths

logger = logging.getLogger(__name__)


def extract_frames_from_video(
    video_path: Path,
    out_dir: Path,
    target_fps: float = 10.0,
    store_size: tuple[int, int] = (128, 128),
    resize_mode: str = "letterbox",
    jpeg_quality: int = 95,
    pack_npy: bool = False,
) -> tuple[int, str]:
    """Extract frames from a video at target_fps and save to out_dir."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if orig_fps <= 0:
        orig_fps = 30.0

    step = max(1.0, orig_fps / target_fps)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    saved_count = 0
    next_sample_frame = 0.0
    packed_frames: list[np.ndarray] = [] if pack_npy else None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx >= int(round(next_sample_frame)):
            resized = resize_frame(frame, target_size=store_size, mode=resize_mode)
            saved_count += 1
            filename = out_dir / f"{saved_count:06d}.jpg"
            cv2.imwrite(str(filename), resized, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])

            if pack_npy:
                # Convert BGR to RGB for pack array
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                packed_frames.append(rgb)

            next_sample_frame += step

        frame_idx += 1

    cap.release()

    if pack_npy and packed_frames:
        npy_path = out_dir.parent / f"{out_dir.name}.npy"
        arr = np.stack(packed_frames, axis=0)  # (N, H, W, 3) uint8
        np.save(npy_path, arr)

    return saved_count, f"{store_size[0]}x{store_size[1]}"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="extract_frames.py",
        description="Trích xuất frames từ video segments cho mô hình CNN-LSTM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--root", type=Path, default=None,
                        help="Thư mục gốc dataset (mặc định: DatasetPaths).")
    parser.add_argument("--work-root", type=Path, default=None,
                        help="Thư mục làm việc cho processed outputs.")
    parser.add_argument("--metadata", type=Path, default=None,
                        help="Đường dẫn metadata.csv.")
    parser.add_argument("--fps", type=float, default=10.0,
                        help="Target FPS trích frame (mặc định: 10.0).")
    parser.add_argument("--size", type=int, nargs=2, default=[128, 128],
                        help="Kích thước lưu (H W), mặc định: 128 128.")
    parser.add_argument("--resize-mode", choices=["letterbox", "squash", "center_crop"], default="letterbox",
                        help="Chế độ resize ảnh (mặc định: letterbox).")
    parser.add_argument("--pack", action="store_true",
                        help="Lưu thêm file .npy uint8 mỗi segment cho Kaggle.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Ghi đè frame đã trích.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Chỉ in thao tác mà không ghi file.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    paths = DatasetPaths(dataset_root=args.root, work_root=args.work_root)
    meta_path = args.metadata or paths.metadata_csv

    if not meta_path.exists():
        logger.error("Metadata CSV not found: %s", meta_path)
        return 1

    df = pd.read_csv(meta_path)
    if "quality" in df.columns:
        df = df[df["quality"].str.lower() != "bad"]

    logger.info("Found %d segments to extract frames.", len(df))

    records = []
    pbar = tqdm(df.iterrows(), total=len(df), desc="Extracting frames")

    for _, row in pbar:
        seg_id = str(row["segment_id"]).strip()
        label = str(row["label"]).strip()
        vid_path = paths.segment_path(label, seg_id)
        out_dir = paths.frames_dir(seg_id)

        if not vid_path.exists():
            logger.warning("Segment video not found: %s", vid_path)
            continue

        if out_dir.exists() and len(list(out_dir.glob("*.jpg"))) > 0 and not args.overwrite:
            saved_count = len(list(out_dir.glob("*.jpg")))
            records.append({
                "segment_id": seg_id,
                "n_frames": saved_count,
                "fps": args.fps,
                "size": f"{args.size[0]}x{args.size[1]}",
            })
            continue

        if args.dry_run:
            continue

        try:
            n_saved, size_str = extract_frames_from_video(
                video_path=vid_path,
                out_dir=out_dir,
                target_fps=args.fps,
                store_size=(args.size[0], args.size[1]),
                resize_mode=args.resize_mode,
                pack_npy=args.pack,
            )
            records.append({
                "segment_id": seg_id,
                "n_frames": n_saved,
                "fps": args.fps,
                "size": size_str,
            })
        except Exception as exc:
            logger.error("Failed extracting frames for %s: %s", seg_id, exc)

    if not args.dry_run:
        index_df = pd.DataFrame(records)
        paths.frames_index_csv.parent.mkdir(parents=True, exist_ok=True)
        index_df.to_csv(paths.frames_index_csv, index=False, encoding="utf-8")
        logger.info("Frames index saved to: %s (%d segments)", paths.frames_index_csv, len(index_df))

    return 0


if __name__ == "__main__":
    sys.exit(main())

