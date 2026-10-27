#!/usr/bin/env python3
"""
extract_landmarks.py — Trích xuất đặc trưng MediaPipe Face Landmarker từ segments.

Lưu features/landmarks/<segment_id>.npz gồm:
  - timestamps_ms (N,)
  - valid (N,) bool
  - fps (float)
  - geo (N, 7) [ear_left, ear_right, ear_avg, mar, yaw, pitch, roll]
  - lm_sel (N, K, 2)
  - full_landmarks (N, 478, 3) float16 (nếu --save-full)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import logging
import sys
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

import numpy as np
import pandas as pd
from tqdm import tqdm

from dbm.features.mediapipe_extractor import MediaPipeFaceExtractor, ensure_model
from dbm.logging_utils import setup_logging
from dbm.paths import DatasetPaths

logger = logging.getLogger(__name__)


def process_single_segment(
    seg_id: str,
    video_path: Path,
    out_npz_path: Path,
    extractor: MediaPipeFaceExtractor,
    save_full: bool = False,
    overwrite: bool = False,
) -> dict:
    """Extract features for one video segment and save to .npz."""
    res_dict = {
        "segment_id": seg_id,
        "status": "PENDING",
        "n_frames": 0,
        "valid_ratio": 0.0,
        "error": "",
    }

    if out_npz_path.exists() and not overwrite:
        res_dict["status"] = "SKIPPED_EXISTING"
        return res_dict

    if not video_path.exists():
        res_dict["status"] = "ERROR"
        res_dict["error"] = f"Video file not found: {video_path}"
        return res_dict

    try:
        data = extractor.extract_from_video(video_path, save_full=save_full)
        out_npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_npz_path, **data)

        n_frames = len(data["valid"])
        valid_count = int(np.sum(data["valid"]))
        ratio = valid_count / n_frames if n_frames > 0 else 0.0

        res_dict["status"] = "SUCCESS"
        res_dict["n_frames"] = n_frames
        res_dict["valid_ratio"] = round(ratio, 4)
    except Exception as exc:
        res_dict["status"] = "ERROR"
        res_dict["error"] = str(exc)

    return res_dict


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="extract_landmarks.py",
        description="Trích xuất đặc trưng MediaPipe từ video segments và lưu file .npz.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--root", type=Path, default=None,
                        help="Thư mục gốc dataset (mặc định theo DatasetPaths).")
    parser.add_argument("--work-root", type=Path, default=None,
                        help="Thư mục làm việc ghi features (mặc định theo DatasetPaths).")
    parser.add_argument("--metadata", type=Path, default=None,
                        help="Đường dẫn metadata.csv. Mặc định: annotations/metadata.csv.")
    parser.add_argument("--model-path", type=Path, default=None,
                        help="Đường dẫn file face_landmarker.task. Mặc định: models/face_landmarker.task.")
    parser.add_argument("--save-full", action="store_true",
                        help="Lưu toàn bộ 478 landmark 3D (float16) vào npz.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Ghi đè file npz đã tồn tại.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Số worker xử lý song song.")
    parser.add_argument("--dry-run", action="store_true",
                        help="In thao tác sẽ làm mà không thực thi.")
    parser.add_argument("--verbose", action="store_true",
                        help="Hiển thị log chi tiết.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    paths = DatasetPaths(dataset_root=args.root, work_root=args.work_root)
    meta_path = args.metadata or paths.metadata_csv

    if not meta_path.exists():
        logger.error("Metadata CSV not found: %s. Run validate_metadata.py first.", meta_path)
        return 1

    try:
        model_file = ensure_model(args.model_path)
        logger.info("MediaPipe model verified at: %s", model_file)
    except Exception as exc:
        logger.error("Cannot proceed without MediaPipe model: %s", exc)
        return 1

    extractor = MediaPipeFaceExtractor(model_path=model_file)
    df = pd.read_csv(meta_path)
    # Exclude bad quality segments
    if "quality" in df.columns:
        df = df[df["quality"].str.lower() != "bad"]

    logger.info("Found %d segments to process.", len(df))

    tasks = []
    for _, row in df.iterrows():
        seg_id = str(row["segment_id"]).strip()
        label = str(row["label"]).strip()
        vid_path = paths.segment_path(label, seg_id)
        out_path = paths.landmark_path(seg_id)
        tasks.append((seg_id, vid_path, out_path, row.to_dict()))

    if args.dry_run:
        logger.info("[DRY-RUN] Would process %d segments to %s", len(tasks), paths.landmarks)
        return 0

    results = []
    pbar = tqdm(tasks, desc="Extracting landmarks")
    for seg_id, vid_path, out_path, meta_row in pbar:
        res = process_single_segment(
            seg_id=seg_id,
            video_path=vid_path,
            out_npz_path=out_path,
            extractor=extractor,
            save_full=args.save_full,
            overwrite=args.overwrite,
        )
        res["lighting"] = meta_row.get("lighting", "unknown")
        res["glasses"] = meta_row.get("glasses", "unknown")
        results.append(res)
        pbar.set_postfix({"id": seg_id, "status": res["status"]})

    res_df = pd.DataFrame(results)
    success_df = res_df[res_df["status"] == "SUCCESS"]

    print("\n" + "=" * 60)
    print("LANDMARK EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"Total Segments Processed: {len(res_df)}")
    print(f"Success: {(res_df['status'] == 'SUCCESS').sum()}")
    print(f"Skipped Existing: {(res_df['status'] == 'SKIPPED_EXISTING').sum()}")
    print(f"Errors: {(res_df['status'] == 'ERROR').sum()}")

    if len(success_df) > 0:
        print(f"Overall Mean Valid Face Ratio: {success_df['valid_ratio'].mean():.2%}")
        if "lighting" in success_df.columns:
            print("\n[Valid Ratio by Lighting]")
            print(success_df.groupby("lighting")["valid_ratio"].mean().to_string())
        if "glasses" in success_df.columns:
            print("\n[Valid Ratio by Glasses]")
            print(success_df.groupby("glasses")["valid_ratio"].mean().to_string())
    print("=" * 60 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())

