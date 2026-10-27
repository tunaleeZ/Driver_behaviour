#!/usr/bin/env python3
"""
visualize_landmarks.py — Trực quan hóa landmark và đặc trưng EAR/MAR/Head pose lên video frame.

Vẽ landmark (và các điểm EAR/MAR đã chọn) lên một vài frame ngẫu nhiên của segment,
xuất file PNG để kiểm tra bằng mắt rằng index landmark hoàn toàn chính xác.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

import cv2
import numpy as np

from dbm.config import load_config
from dbm.features.mediapipe_extractor import MediaPipeFaceExtractor, ensure_model
from dbm.logging_utils import setup_logging
from dbm.paths import DatasetPaths

logger = logging.getLogger(__name__)


def draw_landmarks_and_features(
    frame_bgr: np.ndarray,
    landmarks_norm: np.ndarray,
    ear_r_idx: list[int],
    ear_l_idx: list[int],
    mar_idx: list[int],
    pnp_idx: list[int],
    metrics_text: dict,
) -> np.ndarray:
    """Draw key landmark points and text annotations onto the frame."""
    vis = frame_bgr.copy()
    h, w = vis.shape[:2]

    # Convert to pixel coordinates
    pts = np.zeros((len(landmarks_norm), 2), dtype=np.int32)
    for i, p in enumerate(landmarks_norm):
        pts[i] = [int(round(p[0] * w)), int(round(p[1] * h))]

    # 1. Draw all mesh points subtly
    for p in pts:
        cv2.circle(vis, (p[0], p[1]), 1, (180, 180, 180), -1)

    # 2. Draw Right eye EAR points (Cyan)
    for idx in ear_r_idx:
        cv2.circle(vis, tuple(pts[idx]), 3, (255, 255, 0), -1)

    # 3. Draw Left eye EAR points (Green)
    for idx in ear_l_idx:
        cv2.circle(vis, tuple(pts[idx]), 3, (0, 255, 0), -1)

    # 4. Draw Mouth MAR points (Yellow / Orange)
    for idx in mar_idx:
        cv2.circle(vis, tuple(pts[idx]), 3, (0, 165, 255), -1)

    # 5. Draw Head Pose points (Red)
    for idx in pnp_idx:
        cv2.circle(vis, tuple(pts[idx]), 4, (0, 0, 255), -1)

    # 6. Overlay text metrics
    y0 = 30
    dy = 25
    for i, (k, v) in enumerate(metrics_text.items()):
        text = f"{k}: {v}"
        cv2.putText(vis, text, (20, y0 + i * dy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(vis, text, (20, y0 + i * dy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    return vis


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="visualize_landmarks.py",
        description="Trực quan hoá landmark và đặc trưng hình học lên frame ảnh.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--video", type=Path, required=True,
                        help="Đường dẫn file video segment cần visualize.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/visualizations"),
                        help="Thư mục xuất ảnh PNG kết quả.")
    parser.add_argument("--num-frames", type=int, default=5,
                        help="Số frame ngẫu nhiên cần visualize (mặc định: 5).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    setup_logging()

    if not args.video.exists():
        logger.error("Video file does not exist: %s", args.video)
        return 1

    try:
        model_file = ensure_model()
    except Exception as exc:
        logger.error("%s", exc)
        return 1

    extractor = MediaPipeFaceExtractor(model_path=model_file)
    logger.info("Extracting features from: %s", args.video)
    data = extractor.extract_from_video(args.video, save_full=True)

    cap = cv2.VideoCapture(str(args.video))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Pick random valid frames
    valid_indices = np.where(data["valid"])[0].tolist()
    if not valid_indices:
        logger.warning("No face detected in any frame of %s", args.video)
        return 1

    rng = random.Random(args.seed)
    sampled_indices = rng.sample(valid_indices, min(args.num_frames, len(valid_indices)))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pnp_indices = [1, 152, 263, 33, 291, 61]

    for frame_idx in sampled_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame_bgr = cap.read()
        if not ret:
            continue

        lm = data["full_landmarks"][frame_idx].astype(np.float32)
        geo = data["geo"][frame_idx]  # [ear_l, ear_r, ear_avg, mar, yaw, pitch, roll]

        metrics = {
            "Frame": frame_idx,
            "EAR (L/R/Avg)": f"{geo[0]:.2f} / {geo[1]:.2f} / {geo[2]:.2f}",
            "MAR": f"{geo[3]:.2f}",
            "Yaw": f"{geo[4]:.1f} deg",
            "Pitch": f"{geo[5]:.1f} deg",
            "Roll": f"{geo[6]:.1f} deg",
        }

        vis_frame = draw_landmarks_and_features(
            frame_bgr=frame_bgr,
            landmarks_norm=lm,
            ear_r_idx=extractor.right_eye_indices,
            ear_l_idx=extractor.left_eye_indices,
            mar_idx=extractor.mouth_indices,
            pnp_idx=pnp_indices,
            metrics_text=metrics,
        )

        out_name = args.output_dir / f"frame_{frame_idx:05d}.png"
        cv2.imwrite(str(out_name), vis_frame)
        logger.info("Saved visual check to: %s", out_name)

    cap.release()
    logger.info("Visualization complete. Check PNG files in %s", args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

