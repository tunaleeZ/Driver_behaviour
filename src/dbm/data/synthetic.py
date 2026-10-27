"""Deterministic synthetic artifacts for tests and pipeline smoke runs."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from dbm.labels import label_names

logger = logging.getLogger(__name__)


def generate_synthetic_dataset(root: str | Path, samples_per_class: int = 3,
                               frames_per_segment: int = 12, fps: int = 8,
                               size: int = 128, seed: int = 42) -> Path:
    """Create feature NPZs, JPG frames, videos and split metadata below root.

    Artifacts are marked ``synthetic=true`` and belong only in test/smoke dirs.
    Each class has one subject per split, avoiding subject leakage.
    """
    if samples_per_class < 3 or frames_per_segment < 1 or size < 16 or fps <= 0:
        raise ValueError("Need >=3 samples/class, positive frame count/FPS, and size >=16")
    root = Path(root)
    feature_root = root / "features" / "landmarks"
    frame_root = root / "processed" / "frames"
    video_root = root / "videos"
    annotation_root = root / "annotations"
    for directory in (feature_root, frame_root, video_root, annotation_root):
        directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    metadata, split_rows = [], []
    split_names = ["train", "val", "test"]
    for class_index, label in enumerate(label_names()):
        for sample_index in range(samples_per_class):
            sid = f"SYN_{class_index:02d}_{sample_index:03d}"
            segment_id = f"SYN{class_index:02d}{sample_index:04d}"
            split = split_names[sample_index % 3]
            subject_id = f"{sid}_SUBJECT"
            split_rows.append({"subject_id": subject_id, "split": split, "synthetic": True})
            metadata.append({"segment_id": segment_id, "subject_id": subject_id, "label": label,
                             "quality": "good", "source_video": f"{segment_id}.mp4",
                             "synthetic": True, "fps": fps})
            # Class code dominates modest seeded noise so a small sequence model can overfit.
            geo = np.zeros((frames_per_segment, 7), dtype=np.float32)
            geo[:, class_index % 4] = 1.0 + class_index * 0.15
            geo += rng.normal(0, 0.015, geo.shape).astype(np.float32)
            # Match the 478-point MediaPipe topology so configured landmark indices work.
            lm = np.full((frames_per_segment, 478, 2), 0.1 + class_index * 0.12, dtype=np.float32)
            lm += rng.normal(0, 0.005, lm.shape).astype(np.float32)
            np.savez_compressed(feature_root / f"{segment_id}.npz", timestamps_ms=np.arange(frames_per_segment, dtype=np.int64) * (1000 // fps),
                                valid=np.ones(frames_per_segment, dtype=bool), fps=np.float32(fps), geo=geo,
                                lm_sel=lm, synthetic=np.asarray(True))
            frames_dir = frame_root / segment_id
            frames_dir.mkdir(parents=True, exist_ok=True)
            color = np.zeros(3, dtype=np.uint8)
            color[class_index % 3] = 255
            writer = cv2.VideoWriter(str(video_root / f"{segment_id}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                     float(fps), (size, size))
            if not writer.isOpened():
                raise RuntimeError("OpenCV could not initialize a video writer (mp4v)")
            try:
                for frame_index in range(frames_per_segment):
                    frame = np.full((size, size, 3), int(class_index * 28), dtype=np.uint8)
                    frame[8:32, 8 + (frame_index % max(1, size - 40)):32 + (frame_index % max(1, size - 40))] = color
                    cv2.imwrite(str(frames_dir / f"{frame_index + 1:06d}.jpg"), frame)
                    writer.write(frame)
            finally:
                writer.release()
    pd.DataFrame(metadata).to_csv(annotation_root / "metadata.csv", index=False)
    pd.DataFrame(split_rows).to_csv(annotation_root / "splits.csv", index=False)
    (root / "SYNTHETIC_DATASET.txt").write_text("synthetic=true; tests and smoke runs only\n", encoding="utf-8")
    return root


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Sinh dataset tổng hợp cho test/smoke.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--samples-per-class", type=int, default=3)
    parser.add_argument("--frames-per-segment", type=int, default=12)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    generate_synthetic_dataset(args.root, args.samples_per_class, args.frames_per_segment,
                               args.fps, args.size, args.seed)
    logger.info("Synthetic test data created at %s", args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
