#!/usr/bin/env python3
"""
inventory.py — Quét thư mục raw/ và tạo annotations/inventory.csv.

Ghi các thông tin: source_video (tương đối), fps, số frame, duration,
width, height, codec, dung lượng, sha1 (tùy chọn --hash).

Ví dụ sử dụng:
    python driver_behavior_dataset_v01/scripts/inventory.py
    python driver_behavior_dataset_v01/scripts/inventory.py --hash
    python driver_behavior_dataset_v01/scripts/inventory.py --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from dbm.logging_utils import setup_logging
from dbm.paths import DatasetPaths

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


# ---------------------------------------------------------------------------
# Video probing
# ---------------------------------------------------------------------------

def probe_video(video_path: Path) -> dict:
    """Extract video metadata using cv2.VideoCapture.

    Parameters
    ----------
    video_path:
        Absolute path to the video file.

    Returns
    -------
    dict
        Keys: fps, num_frames, duration, width, height, codec, size_bytes.
        Values may be None if the file cannot be opened.
    """
    try:
        import cv2  # type: ignore
    except ImportError:
        logger.error("opencv-python-headless is required for inventory.py")
        raise

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Cannot open video: %s", video_path)
        return {
            "fps": None, "num_frames": None, "duration": None,
            "width": None, "height": None, "codec": None,
            "size_bytes": video_path.stat().st_size if video_path.exists() else None,
        }

    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00")
    cap.release()

    duration = (num_frames / fps) if fps and fps > 0 else None

    return {
        "fps": round(fps, 3) if fps else None,
        "num_frames": num_frames,
        "duration": round(duration, 3) if duration else None,
        "width": width,
        "height": height,
        "codec": codec,
        "size_bytes": video_path.stat().st_size,
    }


def sha1_hash(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute SHA-1 hash of a file."""
    h = hashlib.sha1()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Main inventory logic
# ---------------------------------------------------------------------------

def build_inventory(
    raw_root: Path,
    compute_hash: bool = False,
) -> list[dict]:
    """Recursively scan raw_root for video files and probe each one.

    Parameters
    ----------
    raw_root:
        Root directory to scan (raw/).
    compute_hash:
        If True, compute SHA-1 hash of each video file.

    Returns
    -------
    list[dict]
        List of dicts, one per video file.
    """
    records = []
    video_files = sorted(
        p for p in raw_root.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS
    )

    logger.info("Found %d video files in %s", len(video_files), raw_root)

    for vid in video_files:
        rel = vid.relative_to(raw_root)
        source_video = str(rel).replace("\\", "/")

        logger.debug("Probing: %s", source_video)
        info = probe_video(vid)
        info["source_video"] = source_video

        if compute_hash:
            info["sha1"] = sha1_hash(vid)
        else:
            info["sha1"] = None

        records.append(info)

    return records


def print_summary(records: list[dict]) -> None:
    """Print summary statistics to the logger."""
    if not records:
        logger.info("No videos found.")
        return

    total_dur = sum(r["duration"] or 0 for r in records)
    logger.info("=== Inventory Summary ===")
    logger.info("Total videos : %d", len(records))
    logger.info("Total duration: %.1f s (%.1f min)", total_dur, total_dur / 60)

    # FPS distribution
    fps_dist: dict = defaultdict(int)
    for r in records:
        fps_dist[r["fps"]].append(r["source_video"]) if isinstance(fps_dist[r["fps"]], list) else None
        if not isinstance(fps_dist[r["fps"]], list):
            fps_dist[r["fps"]] = 1
        else:
            fps_dist[r["fps"]].append(r["source_video"])
    fps_counts: dict[float, int] = defaultdict(int)
    for r in records:
        fps_counts[r["fps"]] += 1
    logger.info("FPS distribution: %s", dict(fps_counts))

    # Unusual FPS
    for r in records:
        if r["fps"] and r["fps"] not in (24.0, 25.0, 29.97, 30.0, 60.0):
            logger.warning("Unusual FPS %.2f: %s", r["fps"], r["source_video"])

    # Duplicate detection by sha1
    sha1_map: dict[str, list[str]] = defaultdict(list)
    for r in records:
        if r.get("sha1"):
            sha1_map[r["sha1"]].append(r["source_video"])
    for sha1, vids in sha1_map.items():
        if len(vids) > 1:
            logger.warning("Duplicate videos (sha1=%s): %s", sha1[:8], vids)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="inventory.py",
        description=(
            "Quét thư mục raw/ tìm video và tạo annotations/inventory.csv.\n"
            "Ghi: source_video, fps, num_frames, duration, width, height, codec, size_bytes, sha1."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--root", type=Path, default=None,
                        help="Dataset root. Mặc định: DatasetPaths.")
    parser.add_argument("--hash", action="store_true",
                        help="Tính SHA-1 hash để phát hiện video trùng (chậm hơn).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Không ghi file, chỉ in thông tin.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Ghi đè inventory.csv đã tồn tại.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    import pandas as pd

    paths = DatasetPaths(dataset_root=args.root) if args.root else DatasetPaths()
    raw_root = paths.raw
    output = paths.inventory_csv

    if not raw_root.exists():
        logger.error("raw/ directory not found: %s", raw_root)
        return 1

    records = build_inventory(raw_root, compute_hash=args.hash)
    print_summary(records)

    if args.dry_run:
        logger.info("[DRY-RUN] Would write %d records to: %s", len(records), output)
        return 0

    if output.exists() and not args.overwrite:
        logger.warning("inventory.csv already exists (use --overwrite): %s", output)
        return 0

    df = pd.DataFrame(records, columns=[
        "source_video", "fps", "num_frames", "duration",
        "width", "height", "codec", "size_bytes", "sha1",
    ])
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False, encoding="utf-8")
    logger.info("Inventory written to: %s (%d rows)", output, len(df))
    return 0


if __name__ == "__main__":
    sys.exit(main())

