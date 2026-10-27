#!/usr/bin/env python3
"""
cut_segments.py — Cut video segments using FFmpeg based on metadata.csv.

Outputs to segments/<label>/<segment_id>.mp4.
Removes audio (-an) for privacy and keeps original video frame rate.
Verifies cut video duration and writes annotations/cut_report.csv.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

import cv2
import pandas as pd

from dbm.logging_utils import setup_logging
from dbm.paths import DatasetPaths
from dbm.timecode import format_time, parse_time

logger = logging.getLogger(__name__)


def get_ffmpeg_binary() -> str:
    """Find FFmpeg binary: PATH first, then imageio-ffmpeg."""
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        raise RuntimeError("FFmpeg not found in PATH and imageio-ffmpeg is not available.")


def build_ffmpeg_cmd(
    ffmpeg_bin: str,
    src_path: Path | str,
    dst_path: Path | str,
    start_time: float | str,
    end_time: float | str,
    fps: Optional[float] = None,
) -> list[str]:
    """Construct the FFmpeg command line arguments list.

    Can be unit tested without executing FFmpeg.
    """
    s_sec = parse_time(start_time)
    e_sec = parse_time(end_time)

    # Use formatted timestamps HH:MM:SS.mmm
    s_str = format_time(s_sec)
    e_str = format_time(e_sec)

    cmd = [
        str(ffmpeg_bin),
        "-y",
        "-ss", s_str,
        "-to", e_str,
        "-i", str(src_path),
        "-an",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-avoid_negative_ts", "make_zero",
    ]
    if fps is not None and fps > 0:
        cmd.extend(["-r", str(fps)])

    cmd.append(str(dst_path))
    return cmd


def verify_cut_video(video_path: Path, expected_duration: float, fps_ref: float = 30.0) -> tuple[bool, float, str]:
    """Check that the output video can be opened and duration matches expected within tolerance.

    Tolerance is max(0.2s, 2 frames).
    Returns: (is_valid, actual_duration, error_message)
    """
    if not video_path.exists():
        return False, 0.0, "Output file does not exist"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False, 0.0, "Cannot open video file with cv2"

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if fps <= 0 or num_frames <= 0:
        return False, 0.0, f"Invalid video stream: frames={num_frames}, fps={fps}"

    actual_duration = num_frames / fps
    tolerance = max(0.2, 2.0 / (fps if fps > 0 else fps_ref))

    if abs(actual_duration - expected_duration) > tolerance:
        return (
            False,
            actual_duration,
            f"Duration mismatch: actual {actual_duration:.3f}s vs expected {expected_duration:.3f}s (tol {tolerance:.3f}s)",
        )

    return True, actual_duration, ""


def process_segment(
    row: dict,
    raw_root: Path,
    work_root: Path,
    ffmpeg_bin: str,
    fps: Optional[float] = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict:
    """Cut and verify a single video segment. Returns report dict."""
    seg_id = str(row["segment_id"]).strip()
    label = str(row["label"]).strip()
    source_video = str(row["source_video"]).strip()
    start_time = row["start_time"]
    end_time = row["end_time"]

    expected_duration = parse_time(end_time) - parse_time(start_time)
    src_file = raw_root / source_video
    dst_dir = work_root / "segments" / label
    dst_file = dst_dir / f"{seg_id}.mp4"

    report = {
        "segment_id": seg_id,
        "label": label,
        "status": "PENDING",
        "actual_duration": 0.0,
        "expected_duration": round(expected_duration, 3),
        "error": "",
    }

    if not src_file.exists():
        report["status"] = "ERROR"
        report["error"] = f"Source video not found: {src_file}"
        return report

    # Check if already exists and valid
    if dst_file.exists() and not overwrite:
        valid, dur, err = verify_cut_video(dst_file, expected_duration)
        if valid:
            report["status"] = "SKIPPED_EXISTING"
            report["actual_duration"] = round(dur, 3)
            return report

    cmd = build_ffmpeg_cmd(
        ffmpeg_bin=ffmpeg_bin,
        src_path=src_file,
        dst_path=dst_file,
        start_time=start_time,
        end_time=end_time,
        fps=fps,
    )

    if dry_run:
        report["status"] = "DRY_RUN"
        report["error"] = " ".join(cmd)
        return report

    dst_dir.mkdir(parents=True, exist_ok=True)

    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        if res.returncode != 0:
            report["status"] = "ERROR"
            report["error"] = f"FFmpeg failed (code {res.returncode}): {res.stderr[-300:] if res.stderr else ''}"
            return report
    except Exception as exc:
        report["status"] = "ERROR"
        report["error"] = str(exc)
        return report

    valid, dur, err = verify_cut_video(dst_file, expected_duration)
    report["actual_duration"] = round(dur, 3)
    if not valid:
        report["status"] = "ERROR"
        report["error"] = err
    else:
        report["status"] = "SUCCESS"

    return report


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cut_segments.py",
        description="Cắt video thành các segment bằng FFmpeg dựa vào metadata.csv.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--root", type=Path, default=None,
                        help="Thư mục gốc dataset (mặc định theo DatasetPaths).")
    parser.add_argument("--work-root", type=Path, default=None,
                        help="Thư mục làm việc cho outputs (mặc định theo DatasetPaths).")
    parser.add_argument("--metadata", type=Path, default=None,
                        help="Đường dẫn metadata.csv. Mặc định: annotations/metadata.csv.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Số luồng xử lý song song (mặc định: 1).")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Lọc chỉ cắt các nhãn này.")
    parser.add_argument("--subjects", nargs="+", default=None,
                        help="Lọc chỉ cắt các subject này.")
    parser.add_argument("--skip-bad", action="store_true", default=True,
                        help="Bỏ qua quality=bad (mặc định: True).")
    parser.add_argument("--no-skip-bad", dest="skip_bad", action="store_false",
                        help="Không bỏ qua quality=bad.")
    parser.add_argument("--fps", type=float, default=None,
                        help="Ép frame rate đầu ra.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Ghi đè segment đã tồn tại.")
    parser.add_argument("--dry-run", action="store_true",
                        help="In lệnh FFmpeg sẽ chạy mà không thực hiện.")
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
        ffmpeg_bin = get_ffmpeg_binary()
        logger.info("Using FFmpeg binary: %s", ffmpeg_bin)
    except Exception as exc:
        logger.error("%s", exc)
        return 1

    df = pd.read_csv(meta_path)
    logger.info("Loaded %d segments from metadata.", len(df))

    if args.skip_bad and "quality" in df.columns:
        init_len = len(df)
        df = df[df["quality"].str.lower() != "bad"]
        logger.info("Filtered out %d 'bad' quality segments. Remaining: %d", init_len - len(df), len(df))

    if args.labels and "label" in df.columns:
        df = df[df["label"].isin(args.labels)]
        logger.info("Filtered by labels %s. Remaining: %d", args.labels, len(df))

    if args.subjects and "subject_id" in df.columns:
        df = df[df["subject_id"].isin(args.subjects)]
        logger.info("Filtered by subjects %s. Remaining: %d", args.subjects, len(df))

    if len(df) == 0:
        logger.warning("No segments to process after filtering.")
        return 0

    rows = df.to_dict(orient="records")
    reports: list[dict] = []

    logger.info("Starting segment cutting with workers=%d...", args.workers)

    if args.workers <= 1:
        for r in rows:
            rep = process_segment(
                row=r,
                raw_root=paths.raw,
                work_root=paths.work_root,
                ffmpeg_bin=ffmpeg_bin,
                fps=args.fps,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )
            reports.append(rep)
            logger.info("Segment %s [%s]: %s %s", rep["segment_id"], rep["label"], rep["status"], rep["error"])
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    process_segment,
                    row=r,
                    raw_root=paths.raw,
                    work_root=paths.work_root,
                    ffmpeg_bin=ffmpeg_bin,
                    fps=args.fps,
                    overwrite=args.overwrite,
                    dry_run=args.dry_run,
                )
                for r in rows
            ]
            for f in concurrent.futures.as_completed(futures):
                rep = f.result()
                reports.append(rep)
                logger.info("Segment %s [%s]: %s %s", rep["segment_id"], rep["label"], rep["status"], rep["error"])

    report_df = pd.DataFrame(reports)
    report_file = paths.cut_report_csv
    if not args.dry_run:
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_df.to_csv(report_file, index=False, encoding="utf-8")
        logger.info("Cut report written to: %s", report_file)

    num_errors = (report_df["status"] == "ERROR").sum()
    num_success = (report_df["status"] == "SUCCESS").sum()
    num_skipped = (report_df["status"] == "SKIPPED_EXISTING").sum()

    logger.info("Summary: %d Success, %d Skipped existing, %d Errors out of %d total segments.",
                num_success, num_skipped, num_errors, len(report_df))

    return 1 if num_errors > 0 else 0


if __name__ == "__main__":
    sys.exit(main())

