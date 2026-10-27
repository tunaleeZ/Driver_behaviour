#!/usr/bin/env python3
"""
validate_metadata.py — Validate annotations/metadata.xlsx and export metadata.csv.

Checks:
  - Required columns and exact names.
  - segment_id format (SEG000001...) and uniqueness.
  - label in 5 valid classes.
  - quality and other enum values valid against schema.yaml.
  - start < end.
  - end <= video duration (checked via inventory or probe).
  - source_video exists in raw/.
  - Segment overlap in the same video: ERROR if different label, WARNING if same label.
  - Segment duration warnings (< 1s or > 60s, configurable in schema.yaml).
  - subject_id non-empty.
  - One source_video belongs to exactly one subject_id.

If valid:
  - Exports annotations/metadata.csv (UTF-8, duration recalculated).
  - Prints summary statistics by label, quality, subject, camera_view, lighting.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

import pandas as pd

from dbm.config import load_config
from dbm.data.metadata import (
    assign_segment_ids,
    compute_duration,
    load_metadata,
    validate_dataframe,
)
from dbm.logging_utils import setup_logging
from dbm.paths import DatasetPaths

logger = logging.getLogger(__name__)


def print_statistics(df: pd.DataFrame) -> None:
    """Print dataset statistics by label, quality, subject, and conditions."""
    print("\n" + "=" * 60)
    print("METADATA SUMMARY STATISTICS")
    print("=" * 60)

    # Convert duration to float for calculations
    df = df.copy()
    df["duration_num"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0.0)

    print(f"\nTotal Segments: {len(df)}")
    print(f"Total Duration: {df['duration_num'].sum():.2f} seconds ({df['duration_num'].sum() / 60:.2f} mins)")

    # By Label
    print("\n[Distribution by Label]")
    label_stats = df.groupby("label")["duration_num"].agg(count="count", total_seconds="sum")
    print(label_stats.to_string())

    # By Quality
    print("\n[Distribution by Quality]")
    quality_stats = df.groupby("quality")["duration_num"].agg(count="count", total_seconds="sum")
    print(quality_stats.to_string())

    # By Subject
    print("\n[Distribution by Subject]")
    subject_stats = df.groupby("subject_id")["duration_num"].agg(count="count", total_seconds="sum")
    print(subject_stats.to_string())

    # By Camera View
    if "camera_view" in df.columns:
        print("\n[Distribution by Camera View]")
        cam_stats = df.groupby("camera_view")["duration_num"].agg(count="count", total_seconds="sum")
        print(cam_stats.to_string())

    # By Lighting
    if "lighting" in df.columns:
        print("\n[Distribution by Lighting]")
        light_stats = df.groupby("lighting")["duration_num"].agg(count="count", total_seconds="sum")
        print(light_stats.to_string())

    print("=" * 60 + "\n")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="validate_metadata.py",
        description="Kiểm tra hợp lệ metadata (xlsx/csv) và xuất metadata.csv.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--root", type=Path, default=None,
                        help="Thư mục gốc dataset (mặc định theo DatasetPaths).")
    parser.add_argument("--input", type=Path, default=None,
                        help="Đường dẫn file metadata (xlsx hoặc csv). Mặc định: annotations/metadata.xlsx.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Đường dẫn xuất csv. Mặc định: annotations/metadata.csv.")
    parser.add_argument("--assign-ids", action="store_true",
                        help="Tự động gán segment_id cho các dòng mới chưa có id.")
    parser.add_argument("--skip-file-check", action="store_true",
                        help="Bỏ qua kiểm tra sự tồn tại của file video trên ổ đĩa.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Không ghi metadata.csv, chỉ in kết quả kiểm tra.")
    parser.add_argument("--verbose", action="store_true",
                        help="Hiển thị log chi tiết.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    paths = DatasetPaths(dataset_root=args.root) if args.root else DatasetPaths()
    input_file = args.input or (paths.metadata_xlsx if paths.metadata_xlsx.exists() else paths.metadata_csv)
    output_file = args.output or paths.metadata_csv

    if not input_file.exists():
        logger.error("Metadata input file not found: %s", input_file)
        return 1

    schema_file = _REPO_ROOT / "configs" / "schema.yaml"
    schema = load_config(schema_file) if schema_file.exists() else {}

    # Load inventory if available
    inventory_df = None
    if paths.inventory_csv.exists():
        try:
            inventory_df = pd.read_csv(paths.inventory_csv)
            logger.info("Loaded inventory from %s", paths.inventory_csv)
        except Exception as e:
            logger.warning("Could not read inventory file: %s", e)

    logger.info("Loading metadata from: %s", input_file)
    try:
        df = load_metadata(input_file)
    except Exception as exc:
        logger.error("Failed to read metadata file: %s", exc)
        return 1

    # Filter out empty placeholder or guide rows if any
    # Only keep rows that have at least some data
    df = df.dropna(how="all").reset_index(drop=True)

    if args.assign_ids:
        df = assign_segment_ids(df)
        logger.info("Assigned segment IDs to missing rows.")

    raw_root = None if args.skip_file_check else paths.raw
    issues = validate_dataframe(df, schema=schema, raw_root=raw_root, inventory_df=inventory_df)

    errors = [issue for issue in issues if issue.level == "ERROR"]
    warnings = [issue for issue in issues if issue.level == "WARNING"]

    for warn in warnings:
        logger.warning(str(warn))
    for err in errors:
        logger.error(str(err))

    logger.info("Validation finished: %d errors, %d warnings.", len(errors), len(warnings))

    if errors:
        logger.error("Validation failed! Fix the errors above before continuing.")
        return 1

    # Recalculate duration
    df_clean = compute_duration(df)
    print_statistics(df_clean)

    if args.dry_run:
        logger.info("[DRY-RUN] Validation passed. Skipping write to %s", output_file)
        return 0

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_file, index=False, encoding="utf-8")
    logger.info("Successfully exported validated metadata to: %s", output_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())

