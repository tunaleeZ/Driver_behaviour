#!/usr/bin/env python3
"""
make_splits.py — Subject-independent Train/Val/Test split.

Ratio: 70% Train / 20% Val / 10% Test (subject-independent).
Invariant: Each subject appears in exactly ONE split.
Generates: annotations/splits.csv (subject_id, split) and prints stats.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

import pandas as pd

from dbm.data.splits import compute_split_stats, make_splits, validate_splits
from dbm.logging_utils import setup_logging
from dbm.paths import DatasetPaths

logger = logging.getLogger(__name__)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="make_splits.py",
        description="Chia tập Train/Val/Test theo subject chống data leakage.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--root", type=Path, default=None,
                        help="Thư mục gốc dataset (mặc định theo DatasetPaths).")
    parser.add_argument("--metadata", type=Path, default=None,
                        help="Đường dẫn metadata.csv. Mặc định: annotations/metadata.csv.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Đường dẫn splits.csv. Mặc định: annotations/splits.csv.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed cho chia tập (mặc định: 42).")
    parser.add_argument("--train-ratio", type=float, default=0.70,
                        help="Tỷ lệ tập train (mặc định: 0.70).")
    parser.add_argument("--val-ratio", type=float, default=0.20,
                        help="Tỷ lệ tập val (mặc định: 0.20).")
    parser.add_argument("--test-ratio", type=float, default=0.10,
                        help="Tỷ lệ tập test (mặc định: 0.10).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Chỉ in thống kê chia tập mà không ghi file.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Ghi đè splits.csv nếu đã tồn tại.")
    parser.add_argument("--verbose", action="store_true",
                        help="Hiển thị log chi tiết.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    paths = DatasetPaths(dataset_root=args.root) if args.root else DatasetPaths()
    meta_path = args.metadata or paths.metadata_csv
    output_path = args.output or paths.splits_csv

    if not meta_path.exists():
        logger.error("Metadata CSV not found: %s. Run validate_metadata.py first.", meta_path)
        return 1

    ratios = {
        "train": args.train_ratio,
        "val": args.val_ratio,
        "test": args.test_ratio,
    }
    if abs(sum(ratios.values()) - 1.0) > 1e-5:
        logger.error("Split ratios must sum to 1.0, got %s (sum=%f)", ratios, sum(ratios.values()))
        return 1

    df_meta = pd.read_csv(meta_path)
    logger.info("Loaded metadata with %d rows from %s", len(df_meta), meta_path)

    try:
        splits_df = make_splits(df_meta, ratios=ratios, seed=args.seed)
    except Exception as exc:
        logger.error("Error creating splits: %s", exc)
        return 1

    # Assertions
    leak_errors = validate_splits(splits_df, df_meta)
    if leak_errors:
        for err in leak_errors:
            logger.error("DATA LEAKAGE DETECTED: %s", err)
        return 1

    logger.info("All data leakage assertions PASSED: no subject/video/segment across multiple splits.")

    # Compute and display stats
    stats_df = compute_split_stats(splits_df, df_meta)
    print("\n" + "=" * 60)
    print("SPLIT DISTRIBUTION STATISTICS")
    print("=" * 60)
    print(stats_df.to_string(index=False))
    print("=" * 60 + "\n")

    # Check for empty or scarce classes in val/test
    for _, row in stats_df.iterrows():
        sp = row["split"]
        if sp in ("val", "test"):
            for col in stats_df.columns:
                if col.startswith("n_") and col != "n_subjects" and col != "n_segments":
                    class_name = col[2:]
                    if row[col] == 0:
                        logger.warning("Class '%s' has 0 samples in %s split!", class_name, sp)
                    elif row[col] < 3:
                        logger.warning("Class '%s' has only %d samples in %s split.", class_name, int(row[col]), sp)

    if args.dry_run:
        logger.info("[DRY-RUN] Skipping write to %s", output_path)
        return 0

    if output_path.exists() and not args.overwrite:
        logger.warning("%s already exists (use --overwrite to replace)", output_path)
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    splits_df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info("Splits saved to: %s", output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

