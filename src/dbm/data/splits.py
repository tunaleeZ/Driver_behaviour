"""
src/dbm/data/splits.py — Subject-level train/val/test split utilities.

Split order: Subject -> Train/Val/Test -> Video -> Segment -> Frame -> Window.
Uses a greedy algorithm with seed to approximate 70/20/10 ratio by total seconds,
while keeping each subject in exactly one split.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

import pandas as pd

from dbm.labels import valid_label_names

logger = logging.getLogger(__name__)

SPLIT_NAMES = ("train", "val", "test")
DEFAULT_RATIOS = {"train": 0.70, "val": 0.20, "test": 0.10}


def make_splits(
    metadata_df: pd.DataFrame,
    ratios: Optional[dict[str, float]] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Assign each subject to a train/val/test split.

    Parameters
    ----------
    metadata_df:
        Validated metadata DataFrame (quality=bad already filtered out by caller).
    ratios:
        Dict with keys "train", "val", "test" summing to 1.0.
        Defaults to 70/20/10.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [subject_id, split].

    Raises
    ------
    ValueError
        If ratios don't sum to approximately 1.0 or contain unknown keys.
    """
    if ratios is None:
        ratios = DEFAULT_RATIOS.copy()

    if abs(sum(ratios.values()) - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {sum(ratios.values())}")

    # Filter quality=bad
    df = metadata_df[metadata_df["quality"].str.lower() != "bad"].copy()

    # Compute total duration per subject (in seconds)
    subject_durations: dict[str, float] = defaultdict(float)
    for _, row in df.iterrows():
        try:
            dur = float(row.get("duration", 0) or 0)
        except (ValueError, TypeError):
            dur = 0.0
        sid = str(row.get("subject_id", "")).strip()
        if sid:
            subject_durations[sid] += dur

    if not subject_durations:
        raise ValueError("No valid subjects found in metadata after filtering quality=bad")

    subjects = sorted(subject_durations.keys())
    total_dur = sum(subject_durations.values())

    # Shuffle subjects with seed
    rng = random.Random(seed)
    rng.shuffle(subjects)

    # Greedy assignment: fill each split bucket up to its target
    target_durs = {s: ratios[s] * total_dur for s in SPLIT_NAMES}
    assigned: dict[str, str] = {}
    current: dict[str, float] = {s: 0.0 for s in SPLIT_NAMES}

    for subject in subjects:
        dur = subject_durations[subject]
        # Pick the split that is most under-filled relative to its target
        best = min(
            SPLIT_NAMES,
            key=lambda s: current[s] / (target_durs[s] + 1e-9),
        )
        assigned[subject] = best
        current[best] += dur

    # Log statistics
    for split in SPLIT_NAMES:
        split_subjects = [s for s, sp in assigned.items() if sp == split]
        split_dur = sum(subject_durations[s] for s in split_subjects)
        pct = 100 * split_dur / (total_dur + 1e-9)
        logger.info(
            "Split %s: %d subjects, %.1f s (%.1f%%)",
            split, len(split_subjects), split_dur, pct,
        )

    result = pd.DataFrame(
        [{"subject_id": s, "split": sp} for s, sp in assigned.items()]
    )
    return result.sort_values("subject_id").reset_index(drop=True)


def validate_splits(
    splits_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
) -> list[str]:
    """Assert data-leakage invariants hold.

    Parameters
    ----------
    splits_df:
        DataFrame with [subject_id, split].
    metadata_df:
        Metadata DataFrame.

    Returns
    -------
    list[str]
        List of error messages. Empty = all assertions pass.
    """
    errors: list[str] = []

    subject_to_split = dict(zip(splits_df["subject_id"], splits_df["split"]))

    # Build segment -> split mapping via subject
    segment_splits: dict[str, set[str]] = defaultdict(set)
    video_splits: dict[str, set[str]] = defaultdict(set)
    subject_splits: dict[str, set[str]] = defaultdict(set)

    for _, row in metadata_df.iterrows():
        sid = str(row.get("subject_id", "")).strip()
        seg_id = str(row.get("segment_id", "")).strip()
        source_video = str(row.get("source_video", "")).strip()
        split = subject_to_split.get(sid, "UNKNOWN")

        if split == "UNKNOWN":
            continue

        subject_splits[sid].add(split)
        if seg_id:
            segment_splits[seg_id].add(split)
        if source_video:
            video_splits[source_video].add(split)

    # No subject in two splits
    for sid, splits in subject_splits.items():
        if len(splits) > 1:
            errors.append(f"Subject {sid!r} appears in multiple splits: {splits}")

    # No source_video in two splits
    for sv, splits in video_splits.items():
        if len(splits) > 1:
            errors.append(f"source_video {sv!r} appears in multiple splits: {splits}")

    # No segment in two splits
    for seg_id, splits in segment_splits.items():
        if len(splits) > 1:
            errors.append(f"segment {seg_id!r} appears in multiple splits: {splits}")

    return errors


def compute_split_stats(
    splits_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per-split statistics: subjects, segments, seconds per label.

    Parameters
    ----------
    splits_df:
        DataFrame with [subject_id, split].
    metadata_df:
        Metadata DataFrame.

    Returns
    -------
    pd.DataFrame
        Statistics table.
    """
    subject_to_split = dict(zip(splits_df["subject_id"], splits_df["split"]))
    valid_labels = sorted(valid_label_names())

    rows = []
    for split in SPLIT_NAMES:
        split_subjects = {s for s, sp in subject_to_split.items() if sp == split}
        split_meta = metadata_df[metadata_df["subject_id"].isin(split_subjects)]
        split_meta = split_meta[split_meta["quality"].str.lower() != "bad"]

        row: dict = {"split": split, "n_subjects": len(split_subjects)}
        row["n_segments"] = len(split_meta)
        row["total_seconds"] = split_meta["duration"].apply(
            lambda x: float(x) if pd.notna(x) else 0.0
        ).sum()

        for label in valid_labels:
            label_meta = split_meta[split_meta["label"] == label]
            row[f"n_{label}"] = len(label_meta)
            row[f"s_{label}"] = label_meta["duration"].apply(
                lambda x: float(x) if pd.notna(x) else 0.0
            ).sum()

        rows.append(row)

    return pd.DataFrame(rows)

