"""
src/dbm/data/metadata.py — Metadata loading and validation utilities.

Shared by scripts (validate_metadata.py) and tests.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from dbm.labels import valid_label_names
from dbm.timecode import parse_time

logger = logging.getLogger(__name__)

# Required columns in the exact order specified in prompt.md
REQUIRED_COLUMNS = [
    "segment_id",
    "source",
    "source_video",
    "subject_id",
    "start_time",
    "end_time",
    "duration",
    "label",
    "camera_view",
    "lighting",
    "glasses",
    "occlusion",
    "quality",
    "notes",
]

SEGMENT_ID_PATTERN = re.compile(r"^SEG\d{6}$")


@dataclass
class ValidationIssue:
    """A single validation error or warning."""

    level: str        # "ERROR" or "WARNING"
    row: Optional[int]  # 1-based row number in the spreadsheet, or None for file-level
    column: Optional[str]
    message: str

    def __str__(self) -> str:
        loc = f"row={self.row}" if self.row is not None else "file"
        col = f", col={self.column}" if self.column else ""
        return f"[{self.level}] ({loc}{col}) {self.message}"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_metadata(path: Path) -> pd.DataFrame:
    """Load metadata from an xlsx or csv file.

    Parameters
    ----------
    path:
        Path to metadata.xlsx or metadata.csv.

    Returns
    -------
    pd.DataFrame
        Raw dataframe (columns not validated yet).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path, sheet_name="metadata", dtype=str)
    elif suffix == ".csv":
        df = pd.read_csv(path, dtype=str)
    else:
        raise ValueError(f"Unsupported metadata file format: {suffix!r}")

    # Strip whitespace from string cells
    df = df.apply(lambda col: col.str.strip() if col.dtype == object else col)
    # Replace empty strings with NaN
    df = df.replace("", pd.NA)

    logger.debug("Loaded %d rows from %s", len(df), path)
    return df


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_dataframe(
    df: pd.DataFrame,
    schema: dict,
    raw_root: Optional[Path] = None,
    inventory_df: Optional[pd.DataFrame] = None,
) -> list[ValidationIssue]:
    """Validate a metadata DataFrame and return a list of issues.

    Parameters
    ----------
    df:
        Raw metadata DataFrame as loaded by load_metadata().
    schema:
        Parsed schema.yaml dict (for valid enum values and thresholds).
    raw_root:
        Root directory for raw videos. Used to check source_video existence.
        If None, existence check is skipped.
    inventory_df:
        Pre-loaded inventory CSV. Used to check video duration.
        If None, duration check is skipped.

    Returns
    -------
    list[ValidationIssue]
        List of errors and warnings. Empty list = all clear.
    """
    issues: list[ValidationIssue] = []

    # --- 1. Check required columns ---
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        issues.append(ValidationIssue(
            level="ERROR", row=None, column=None,
            message=f"Missing required columns: {missing_cols}",
        ))
        # Cannot proceed with row-level checks
        return issues

    # Pre-compute inventory lookup {source_video -> duration_s}
    inv_lookup: dict[str, float] = {}
    if inventory_df is not None and not inventory_df.empty:
        for _, inv_row in inventory_df.iterrows():
            sv = str(inv_row.get("source_video", "")).strip()
            try:
                dur = float(inv_row.get("duration", 0))
                inv_lookup[sv] = dur
            except (ValueError, TypeError):
                pass

    seen_segment_ids: dict[str, int] = {}  # segment_id -> first row
    # For overlap detection: {source_video -> list of (start, end, label, row)}
    video_segments: dict[str, list[tuple]] = {}
    # For subject consistency: {source_video -> set of subject_ids}
    video_subjects: dict[str, set] = {}

    valid_labels = valid_label_names()
    valid_quality = set(schema.get("quality", ["good", "acceptable", "bad"]))
    valid_camera = set(schema.get("camera_view", []))
    valid_lighting = set(schema.get("lighting", []))
    valid_glasses = set(schema.get("glasses", []))
    valid_occlusion = set(schema.get("occlusion", []))
    min_dur = float(schema.get("min_segment_duration", 1.0))
    max_dur = float(schema.get("max_segment_duration", 60.0))

    for i, row in df.iterrows():
        row_num = int(i) + 2  # 1-based, +1 for header

        seg_id = _str(row, "segment_id")
        source_video = _str(row, "source_video")
        subject_id = _str(row, "subject_id")
        label = _str(row, "label")
        quality = _str(row, "quality")
        camera_view = _str(row, "camera_view")
        lighting = _str(row, "lighting")
        glasses = _str(row, "glasses")
        occlusion = _str(row, "occlusion")

        # --- segment_id format ---
        if not seg_id:
            issues.append(ValidationIssue(
                level="ERROR", row=row_num, column="segment_id",
                message="segment_id is empty",
            ))
        elif not SEGMENT_ID_PATTERN.match(seg_id):
            issues.append(ValidationIssue(
                level="ERROR", row=row_num, column="segment_id",
                message=f"segment_id {seg_id!r} does not match SEGxxxxxx format",
            ))
        elif seg_id in seen_segment_ids:
            issues.append(ValidationIssue(
                level="ERROR", row=row_num, column="segment_id",
                message=f"Duplicate segment_id {seg_id!r} (first at row {seen_segment_ids[seg_id]})",
            ))
        else:
            seen_segment_ids[seg_id] = row_num

        # --- label ---
        if not label:
            issues.append(ValidationIssue(
                level="ERROR", row=row_num, column="label",
                message="label is empty",
            ))
        elif label not in valid_labels:
            issues.append(ValidationIssue(
                level="ERROR", row=row_num, column="label",
                message=f"Invalid label {label!r}. Valid: {sorted(valid_labels)}",
            ))

        # --- quality ---
        if quality and quality not in valid_quality:
            issues.append(ValidationIssue(
                level="ERROR", row=row_num, column="quality",
                message=f"Invalid quality {quality!r}. Valid: {sorted(valid_quality)}",
            ))

        # --- enum fields ---
        _check_enum(issues, row_num, "camera_view", camera_view, valid_camera)
        _check_enum(issues, row_num, "lighting", lighting, valid_lighting)
        _check_enum(issues, row_num, "glasses", glasses, valid_glasses)
        _check_enum(issues, row_num, "occlusion", occlusion, valid_occlusion)

        # --- subject_id ---
        if not subject_id:
            issues.append(ValidationIssue(
                level="ERROR", row=row_num, column="subject_id",
                message="subject_id is empty",
            ))

        # --- timestamps ---
        start_s = _parse_time_safe(issues, row_num, "start_time", row.get("start_time"))
        end_s = _parse_time_safe(issues, row_num, "end_time", row.get("end_time"))

        if start_s is not None and end_s is not None:
            if start_s >= end_s:
                issues.append(ValidationIssue(
                    level="ERROR", row=row_num, column="start_time",
                    message=f"start_time ({start_s:.3f}s) >= end_time ({end_s:.3f}s)",
                ))
            else:
                seg_dur = end_s - start_s
                if seg_dur < min_dur:
                    issues.append(ValidationIssue(
                        level="WARNING", row=row_num, column="duration",
                        message=f"Segment too short: {seg_dur:.3f}s < {min_dur}s",
                    ))
                if seg_dur > max_dur:
                    issues.append(ValidationIssue(
                        level="WARNING", row=row_num, column="duration",
                        message=f"Segment too long: {seg_dur:.3f}s > {max_dur}s",
                    ))

                # duration mismatch
                dur_cell = _str(row, "duration")
                if dur_cell:
                    try:
                        dur_val = float(dur_cell)
                        if abs(dur_val - seg_dur) > 0.01:
                            issues.append(ValidationIssue(
                                level="WARNING", row=row_num, column="duration",
                                message=(
                                    f"duration cell ({dur_val:.3f}s) differs from "
                                    f"end-start ({seg_dur:.3f}s); will be recalculated"
                                ),
                            ))
                    except (ValueError, TypeError):
                        pass

                # Check against video duration from inventory
                if source_video and source_video in inv_lookup:
                    video_dur = inv_lookup[source_video]
                    if end_s > video_dur + 0.1:  # 100ms tolerance
                        issues.append(ValidationIssue(
                            level="ERROR", row=row_num, column="end_time",
                            message=(
                                f"end_time {end_s:.3f}s exceeds video duration "
                                f"{video_dur:.3f}s for {source_video!r}"
                            ),
                        ))

                # Overlap detection within same video
                if source_video:
                    segs = video_segments.setdefault(source_video, [])
                    for (prev_s, prev_e, prev_label, prev_row) in segs:
                        if start_s < prev_e and end_s > prev_s:
                            # overlapping
                            if label != prev_label:
                                issues.append(ValidationIssue(
                                    level="ERROR", row=row_num, column="start_time",
                                    message=(
                                        f"Segment overlaps with row {prev_row} "
                                        f"({source_video!r}) with different label "
                                        f"({label!r} vs {prev_label!r})"
                                    ),
                                ))
                            else:
                                issues.append(ValidationIssue(
                                    level="WARNING", row=row_num, column="start_time",
                                    message=(
                                        f"Segment overlaps with row {prev_row} "
                                        f"({source_video!r}) with same label {label!r}"
                                    ),
                                ))
                    segs.append((start_s, end_s, label, row_num))

        # --- source_video existence ---
        if source_video:
            if raw_root is not None:
                vid_path = raw_root / source_video
                if not vid_path.exists():
                    issues.append(ValidationIssue(
                        level="ERROR", row=row_num, column="source_video",
                        message=f"Video file not found: {vid_path}",
                    ))

            # Track subject per video
            if subject_id:
                subs = video_subjects.setdefault(source_video, set())
                subs.add(subject_id)

    # --- source_video -> exactly one subject_id ---
    for sv, subs in video_subjects.items():
        if len(subs) > 1:
            issues.append(ValidationIssue(
                level="ERROR", row=None, column="subject_id",
                message=f"source_video {sv!r} has multiple subject_ids: {sorted(subs)}",
            ))

    return issues


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _str(row: pd.Series, col: str) -> str:
    """Return stripped string value or empty string."""
    val = row.get(col, pd.NA)
    if pd.isna(val):
        return ""
    return str(val).strip()


def _parse_time_safe(
    issues: list[ValidationIssue],
    row_num: int,
    col: str,
    val,
) -> Optional[float]:
    """Parse a time value; append an error issue on failure and return None."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        issues.append(ValidationIssue(
            level="ERROR", row=row_num, column=col,
            message=f"{col} is empty",
        ))
        return None
    try:
        return parse_time(val)
    except (ValueError, TypeError) as exc:
        issues.append(ValidationIssue(
            level="ERROR", row=row_num, column=col,
            message=f"Cannot parse {col}={val!r}: {exc}",
        ))
        return None


def _check_enum(
    issues: list[ValidationIssue],
    row_num: int,
    col: str,
    val: str,
    valid: set[str],
) -> None:
    """Append an error if val is non-empty and not in valid set."""
    if val and valid and val not in valid:
        issues.append(ValidationIssue(
            level="ERROR", row=row_num, column=col,
            message=f"Invalid {col}={val!r}. Valid: {sorted(valid)}",
        ))


# ---------------------------------------------------------------------------
# Post-validation: compute duration and export
# ---------------------------------------------------------------------------

def compute_duration(df: pd.DataFrame) -> pd.DataFrame:
    """Recalculate the duration column from end_time - start_time.

    Parameters
    ----------
    df:
        DataFrame with start_time and end_time columns already parsed
        (should be called after validation passes).

    Returns
    -------
    pd.DataFrame
        Copy of df with duration recalculated.
    """
    df = df.copy()
    durations = []
    for _, row in df.iterrows():
        try:
            s = parse_time(row["start_time"])
            e = parse_time(row["end_time"])
            durations.append(round(e - s, 3))
        except Exception:
            durations.append(pd.NA)
    df["duration"] = durations
    return df


def assign_segment_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Assign segment_ids to rows where segment_id is empty.

    IDs are sequential starting from the maximum existing ID + 1.

    Parameters
    ----------
    df:
        Raw metadata DataFrame.

    Returns
    -------
    pd.DataFrame
        Copy with missing segment_ids filled in.
    """
    df = df.copy()

    # Find max existing numeric ID
    max_id = 0
    for val in df["segment_id"].dropna():
        val = str(val).strip()
        if SEGMENT_ID_PATTERN.match(val):
            num = int(val[3:])
            if num > max_id:
                max_id = num

    next_id = max_id + 1
    for i, row in df.iterrows():
        val = str(row.get("segment_id", "")).strip() if pd.notna(row.get("segment_id")) else ""
        if not val:
            df.at[i, "segment_id"] = f"SEG{next_id:06d}"
            next_id += 1

    return df

