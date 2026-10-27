"""
tests/test_validate_metadata.py — Unit tests for metadata validation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dbm.data.metadata import (
    REQUIRED_COLUMNS,
    assign_segment_ids,
    compute_duration,
    validate_dataframe,
)


@pytest.fixture
def base_schema():
    return {
        "camera_view": ["front", "left", "right", "dashboard", "unknown"],
        "lighting": ["day", "night", "low_light", "mixed", "unknown"],
        "glasses": ["none", "glasses", "sunglasses", "unknown"],
        "occlusion": ["none", "partial", "heavy", "unknown"],
        "quality": ["good", "acceptable", "bad"],
        "min_segment_duration": 1.0,
        "max_segment_duration": 60.0,
    }


@pytest.fixture
def valid_row():
    return {
        "segment_id": "SEG000001",
        "source": "public",
        "source_video": "public/vid1.mp4",
        "subject_id": "PUB_001",
        "start_time": "00:00:02.000",
        "end_time": "00:00:07.000",
        "duration": "5.000",
        "label": "normal",
        "camera_view": "front",
        "lighting": "day",
        "glasses": "none",
        "occlusion": "none",
        "quality": "good",
        "notes": "",
    }


class TestValidateMetadata:
    def test_valid_dataframe_passes(self, base_schema, valid_row):
        df = pd.DataFrame([valid_row])
        issues = validate_dataframe(df, schema=base_schema)
        errors = [i for i in issues if i.level == "ERROR"]
        assert len(errors) == 0

    def test_missing_column_error(self, base_schema, valid_row):
        del valid_row["segment_id"]
        df = pd.DataFrame([valid_row])
        issues = validate_dataframe(df, schema=base_schema)
        errors = [i for i in issues if i.level == "ERROR"]
        assert any("Missing required columns" in e.message for e in errors)

    def test_invalid_segment_id_format(self, base_schema, valid_row):
        valid_row["segment_id"] = "SEG123"  # Not 6 digits
        df = pd.DataFrame([valid_row])
        issues = validate_dataframe(df, schema=base_schema)
        errors = [i for i in issues if i.level == "ERROR"]
        assert any("does not match SEGxxxxxx format" in e.message for e in errors)

    def test_duplicate_segment_id(self, base_schema, valid_row):
        row2 = valid_row.copy()
        row2["start_time"] = "00:00:10.000"
        row2["end_time"] = "00:00:15.000"
        df = pd.DataFrame([valid_row, row2])
        issues = validate_dataframe(df, schema=base_schema)
        errors = [i for i in issues if i.level == "ERROR"]
        assert any("Duplicate segment_id" in e.message for e in errors)

    def test_invalid_label(self, base_schema, valid_row):
        valid_row["label"] = "dancing"
        df = pd.DataFrame([valid_row])
        issues = validate_dataframe(df, schema=base_schema)
        errors = [i for i in issues if i.level == "ERROR"]
        assert any("Invalid label" in e.message for e in errors)

    def test_invalid_quality_enum(self, base_schema, valid_row):
        valid_row["quality"] = "super_good"
        df = pd.DataFrame([valid_row])
        issues = validate_dataframe(df, schema=base_schema)
        errors = [i for i in issues if i.level == "ERROR"]
        assert any("Invalid quality" in e.message for e in errors)

    def test_start_greater_than_end(self, base_schema, valid_row):
        valid_row["start_time"] = "00:00:10.000"
        valid_row["end_time"] = "00:00:05.000"
        df = pd.DataFrame([valid_row])
        issues = validate_dataframe(df, schema=base_schema)
        errors = [i for i in issues if i.level == "ERROR"]
        assert any("start_time" in e.message and ">=" in e.message for e in errors)

    def test_duration_warnings(self, base_schema, valid_row):
        valid_row["start_time"] = "00:00:01.000"
        valid_row["end_time"] = "00:00:01.500"  # 0.5s < 1.0s
        df = pd.DataFrame([valid_row])
        issues = validate_dataframe(df, schema=base_schema)
        warnings = [i for i in issues if i.level == "WARNING"]
        assert any("Segment too short" in w.message for w in warnings)

    def test_end_time_exceeds_video_duration(self, base_schema, valid_row):
        inv_df = pd.DataFrame([{"source_video": "public/vid1.mp4", "duration": 10.0}])
        valid_row["start_time"] = "00:00:05.000"
        valid_row["end_time"] = "00:00:15.000"  # 15s > 10s
        df = pd.DataFrame([valid_row])
        issues = validate_dataframe(df, schema=base_schema, inventory_df=inv_df)
        errors = [i for i in issues if i.level == "ERROR"]
        assert any("exceeds video duration" in e.message for e in errors)

    def test_overlap_different_label_is_error(self, base_schema, valid_row):
        row2 = valid_row.copy()
        row2["segment_id"] = "SEG000002"
        row2["start_time"] = "00:00:04.000"  # overlaps 2s-7s
        row2["end_time"] = "00:00:09.000"
        row2["label"] = "drowsy"  # different label
        df = pd.DataFrame([valid_row, row2])
        issues = validate_dataframe(df, schema=base_schema)
        errors = [i for i in issues if i.level == "ERROR"]
        assert any("overlaps" in e.message and "different label" in e.message for e in errors)

    def test_overlap_same_label_is_warning(self, base_schema, valid_row):
        row2 = valid_row.copy()
        row2["segment_id"] = "SEG000002"
        row2["start_time"] = "00:00:04.000"  # overlaps 2s-7s
        row2["end_time"] = "00:00:09.000"
        row2["label"] = "normal"  # same label
        df = pd.DataFrame([valid_row, row2])
        issues = validate_dataframe(df, schema=base_schema)
        errors = [i for i in issues if i.level == "ERROR"]
        warnings = [i for i in issues if i.level == "WARNING"]
        assert len(errors) == 0
        assert any("overlaps" in w.message and "same label" in w.message for w in warnings)

    def test_multiple_subjects_for_one_video_is_error(self, base_schema, valid_row):
        row2 = valid_row.copy()
        row2["segment_id"] = "SEG000002"
        row2["start_time"] = "00:00:10.000"
        row2["end_time"] = "00:00:15.000"
        row2["subject_id"] = "PUB_002"  # Different subject for same video
        df = pd.DataFrame([valid_row, row2])
        issues = validate_dataframe(df, schema=base_schema)
        errors = [i for i in issues if i.level == "ERROR"]
        assert any("multiple subject_ids" in e.message for e in errors)


class TestHelpers:
    def test_compute_duration(self, valid_row):
        df = pd.DataFrame([valid_row])
        df["duration"] = "0.0"
        df_out = compute_duration(df)
        assert df_out["duration"].iloc[0] == pytest.approx(5.0)

    def test_assign_segment_ids(self, valid_row):
        row1 = valid_row.copy()
        row1["segment_id"] = ""
        row2 = valid_row.copy()
        row2["segment_id"] = "SEG000005"
        row3 = valid_row.copy()
        row3["segment_id"] = None
        df = pd.DataFrame([row1, row2, row3])
        df_out = assign_segment_ids(df)
        # Max existing is SEG000005, so new IDs should start at SEG000006
        assert df_out["segment_id"].iloc[0] == "SEG000006"
        assert df_out["segment_id"].iloc[1] == "SEG000005"
        assert df_out["segment_id"].iloc[2] == "SEG000007"

