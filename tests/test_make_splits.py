"""
tests/test_make_splits.py — Unit tests for subject-independent splits.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dbm.data.splits import compute_split_stats, make_splits, validate_splits


@pytest.fixture
def mock_dataset_metadata():
    """Generates synthetic metadata with 10 subjects and various labels and durations."""
    rows = []
    labels = ["normal", "drowsy", "sleepy", "drinking", "phone"]
    for s_idx in range(1, 11):
        subj = f"S{s_idx:03d}"
        for c_idx, label in enumerate(labels):
            rows.append({
                "segment_id": f"SEG{s_idx:02d}{c_idx:02d}01",
                "source": "mock",
                "source_video": f"mock/vid_{subj}.mp4",
                "subject_id": subj,
                "start_time": "00:00:00.000",
                "end_time": "00:00:10.000",
                "duration": 10.0,
                "label": label,
                "quality": "good",
            })
    # Add one bad quality segment
    rows.append({
        "segment_id": "SEG999999",
        "source": "mock",
        "source_video": "mock/vid_bad.mp4",
        "subject_id": "S999",
        "start_time": "00:00:00.000",
        "end_time": "00:00:10.000",
        "duration": 10.0,
        "label": "normal",
        "quality": "bad",
    })
    return pd.DataFrame(rows)


class TestMakeSplits:
    def test_no_data_leakage(self, mock_dataset_metadata):
        splits_df = make_splits(mock_dataset_metadata, seed=42)
        errors = validate_splits(splits_df, mock_dataset_metadata)
        assert len(errors) == 0

    def test_each_subject_assigned_once(self, mock_dataset_metadata):
        splits_df = make_splits(mock_dataset_metadata, seed=42)
        assert len(splits_df) == 10  # S001..S010 (S999 excluded because quality=bad)
        assert splits_df["subject_id"].nunique() == 10
        assert set(splits_df["split"].unique()).issubset({"train", "val", "test"})

    def test_quality_bad_excluded(self, mock_dataset_metadata):
        splits_df = make_splits(mock_dataset_metadata, seed=42)
        assert "S999" not in splits_df["subject_id"].values

    def test_reproducible_with_seed(self, mock_dataset_metadata):
        splits1 = make_splits(mock_dataset_metadata, seed=123)
        splits2 = make_splits(mock_dataset_metadata, seed=123)
        pd.testing.assert_frame_equal(splits1, splits2)

    def test_different_seed_different_split(self, mock_dataset_metadata):
        splits1 = make_splits(mock_dataset_metadata, seed=42)
        splits2 = make_splits(mock_dataset_metadata, seed=9999)
        # With 10 subjects, different seeds should likely produce different splits
        assert not splits1["split"].equals(splits2["split"])

    def test_ratios_approximately_match(self, mock_dataset_metadata):
        splits_df = make_splits(mock_dataset_metadata, seed=42)
        counts = splits_df["split"].value_counts()
        # 10 subjects: 7 train, 2 val, 1 test or similar
        assert counts.get("train", 0) >= 6
        assert counts.get("val", 0) >= 1
        assert counts.get("test", 0) >= 1

    def test_validate_splits_detects_leakage(self, mock_dataset_metadata):
        # Artificially create a leakage in splits
        splits_df = pd.DataFrame([
            {"subject_id": "S001", "split": "train"},
            {"subject_id": "S002", "split": "val"},
        ])
        # Modify metadata so S001 and S002 share the same source_video
        meta = mock_dataset_metadata.copy()
        meta.loc[meta["subject_id"] == "S002", "source_video"] = "mock/vid_S001.mp4"

        errors = validate_splits(splits_df, meta)
        assert len(errors) > 0
        assert any("source_video 'mock/vid_S001.mp4' appears in multiple splits" in e for e in errors)

    def test_compute_split_stats(self, mock_dataset_metadata):
        splits_df = make_splits(mock_dataset_metadata, seed=42)
        stats = compute_split_stats(splits_df, mock_dataset_metadata)
        assert len(stats) == 3
        assert set(stats["split"]) == {"train", "val", "test"}
        assert "n_subjects" in stats.columns
        assert "n_segments" in stats.columns
        assert "total_seconds" in stats.columns
        assert "n_normal" in stats.columns
        assert "s_drowsy" in stats.columns

