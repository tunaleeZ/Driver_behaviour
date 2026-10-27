"""
tests/test_init_dataset.py — Tests for init_dataset.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Import the module under test
_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "driver_behavior_dataset_v01" / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

from init_dataset import init_dataset


class TestInitDataset:
    """Tests for init_dataset()."""

    EXPECTED_DIRS = [
        "annotations",
        "features",
        "features/landmarks",
        "processed",
        "processed/frames",
        "raw",
        "raw/public",
        "scripts",
        "segments/drinking",
        "segments/drowsy",
        "segments/normal",
        "segments/phone",
        "segments/sleepy",
    ]

    def test_creates_all_directories(self, tmp_path):
        init_dataset(tmp_path)
        for rel in self.EXPECTED_DIRS:
            assert (tmp_path / rel).is_dir(), f"Missing directory: {rel}"

    def test_creates_gitignore(self, tmp_path):
        init_dataset(tmp_path)
        gi = tmp_path / ".gitignore"
        assert gi.exists()
        content = gi.read_text()
        # Should contain raw/ and segments entries
        assert "raw/" in content
        assert "segments/" in content

    def test_creates_readme(self, tmp_path):
        init_dataset(tmp_path)
        readme = tmp_path / "README_Driver_Behavior_Dataset.md"
        assert readme.exists()
        content = readme.read_text(encoding="utf-8")
        assert "Driver Behavior" in content
        assert "normal" in content
        assert "drowsy" in content

    def test_creates_gitkeep_files(self, tmp_path):
        init_dataset(tmp_path)
        for rel_dir in ["features", "processed", "raw/public"]:
            gk = tmp_path / rel_dir / ".gitkeep"
            assert gk.exists(), f"Missing .gitkeep in {rel_dir}"

    def test_idempotent_no_overwrite(self, tmp_path):
        """Running init_dataset twice should not raise and should not overwrite files."""
        init_dataset(tmp_path)
        # Modify the .gitignore to check it isn't overwritten
        gi = tmp_path / ".gitignore"
        gi.write_text("custom content")
        init_dataset(tmp_path, overwrite=False)
        assert gi.read_text() == "custom content"

    def test_overwrite_flag(self, tmp_path):
        """Running with overwrite=True should regenerate text files."""
        init_dataset(tmp_path)
        gi = tmp_path / ".gitignore"
        gi.write_text("custom content")
        init_dataset(tmp_path, overwrite=True)
        assert gi.read_text() != "custom content"

    def test_dry_run_does_not_create_dirs(self, tmp_path):
        """Dry run should not create any directories or files."""
        new_root = tmp_path / "new_dataset"
        init_dataset(new_root, dry_run=True)
        assert not new_root.exists()
