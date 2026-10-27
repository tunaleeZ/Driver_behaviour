"""
tests/test_paths_config.py — Unit tests for paths.py, config.py, and labels.py.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

# Make the package importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dbm.config import load_config, merge_configs, save_config
from dbm.labels import (
    index_to_label,
    label_names,
    label_to_index,
    num_classes,
    valid_label_names,
)
from dbm.paths import DatasetPaths


# ---------------------------------------------------------------------------
# DatasetPaths tests
# ---------------------------------------------------------------------------

class TestDatasetPaths:
    """Tests for DatasetPaths."""

    def test_default_dataset_root_exists_or_resolvable(self):
        """DatasetPaths should resolve without error in a normal environment."""
        paths = DatasetPaths()
        assert isinstance(paths.dataset_root, Path)

    def test_override_dataset_root(self, tmp_path):
        paths = DatasetPaths(dataset_root=tmp_path)
        assert paths.dataset_root == tmp_path

    def test_override_work_root(self, tmp_path):
        paths = DatasetPaths(work_root=tmp_path)
        assert paths.work_root == tmp_path

    def test_env_var_dataset_root(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DBM_RAW_ROOT", str(tmp_path))
        paths = DatasetPaths()
        assert paths.dataset_root == tmp_path

    def test_env_var_work_root(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DBM_WORK_ROOT", str(tmp_path))
        paths = DatasetPaths()
        assert paths.work_root == tmp_path

    def test_raw_property(self, tmp_path):
        paths = DatasetPaths(dataset_root=tmp_path)
        assert paths.raw == tmp_path / "raw"

    def test_segments_property(self, tmp_path):
        paths = DatasetPaths(work_root=tmp_path)
        assert paths.segments == tmp_path / "segments"

    def test_segment_dir(self, tmp_path):
        paths = DatasetPaths(work_root=tmp_path)
        assert paths.segment_dir("drowsy") == tmp_path / "segments" / "drowsy"

    def test_segment_path(self, tmp_path):
        paths = DatasetPaths(work_root=tmp_path)
        p = paths.segment_path("drowsy", "SEG000001")
        assert p == tmp_path / "segments" / "drowsy" / "SEG000001.mp4"

    def test_landmark_path(self, tmp_path):
        paths = DatasetPaths(work_root=tmp_path)
        p = paths.landmark_path("SEG000001")
        assert p == tmp_path / "features" / "landmarks" / "SEG000001.npz"

    def test_frames_dir(self, tmp_path):
        paths = DatasetPaths(work_root=tmp_path)
        assert paths.frames_dir("SEG000001") == tmp_path / "processed" / "frames" / "SEG000001"

    def test_env_var_face_landmarker(self, tmp_path, monkeypatch):
        fake = tmp_path / "face_landmarker.task"
        monkeypatch.setenv("DBM_FACE_LANDMARKER_PATH", str(fake))
        paths = DatasetPaths()
        assert paths.face_landmarker == fake

    def test_repr_contains_paths(self, tmp_path):
        paths = DatasetPaths(dataset_root=tmp_path, work_root=tmp_path)
        assert "DatasetPaths" in repr(paths)


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfig:
    """Tests for load_config, merge_configs, save_config."""

    def _make_yaml(self, tmp_path: Path, data: dict) -> Path:
        p = tmp_path / "test_cfg.yaml"
        with p.open("w") as fh:
            yaml.dump(data, fh)
        return p

    def test_load_simple(self, tmp_path):
        p = self._make_yaml(tmp_path, {"lr": 0.001, "batch_size": 32})
        cfg = load_config(p)
        assert cfg["lr"] == pytest.approx(0.001)
        assert cfg["batch_size"] == 32

    def test_load_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")

    def test_override_int(self, tmp_path):
        p = self._make_yaml(tmp_path, {"lr": 0.001, "batch_size": 32})
        cfg = load_config(p, overrides=["batch_size=64"])
        assert cfg["batch_size"] == 64

    def test_override_float(self, tmp_path):
        p = self._make_yaml(tmp_path, {"lr": 0.001})
        cfg = load_config(p, overrides=["lr=0.0003"])
        assert cfg["lr"] == pytest.approx(0.0003)

    def test_override_string(self, tmp_path):
        p = self._make_yaml(tmp_path, {"loss": "cross_entropy"})
        cfg = load_config(p, overrides=["loss=focal"])
        assert cfg["loss"] == "focal"

    def test_override_invalid_format_raises(self, tmp_path):
        p = self._make_yaml(tmp_path, {"lr": 0.001})
        with pytest.raises(ValueError):
            load_config(p, overrides=["no_equals_sign"])

    def test_merge_configs(self):
        base = {"lr": 0.001, "epochs": 50, "model": {"hidden": 64}}
        override = {"lr": 0.0003, "model": {"hidden": 128, "layers": 2}}
        merged = merge_configs(base, override)
        assert merged["lr"] == pytest.approx(0.0003)
        assert merged["epochs"] == 50
        assert merged["model"]["hidden"] == 128
        assert merged["model"]["layers"] == 2

    def test_save_and_reload(self, tmp_path):
        cfg = {"lr": 0.001, "batch_size": 64, "nested": {"x": 1}}
        out = tmp_path / "saved.yaml"
        save_config(cfg, out)
        loaded = load_config(out)
        assert loaded == cfg

    def test_save_creates_parent_dirs(self, tmp_path):
        cfg = {"key": "value"}
        out = tmp_path / "deep" / "nested" / "cfg.yaml"
        save_config(cfg, out)
        assert out.exists()


# ---------------------------------------------------------------------------
# Labels tests
# ---------------------------------------------------------------------------

class TestLabels:
    """Tests for labels.py."""

    def test_num_classes(self):
        assert num_classes() == 5

    def test_label_names_ordered(self):
        names = label_names()
        assert names == ["normal", "drowsy", "sleepy", "drinking", "phone"]

    def test_label_to_index(self):
        assert label_to_index("normal") == 0
        assert label_to_index("drowsy") == 1
        assert label_to_index("sleepy") == 2
        assert label_to_index("drinking") == 3
        assert label_to_index("phone") == 4

    def test_index_to_label(self):
        assert index_to_label(0) == "normal"
        assert index_to_label(4) == "phone"

    def test_invalid_label_raises(self):
        with pytest.raises(ValueError):
            label_to_index("unknown_label")

    def test_invalid_index_raises(self):
        with pytest.raises(ValueError):
            index_to_label(99)

    def test_valid_label_names_set(self):
        valid = valid_label_names()
        assert "normal" in valid
        assert "phone" in valid
        assert len(valid) == 5

    def test_roundtrip(self):
        for name in label_names():
            idx = label_to_index(name)
            assert index_to_label(idx) == name

