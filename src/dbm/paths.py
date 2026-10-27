"""
paths.py — Centralised path resolution for Driver Behavior Monitoring.

Priority order for each root:
  1. Environment variable (DBM_RAW_ROOT, DBM_WORK_ROOT, DBM_FACE_LANDMARKER_PATH)
  2. configs/paths.yaml
  3. Kaggle auto-detection (/kaggle/input, /kaggle/working)
  4. Relative to the repository root

All modules should obtain paths through DatasetPaths, never hard-code them.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    """Return the repository root (directory containing this file's package)."""
    return Path(__file__).resolve().parents[2]


def _load_paths_yaml() -> dict:
    """Load configs/paths.yaml relative to the repo root."""
    cfg_path = _repo_root() / "configs" / "paths.yaml"
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    return {}


def _is_kaggle() -> bool:
    """Return True when running inside a Kaggle environment."""
    return Path("/kaggle").exists()


# ---------------------------------------------------------------------------
# DatasetPaths
# ---------------------------------------------------------------------------

class DatasetPaths:
    """Resolve and expose all important dataset and output paths.

    Parameters
    ----------
    dataset_root:
        Override the dataset root directory (raw, segments, processed are
        expected as subdirectories).  If None, resolved from env var
        ``DBM_RAW_ROOT``, then paths.yaml, then Kaggle auto-detection, then
        ``<repo_root>/driver_behavior_dataset_v01``.
    work_root:
        Override the writable working directory (for segments, processed,
        features output).  If None, resolved from env var ``DBM_WORK_ROOT``,
        then paths.yaml, then Kaggle auto-detection.
    """

    def __init__(
        self,
        dataset_root: Optional[Path | str] = None,
        work_root: Optional[Path | str] = None,
    ) -> None:
        cfg = _load_paths_yaml()
        repo = _repo_root()

        # --- dataset root (read-only raw data) ---
        if dataset_root is not None:
            self._dataset_root = Path(dataset_root).resolve()
        elif "DBM_RAW_ROOT" in os.environ:
            self._dataset_root = Path(os.environ["DBM_RAW_ROOT"]).resolve()
        elif "dataset_root" in cfg:
            p = Path(cfg["dataset_root"])
            self._dataset_root = p if p.is_absolute() else (repo / p).resolve()
        elif _is_kaggle():
            # Try to find the dataset slug directory under /kaggle/input
            kaggle_input = Path("/kaggle/input")
            candidates = sorted(kaggle_input.iterdir()) if kaggle_input.exists() else []
            self._dataset_root = candidates[0] if candidates else kaggle_input
        else:
            self._dataset_root = (repo / "driver_behavior_dataset_v01").resolve()

        # --- work root (writable) ---
        if work_root is not None:
            self._work_root = Path(work_root).resolve()
        elif "DBM_WORK_ROOT" in os.environ:
            self._work_root = Path(os.environ["DBM_WORK_ROOT"]).resolve()
        elif dataset_root is not None:
            self._work_root = self._dataset_root
        elif "work_root" in cfg:
            p = Path(cfg["work_root"])
            self._work_root = p if p.is_absolute() else (repo / p).resolve()
        elif _is_kaggle():
            self._work_root = Path("/kaggle/working/dbd_work").resolve()
        else:
            self._work_root = self._dataset_root

    # ------------------------------------------------------------------
    # Dataset read-only subdirectories
    # ------------------------------------------------------------------

    @property
    def dataset_root(self) -> Path:
        """Root of the dataset directory."""
        return self._dataset_root

    @property
    def raw(self) -> Path:
        """raw/ — original videos, NEVER modified by any script."""
        return self._dataset_root / "raw"

    @property
    def annotations(self) -> Path:
        """annotations/ — metadata xlsx/csv and splits.csv (tracked by git)."""
        return self._dataset_root / "annotations"

    @property
    def scripts(self) -> Path:
        """scripts/ — dataset scripts (tracked by git)."""
        return self._dataset_root / "scripts"

    # ------------------------------------------------------------------
    # Working (writable) subdirectories
    # ------------------------------------------------------------------

    @property
    def work_root(self) -> Path:
        """Writable work root."""
        return self._work_root

    @property
    def segments(self) -> Path:
        """segments/ — cut video clips by label."""
        return self._work_root / "segments"

    @property
    def processed(self) -> Path:
        """processed/ — extracted frames and other intermediate files."""
        return self._work_root / "processed"

    @property
    def frames(self) -> Path:
        """processed/frames/ — extracted JPEG frames per segment."""
        return self._work_root / "processed" / "frames"

    @property
    def features(self) -> Path:
        """features/ — extracted landmark .npz files."""
        return self._work_root / "features"

    @property
    def landmarks(self) -> Path:
        """features/landmarks/ — per-segment landmark .npz files."""
        return self._work_root / "features" / "landmarks"

    # ------------------------------------------------------------------
    # Annotation files
    # ------------------------------------------------------------------

    @property
    def metadata_xlsx(self) -> Path:
        """annotations/metadata.xlsx — primary annotation file."""
        return self.annotations / "metadata.xlsx"

    @property
    def metadata_csv(self) -> Path:
        """annotations/metadata.csv — validated export of metadata.xlsx."""
        return self.annotations / "metadata.csv"

    @property
    def splits_csv(self) -> Path:
        """annotations/splits.csv — subject-level train/val/test assignments."""
        return self.annotations / "splits.csv"

    @property
    def inventory_csv(self) -> Path:
        """annotations/inventory.csv — video inventory produced by inventory.py."""
        return self.annotations / "inventory.csv"

    @property
    def cut_report_csv(self) -> Path:
        """annotations/cut_report.csv — output of cut_segments.py."""
        return self.annotations / "cut_report.csv"

    @property
    def frames_index_csv(self) -> Path:
        """processed/frames_index.csv — index of extracted frames."""
        return self.processed / "frames_index.csv"

    # ------------------------------------------------------------------
    # Model & output paths
    # ------------------------------------------------------------------

    @property
    def face_landmarker(self) -> Path:
        """Path to face_landmarker.task (MediaPipe model)."""
        if "DBM_FACE_LANDMARKER_PATH" in os.environ:
            return Path(os.environ["DBM_FACE_LANDMARKER_PATH"]).resolve()
        cfg = _load_paths_yaml()
        if "face_landmarker_path" in cfg:
            p = Path(cfg["face_landmarker_path"])
            return p if p.is_absolute() else (_repo_root() / p).resolve()
        return (_repo_root() / "models" / "face_landmarker.task").resolve()

    @property
    def outputs(self) -> Path:
        """outputs/ — training run directories."""
        cfg = _load_paths_yaml()
        if "outputs_dir" in cfg:
            p = Path(cfg["outputs_dir"])
            return p if p.is_absolute() else (_repo_root() / p).resolve()
        return (_repo_root() / "outputs").resolve()

    # ------------------------------------------------------------------
    # Per-label segment directory helper
    # ------------------------------------------------------------------

    def segment_dir(self, label: str) -> Path:
        """Return segments/<label>/ directory."""
        return self.segments / label

    def segment_path(self, label: str, segment_id: str) -> Path:
        """Return the expected path for a segment video file."""
        return self.segment_dir(label) / f"{segment_id}.mp4"

    def landmark_path(self, segment_id: str) -> Path:
        """Return the expected path for a landmark .npz file."""
        return self.landmarks / f"{segment_id}.npz"

    def frames_dir(self, segment_id: str) -> Path:
        """Return processed/frames/<segment_id>/ directory."""
        return self.frames / segment_id

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"DatasetPaths(\n"
            f"  dataset_root={self._dataset_root},\n"
            f"  work_root={self._work_root}\n"
            f")"
        )

