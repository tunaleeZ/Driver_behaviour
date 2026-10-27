"""
labels.py — Label utilities for Driver Behavior Monitoring.

All label definitions are read from configs/labels.yaml (the single source of
truth).  No label name or index is hard-coded in this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml


# ---------------------------------------------------------------------------
# Internal cache
# ---------------------------------------------------------------------------

_LABEL_LIST: list[dict] | None = None
_LABELS_YAML_PATH: Path | None = None


def _get_labels_yaml_path() -> Path:
    """Locate configs/labels.yaml relative to repo root."""
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "configs" / "labels.yaml"


def _load_labels() -> list[dict]:
    """Load and cache label definitions from configs/labels.yaml."""
    global _LABEL_LIST, _LABELS_YAML_PATH
    path = _get_labels_yaml_path()
    if _LABEL_LIST is None or _LABELS_YAML_PATH != path:
        if not path.exists():
            raise FileNotFoundError(f"labels.yaml not found at {path}")
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        _LABEL_LIST = data["labels"]
        _LABELS_YAML_PATH = path
    return _LABEL_LIST


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def label_names() -> list[str]:
    """Return ordered list of label names (index = class index)."""
    return [item["name"] for item in _load_labels()]


def num_classes() -> int:
    """Return number of classes."""
    return len(_load_labels())


def label_to_index(name: str) -> int:
    """Convert a label name to its integer index.

    Parameters
    ----------
    name:
        Label name (e.g. ``"drowsy"``).

    Returns
    -------
    int
        Class index.

    Raises
    ------
    ValueError
        If the name is not a valid label.
    """
    mapping = {item["name"]: item["index"] for item in _load_labels()}
    if name not in mapping:
        valid = list(mapping.keys())
        raise ValueError(f"Unknown label {name!r}. Valid labels: {valid}")
    return mapping[name]


def index_to_label(index: int) -> str:
    """Convert a class index to its label name.

    Parameters
    ----------
    index:
        Integer class index.

    Returns
    -------
    str
        Label name.

    Raises
    ------
    ValueError
        If the index is out of range.
    """
    mapping = {item["index"]: item["name"] for item in _load_labels()}
    if index not in mapping:
        raise ValueError(f"Unknown class index {index}. Valid indices: {sorted(mapping.keys())}")
    return mapping[index]


def valid_label_names() -> set[str]:
    """Return the set of valid label name strings."""
    return {item["name"] for item in _load_labels()}


def label_descriptions() -> dict[str, str]:
    """Return a dict mapping label name to its description."""
    return {item["name"]: item.get("description", "") for item in _load_labels()}

