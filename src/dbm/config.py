"""
config.py — YAML configuration loader with override support.

Usage
-----
    cfg = load_config("configs/models/geo_lstm.yaml")
    cfg = load_config("configs/models/geo_lstm.yaml", overrides=["lr=0.001", "batch_size=32"])
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load a YAML config file and apply key=value overrides.

    Parameters
    ----------
    path:
        Path to the YAML configuration file.
    overrides:
        List of ``key=value`` strings that override config entries.
        Nested keys use dot notation: ``scheduler.patience=3``.

    Returns
    -------
    dict
        Merged configuration dictionary.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh) or {}

    logger.debug("Loaded config from %s", path)

    if overrides:
        for item in overrides:
            if "=" not in item:
                raise ValueError(f"Override must be 'key=value', got: {item!r}")
            key_path, _, value_str = item.partition("=")
            _set_nested(cfg, key_path.strip().split("."), _parse_value(value_str.strip()))
            logger.debug("Override applied: %s = %r", key_path, _parse_value(value_str))

    return cfg


def merge_configs(*configs: dict[str, Any]) -> dict[str, Any]:
    """Merge multiple config dicts (later dicts override earlier ones).

    Parameters
    ----------
    *configs:
        Dicts to merge in order. Each successive dict overrides the previous.

    Returns
    -------
    dict
        Merged configuration.
    """
    result: dict[str, Any] = {}
    for cfg in configs:
        result = _deep_merge(result, copy.deepcopy(cfg))
    return result


def save_config(cfg: dict[str, Any], path: str | Path) -> None:
    """Save a configuration dict to a YAML file.

    Parameters
    ----------
    cfg:
        Configuration dictionary to save.
    path:
        Destination file path (parent directories are created automatically).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.dump(cfg, fh, default_flow_style=False, allow_unicode=True, sort_keys=False)
    logger.debug("Config saved to %s", path)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _set_nested(d: dict, keys: list[str], value: Any) -> None:
    """Set a nested dict value using a list of key segments."""
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def _parse_value(s: str) -> Any:
    """Parse a string value from a CLI override into a Python object."""
    # Try YAML parsing for ints, floats, booleans, null, lists, etc.
    try:
        return yaml.safe_load(s)
    except yaml.YAMLError:
        return s  # fall back to raw string


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result

