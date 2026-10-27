"""
logging_utils.py — Logging configuration helpers.

Use ``setup_logging()`` at the entry point of every script.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: int | str = logging.INFO,
    log_file: Optional[Path | str] = None,
    fmt: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> None:
    """Configure root logger with console (and optionally file) handlers.

    Parameters
    ----------
    level:
        Log level for the root logger (e.g. ``logging.DEBUG`` or ``"DEBUG"``).
    log_file:
        If given, also write logs to this file (appended).
    fmt:
        Log format string.
    datefmt:
        Date/time format string.
    """
    root = logging.getLogger()
    root.setLevel(level)

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    # Console handler (stdout so it doesn't mix with stderr progress bars)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        root.addHandler(fh)


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger.

    Parameters
    ----------
    name:
        Typically ``__name__``.

    Returns
    -------
    logging.Logger
    """
    return logging.getLogger(name)

