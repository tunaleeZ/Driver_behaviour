"""
timecode.py — Timestamp parsing and formatting utilities.

``parse_time(x)`` accepts multiple input formats and returns seconds as float.
``format_time(sec)`` returns a canonical "HH:MM:SS.mmm" string.

Supported input formats for parse_time:
  - "HH:MM:SS.mmm"  (e.g. "00:01:23.456")
  - "MM:SS.mmm"     (e.g. "01:23.456" or "00:05.000")
  - "SS.mmm"        (e.g. "5.25" or "83.0")
  - float / int     (seconds as number)
  - datetime.time   (Excel may produce these)
  - datetime.timedelta
"""

from __future__ import annotations

import datetime
import re
from typing import Union

# Type alias for accepted inputs
TimecodeInput = Union[str, float, int, datetime.time, datetime.timedelta]


def parse_time(x: TimecodeInput) -> float:
    """Parse a timecode value to seconds (float).

    Parameters
    ----------
    x:
        Timestamp in one of the supported formats.

    Returns
    -------
    float
        Time in seconds.

    Raises
    ------
    ValueError
        If the input cannot be parsed.
    """
    if isinstance(x, (int, float)):
        return float(x)

    if isinstance(x, datetime.timedelta):
        return x.total_seconds()

    if isinstance(x, datetime.time):
        return (
            x.hour * 3600
            + x.minute * 60
            + x.second
            + x.microsecond / 1_000_000
        )

    if isinstance(x, str):
        x = x.strip()
        if not x:
            raise ValueError("Empty timecode string")

        # HH:MM:SS.mmm  or  HH:MM:SS
        m = re.fullmatch(r"(\d+):(\d+):(\d+)(?:\.(\d+))?", x)
        if m:
            h, mi, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
            frac = _frac_seconds(m.group(4))
            return h * 3600 + mi * 60 + s + frac

        # MM:SS.mmm  or  MM:SS
        m = re.fullmatch(r"(\d+):(\d+)(?:\.(\d+))?", x)
        if m:
            mi, s = int(m.group(1)), int(m.group(2))
            frac = _frac_seconds(m.group(3))
            return mi * 60 + s + frac

        # Plain float/int string: "5.25", "83", "5"
        m = re.fullmatch(r"\d+(?:\.\d+)?", x)
        if m:
            return float(x)

        raise ValueError(f"Cannot parse timecode: {x!r}")

    raise TypeError(f"Unsupported timecode type: {type(x).__name__}")


def format_time(sec: float) -> str:
    """Format seconds as "HH:MM:SS.mmm".

    Parameters
    ----------
    sec:
        Time in seconds (non-negative).

    Returns
    -------
    str
        Formatted timecode string.
    """
    if sec < 0:
        raise ValueError(f"Time must be non-negative, got {sec}")
    total_ms = round(sec * 1000)
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _frac_seconds(frac_str: str | None) -> float:
    """Convert the fractional part of a timecode to seconds."""
    if frac_str is None:
        return 0.0
    # Normalise to at most 3 digits (milliseconds)
    frac_str = frac_str[:3].ljust(3, "0")
    return int(frac_str) / 1000.0

