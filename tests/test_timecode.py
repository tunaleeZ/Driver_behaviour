"""
tests/test_timecode.py — Unit tests for src/dbm/timecode.py

Tests cover all input formats documented in prompt.md section 4.3:
  - "HH:MM:SS.mmm"
  - "MM:SS.mmm"
  - plain float string / int string
  - float / int literals
  - datetime.time (Excel conversion)
  - datetime.timedelta
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

import pytest

# Make sure the package is importable regardless of install state
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dbm.timecode import format_time, parse_time


class TestParseTime:
    """Tests for parse_time()."""

    # ------------------------------------------------------------------
    # Numeric inputs
    # ------------------------------------------------------------------

    def test_int_zero(self):
        assert parse_time(0) == pytest.approx(0.0)

    def test_int_positive(self):
        assert parse_time(90) == pytest.approx(90.0)

    def test_float(self):
        assert parse_time(5.25) == pytest.approx(5.25)

    def test_float_string(self):
        assert parse_time("5.25") == pytest.approx(5.25)

    def test_int_string(self):
        assert parse_time("83") == pytest.approx(83.0)

    # ------------------------------------------------------------------
    # "MM:SS.mmm" format
    # ------------------------------------------------------------------

    def test_mm_ss_mmm(self):
        # "00:05.000" -> 5.0 s
        assert parse_time("00:05.000") == pytest.approx(5.0)

    def test_mm_ss_no_frac(self):
        assert parse_time("01:30") == pytest.approx(90.0)

    def test_mm_ss_with_frac(self):
        assert parse_time("01:23.456") == pytest.approx(83.456)

    # ------------------------------------------------------------------
    # "HH:MM:SS.mmm" format
    # ------------------------------------------------------------------

    def test_hh_mm_ss_mmm(self):
        assert parse_time("00:01:23.456") == pytest.approx(83.456)

    def test_hh_mm_ss_no_frac(self):
        assert parse_time("01:00:00") == pytest.approx(3600.0)

    def test_hh_mm_ss_zeros(self):
        assert parse_time("00:00:00.000") == pytest.approx(0.0)

    # ------------------------------------------------------------------
    # datetime.time (Excel may produce these)
    # ------------------------------------------------------------------

    def test_datetime_time_zero(self):
        t = datetime.time(0, 0, 0)
        assert parse_time(t) == pytest.approx(0.0)

    def test_datetime_time_hms(self):
        t = datetime.time(0, 1, 23, 456000)  # microseconds
        assert parse_time(t) == pytest.approx(83.456)

    def test_datetime_time_hour(self):
        t = datetime.time(1, 0, 0)
        assert parse_time(t) == pytest.approx(3600.0)

    # ------------------------------------------------------------------
    # datetime.timedelta
    # ------------------------------------------------------------------

    def test_timedelta_zero(self):
        assert parse_time(datetime.timedelta(0)) == pytest.approx(0.0)

    def test_timedelta_seconds(self):
        assert parse_time(datetime.timedelta(seconds=90)) == pytest.approx(90.0)

    def test_timedelta_complex(self):
        td = datetime.timedelta(hours=1, minutes=2, seconds=3, milliseconds=456)
        expected = 3600 + 120 + 3 + 0.456
        assert parse_time(td) == pytest.approx(expected)

    # ------------------------------------------------------------------
    # Error cases
    # ------------------------------------------------------------------

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            parse_time("")

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError):
            parse_time("abc")

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            parse_time([1, 2, 3])  # type: ignore


class TestFormatTime:
    """Tests for format_time()."""

    def test_zero(self):
        assert format_time(0.0) == "00:00:00.000"

    def test_one_hour(self):
        assert format_time(3600.0) == "01:00:00.000"

    def test_millis(self):
        assert format_time(83.456) == "00:01:23.456"

    def test_rounding(self):
        # 5.5 seconds = 5500 ms -> "00:00:05.500"
        result = format_time(5.5)
        assert result == "00:00:05.500"

    def test_large_value(self):
        # 2 hours, 30 minutes, 15 seconds, 500 ms
        sec = 2 * 3600 + 30 * 60 + 15 + 0.5
        assert format_time(sec) == "02:30:15.500"

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            format_time(-1.0)


class TestRoundTrip:
    """parse_time -> format_time -> parse_time roundtrip."""

    @pytest.mark.parametrize("ts", [
        "00:00:00.000",
        "00:01:23.456",
        "01:00:00.000",
        "02:30:15.500",
    ])
    def test_roundtrip(self, ts: str):
        parsed = parse_time(ts)
        formatted = format_time(parsed)
        assert formatted == ts
