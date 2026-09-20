"""Unit tests for timestamp formatting (TXT ``HH:MM:SS`` and SRT ``HH:MM:SS,mmm``)."""

from __future__ import annotations

from whisper_transcriber.output import format_srt_timestamp, format_txt_timestamp


def test_txt_timestamp_zero() -> None:
    assert format_txt_timestamp(0) == "00:00:00"


def test_txt_timestamp_seconds_only() -> None:
    assert format_txt_timestamp(9) == "00:00:09"
    assert format_txt_timestamp(59) == "00:00:59"


def test_txt_timestamp_minutes() -> None:
    assert format_txt_timestamp(65) == "00:01:05"
    assert format_txt_timestamp(3599) == "00:59:59"


def test_txt_timestamp_hours() -> None:
    assert format_txt_timestamp(3600) == "01:00:00"
    assert format_txt_timestamp(7325) == "02:02:05"


def test_txt_timestamp_multi_hour_long_recording() -> None:
    # 3 hours, 25 minutes, 40 seconds - must not assume < 30 minute media.
    assert format_txt_timestamp(3 * 3600 + 25 * 60 + 40) == "03:25:40"


def test_txt_timestamp_rounds_to_nearest_second() -> None:
    assert format_txt_timestamp(3.6) == "00:00:04"
    assert format_txt_timestamp(3.4) == "00:00:03"


def test_srt_timestamp_zero() -> None:
    assert format_srt_timestamp(0) == "00:00:00,000"


def test_srt_timestamp_has_millisecond_precision() -> None:
    assert format_srt_timestamp(3.123) == "00:00:03,123"
    assert format_srt_timestamp(9.5) == "00:00:09,500"


def test_srt_timestamp_hours() -> None:
    assert format_srt_timestamp(3661.25) == "01:01:01,250"


def test_srt_timestamp_negative_clamped_to_zero() -> None:
    assert format_srt_timestamp(-1) == "00:00:00,000"
    assert format_txt_timestamp(-1) == "00:00:00"
