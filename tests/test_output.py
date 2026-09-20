"""Unit tests for TXT/SRT line formatting and incremental file writing.

No GPU and no Whisper model download are required: segments are plain
``(start, end, text)`` tuples fed directly into the formatting/writer code.
"""

from __future__ import annotations

from pathlib import Path

from whisper_transcriber.output import (
    TranscriptWriter,
    format_srt_block,
    format_txt_line,
    output_paths_for,
)


def test_format_txt_line() -> None:
    line = format_txt_line(3, 9, "Вітаю всіх сьогодні")
    assert line == "[00:00:03 \u2192 00:00:09] Вітаю всіх сьогодні"


def test_format_txt_line_strips_whitespace() -> None:
    line = format_txt_line(0, 1, "  hello world  ")
    assert line == "[00:00:00 \u2192 00:00:01] hello world"


def test_format_srt_block() -> None:
    block = format_srt_block(1, 3, 9.5, "Hello there")
    assert block == "1\n00:00:03,000 --> 00:00:09,500\nHello there"


def test_output_paths_for(tmp_path: Path) -> None:
    media = tmp_path / "video.mp4"
    txt_path, srt_path = output_paths_for(media)
    assert txt_path == tmp_path / "video.txt"
    assert srt_path == tmp_path / "video.srt"


def test_transcript_writer_creates_valid_txt_and_srt(tmp_path: Path) -> None:
    txt_path = tmp_path / "out.txt"
    srt_path = tmp_path / "out.srt"

    segments = [
        (0.0, 3.2, "Вітаю всіх сьогодні"),
        (3.2, 7.0, "Сьогодні ми поговоримо"),
    ]

    with TranscriptWriter(txt_path, srt_path) as writer:
        for start, end, text in segments:
            writer.write_segment(start, end, text)

    txt_content = txt_path.read_text(encoding="utf-8")
    assert txt_content == (
        "[00:00:00 \u2192 00:00:03] Вітаю всіх сьогодні\n"
        "[00:00:03 \u2192 00:00:07] Сьогодні ми поговоримо\n"
    )

    srt_content = srt_path.read_text(encoding="utf-8")
    assert srt_content == (
        "1\n00:00:00,000 --> 00:00:03,200\nВітаю всіх сьогодні\n\n"
        "2\n00:00:03,200 --> 00:00:07,000\nСьогодні ми поговоримо\n\n"
    )


def test_transcript_writer_skips_empty_segments(tmp_path: Path) -> None:
    txt_path = tmp_path / "out.txt"
    srt_path = tmp_path / "out.srt"

    with TranscriptWriter(txt_path, srt_path) as writer:
        writer.write_segment(0.0, 1.0, "   ")
        writer.write_segment(1.0, 2.0, "real text")

    assert "real text" in txt_path.read_text(encoding="utf-8")
    assert srt_path.read_text(encoding="utf-8").startswith("1\n")


def test_transcript_writer_incremental_flush(tmp_path: Path) -> None:
    """Each segment must be flushed to disk immediately, not buffered until close."""
    txt_path = tmp_path / "out.txt"
    srt_path = tmp_path / "out.srt"

    writer = TranscriptWriter(txt_path, srt_path)
    try:
        writer.write_segment(0.0, 1.0, "first segment")
        assert "first segment" in txt_path.read_text(encoding="utf-8")
    finally:
        writer.close()
