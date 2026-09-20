"""Formatting and incremental writing of transcription output files (TXT and SRT).

These functions and classes operate on plain ``(start, end, text)`` data so
they can be unit-tested without any dependency on faster-whisper or a GPU.
"""

from __future__ import annotations

from pathlib import Path
from types import TracebackType


def format_txt_timestamp(seconds: float) -> str:
    """Format seconds as ``HH:MM:SS`` for human-readable transcripts."""
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_srt_timestamp(seconds: float) -> str:
    """Format seconds as ``HH:MM:SS,mmm`` as required by the SRT standard."""
    total_ms = max(0, round(seconds * 1000))
    hours, remainder_ms = divmod(total_ms, 3_600_000)
    minutes, remainder_ms = divmod(remainder_ms, 60_000)
    secs, millis = divmod(remainder_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_txt_line(start: float, end: float, text: str) -> str:
    """Format a single TXT transcript line, e.g. ``[00:00:03 → 00:00:09] Text``."""
    return f"[{format_txt_timestamp(start)} \u2192 {format_txt_timestamp(end)}] {text.strip()}"


def format_srt_block(index: int, start: float, end: float, text: str) -> str:
    """Format a single SRT subtitle block (without trailing blank line)."""
    return f"{index}\n{format_srt_timestamp(start)} --> {format_srt_timestamp(end)}\n{text.strip()}"


class TranscriptWriter:
    """Incrementally writes TXT and SRT files, one segment at a time.

    Segments are written and flushed as soon as they arrive so that a
    transcription of a multi-hour file never needs to hold the full
    transcript in memory.
    """

    def __init__(self, txt_path: Path, srt_path: Path) -> None:
        self.txt_path = txt_path
        self.srt_path = srt_path
        self._txt_file = txt_path.open("w", encoding="utf-8")
        self._srt_file = srt_path.open("w", encoding="utf-8")
        self._index = 1

    def write_segment(self, start: float, end: float, text: str) -> None:
        """Append one segment to both output files, skipping empty text."""
        text = text.strip()
        if not text:
            return
        self._txt_file.write(format_txt_line(start, end, text) + "\n")
        self._txt_file.flush()
        self._srt_file.write(format_srt_block(self._index, start, end, text) + "\n\n")
        self._srt_file.flush()
        self._index += 1

    def close(self) -> None:
        self._txt_file.close()
        self._srt_file.close()

    def __enter__(self) -> TranscriptWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def output_paths_for(media_path: Path) -> tuple[Path, Path]:
    """Return the ``(txt_path, srt_path)`` placed next to the source media file."""
    return media_path.with_suffix(".txt"), media_path.with_suffix(".srt")
