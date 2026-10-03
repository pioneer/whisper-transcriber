"""Formatting and incremental writing of transcription output files (TXT and SRT).

These functions and classes operate on plain ``(start, end, text)`` data so
they can be unit-tested without any dependency on faster-whisper or a GPU.
"""

from __future__ import annotations

import re
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

    def __init__(
        self,
        txt_path: Path,
        srt_path: Path,
        append: bool = False,
        start_index: int = 1,
    ) -> None:
        """``append``/``start_index`` resume a previous run instead of
        overwriting it, e.g. after retrying a failed transcription attempt.
        """
        self.txt_path = txt_path
        self.srt_path = srt_path
        mode = "a" if append else "w"
        self._txt_file = txt_path.open(mode, encoding="utf-8")
        self._srt_file = srt_path.open(mode, encoding="utf-8")
        self._index = start_index

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


def read_resume_state(txt_path: Path, srt_path: Path) -> tuple[float, int]:
    """Validate paired transcript outputs and return the last end time and cue count."""
    if not txt_path.exists() and not srt_path.exists():
        return 0.0, 0
    if not txt_path.is_file() or not srt_path.is_file():
        raise ValueError("Resuming requires both the existing TXT and SRT files.")

    timestamp_pattern = re.compile(
        r"(\d{2,}):([0-5]\d):([0-5]\d),(\d{3}) --> "
        r"(\d{2,}):([0-5]\d):([0-5]\d),(\d{3})"
    )
    txt_pattern = re.compile(r"\[\d{2,}:\d{2}:\d{2} \u2192 \d{2,}:\d{2}:\d{2}\] ")
    end_time = 0.0
    cue_count = 0
    with txt_path.open(encoding="utf-8") as txt, srt_path.open(encoding="utf-8") as srt:
        while index_line := srt.readline():
            if index_line.strip() != str(cue_count + 1):
                raise ValueError("SRT cue numbering is invalid; cannot safely resume.")
            match = timestamp_pattern.fullmatch(srt.readline().strip())
            if match is None:
                raise ValueError("SRT timestamps are invalid; cannot safely resume.")
            values = [int(value) for value in match.groups()]
            start = values[0] * 3600 + values[1] * 60 + values[2] + values[3] / 1000
            end = values[4] * 3600 + values[5] * 60 + values[6] + values[7] / 1000
            if end < start or end < end_time:
                raise ValueError("SRT timestamps are out of order; cannot safely resume.")

            text_lines: list[str] = []
            while True:
                line = srt.readline()
                if not line:
                    raise ValueError("SRT ends with an incomplete cue; cannot safely resume.")
                if not line.strip():
                    break
                text_lines.append(line)
            if not text_lines:
                raise ValueError("SRT contains an empty cue; cannot safely resume.")
            txt_line = txt.readline()
            prefix = txt_pattern.match(txt_line)
            if prefix is None:
                raise ValueError("TXT and SRT do not match; cannot safely resume.")
            saved_text = txt_line[prefix.end() :] + "".join(txt.readline() for _ in text_lines[1:])
            if saved_text != "".join(text_lines):
                raise ValueError("TXT and SRT do not match; cannot safely resume.")
            cue_count += 1
            end_time = end
        if txt.read(1):
            raise ValueError("TXT has unmatched content; cannot safely resume.")
    return end_time, cue_count


def summary_output_path_for(path: Path) -> Path:
    """Return the ``<base>.summary.md`` path for a media or transcript file."""
    if path.name.endswith(".summary.md"):
        return path
    return path.with_suffix(".summary.md")


def print_markdown(text: str, console: object | None = None) -> None:
    """Render markdown with rich terminal formatting."""
    from rich.console import Console
    from rich.markdown import Markdown

    c: Console = console if isinstance(console, Console) else Console()
    c.print()
    c.rule("[bold]AI Summary[/bold]")
    c.print(Markdown(text))
    c.rule()
