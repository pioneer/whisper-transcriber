"""CLI/Invoke integration layer: argument handling, progress printing,
error-to-exit-code mapping, and orchestration of transcriber + output.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from . import diagnostics as diag
from .config import (
    DEFAULT_CPU_FALLBACK,
    DEFAULT_DELETE_VIDEO,
    DEFAULT_VIDEO_DOWNLOAD_COMMAND,
    DEFAULT_VIDEO_DOWNLOAD_DIR,
    TranscriptionConfig,
)
from .downloader import VideoDownloadError, download_video, is_video_url
from .output import TranscriptWriter, output_paths_for
from .transcriber import OutOfMemoryError, TranscriptionError, transcribe


def _format_hms(seconds: float) -> str:
    total = max(0, int(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def run_transcribe(
    file: str,
    model: str,
    device: str,
    compute_type: str,
    language: str | None,
    beam_size: int,
    video_download_dir: str = DEFAULT_VIDEO_DOWNLOAD_DIR,
    video_download_command: str = DEFAULT_VIDEO_DOWNLOAD_COMMAND,
    delete_video: bool = DEFAULT_DELETE_VIDEO,
    cpu_fallback: bool = DEFAULT_CPU_FALLBACK,
) -> int:
    """Run a full transcription and write TXT/SRT next to the source file.

    ``file`` may be a local path, or an ``http(s)://`` URL, in which case the
    video is downloaded first (to ``video_download_dir``, via
    ``video_download_command``).

    If ``cpu_fallback`` is set and a CUDA out-of-memory error occurs (at model
    load or partway through transcription), this loudly retries on the CPU
    instead of failing, keeping the same model/quality. If some segments were
    already transcribed before the failure, only the remainder is
    re-transcribed (resuming from that point), not the whole file.

    Returns the process exit code (0 on success).
    """
    downloaded_path: Path | None = None
    if is_video_url(file):
        download_dir = Path(video_download_dir).expanduser()
        print(f"Downloading video: {file}", flush=True)
        print(f"Download folder: {download_dir}", flush=True)
        try:
            downloaded_path = download_video(file, download_dir, video_download_command)
        except VideoDownloadError as exc:
            print(f"\nError: {exc}", file=sys.stderr)
            return 1
        path = downloaded_path
        print(f"Downloaded: {path}", flush=True)
    else:
        path = Path(file).expanduser().resolve()

    txt_path, srt_path = output_paths_for(path)

    devices_to_try = [device]
    if cpu_fallback and device != "cpu":
        devices_to_try.append("cpu")

    resume_from = 0.0
    total_segment_count = 0
    overall_start = time.monotonic()

    for attempt, attempt_device in enumerate(devices_to_try):
        is_last_attempt = attempt == len(devices_to_try) - 1
        resuming = resume_from > 0.0
        resume_note = f" from {_format_hms(resume_from)}" if resuming else ""
        config = TranscriptionConfig(
            model=model,
            device=attempt_device,
            compute_type=compute_type,
            beam_size=beam_size,
            language=language,
        )

        print(
            f"Model: {config.model}   Device: {config.device}   "
            f"Compute type: {config.compute_type}",
            flush=True,
        )
        print(f"File:  {path}", flush=True)
        if resuming:
            print(
                f"Resuming{resume_note} (already-transcribed segments are kept)...",
                flush=True,
            )
        print(
            "Loading model (if not cached yet, this downloads it from Hugging Face "
            "and may take a while depending on model size and connection speed)...",
            flush=True,
        )

        try:
            segments, info = transcribe(path, config, start_time=resume_from)
        except OutOfMemoryError as exc:
            if not is_last_attempt:
                print(f"\n{exc}\n\nRetrying on --device=cpu{resume_note} (this may be slower)...\n")
                continue
            print(f"\nError: {exc}", file=sys.stderr)
            return 1
        except TranscriptionError as exc:
            print(f"\nError: {exc}", file=sys.stderr)
            return 1
        except KeyboardInterrupt:
            print("\nCancelled before transcription started.", file=sys.stderr)
            return 130

        print(
            f"Detected language: {info.language} (probability {info.language_probability:.2f})",
            flush=True,
        )
        duration = info.duration
        if duration:
            print(f"Media duration: {_format_hms(duration)}", flush=True)

        print(f"Writing: {txt_path.name}, {srt_path.name}", flush=True)

        try:
            with TranscriptWriter(
                txt_path, srt_path, append=resuming, start_index=total_segment_count + 1
            ) as writer:
                for segment in segments:
                    writer.write_segment(segment.start, segment.end, segment.text)
                    total_segment_count += 1
                    resume_from = segment.end

                    elapsed_hms = _format_hms(segment.end)
                    if duration:
                        percent = min(100.0, (segment.end / duration) * 100)
                        progress = f"[{elapsed_hms} / {_format_hms(duration)}] {percent:5.1f}%"
                    else:
                        progress = f"[{elapsed_hms}]"
                    print(f"\r{progress}", end="", flush=True)
        except KeyboardInterrupt:
            print(
                f"\n\nInterrupted by user after {total_segment_count} segment(s). "
                f"Partial output kept at:\n  {txt_path}\n  {srt_path}",
                file=sys.stderr,
            )
            return 130
        except OutOfMemoryError as exc:
            if not is_last_attempt:
                resume_note = f" from {_format_hms(resume_from)}" if resume_from > 0.0 else ""
                print(
                    f"\n\n{exc}\n\nRetrying on --device=cpu{resume_note} (this may be slower)...\n"
                )
                continue
            print(f"\n\nError during transcription: {exc}", file=sys.stderr)
            return 1
        except TranscriptionError as exc:
            print(f"\n\nError during transcription: {exc}", file=sys.stderr)
            return 1

        wall_time = time.monotonic() - overall_start
        print(f"\n\nDone in {wall_time:.1f}s. Wrote {total_segment_count} segments to:")
        print(f"  {txt_path}")
        print(f"  {srt_path}")

        if delete_video and downloaded_path is not None:
            downloaded_path.unlink(missing_ok=True)
            print(f"Deleted downloaded video: {downloaded_path}")

        return 0

    return 1  # pragma: no cover - unreachable: the loop always returns or retries

    return 1  # pragma: no cover - unreachable: the loop always returns or retries


def run_diagnose() -> int:
    """Print the diagnostics report. Always returns 0 (informational only)."""
    report = diag.collect_diagnostics()
    print(diag.format_report(report))
    return 0
