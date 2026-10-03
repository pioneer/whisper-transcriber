"""CLI/Invoke integration layer: argument handling, progress printing,
error-to-exit-code mapping, and orchestration of transcriber + output.
"""

from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path

from . import diagnostics as diag
from .config import (
    DEFAULT_COOKIES_FILE,
    DEFAULT_COOKIES_FROM_BROWSER,
    DEFAULT_CPU_FALLBACK,
    DEFAULT_CUDA_RETRIES,
    DEFAULT_DELETE_VIDEO,
    DEFAULT_DISPLAY_SUMMARY,
    DEFAULT_MULTILINGUAL,
    DEFAULT_SUMMARIZE,
    DEFAULT_SUMMARY_BASE_URL,
    DEFAULT_SUMMARY_MODEL,
    DEFAULT_VIDEO_DOWNLOAD_COMMAND,
    DEFAULT_VIDEO_DOWNLOAD_DIR,
    SummaryConfig,
    TranscriptionConfig,
)
from .downloader import VideoDownloadError, download_video, is_video_url
from .output import TranscriptWriter, output_paths_for, print_markdown, read_resume_state
from .summarizer import (
    SummaryError,
    detect_text_language,
    load_dotenv,
    load_transcript_text,
    resolve_language_name,
    resolve_summary_config,
    resolve_transcript_path,
    summarize_file,
)
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
    cookies_from_browser: str | None = DEFAULT_COOKIES_FROM_BROWSER,
    cookies: str | None = DEFAULT_COOKIES_FILE,
    delete_video: bool = DEFAULT_DELETE_VIDEO,
    cpu_fallback: bool = DEFAULT_CPU_FALLBACK,
    summarize: bool = DEFAULT_SUMMARIZE,
    summary_model: str = DEFAULT_SUMMARY_MODEL,
    summary_api_key: str | None = None,
    summary_base_url: str | None = None,
    summary_prompt: str | None = None,
    summary_language: str | None = None,
    display_summary: bool = DEFAULT_DISPLAY_SUMMARY,
    cuda_retries: int = DEFAULT_CUDA_RETRIES,
    resume: bool = False,
    multilingual: bool = DEFAULT_MULTILINGUAL,
) -> int:
    """Run a full transcription and write TXT/SRT next to the source file.

    ``file`` may be a local path, or an ``http(s)://`` URL, in which case the
    video is downloaded first (to ``video_download_dir``, via
    ``video_download_command``).

    CUDA out-of-memory errors resume on CUDA while media progress advances.
    ``cuda_retries`` limits additional attempts without progress before CPU
    fallback (if enabled), keeping the same model/quality.
    If some segments were
    already transcribed before the failure, only the remainder is
    re-transcribed (resuming from that point), not the whole file.

    With ``resume``, existing paired TXT/SRT output supplies the last completed
    segment's end time and subtitle index, including after a previous Ctrl-C.

    Returns the process exit code (0 on success).
    """
    if cuda_retries < 0:
        print("Error: --cuda-retries must be non-negative.", file=sys.stderr)
        return 1
    load_dotenv()
    effective_cookies_browser = (
        cookies_from_browser
        or os.environ.get("YTDLP_COOKIES_FROM_BROWSER")
        or os.environ.get("COOKIES_FROM_BROWSER")
    )
    effective_cookies_file = (
        cookies or os.environ.get("YTDLP_COOKIES_FILE") or os.environ.get("COOKIES_FILE")
    )

    downloaded_path: Path | None = None
    if is_video_url(file):
        download_dir = Path(video_download_dir).expanduser()
        print(f"Downloading video: {file}", flush=True)
        print(f"Download folder: {download_dir}", flush=True)
        try:
            downloaded_path = download_video(
                file,
                download_dir,
                video_download_command,
                cookies_from_browser=effective_cookies_browser,
                cookies_file=effective_cookies_file,
            )
        except VideoDownloadError as exc:
            print(f"\nError: {exc}", file=sys.stderr)
            return 1
        path = downloaded_path
        print(f"Downloaded: {path}", flush=True)
    else:
        path = Path(file).expanduser().resolve()

    txt_path, srt_path = output_paths_for(path)
    resume_from = 0.0
    total_segment_count = 0
    if resume:
        try:
            resume_from, total_segment_count = read_resume_state(txt_path, srt_path)
        except (OSError, ValueError) as exc:
            print(f"Error: Cannot resume: {exc} Existing output was not changed.", file=sys.stderr)
            return 1

    overall_start = time.monotonic()
    segments = None
    attempt_device = device
    stalled_retries = 0
    retrying = False

    def retry_after_oom(exc: OutOfMemoryError) -> bool:
        nonlocal attempt_device, stalled_retries
        if attempt_device == "cuda" and resume_from > attempt_resume_from:
            stalled_retries = 0
            retry_label = "CUDA resume (progress made; retry budget reset)"
        elif attempt_device == "cuda" and stalled_retries < cuda_retries:
            stalled_retries += 1
            retry_label = f"CUDA retry {stalled_retries}/{cuda_retries} without progress"
        elif cpu_fallback and attempt_device != "cpu":
            attempt_device = "cpu"
            retry_label = "CPU fallback (CUDA stopped progressing; this may be slower)"
        else:
            print(f"\n\nError during transcription: {exc}", file=sys.stderr)
            return False
        resume_note = f" from {_format_hms(resume_from)}" if resume_from > 0 else ""
        print(
            f"\n\n{exc}\n\nRetrying on --device={attempt_device}{resume_note}: {retry_label}...\n"
        )
        return True

    while True:
        if retrying:
            segments = None
            gc.collect()
        attempt_resume_from = resume_from
        resuming = total_segment_count > 0 or resume_from > 0.0
        resume_note = f" from {_format_hms(resume_from)}" if resuming else ""
        config = TranscriptionConfig(
            model=model,
            device=attempt_device,
            compute_type=compute_type,
            beam_size=beam_size,
            language=language,
            multilingual=multilingual,
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
            retrying = retry_after_oom(exc)
            if retrying:
                continue
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
        attempt_start = time.monotonic()
        attempt_media_start = resume_from

        try:
            with TranscriptWriter(
                txt_path, srt_path, append=resuming, start_index=total_segment_count + 1
            ) as writer:
                for segment in segments:
                    writer.write_segment(segment.start, segment.end, segment.text)
                    if segment.text.strip():
                        total_segment_count += 1
                    resume_from = max(resume_from, segment.end)

                    now = time.monotonic()
                    elapsed_wall_hms = _format_hms(now - overall_start)
                    elapsed_hms = _format_hms(segment.end)
                    processed = segment.end - attempt_media_start
                    remaining_label = "remaining unknown"
                    if duration and processed > 0:
                        remaining_seconds = (
                            max(0.0, duration - segment.end) * (now - attempt_start) / processed
                        )
                        remaining_label = f"remaining ~{_format_hms(remaining_seconds)}"
                    if duration:
                        percent = min(100.0, (segment.end / duration) * 100)
                        progress = (
                            f"[{elapsed_hms} / {_format_hms(duration)}] {percent:5.1f}%   "
                            f"(elapsed {elapsed_wall_hms}, {remaining_label})"
                        )
                    else:
                        progress = (
                            f"[{elapsed_hms}]   (elapsed {elapsed_wall_hms}, remaining unknown)"
                        )
                    print(f"\r{progress}", end="", flush=True)
        except KeyboardInterrupt:
            print(
                f"\n\nInterrupted by user after {total_segment_count} segment(s). "
                f"Partial output kept at:\n  {txt_path}\n  {srt_path}\n"
                "Run the same command with --resume to continue from the last completed segment.",
                file=sys.stderr,
            )
            return 130
        except OutOfMemoryError as exc:
            retrying = retry_after_oom(exc)
            if retrying:
                continue
            return 1
        except TranscriptionError as exc:
            print(f"\n\nError during transcription: {exc}", file=sys.stderr)
            return 1

        wall_time = time.monotonic() - overall_start
        print(f"\n\nDone in {wall_time:.1f}s. Wrote {total_segment_count} segments to:")
        print(f"  {txt_path}")
        print(f"  {srt_path}")

        if summarize:
            detected_lang = resolve_language_name(info.language)
            effective_summary_lang = (
                summary_language
                or (resolve_language_name(language) if language else None)
                or detected_lang
            )
            lang_label = f" ({effective_summary_lang})" if effective_summary_lang else ""
            print(f"\nGenerating AI summary{lang_label}...", flush=True)
            summary_start = time.monotonic()
            summary_config = SummaryConfig(
                model=summary_model,
                base_url=summary_base_url or DEFAULT_SUMMARY_BASE_URL,
                api_key=summary_api_key,
                system_prompt=summary_prompt,
                language=effective_summary_lang,
                display=display_summary,
            )
            try:

                def _on_chunk(current: int, total: int) -> None:
                    print(f"\rSummarizing chunk {current}/{total}...", end="", flush=True)

                summary_path, summary_text = summarize_file(
                    txt_path,
                    summary_config,
                    on_chunk_progress=_on_chunk,
                )
                summary_elapsed = time.monotonic() - summary_start
                print(f"\nSummary generated in {summary_elapsed:.1f}s. Wrote to:")
                print(f"  {summary_path}")
                if summary_config.display:
                    print_markdown(summary_text)
            except SummaryError as exc:
                print(f"\nError generating summary: {exc}", file=sys.stderr)
                if delete_video and downloaded_path is not None:
                    downloaded_path.unlink(missing_ok=True)
                    print(f"Deleted downloaded video: {downloaded_path}")
                return 1

        if delete_video and downloaded_path is not None:
            downloaded_path.unlink(missing_ok=True)
            print(f"Deleted downloaded video: {downloaded_path}")

        return 0


def run_summarize(
    file: str,
    model: str = DEFAULT_SUMMARY_MODEL,
    api_key: str | None = None,
    base_url: str | None = None,
    prompt: str | None = None,
    output: str | None = None,
    language: str | None = None,
    display: bool = DEFAULT_DISPLAY_SUMMARY,
) -> int:
    """Generate an AI summary from a transcript file or media file.

    Returns the process exit code (0 on success).
    """
    path = Path(file).expanduser().resolve()
    target_output = Path(output).expanduser().resolve() if output else None

    resolved_lang = resolve_language_name(language)

    config = SummaryConfig(
        model=model,
        base_url=base_url or DEFAULT_SUMMARY_BASE_URL,
        api_key=api_key,
        system_prompt=prompt,
        language=resolved_lang,
        display=display,
    )
    config = resolve_summary_config(config)

    print(f"Summarizing transcript for: {path}", flush=True)
    print(f"Model: {config.model}", flush=True)

    try:
        resolved_path = resolve_transcript_path(path)
        transcript_text = load_transcript_text(resolved_path)
        effective_lang = config.language
        if not effective_lang:
            detected = detect_text_language(transcript_text)
            if detected:
                effective_lang = f"{detected} (detected)"
                config = SummaryConfig(
                    model=config.model,
                    base_url=config.base_url,
                    api_key=config.api_key,
                    system_prompt=config.system_prompt,
                    chunk_size=config.chunk_size,
                    language=detected,
                    display=config.display,
                )
        if effective_lang:
            print(f"Language: {effective_lang}", flush=True)
    except Exception:
        if config.language:
            print(f"Language: {config.language}", flush=True)

    start_time = time.monotonic()
    try:

        def _on_chunk(current: int, total: int) -> None:
            print(f"\rSummarizing chunk {current}/{total}...", end="", flush=True)

        summary_path, summary_text = summarize_file(
            path,
            config,
            output_path=target_output,
            on_chunk_progress=_on_chunk,
        )
    except SummaryError as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nSummary cancelled by user.", file=sys.stderr)
        return 130

    elapsed = time.monotonic() - start_time
    print(f"\nSummary generated in {elapsed:.1f}s. Wrote to:")
    print(f"  {summary_path}")
    if config.display:
        print_markdown(summary_text)
    return 0


def run_diagnose() -> int:
    """Print the diagnostics report. Always returns 0 (informational only)."""
    report = diag.collect_diagnostics()
    print(diag.format_report(report))
    return 0
