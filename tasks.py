"""Invoke entry points. Thin wrappers around ``whisper_transcriber.cli``.

Run with e.g.:
    uv run inv transcribe path/to/file.mp4
    uv run inv transcribe path/to/file.mp4 --language=uk
    uv run inv diagnose
    uv run inv test
"""

from __future__ import annotations

import os
import sys

from invoke import Context, task

from whisper_transcriber.config import (
    DEFAULT_BEAM_SIZE,
    DEFAULT_COMPUTE_TYPE,
    DEFAULT_DELETE_VIDEO,
    DEFAULT_DEVICE,
    DEFAULT_MODEL,
    DEFAULT_VIDEO_DOWNLOAD_COMMAND,
    DEFAULT_VIDEO_DOWNLOAD_DIR,
)
from whisper_transcriber.diagnostics import cuda_library_dirs


def _ensure_cuda_ld_library_path(device: str) -> None:
    """Re-exec this process with LD_LIBRARY_PATH set for the project-local
    CUDA libraries (nvidia-cublas-cu12 / nvidia-cudnn-cu12), if needed.

    This must happen *before* faster-whisper/CTranslate2 is imported: the
    dynamic linker only reads LD_LIBRARY_PATH once, at process start.
    """
    if device != "cuda":
        return

    lib_dirs = cuda_library_dirs()
    if not lib_dirs:
        return

    current = os.environ.get("LD_LIBRARY_PATH", "")
    current_parts = [p for p in current.split(":") if p]
    missing = [d for d in lib_dirs if d not in current_parts]
    if not missing:
        return

    os.environ["LD_LIBRARY_PATH"] = ":".join(missing + current_parts)
    os.execv(sys.executable, [sys.executable, *sys.argv])


@task
def transcribe(
    c: Context,
    file: str,
    model: str = DEFAULT_MODEL,
    device: str = DEFAULT_DEVICE,
    compute_type: str = DEFAULT_COMPUTE_TYPE,
    language: str | None = None,
    beam_size: int = DEFAULT_BEAM_SIZE,
    video_download_dir: str = DEFAULT_VIDEO_DOWNLOAD_DIR,
    video_download_command: str = DEFAULT_VIDEO_DOWNLOAD_COMMAND,
    delete_video: bool = DEFAULT_DELETE_VIDEO,
) -> None:
    """Transcribe a video/audio file, or a video URL, to <file>.txt and <file>.srt."""
    _ensure_cuda_ld_library_path(device)

    from whisper_transcriber.cli import run_transcribe

    exit_code = run_transcribe(
        file=file,
        model=model,
        device=device,
        compute_type=compute_type,
        language=language,
        beam_size=beam_size,
        video_download_dir=video_download_dir,
        video_download_command=video_download_command,
        delete_video=delete_video,
    )
    raise SystemExit(exit_code)


@task
def diagnose(c: Context) -> None:
    """Print Python/GPU/CUDA/library diagnostics."""
    from whisper_transcriber.cli import run_diagnose

    raise SystemExit(run_diagnose())


@task
def cuda_env(c: Context) -> None:
    """Print a shell export command for LD_LIBRARY_PATH pointing at the
    project-local CUDA libraries, for use outside of `inv transcribe`.
    """
    lib_dirs = cuda_library_dirs()
    if not lib_dirs:
        print(
            "No bundled CUDA libraries found. Install them with:\n"
            "  uv add nvidia-cublas-cu12 nvidia-cudnn-cu12",
            file=sys.stderr,
        )
        raise SystemExit(1)
    print(f'export LD_LIBRARY_PATH="{":".join(lib_dirs)}:$LD_LIBRARY_PATH"')


@task
def test(c: Context) -> None:
    """Run the test suite (no GPU or model download required)."""
    c.run("uv run pytest", pty=True)


@task
def lint(c: Context) -> None:
    """Run ruff checks and formatting checks."""
    c.run("uv run ruff check .", pty=True)
    c.run("uv run ruff format --check .", pty=True)
