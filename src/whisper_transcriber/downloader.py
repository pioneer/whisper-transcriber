"""Downloading a video from a URL (e.g. via yt-dlp) before transcription."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4


class VideoDownloadError(Exception):
    """A video URL could not be downloaded."""


def is_video_url(value: str) -> bool:
    """Return True if ``value`` looks like a URL rather than a local file path."""
    return urlparse(value).scheme in ("http", "https")


def download_video(url: str, download_dir: Path, command_template: str) -> Path:
    """Download ``url`` into ``download_dir`` and return the downloaded file's path.

    ``command_template`` is a whitespace/shell-tokenized command whose tokens
    may contain the placeholders ``{url}`` and ``{output}`` (e.g. ``"yt-dlp -o
    {output} {url}"``). The command is executed directly as an argv list (no
    shell), so ``url``/``output`` need no extra quoting even if they contain
    spaces.
    """
    download_dir = download_dir.expanduser()
    download_dir.mkdir(parents=True, exist_ok=True)

    # A unique basename lets us reliably locate the downloaded file afterwards,
    # regardless of the title/extension the download tool picks.
    unique_id = uuid4().hex
    output_template = str(download_dir / f"{unique_id}.%(ext)s")

    try:
        tokens = shlex.split(command_template)
    except ValueError as exc:
        raise VideoDownloadError(f"Invalid video download command template: {exc}") from exc
    if not tokens:
        raise VideoDownloadError("Video download command template is empty.")
    argv = [token.format(url=url, output=output_template) for token in tokens]

    try:
        result = subprocess.run(argv, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise VideoDownloadError(
            f"Download command '{argv[0]}' not found. Install it (e.g. `pip install "
            f"yt-dlp` or via your system package manager) or change the download "
            f"command setting.\nOriginal error: {exc}"
        ) from exc

    if result.returncode != 0:
        raise VideoDownloadError(
            f"Video download command failed (exit code {result.returncode}):\n"
            f"{result.stderr.strip() or result.stdout.strip()}"
        )

    matches = sorted(download_dir.glob(f"{unique_id}.*"))
    if not matches:
        raise VideoDownloadError(
            f"Download command finished but no output file matching '{unique_id}.*' "
            f"was found in {download_dir}."
        )
    return matches[0]
