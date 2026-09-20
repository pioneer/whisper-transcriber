"""Downloading a video from a URL (e.g. via yt-dlp) before transcription."""

from __future__ import annotations

import hashlib
import shlex
import subprocess
from pathlib import Path
from urllib.parse import urlparse


class VideoDownloadError(Exception):
    """A video URL could not be downloaded."""


def is_video_url(value: str) -> bool:
    """Return True if ``value`` looks like a URL rather than a local file path."""
    return urlparse(value).scheme in ("http", "https")


def download_video(url: str, download_dir: Path, command_template: str) -> Path:
    """Download ``url`` into ``download_dir`` and return the downloaded file's path.

    The output filename is based on the video's title (e.g. ``My Video.<tag>.mp4``),
    with a short tag derived from the URL to keep it unique and greppable. The
    tag is deterministic (not random), so re-running the same URL reuses the
    same destination path and lets the download tool resume a partial
    download, or skip one that already finished, instead of starting over.

    ``command_template`` is a whitespace/shell-tokenized command whose tokens
    may contain the placeholders ``{url}`` and ``{output}`` (e.g. ``"yt-dlp -o
    {output} {url}"``). The command is executed directly as an argv list (no
    shell), so ``url``/``output`` need no extra quoting even if they contain
    spaces.
    """
    download_dir = download_dir.expanduser()
    download_dir.mkdir(parents=True, exist_ok=True)

    # Derived from the URL (not random) so re-running the same URL reuses the
    # same destination path: partial downloads resume and finished ones are
    # skipped, instead of always starting a fresh download. The tag also lets
    # us reliably locate the downloaded file afterwards, whatever title/
    # extension the download tool picks.
    tag = hashlib.sha256(url.encode("utf-8")).hexdigest()[:10]
    output_template = str(download_dir / f"%(title)s.{tag}.%(ext)s")

    try:
        tokens = shlex.split(command_template)
    except ValueError as exc:
        raise VideoDownloadError(f"Invalid video download command template: {exc}") from exc
    if not tokens:
        raise VideoDownloadError("Video download command template is empty.")
    argv = [token.format(url=url, output=output_template) for token in tokens]

    try:
        # Inherit stdout/stderr so the download tool's own progress output
        # (e.g. yt-dlp's live progress bar) is visible to the user.
        result = subprocess.run(argv, check=False)
    except FileNotFoundError as exc:
        raise VideoDownloadError(
            f"Download command '{argv[0]}' not found. Install it (e.g. `pip install "
            f"yt-dlp` or via your system package manager) or change the download "
            f"command setting.\nOriginal error: {exc}"
        ) from exc

    if result.returncode != 0:
        raise VideoDownloadError(
            f"Video download command failed (exit code {result.returncode}). "
            "See its output above for details."
        )

    matches = sorted(download_dir.glob(f"*.{tag}.*"))
    if not matches:
        raise VideoDownloadError(
            f"Download command finished but no output file matching '*.{tag}.*' "
            f"was found in {download_dir}."
        )
    return matches[0]
