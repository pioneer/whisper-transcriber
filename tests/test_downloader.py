"""Unit tests for URL detection and video downloading.

``subprocess.run`` is monkeypatched so no real network access or external
tool (yt-dlp) is required.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from whisper_transcriber.downloader import (
    VideoDownloadError,
    download_video,
    is_video_url,
)


@pytest.mark.parametrize(
    "value",
    ["https://example.com/watch?v=abc", "http://example.com/video.mp4"],
)
def test_is_video_url_true_for_http_urls(value: str) -> None:
    assert is_video_url(value)


@pytest.mark.parametrize("value", ["video.mp4", "/home/user/video.mkv", "relative/path.wav"])
def test_is_video_url_false_for_local_paths(value: str) -> None:
    assert not is_video_url(value)


def test_download_video_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured_argv: list[str] = []

    def fake_run(argv: list[str], check: bool):
        captured_argv.extend(argv)
        output_arg = argv[argv.index("-o") + 1]
        actual_path = Path(
            output_arg.replace("%(title)s", "My Video Title").replace("%(ext)s", "mp4")
        )
        actual_path.write_bytes(b"fake video data")
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr("whisper_transcriber.downloader.subprocess.run", fake_run)

    result = download_video("https://example.com/watch?v=abc", tmp_path, "yt-dlp -o {output} {url}")

    assert result.exists()
    assert result.suffix == ".mp4"
    assert result.parent == tmp_path
    assert result.name.startswith("My Video Title.")
    assert captured_argv[0] == "yt-dlp"
    assert captured_argv[-1] == "https://example.com/watch?v=abc"


def test_download_video_output_path_is_deterministic_per_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured_output_args: list[str] = []

    def fake_run(argv: list[str], check: bool):
        output_arg = argv[argv.index("-o") + 1]
        captured_output_args.append(output_arg)
        Path(output_arg.replace("%(title)s", "Title").replace("%(ext)s", "mp4")).write_bytes(
            b"data"
        )
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr("whisper_transcriber.downloader.subprocess.run", fake_run)

    url = "https://example.com/watch?v=same"
    download_video(url, tmp_path, "yt-dlp -o {output} {url}")
    download_video(url, tmp_path, "yt-dlp -o {output} {url}")

    assert captured_output_args[0] == captured_output_args[1]


def test_download_video_command_not_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(argv: list[str], check: bool):
        raise FileNotFoundError("no such file")

    monkeypatch.setattr("whisper_transcriber.downloader.subprocess.run", fake_run)

    with pytest.raises(VideoDownloadError, match="not found"):
        download_video("https://example.com/x", tmp_path, "yt-dlp -o {output} {url}")


def test_download_video_nonzero_exit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(argv: list[str], check: bool):
        return subprocess.CompletedProcess(argv, 1)

    monkeypatch.setattr("whisper_transcriber.downloader.subprocess.run", fake_run)

    with pytest.raises(VideoDownloadError, match="exit code 1"):
        download_video("https://example.com/x", tmp_path, "yt-dlp -o {output} {url}")


def test_download_video_no_output_file_produced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(argv: list[str], check: bool):
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr("whisper_transcriber.downloader.subprocess.run", fake_run)

    with pytest.raises(VideoDownloadError, match="no output file"):
        download_video("https://example.com/x", tmp_path, "yt-dlp -o {output} {url}")


def test_download_video_creates_download_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    download_dir = tmp_path / "nested" / "videos"

    def fake_run(argv: list[str], check: bool):
        output_arg = argv[argv.index("-o") + 1]
        actual_path = Path(
            output_arg.replace("%(title)s", "Nested Video").replace("%(ext)s", "webm")
        )
        actual_path.write_bytes(b"data")
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr("whisper_transcriber.downloader.subprocess.run", fake_run)

    result = download_video("https://example.com/x", download_dir, "yt-dlp -o {output} {url}")
    assert result.parent == download_dir
