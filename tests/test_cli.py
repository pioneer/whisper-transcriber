"""Unit tests for the ``run_transcribe`` CLI orchestration, focusing on the
video-URL download flow and the delete-after-transcribe option.

Downloading and the faster-whisper transcribe() call are both monkeypatched;
no network access, external tools, or GPU/model are required.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from whisper_transcriber import cli
from whisper_transcriber.transcriber import Segment, TranscriptionInfo


def _fake_transcribe(path, config):
    info = TranscriptionInfo(language="en", language_probability=0.9, duration=2.0)
    segments = iter([Segment(start=0.0, end=2.0, text="hello")])
    return segments, info


def test_run_transcribe_downloads_url_and_deletes_after(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    downloaded_file = tmp_path / "abc123.mp4"
    downloaded_file.write_bytes(b"fake video")

    monkeypatch.setattr(cli, "is_video_url", lambda value: True)
    monkeypatch.setattr(cli, "download_video", lambda url, download_dir, command: downloaded_file)
    monkeypatch.setattr(cli, "transcribe", _fake_transcribe)

    exit_code = cli.run_transcribe(
        file="https://example.com/video",
        model="tiny",
        device="cpu",
        compute_type="int8",
        language=None,
        beam_size=1,
        video_download_dir=str(tmp_path),
        delete_video=True,
    )

    assert exit_code == 0
    assert not downloaded_file.exists()
    assert (tmp_path / "abc123.txt").exists()
    assert (tmp_path / "abc123.srt").exists()


def test_run_transcribe_keeps_video_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    downloaded_file = tmp_path / "abc123.mp4"
    downloaded_file.write_bytes(b"fake video")

    monkeypatch.setattr(cli, "is_video_url", lambda value: True)
    monkeypatch.setattr(cli, "download_video", lambda url, download_dir, command: downloaded_file)
    monkeypatch.setattr(cli, "transcribe", _fake_transcribe)

    exit_code = cli.run_transcribe(
        file="https://example.com/video",
        model="tiny",
        device="cpu",
        compute_type="int8",
        language=None,
        beam_size=1,
        video_download_dir=str(tmp_path),
    )

    assert exit_code == 0
    assert downloaded_file.exists()


def test_run_transcribe_download_error_returns_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "is_video_url", lambda value: True)

    def fake_download(url, download_dir, command):
        raise cli.VideoDownloadError("boom")

    monkeypatch.setattr(cli, "download_video", fake_download)

    exit_code = cli.run_transcribe(
        file="https://example.com/video",
        model="tiny",
        device="cpu",
        compute_type="int8",
        language=None,
        beam_size=1,
        video_download_dir=str(tmp_path),
    )

    assert exit_code == 1


def test_run_transcribe_local_file_not_treated_as_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media_file = tmp_path / "video.wav"
    media_file.write_bytes(b"fake data")

    download_called = MagicMock()
    monkeypatch.setattr(cli, "download_video", download_called)
    monkeypatch.setattr(cli, "transcribe", _fake_transcribe)

    exit_code = cli.run_transcribe(
        file=str(media_file),
        model="tiny",
        device="cpu",
        compute_type="int8",
        language=None,
        beam_size=1,
    )

    assert exit_code == 0
    download_called.assert_not_called()
