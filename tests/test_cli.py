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
from whisper_transcriber.transcriber import OutOfMemoryError, Segment, TranscriptionInfo


def _fake_transcribe(path, config, start_time=0.0):
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


def test_run_transcribe_cpu_fallback_retries_after_oom_at_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media_file = tmp_path / "video.wav"
    media_file.write_bytes(b"fake data")

    attempted_devices: list[str] = []

    def fake_transcribe(path, config, start_time=0.0):
        attempted_devices.append(config.device)
        if config.device == "cuda":
            raise OutOfMemoryError("GPU out of memory")
        return _fake_transcribe(path, config, start_time)

    monkeypatch.setattr(cli, "transcribe", fake_transcribe)

    exit_code = cli.run_transcribe(
        file=str(media_file),
        model="large-v3",
        device="cuda",
        compute_type="int8_float32",
        language=None,
        beam_size=1,
        cpu_fallback=True,
    )

    assert exit_code == 0
    assert attempted_devices == ["cuda", "cpu"]


def test_run_transcribe_cpu_fallback_resumes_after_mid_stream_oom(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On a mid-stream OOM, only the remainder should be re-transcribed, not the
    whole file: the retry must receive the last successful segment's end time,
    and previously-written output must be kept (appended to), not overwritten.
    """
    media_file = tmp_path / "video.wav"
    media_file.write_bytes(b"fake data")

    attempted: list[tuple[str, float]] = []

    def failing_segments():
        yield Segment(start=0.0, end=1.0, text="first half")
        raise OutOfMemoryError("GPU out of memory mid-stream")

    def fake_transcribe(path, config, start_time=0.0):
        attempted.append((config.device, start_time))
        info = TranscriptionInfo(language="en", language_probability=0.9, duration=2.0)
        if config.device == "cuda":
            return failing_segments(), info
        assert start_time == 1.0
        return iter([Segment(start=1.0, end=2.0, text="second half")]), info

    monkeypatch.setattr(cli, "transcribe", fake_transcribe)

    exit_code = cli.run_transcribe(
        file=str(media_file),
        model="large-v3",
        device="cuda",
        compute_type="int8_float32",
        language=None,
        beam_size=1,
        cpu_fallback=True,
    )

    assert exit_code == 0
    assert attempted == [("cuda", 0.0), ("cpu", 1.0)]

    txt_content = (tmp_path / "video.txt").read_text(encoding="utf-8")
    assert "first half" in txt_content
    assert "second half" in txt_content
    assert txt_content.index("first half") < txt_content.index("second half")

    srt_content = (tmp_path / "video.srt").read_text(encoding="utf-8")
    assert srt_content.startswith("1\n")
    assert "\n2\n" in srt_content


def test_run_transcribe_oom_without_cpu_fallback_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media_file = tmp_path / "video.wav"
    media_file.write_bytes(b"fake data")

    def fake_transcribe(path, config, start_time=0.0):
        raise OutOfMemoryError("GPU out of memory")

    monkeypatch.setattr(cli, "transcribe", fake_transcribe)

    exit_code = cli.run_transcribe(
        file=str(media_file),
        model="large-v3",
        device="cuda",
        compute_type="int8_float32",
        language=None,
        beam_size=1,
    )

    assert exit_code == 1
