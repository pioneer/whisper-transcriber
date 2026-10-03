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
    monkeypatch.setattr(
        cli,
        "download_video",
        lambda url, download_dir, command, cookies_from_browser=None, cookies_file=None: (
            downloaded_file
        ),
    )
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
    monkeypatch.setattr(
        cli,
        "download_video",
        lambda url, download_dir, command, cookies_from_browser=None, cookies_file=None: (
            downloaded_file
        ),
    )
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

    def fake_download(
        url,
        download_dir,
        command,
        cookies_from_browser=None,
        cookies_file=None,
    ):
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


def test_run_transcribe_passes_cookies_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    downloaded_file = tmp_path / "abc123.mp4"
    downloaded_file.write_bytes(b"fake video")

    captured_kwargs: dict = {}

    def fake_download(url, download_dir, command, **kwargs):
        captured_kwargs.update(kwargs)
        return downloaded_file

    monkeypatch.setattr(cli, "is_video_url", lambda value: True)
    monkeypatch.setattr(cli, "download_video", fake_download)
    monkeypatch.setattr(cli, "transcribe", _fake_transcribe)

    exit_code = cli.run_transcribe(
        file="https://example.com/video",
        model="tiny",
        device="cpu",
        compute_type="int8",
        language=None,
        beam_size=1,
        video_download_dir=str(tmp_path),
        cookies_from_browser="firefox",
        cookies="cookies.txt",
    )

    assert exit_code == 0
    assert captured_kwargs["cookies_from_browser"] == "firefox"
    assert captured_kwargs["cookies_file"] == "cookies.txt"


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
    assert attempted_devices == ["cuda", "cuda", "cuda", "cpu"]


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
            if start_time > 0:
                raise OutOfMemoryError("GPU out of memory without progress")
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
        cuda_retries=0,
    )

    assert exit_code == 0
    assert attempted == [("cuda", 0.0), ("cuda", 1.0), ("cpu", 1.0)]

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
        cpu_fallback=False,
    )

    assert exit_code == 1


@pytest.mark.parametrize("stage", ["load", "stream"])
def test_cuda_retry_recovers_without_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    attempted: list[tuple[str, float]] = []

    def failing_segments():
        yield Segment(start=0.0, end=1.0, text="first half")
        raise OutOfMemoryError("OOM")

    def fake_transcribe(path, config, start_time=0.0):
        attempted.append((config.device, start_time))
        if len(attempted) == 1:
            if stage == "load":
                raise OutOfMemoryError("OOM")
            info = TranscriptionInfo(language="en", language_probability=0.9, duration=2.0)
            return failing_segments(), info
        if stage == "load":
            return _fake_transcribe(path, config, start_time)
        info = TranscriptionInfo(language="en", language_probability=0.9, duration=2.0)
        return iter([Segment(start=1.0, end=2.0, text="second half")]), info

    monkeypatch.setattr(cli, "transcribe", fake_transcribe)
    assert cli.run_transcribe(str(tmp_path / "video.wav"), "tiny", "cuda", "int8", None, 1) == 0
    assert attempted == [("cuda", 0.0), ("cuda", 1.0 if stage == "stream" else 0.0)]
    if stage == "stream":
        text = (tmp_path / "video.txt").read_text(encoding="utf-8")
        assert text.count("first half") == 1
        assert text.count("second half") == 1
        assert "\n2\n" in (tmp_path / "video.srt").read_text(encoding="utf-8")


@pytest.mark.parametrize("cuda_retries", [0, 1, 3])
@pytest.mark.parametrize("cpu_fallback", [False, True])
def test_cuda_retry_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cuda_retries: int, cpu_fallback: bool
) -> None:
    attempted: list[str] = []

    def fake_transcribe(path, config, start_time=0.0):
        attempted.append(config.device)
        if config.device == "cuda":
            raise OutOfMemoryError("OOM")
        return _fake_transcribe(path, config, start_time)

    monkeypatch.setattr(cli, "transcribe", fake_transcribe)
    result = cli.run_transcribe(
        str(tmp_path / "video.wav"),
        "tiny",
        "cuda",
        "int8",
        None,
        1,
        cuda_retries=cuda_retries,
        cpu_fallback=cpu_fallback,
    )
    assert result == (0 if cpu_fallback else 1)
    assert attempted == ["cuda"] * (cuda_retries + 1) + (["cpu"] if cpu_fallback else [])


@pytest.mark.parametrize("cuda_retries", [0, 1, 2])
@pytest.mark.parametrize("cpu_fallback", [False, True])
def test_cuda_keeps_retrying_while_progressing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cuda_retries: int, cpu_fallback: bool
) -> None:
    attempted: list[tuple[str, float]] = []

    def progress_then_oom(start_time):
        yield Segment(start=start_time, end=start_time + 1, text=f"part {int(start_time)}")
        raise OutOfMemoryError("OOM after progress")

    def fake_transcribe(path, config, start_time=0.0):
        attempted.append((config.device, start_time))
        assert len(attempted) <= 6
        info = TranscriptionInfo(language="en", language_probability=0.9, duration=6.0)
        if start_time < 5:
            return progress_then_oom(start_time), info
        return iter([Segment(start=5.0, end=6.0, text="final")]), info

    monkeypatch.setattr(cli, "transcribe", fake_transcribe)
    media_file = tmp_path / "video.wav"
    assert (
        cli.run_transcribe(
            str(media_file),
            "tiny",
            "cuda",
            "int8",
            None,
            1,
            cuda_retries=cuda_retries,
            cpu_fallback=cpu_fallback,
        )
        == 0
    )
    assert attempted == [("cuda", float(position)) for position in range(6)]
    txt = media_file.with_suffix(".txt").read_text(encoding="utf-8")
    for position in range(5):
        assert txt.count(f"part {position}") == 1
    assert "\n6\n" in media_file.with_suffix(".srt").read_text(encoding="utf-8")


@pytest.mark.parametrize("stage", ["load", "stream"])
@pytest.mark.parametrize("cpu_fallback", [False, True])
def test_progress_resets_stalled_retry_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str, cpu_fallback: bool
) -> None:
    attempted: list[tuple[str, float]] = []

    def stalled_segments(start_time):
        yield Segment(start=start_time, end=start_time, text=" ")
        raise OutOfMemoryError("OOM without progress")

    def progressing_segments():
        yield Segment(start=0.0, end=1.0, text="saved")
        raise OutOfMemoryError("OOM after progress")

    def fake_transcribe(path, config, start_time=0.0):
        attempted.append((config.device, start_time))
        assert len(attempted) <= 6
        info = TranscriptionInfo(language="en", language_probability=0.9, duration=2.0)
        if config.device == "cpu":
            return iter([Segment(start=1.0, end=2.0, text="remaining")]), info
        if len(attempted) == 2:
            return progressing_segments(), info
        if stage == "load":
            raise OutOfMemoryError("OOM without progress")
        return stalled_segments(start_time), info

    monkeypatch.setattr(cli, "transcribe", fake_transcribe)
    result = cli.run_transcribe(
        str(tmp_path / "video.wav"),
        "tiny",
        "cuda",
        "int8",
        None,
        1,
        cuda_retries=2,
        cpu_fallback=cpu_fallback,
    )
    assert result == (0 if cpu_fallback else 1)
    expected = [("cuda", 0.0), ("cuda", 0.0)] + [("cuda", 1.0)] * 3
    assert attempted == expected + ([("cpu", 1.0)] if cpu_fallback else [])
    assert (tmp_path / "video.txt").read_text(encoding="utf-8").count("saved") == 1


def test_cpu_oom_does_not_retry_forever(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    attempted: list[str] = []

    def fake_transcribe(path, config, start_time=0.0):
        attempted.append(config.device)
        assert len(attempted) <= 2
        raise OutOfMemoryError("OOM")

    monkeypatch.setattr(cli, "transcribe", fake_transcribe)
    assert (
        cli.run_transcribe(
            str(tmp_path / "video.wav"), "tiny", "cuda", "int8", None, 1, cuda_retries=0
        )
        == 1
    )
    assert attempted == ["cuda", "cpu"]


def test_negative_cuda_retries_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    mocked_transcribe = MagicMock()
    monkeypatch.setattr(cli, "transcribe", mocked_transcribe)
    assert cli.run_transcribe("video.wav", "tiny", "cuda", "int8", None, 1, cuda_retries=-1) == 1
    mocked_transcribe.assert_not_called()


@pytest.mark.parametrize(
    ("duration", "position", "expected"),
    [
        (10.0, 2.0, "remaining ~00:00:16"),
        (0.0, 2.0, "remaining unknown"),
        (10.0, 0.0, "remaining unknown"),
        (10.0, 12.0, "remaining ~00:00:00"),
    ],
)
def test_transcription_eta(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    duration: float,
    position: float,
    expected: str,
) -> None:
    def fake_transcribe(path, config, start_time=0.0):
        info = TranscriptionInfo(language="en", language_probability=0.9, duration=duration)
        return iter([Segment(start=0.0, end=position, text="hello")]), info

    clock = iter([100.0, 110.0, 114.0, 114.0])
    monkeypatch.setattr(cli.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(cli, "transcribe", fake_transcribe)
    assert cli.run_transcribe(str(tmp_path / "video.wav"), "tiny", "cpu", "int8", None, 1) == 0
    output = capsys.readouterr().out
    assert "elapsed 00:00:14" in output
    assert expected in output


def test_eta_resets_after_cpu_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def failing_segments():
        yield Segment(start=0.0, end=2.0, text="first")
        raise OutOfMemoryError("OOM")

    def fake_transcribe(path, config, start_time=0.0):
        info = TranscriptionInfo(language="en", language_probability=0.9, duration=10.0)
        if config.device == "cuda":
            if start_time > 0:
                raise OutOfMemoryError("OOM without progress")
            return failing_segments(), info
        return iter([Segment(start=2.0, end=4.0, text="second")]), info

    clock = iter([100.0, 100.0, 102.0, 110.0, 114.0, 114.0])
    monkeypatch.setattr(cli.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(cli, "transcribe", fake_transcribe)
    assert (
        cli.run_transcribe(
            str(tmp_path / "video.wav"), "tiny", "cuda", "int8", None, 1, cuda_retries=0
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "elapsed 00:00:14, remaining ~00:00:12" in output


@pytest.mark.parametrize("source", ["local", "url"])
def test_resume_after_ctrl_c(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    source: str,
) -> None:
    media_file = tmp_path / "video.wav"
    file = str(media_file) if source == "local" else "https://example.com/video"
    if source == "url":
        monkeypatch.setattr(cli, "download_video", lambda *args, **kwargs: media_file)
    attempts: list[tuple[str, float]] = []

    def interrupted_segments():
        yield Segment(start=0.0, end=1.125, text="first half")
        raise KeyboardInterrupt

    def fake_transcribe(path, config, start_time=0.0):
        attempts.append((config.device, start_time))
        info = TranscriptionInfo(language="en", language_probability=0.9, duration=2.5)
        if len(attempts) == 1:
            return interrupted_segments(), info
        return iter([Segment(start=1.125, end=2.5, text="second half")]), info

    monkeypatch.setattr(cli, "transcribe", fake_transcribe)
    assert cli.run_transcribe(file, "tiny", "cuda", "int8", None, 1) == 130
    assert "--resume" in capsys.readouterr().err
    saved_txt = media_file.with_suffix(".txt").read_bytes()
    saved_srt = media_file.with_suffix(".srt").read_bytes()
    assert cli.run_transcribe(file, "tiny", "cuda", "int8", None, 1, resume=True) == 0
    assert attempts == [("cuda", 0.0), ("cuda", 1.125)]
    new_txt = media_file.with_suffix(".txt").read_bytes()
    new_srt = media_file.with_suffix(".srt").read_bytes()
    assert new_txt.startswith(saved_txt)
    assert new_srt.startswith(saved_srt)
    assert new_txt.count(b"first half") == 1
    assert new_txt.count(b"second half") == 1
    assert b"\n2\n" in new_srt


def test_resume_then_cuda_oom_preserves_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media_file = tmp_path / "video.wav"
    with cli.TranscriptWriter(
        media_file.with_suffix(".txt"), media_file.with_suffix(".srt")
    ) as writer:
        writer.write_segment(0.0, 1.125, "saved segment")
    attempts: list[float] = []

    def failing_segments():
        yield Segment(start=1.125, end=1.5, text="   ")
        yield Segment(start=1.5, end=2.0, text="new segment")
        raise OutOfMemoryError("OOM")

    def fake_transcribe(path, config, start_time=0.0):
        attempts.append(start_time)
        info = TranscriptionInfo(language="en", language_probability=0.9, duration=3.0)
        if len(attempts) == 1:
            return failing_segments(), info
        return iter([Segment(start=2.0, end=3.0, text="final segment")]), info

    monkeypatch.setattr(cli, "transcribe", fake_transcribe)
    assert cli.run_transcribe(str(media_file), "tiny", "cuda", "int8", None, 1, resume=True) == 0
    assert attempts == [1.125, 2.0]
    srt = media_file.with_suffix(".srt").read_text(encoding="utf-8")
    assert "\n2\n" in srt
    assert "\n3\n" in srt
    assert "\n4\n" not in srt
    assert srt.count("saved segment") == 1


def test_resume_rejects_missing_srt_without_overwriting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media_file = tmp_path / "video.wav"
    txt_path = media_file.with_suffix(".txt")
    txt_path.write_text("partial transcript", encoding="utf-8")
    mocked_transcribe = MagicMock()
    monkeypatch.setattr(cli, "transcribe", mocked_transcribe)
    assert cli.run_transcribe(str(media_file), "tiny", "cuda", "int8", None, 1, resume=True) == 1
    mocked_transcribe.assert_not_called()
    assert txt_path.read_text(encoding="utf-8") == "partial transcript"


def test_resume_without_output_starts_normally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "transcribe", _fake_transcribe)
    media_file = tmp_path / "video.wav"
    assert cli.run_transcribe(str(media_file), "tiny", "cpu", "int8", None, 1, resume=True) == 0
    assert media_file.with_suffix(".srt").read_text(encoding="utf-8").startswith("1\n")


def test_run_transcribe_with_summarize(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    media_file = tmp_path / "video.wav"
    media_file.write_bytes(b"fake data")

    monkeypatch.setattr(cli, "transcribe", _fake_transcribe)

    def fake_summarize_file(path, config, output_path=None, on_chunk_progress=None):
        out = output_path or path.with_suffix(".summary.md")
        out.write_text("# Summary\nAll good.", encoding="utf-8")
        return out, "# Summary\nAll good."

    monkeypatch.setattr(cli, "summarize_file", fake_summarize_file)

    exit_code = cli.run_transcribe(
        file=str(media_file),
        model="tiny",
        device="cpu",
        compute_type="int8",
        language=None,
        beam_size=1,
        summarize=True,
    )

    assert exit_code == 0
    assert (tmp_path / "video.txt").exists()
    assert (tmp_path / "video.srt").exists()
    assert (tmp_path / "video.summary.md").exists()
    assert "# Summary" in (tmp_path / "video.summary.md").read_text(encoding="utf-8")


def test_run_transcribe_with_summarize_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media_file = tmp_path / "video.wav"
    media_file.write_bytes(b"fake data")

    monkeypatch.setattr(cli, "transcribe", _fake_transcribe)

    def fake_summarize_fail(path, config, output_path=None, on_chunk_progress=None):
        raise cli.SummaryError("API failed")

    monkeypatch.setattr(cli, "summarize_file", fake_summarize_fail)

    exit_code = cli.run_transcribe(
        file=str(media_file),
        model="tiny",
        device="cpu",
        compute_type="int8",
        language=None,
        beam_size=1,
        summarize=True,
    )

    assert exit_code == 1
    # Transcripts are still preserved even if summarization fails
    assert (tmp_path / "video.txt").exists()
    assert (tmp_path / "video.srt").exists()


def test_run_summarize_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    txt_file = tmp_path / "meeting.txt"
    txt_file.write_text("Spoken words", encoding="utf-8")

    def fake_summarize_file(path, config, output_path=None, on_chunk_progress=None):
        out = output_path or path.with_suffix(".summary.md")
        out.write_text("# Meeting Summary", encoding="utf-8")
        return out, "# Meeting Summary"

    monkeypatch.setattr(cli, "summarize_file", fake_summarize_file)

    exit_code = cli.run_summarize(file=str(txt_file))
    assert exit_code == 0
    assert (tmp_path / "meeting.summary.md").exists()


@pytest.mark.parametrize("model", [None, "explicit-model"])
def test_run_summarize_reports_resolved_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    model: str | None,
) -> None:
    txt_file = tmp_path / "meeting.txt"
    txt_file.write_text("Spoken words", encoding="utf-8")
    monkeypatch.setenv("SUMMARY_MODEL", "gpt-6.1-sol")
    monkeypatch.setenv("SUMMARY_API_KEY", "test-key")
    expected_model = model or "gpt-6.1-sol"

    def fake_summarize_file(path, config, output_path=None, on_chunk_progress=None):
        assert config.model == expected_model
        return path.with_suffix(".summary.md"), "# Summary"

    monkeypatch.setattr(cli, "summarize_file", fake_summarize_file)
    exit_code = (
        cli.run_summarize(file=str(txt_file), model=model)
        if model is not None
        else cli.run_summarize(file=str(txt_file))
    )
    assert exit_code == 0
    assert f"Model: {expected_model}" in capsys.readouterr().out


def test_run_summarize_custom_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    txt_file = tmp_path / "meeting.txt"
    txt_file.write_text("Spoken words", encoding="utf-8")
    custom_out = tmp_path / "custom.md"

    def fake_summarize_file(path, config, output_path=None, on_chunk_progress=None):
        out = output_path or path.with_suffix(".summary.md")
        out.write_text("# Custom Summary", encoding="utf-8")
        return out, "# Custom Summary"

    monkeypatch.setattr(cli, "summarize_file", fake_summarize_file)

    exit_code = cli.run_summarize(file=str(txt_file), output=str(custom_out))
    assert exit_code == 0
    assert custom_out.exists()


def test_run_summarize_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_summarize_fail(path, config, output_path=None, on_chunk_progress=None):
        raise cli.SummaryError("File missing")

    monkeypatch.setattr(cli, "summarize_file", fake_summarize_fail)

    exit_code = cli.run_summarize(file=str(tmp_path / "nonexistent.txt"))
    assert exit_code == 1


def test_run_summarize_keyboard_interrupt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_summarize_cancel(path, config, output_path=None, on_chunk_progress=None):
        raise KeyboardInterrupt()

    monkeypatch.setattr(cli, "summarize_file", fake_summarize_cancel)

    exit_code = cli.run_summarize(file=str(tmp_path / "some.txt"))
    assert exit_code == 130


def test_run_summarize_with_language(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    txt_file = tmp_path / "meeting.txt"
    txt_file.write_text("Spoken words", encoding="utf-8")

    captured_config: list[cli.SummaryConfig] = []

    def fake_summarize_file(path, config, output_path=None, on_chunk_progress=None):
        captured_config.append(config)
        out = output_path or path.with_suffix(".summary.md")
        out.write_text("# Summary", encoding="utf-8")
        return out, "# Summary"

    monkeypatch.setattr(cli, "summarize_file", fake_summarize_file)

    exit_code = cli.run_summarize(file=str(txt_file), language="Ukrainian")
    assert exit_code == 0
    assert len(captured_config) == 1
    assert captured_config[0].language == "Ukrainian"


def test_run_transcribe_with_summarize_language(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media_file = tmp_path / "video.wav"
    media_file.write_bytes(b"fake data")

    monkeypatch.setattr(cli, "transcribe", _fake_transcribe)

    captured_config: list[cli.SummaryConfig] = []

    def fake_summarize_file(path, config, output_path=None, on_chunk_progress=None):
        captured_config.append(config)
        out = output_path or path.with_suffix(".summary.md")
        out.write_text("# Summary", encoding="utf-8")
        return out, "# Summary"

    monkeypatch.setattr(cli, "summarize_file", fake_summarize_file)

    exit_code = cli.run_transcribe(
        file=str(media_file),
        model="tiny",
        device="cpu",
        compute_type="int8",
        language=None,
        beam_size=1,
        summarize=True,
        summary_language="Ukrainian",
    )

    assert exit_code == 0
    assert len(captured_config) == 1
    assert captured_config[0].language == "Ukrainian"


def test_run_transcribe_with_summarize_uses_detected_language(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media_file = tmp_path / "video.wav"
    media_file.write_bytes(b"fake data")

    def _fake_transcribe_uk(path, config, start_time=0.0):
        info = TranscriptionInfo(language="uk", language_probability=0.98, duration=2.0)
        segments = iter([Segment(start=0.0, end=2.0, text="Вітаю всіх")])
        return segments, info

    monkeypatch.setattr(cli, "transcribe", _fake_transcribe_uk)

    captured_config: list[cli.SummaryConfig] = []

    def fake_summarize_file(path, config, output_path=None, on_chunk_progress=None):
        captured_config.append(config)
        out = output_path or path.with_suffix(".summary.md")
        out.write_text("# Підсумок", encoding="utf-8")
        return out, "# Підсумок"

    monkeypatch.setattr(cli, "summarize_file", fake_summarize_file)

    exit_code = cli.run_transcribe(
        file=str(media_file),
        model="tiny",
        device="cpu",
        compute_type="int8",
        language=None,
        beam_size=1,
        summarize=True,
    )

    assert exit_code == 0
    assert len(captured_config) == 1
    assert captured_config[0].language == "Ukrainian"


def test_run_summarize_auto_detects_language(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    txt_file = tmp_path / "meeting.txt"
    txt_file.write_text(
        "Вітаю всіх на нашому каналі. Сьогодні ми обговоримо новини.",
        encoding="utf-8",
    )

    captured_config: list[cli.SummaryConfig] = []

    def fake_summarize_file(path, config, output_path=None, on_chunk_progress=None):
        captured_config.append(config)
        out = output_path or path.with_suffix(".summary.md")
        out.write_text("# Підсумок", encoding="utf-8")
        return out, "# Підсумок"

    monkeypatch.setattr(cli, "summarize_file", fake_summarize_file)

    exit_code = cli.run_summarize(file=str(txt_file))
    assert exit_code == 0
    assert len(captured_config) == 1
    assert captured_config[0].language == "Ukrainian"


def test_run_summarize_display_option(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    txt_file = tmp_path / "meeting.txt"
    txt_file.write_text("Spoken words", encoding="utf-8")

    def fake_summarize_file(path, config, output_path=None, on_chunk_progress=None):
        out = output_path or path.with_suffix(".summary.md")
        out.write_text("# Summary", encoding="utf-8")
        return out, "# Summary"

    monkeypatch.setattr(cli, "summarize_file", fake_summarize_file)
    mock_print = MagicMock()
    monkeypatch.setattr(cli, "print_markdown", mock_print)

    # By default, display is True
    exit_code = cli.run_summarize(file=str(txt_file))
    assert exit_code == 0
    mock_print.assert_called_once_with("# Summary")

    # With display=False, print_markdown is not called
    mock_print.reset_mock()
    exit_code = cli.run_summarize(file=str(txt_file), display=False)
    assert exit_code == 0
    mock_print.assert_not_called()


def test_run_transcribe_display_summary_option(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media_file = tmp_path / "video.wav"
    media_file.write_bytes(b"fake data")

    monkeypatch.setattr(cli, "transcribe", _fake_transcribe)

    def fake_summarize_file(path, config, output_path=None, on_chunk_progress=None):
        out = output_path or path.with_suffix(".summary.md")
        out.write_text("# Summary", encoding="utf-8")
        return out, "# Summary"

    monkeypatch.setattr(cli, "summarize_file", fake_summarize_file)
    mock_print = MagicMock()
    monkeypatch.setattr(cli, "print_markdown", mock_print)

    # By default, display_summary is True
    exit_code = cli.run_transcribe(
        file=str(media_file),
        model="tiny",
        device="cpu",
        compute_type="int8",
        language=None,
        beam_size=1,
        summarize=True,
    )
    assert exit_code == 0
    mock_print.assert_called_once_with("# Summary")

    # With display_summary=False, print_markdown is not called
    mock_print.reset_mock()
    exit_code = cli.run_transcribe(
        file=str(media_file),
        model="tiny",
        device="cpu",
        compute_type="int8",
        language=None,
        beam_size=1,
        summarize=True,
        display_summary=False,
    )
    assert exit_code == 0
    mock_print.assert_not_called()
