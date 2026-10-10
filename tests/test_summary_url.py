"""Summary-only URL routing tests with mocked providers and transcription."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from invoke import Context

import tasks
from whisper_transcriber import cli
from whisper_transcriber.transcriber import Segment, TranscriptionInfo

URL = "https://www.youtube.com/watch?v=9hE5-98ZeCg"


@pytest.fixture(autouse=True)
def isolated_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "resolve_summary_config", lambda config: config)
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    for key in (
        "GEMINI_API_KEY",
        "GEMINI_MODEL",
        "YTDLP_COOKIES_FROM_BROWSER",
        "YTDLP_COOKIES_FILE",
        "COOKIES_FROM_BROWSER",
        "COOKIES_FILE",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.mark.parametrize("custom_output", [False, True])
def test_gemini_success_skips_all_local_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, custom_output: bool
) -> None:
    gemini = MagicMock(return_value="# Gemini summary")
    download = MagicMock()
    transcribe = MagicMock()
    local_summary = MagicMock()
    display = MagicMock()
    monkeypatch.setattr(cli, "summarize_youtube", gemini)
    monkeypatch.setattr(cli, "download_video", download)
    monkeypatch.setattr(cli, "run_transcribe", transcribe)
    monkeypatch.setattr(cli, "summarize_file", local_summary)
    monkeypatch.setattr(cli, "print_markdown", display)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "env-model")
    target = (
        tmp_path / "nested/custom.md"
        if custom_output
        else (tmp_path / "youtube.9hE5-98ZeCg.summary.md")
    )
    assert (
        cli.run_summarize(
            URL,
            video_download_dir=str(tmp_path),
            output=str(target) if custom_output else None,
            language="uk",
            prompt="Focus on conclusions.",
            delete_video=True,
        )
        == 0
    )
    assert target.read_text() == "# Gemini summary\n"
    assert list(tmp_path.rglob("*.*")) == [target]
    download.assert_not_called()
    transcribe.assert_not_called()
    local_summary.assert_not_called()
    display.assert_called_once_with("# Gemini summary")
    assert gemini.call_args.kwargs == {"api_key": "test-key", "model": "env-model"}
    assert gemini.call_args.args[1].language == "Ukrainian"
    assert gemini.call_args.args[1].system_prompt == "Focus on conclusions."


def test_gemini_cli_overrides_and_no_display(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gemini = MagicMock(return_value="# Summary")
    display = MagicMock()
    monkeypatch.setattr(cli, "summarize_youtube", gemini)
    monkeypatch.setattr(cli, "print_markdown", display)
    monkeypatch.setenv("GEMINI_MODEL", "env-model")
    monkeypatch.setenv("GEMINI_API_KEY", "env-key")
    assert (
        cli.run_summarize(
            URL,
            gemini_api_key="cli-key",
            gemini_model="cli-model",
            video_download_dir=str(tmp_path),
            display=False,
        )
        == 0
    )
    assert gemini.call_args.kwargs == {"api_key": "cli-key", "model": "cli-model"}
    display.assert_not_called()


@pytest.mark.parametrize("route", ["api_failure", "missing_key", "disabled", "other_host"])
@pytest.mark.parametrize("delete_video", [False, True])
def test_local_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    route: str,
    delete_video: bool,
) -> None:
    video = tmp_path / "downloaded.mp4"
    video.write_bytes(b"fake video")
    download = MagicMock(return_value=video)
    if route == "missing_key":
        gemini = MagicMock(wraps=cli.summarize_youtube)
    else:
        gemini = MagicMock(side_effect=cli.SummaryError("quota exceeded"))
    monkeypatch.setattr(cli, "summarize_youtube", gemini)
    monkeypatch.setattr(cli, "download_video", download)
    monkeypatch.setattr(cli, "is_model_cached", lambda *a: True)
    monkeypatch.setattr(
        cli,
        "transcribe",
        lambda *a, **kw: (
            iter([Segment(0.0, 2.0, "The video content.")]),
            TranscriptionInfo(language="en", language_probability=0.9, duration=2.0),
        ),
    )
    summary_config = []

    def fake_summary(path, config, output_path=None, on_chunk_progress=None):
        assert path == video
        assert video.with_suffix(".txt").exists()
        summary_config.append(config)
        output_path.write_text("# Local summary")
        return output_path, "# Local summary"

    monkeypatch.setattr(cli, "summarize_file", fake_summary)
    monkeypatch.setenv("YTDLP_COOKIES_FROM_BROWSER", "firefox")
    monkeypatch.setenv("YTDLP_COOKIES_FILE", "cookies.txt")
    target = tmp_path / "result.md"
    url = URL if route != "other_host" else "https://example.com/video"
    assert (
        cli.run_summarize(
            url,
            gemini=route != "disabled",
            transcription_model="tiny",
            device="cpu",
            compute_type="int8",
            video_download_dir=str(tmp_path),
            video_download_command="custom {url} {output}",
            output=str(target),
            api_key="local-key",
            model="local-model",
            language="uk",
            display=False,
            delete_video=delete_video,
        )
        == 0
    )
    assert target.read_text() == "# Local summary"
    assert video.with_suffix(".txt").exists()
    assert video.with_suffix(".srt").exists()
    assert video.exists() is not delete_video
    assert summary_config[0].model == "local-model"
    assert summary_config[0].api_key == "local-key"
    assert summary_config[0].language == "Ukrainian"
    assert download.call_args.kwargs == {
        "cookies_from_browser": "firefox",
        "cookies_file": "cookies.txt",
    }
    if route in {"disabled", "other_host"}:
        gemini.assert_not_called()
    else:
        assert "Falling back to local transcription" in capsys.readouterr().err


@pytest.mark.parametrize("stage", ["gemini", "download", "transcribe", "summary"])
def test_url_cancellation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str) -> None:
    video = tmp_path / "download.mp4"
    video.write_bytes(b"fake")
    gemini = MagicMock(
        side_effect=(KeyboardInterrupt if stage == "gemini" else cli.SummaryError("unavailable"))
    )
    download = MagicMock(
        side_effect=KeyboardInterrupt if stage == "download" else None,
        return_value=video,
    )
    transcribe = MagicMock(return_value=130 if stage == "transcribe" else 0)
    summary = MagicMock(side_effect=KeyboardInterrupt)
    monkeypatch.setattr(cli, "summarize_youtube", gemini)
    monkeypatch.setattr(cli, "download_video", download)
    monkeypatch.setattr(cli, "run_transcribe", transcribe)
    monkeypatch.setattr(cli, "summarize_file", summary)
    assert cli.run_summarize(URL, video_download_dir=str(tmp_path), display=False) == 130
    if stage == "gemini":
        download.assert_not_called()
    if stage in {"gemini", "download"}:
        transcribe.assert_not_called()
    if stage != "summary":
        summary.assert_not_called()


def test_save_failure_does_not_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    blocked_parent = tmp_path / "file"
    blocked_parent.write_text("not a directory")
    monkeypatch.setattr(cli, "summarize_youtube", lambda *a, **kw: "# Summary")
    download = MagicMock()
    monkeypatch.setattr(cli, "download_video", download)
    assert cli.run_summarize(URL, output=str(blocked_parent / "summary.md")) == 1
    download.assert_not_called()
    assert blocked_parent.read_text() == "not a directory"


@pytest.mark.parametrize("stage", ["download", "transcribe", "summary"])
def test_fallback_failure_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    video = tmp_path / "download.mp4"
    video.write_bytes(b"fake")
    monkeypatch.setattr(cli, "summarize_youtube", MagicMock(side_effect=cli.SummaryError("fail")))
    download = MagicMock(
        side_effect=cli.VideoDownloadError("download failed") if stage == "download" else None,
        return_value=video,
    )
    transcribe = MagicMock(return_value=1 if stage == "transcribe" else 0)
    summary = MagicMock(side_effect=cli.SummaryError("summary failed"))
    monkeypatch.setattr(cli, "download_video", download)
    monkeypatch.setattr(cli, "run_transcribe", transcribe)
    monkeypatch.setattr(cli, "summarize_file", summary)
    assert cli.run_summarize(URL, display=False) == 1
    if stage == "download":
        transcribe.assert_not_called()
    if stage != "summary":
        summary.assert_not_called()


def test_summary_task_wires_options(monkeypatch: pytest.MonkeyPatch) -> None:
    run = MagicMock(return_value=0)
    bootstrap = MagicMock()
    monkeypatch.setattr(cli, "run_summarize", run)
    monkeypatch.setattr(tasks, "_ensure_cuda_ld_library_path", bootstrap)
    with pytest.raises(SystemExit) as exc:
        tasks.summarize.body(
            Context(),
            URL,
            gemini=False,
            gemini_api_key="key",
            gemini_model="gemini-model",
            transcription_model="tiny",
            device="cpu",
            compute_type="int8",
            cookies_from_browser="firefox",
            cookies="cookies.txt",
            delete_video=True,
            display_summary=False,
        )
    assert exc.value.code == 0
    bootstrap.assert_called_once_with("cpu")
    kwargs = run.call_args.kwargs
    assert kwargs["gemini"] is False
    assert kwargs["gemini_api_key"] == "key"
    assert kwargs["gemini_model"] == "gemini-model"
    assert kwargs["transcription_model"] == "tiny"
    assert kwargs["device"] == "cpu"
    assert kwargs["compute_type"] == "int8"
    assert kwargs["cookies_from_browser"] == "firefox"
    assert kwargs["cookies"] == "cookies.txt"
    assert kwargs["delete_video"] is True
    assert kwargs["display"] is False


def test_transcript_and_summary_never_uses_gemini(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "download.mp4"
    video.write_bytes(b"fake")
    gemini = MagicMock()
    download = MagicMock(return_value=video)
    monkeypatch.setattr(cli, "summarize_youtube", gemini)
    monkeypatch.setattr(cli, "download_video", download)
    monkeypatch.setattr(cli, "is_model_cached", lambda *a: True)
    monkeypatch.setattr(
        cli,
        "transcribe",
        lambda *a, **kw: (
            iter([Segment(0.0, 2.0, "The content.")]),
            TranscriptionInfo(language="en", language_probability=0.9, duration=2.0),
        ),
    )

    def fake_summary(path, config, on_chunk_progress=None):
        target = path.with_suffix(".summary.md")
        target.write_text("# Summary")
        return target, "# Summary"

    monkeypatch.setattr(cli, "summarize_file", fake_summary)
    assert (
        cli.run_transcribe(
            URL, "tiny", "cpu", "int8", None, 1, summarize=True, display_summary=False
        )
        == 0
    )
    gemini.assert_not_called()
    download.assert_called_once()
    assert video.with_suffix(".txt").exists()
    assert video.with_suffix(".srt").exists()
    assert video.with_suffix(".summary.md").exists()
