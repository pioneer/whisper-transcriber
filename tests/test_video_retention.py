"""Downloaded-media retention defaults and CLI/environment precedence."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from invoke import Context
from invoke.parser import Parser, ParserContext

import tasks
from whisper_transcriber import cli
from whisper_transcriber.config import resolve_delete_video
from whisper_transcriber.summarizer import load_dotenv
from whisper_transcriber.transcriber import Segment, TranscriptionInfo


@pytest.mark.parametrize(
    ("env", "override", "expected"),
    [
        (None, None, True),
        ("false", None, False),
        ("  OFF ", None, False),
        ("0", None, False),
        ("no", None, False),
        ("true", None, True),
        ("YES", None, True),
        ("on", None, True),
        ("1", None, True),
        ("false", True, True),
        ("true", False, False),
        ("invalid", True, True),
        ("invalid", False, False),
    ],
)
def test_retention_precedence(
    monkeypatch: pytest.MonkeyPatch, env: str | None, override: bool | None, expected: bool
) -> None:
    monkeypatch.delenv("DELETE_VIDEO", raising=False)
    if env is not None:
        monkeypatch.setenv("DELETE_VIDEO", env)
    assert resolve_delete_video(override) is expected


@pytest.mark.parametrize("value", ["", "maybe", "2"])
def test_invalid_environment(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("DELETE_VIDEO", value)
    with pytest.raises(ValueError, match="DELETE_VIDEO"):
        resolve_delete_video(None)


def test_retention_from_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("DELETE_VIDEO=false\n")
    environment: dict[str, str] = {}
    load_dotenv(env_file, env=environment)
    monkeypatch.setenv("DELETE_VIDEO", environment["DELETE_VIDEO"])
    assert resolve_delete_video(None) is False
    assert resolve_delete_video(True) is True


@pytest.mark.parametrize("command", ["transcribe", "summarize"])
@pytest.mark.parametrize(
    ("flags", "expected"),
    [([], None), (["--keep-video"], False), (["--delete-video"], True)],
)
def test_actual_invoke_flags(
    monkeypatch: pytest.MonkeyPatch, command: str, flags: list[str], expected: bool | None
) -> None:
    task = getattr(tasks, command)
    parser = Parser(contexts=[ParserContext(name=command, args=task.get_arguments())])
    parsed = parser.parse_argv([command, "video.wav", *flags])
    run = MagicMock(return_value=0)
    monkeypatch.setattr(cli, f"run_{command}", run)
    monkeypatch.setattr(tasks, "_ensure_cuda_ld_library_path", lambda *args: None)
    with pytest.raises(SystemExit) as exc:
        task.body(Context(), **parsed[0].as_kwargs)
    assert exc.value.code == 0
    assert run.call_args.kwargs["delete_video"] is expected


@pytest.mark.parametrize("command", ["transcribe", "summarize"])
def test_conflicting_flags_rejected(
    monkeypatch: pytest.MonkeyPatch, command: str, capsys: pytest.CaptureFixture[str]
) -> None:
    run = MagicMock()
    monkeypatch.setattr(cli, f"run_{command}", run)
    with pytest.raises(SystemExit) as exc:
        getattr(tasks, command).body(Context(), "video.wav", delete_video=True, keep_video=True)
    assert exc.value.code == 1
    assert "cannot be used together" in capsys.readouterr().err
    run.assert_not_called()


@pytest.mark.parametrize("command", ["transcribe", "summarize"])
@pytest.mark.parametrize(
    ("env", "override", "deleted"),
    [(None, None, True), ("false", None, False), ("false", True, True), ("true", False, False)],
)
def test_url_retention(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    env: str | None,
    override: bool | None,
    deleted: bool,
) -> None:
    video = tmp_path / "video.wav"
    video.write_bytes(b"media")
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setattr(cli, "resolve_summary_config", lambda config: config)
    monkeypatch.delenv("DELETE_VIDEO", raising=False)
    if env is not None:
        monkeypatch.setenv("DELETE_VIDEO", env)
    monkeypatch.setattr(cli, "download_video", lambda *a, **kw: video)
    monkeypatch.setattr(cli, "is_model_cached", lambda *a: True)
    monkeypatch.setattr(
        cli,
        "transcribe",
        lambda *a, **kw: (
            iter([Segment(0, 1, "speech")]),
            TranscriptionInfo(language="en", language_probability=1, duration=1),
        ),
    )

    def summary(path, config, output_path=None, on_chunk_progress=None):
        target = path.with_suffix(".summary.md")
        target.write_text("# Summary")
        return target, "# Summary"

    monkeypatch.setattr(cli, "summarize_file", summary)
    if command == "transcribe":
        result = cli.run_transcribe(
            "https://example.com/video",
            "tiny",
            "cpu",
            "int8",
            None,
            1,
            delete_video=override,
        )
    else:
        result = cli.run_summarize(
            "https://example.com/video", device="cpu", delete_video=override, display=False
        )
    assert result == 0
    assert video.exists() is not deleted
    assert video.with_suffix(".txt").exists()
    assert video.with_suffix(".srt").exists()


@pytest.mark.parametrize("command", ["transcribe", "summarize"])
def test_invalid_retention_fails_before_download(
    monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setattr(cli, "resolve_summary_config", lambda config: config)
    monkeypatch.setenv("DELETE_VIDEO", "invalid")
    download = MagicMock()
    monkeypatch.setattr(cli, "download_video", download)
    if command == "transcribe":
        result = cli.run_transcribe("https://example.com/video", "tiny", "cpu", "int8", None, 1)
    else:
        result = cli.run_summarize("https://example.com/video")
    assert result == 1
    download.assert_not_called()


@pytest.mark.parametrize("failure", ["error", "cancel"])
def test_failed_transcription_keeps_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    video = tmp_path / "video.wav"
    video.write_bytes(b"media")
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.delenv("DELETE_VIDEO", raising=False)
    monkeypatch.setattr(cli, "download_video", lambda *a, **kw: video)
    monkeypatch.setattr(cli, "is_model_cached", lambda *a: True)
    monkeypatch.setattr(
        cli,
        "transcribe",
        MagicMock(
            side_effect=(
                cli.TranscriptionError("failed") if failure == "error" else KeyboardInterrupt()
            )
        ),
    )
    assert cli.run_transcribe("https://example.com/video", "tiny", "cpu", "int8", None, 1) == (
        1 if failure == "error" else 130
    )
    assert video.exists()


def test_local_source_never_deleted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video = tmp_path / "local.wav"
    video.write_bytes(b"media")
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setattr(cli, "is_model_cached", lambda *a: True)
    monkeypatch.setattr(
        cli,
        "transcribe",
        lambda *a, **kw: (
            iter([Segment(0, 1, "speech")]),
            TranscriptionInfo(language="en", language_probability=1, duration=1),
        ),
    )
    assert cli.run_transcribe(str(video), "tiny", "cpu", "int8", None, 1, delete_video=True) == 0
    assert video.read_bytes() == b"media"
