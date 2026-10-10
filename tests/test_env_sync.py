"""Environment synchronization tests use temporary files, never real credentials."""

from pathlib import Path

import pytest
from invoke import Context

import tasks
from whisper_transcriber.env_sync import sync_env


def test_appends_missing_variables_preserving_existing_values(tmp_path: Path) -> None:
    example = tmp_path / ".env.example"
    env = tmp_path / ".env"
    example.write_text(
        "# Configuration\n"
        "EXISTING=placeholder\n"
        "# OPTIONAL=default\n"
        "# GEMINI_API_KEY=placeholder\n"
        "NEW_SETTING=value\n"
        "# CUSTOM=example\n"
    )
    original = b"EXISTING=private-value\n# OPTIONAL=user-default\nCUSTOM=custom\n"
    env.write_bytes(original)

    assert sync_env(example, env) == ["GEMINI_API_KEY", "NEW_SETTING"]
    assert env.read_bytes() == (original + b"# GEMINI_API_KEY=placeholder\nNEW_SETTING=value\n")
    assert sync_env(example, env) == []


@pytest.mark.parametrize("original", [b"", b"EXISTING=mine", b"EXISTING=mine\n"])
def test_empty_or_unterminated_target(tmp_path: Path, original: bytes) -> None:
    example = tmp_path / ".env.example"
    env = tmp_path / ".env"
    example.write_text("NEW=value\n")
    env.write_bytes(original)
    assert sync_env(example, env) == ["NEW"]
    separator = b"\n" if original and not original.endswith(b"\n") else b""
    assert env.read_bytes() == original + separator + b"NEW=value\n"


def test_preserves_crlf_and_unicode(tmp_path: Path) -> None:
    example = tmp_path / ".env.example"
    env = tmp_path / ".env"
    example.write_text("# NEW=value\n")
    original = "LANGUAGE=Українська\r\n# Comment\r\n".encode()
    env.write_bytes(original)
    assert sync_env(example, env) == ["NEW"]
    assert env.read_bytes() == original + b"# NEW=value\r\n"


def test_missing_env_copies_complete_example(tmp_path: Path) -> None:
    example = tmp_path / ".env.example"
    env = tmp_path / ".env"
    template = b"# Required settings\r\nKEY=placeholder\r\n\r\n# Optional\r\n# MODEL=default\r\n"
    example.write_bytes(template)
    assert sync_env(example, env) == ["KEY", "MODEL"]
    assert env.read_bytes() == template


def test_recognizes_assignment_spacing_and_export(tmp_path: Path) -> None:
    example = tmp_path / ".env.example"
    env = tmp_path / ".env"
    example.write_text(
        "ACTIVE=example\n# OPTIONAL=example\nEXPORTED=example\nDISABLED=example\n"
        " NEW = first\n# NEW=duplicate\n"
        "# Not a variable\nINVALID-LINE=value\n"
    )
    original = (
        b" ACTIVE = mine\n  # OPTIONAL = mine\nexport EXPORTED=mine\n# export DISABLED=mine\n"
    )
    env.write_bytes(original)
    assert sync_env(example, env) == ["NEW"]
    assert env.read_bytes() == original + b" NEW = first\n"


def test_no_missing_variables_does_not_write(tmp_path: Path) -> None:
    example = tmp_path / ".env.example"
    env = tmp_path / ".env"
    example.write_text("KEY=placeholder\n")
    env.write_bytes(b"KEY=mine")
    before = env.stat().st_mtime_ns
    assert sync_env(example, env) == []
    assert env.read_bytes() == b"KEY=mine"
    assert env.stat().st_mtime_ns == before


def test_missing_example_leaves_target_unchanged(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("KEY=mine")
    with pytest.raises(FileNotFoundError):
        sync_env(tmp_path / ".env.example", env)
    assert env.read_text() == "KEY=mine"


@pytest.mark.parametrize("invalid_file", ["example", "env"])
def test_invalid_encoding_does_not_mutate_target(tmp_path: Path, invalid_file: str) -> None:
    example = tmp_path / ".env.example"
    env = tmp_path / ".env"
    example.write_bytes(b"NEW=value\n" if invalid_file != "example" else b"\xff")
    original = b"KEY=mine\n" if invalid_file != "env" else b"\xff"
    env.write_bytes(original)
    with pytest.raises(UnicodeError):
        sync_env(example, env)
    assert env.read_bytes() == original


def test_env_sync_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env.example").write_text("API_KEY=placeholder\n# NEW=default\n")
    (tmp_path / ".env").write_text("API_KEY=private-value\n")
    tasks.env_sync.body(Context())
    output = capsys.readouterr().out
    assert "NEW" in output
    assert "private-value" not in output
    assert "default" not in output
    assert (tmp_path / ".env").read_text() == "API_KEY=private-value\n# NEW=default\n"
    tasks.env_sync.body(Context())
    assert "up to date" in capsys.readouterr().out


def test_env_sync_task_reports_missing_example(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as exc:
        tasks.env_sync.body(Context())
    assert exc.value.code == 1
    assert "Error syncing" in capsys.readouterr().err
    assert not (tmp_path / ".env").exists()


def test_env_sync_task_reports_encoding_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env.example").write_bytes(b"\xff")
    with pytest.raises(SystemExit) as exc:
        tasks.env_sync.body(Context())
    assert exc.value.code == 1
    assert "UTF-8" in capsys.readouterr().err
