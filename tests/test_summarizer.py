"""Unit tests for the AI summarizer module.

All network calls are mocked; no actual API keys or remote servers required.
"""

from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from whisper_transcriber.config import SummaryConfig
from whisper_transcriber.summarizer import (
    EmptyTranscriptError,
    SummaryAPIError,
    SummaryConfigError,
    TranscriptNotFoundError,
    chunk_text,
    detect_local_ollama,
    load_dotenv,
    load_transcript_text,
    resolve_summary_config,
    resolve_transcript_path,
    summarize_file,
    summarize_text,
)


def test_resolve_transcript_path_direct_txt(tmp_path: Path) -> None:
    txt_file = tmp_path / "meeting.txt"
    txt_file.write_text("transcript text", encoding="utf-8")
    assert resolve_transcript_path(txt_file) == txt_file


def test_resolve_transcript_path_direct_srt(tmp_path: Path) -> None:
    srt_file = tmp_path / "meeting.srt"
    srt_file.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello", encoding="utf-8")
    assert resolve_transcript_path(srt_file) == srt_file


def test_resolve_transcript_path_from_media_file_finds_txt(tmp_path: Path) -> None:
    media = tmp_path / "meeting.mp4"
    media.write_bytes(b"dummy video")
    txt_file = tmp_path / "meeting.txt"
    txt_file.write_text("transcript text", encoding="utf-8")
    assert resolve_transcript_path(media) == txt_file


def test_resolve_transcript_path_from_media_file_finds_srt(tmp_path: Path) -> None:
    media = tmp_path / "meeting.mp4"
    media.write_bytes(b"dummy video")
    srt_file = tmp_path / "meeting.srt"
    srt_file.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello", encoding="utf-8")
    assert resolve_transcript_path(media) == srt_file


def test_resolve_transcript_path_missing_raises_error(tmp_path: Path) -> None:
    media = tmp_path / "meeting.mp4"
    media.write_bytes(b"dummy video")
    with pytest.raises(TranscriptNotFoundError, match="No transcript found"):
        resolve_transcript_path(media)


def test_resolve_transcript_path_nonexistent_raises_error(tmp_path: Path) -> None:
    with pytest.raises(TranscriptNotFoundError, match="Transcript file not found"):
        resolve_transcript_path(tmp_path / "nonexistent.txt")


def test_load_transcript_text_txt(tmp_path: Path) -> None:
    txt_file = tmp_path / "audio.txt"
    txt_file.write_text("[00:00:00 → 00:00:05] Hello world", encoding="utf-8")
    assert load_transcript_text(txt_file) == "[00:00:00 → 00:00:05] Hello world"


def test_load_transcript_text_empty_txt(tmp_path: Path) -> None:
    txt_file = tmp_path / "empty.txt"
    txt_file.write_text("   \n  \n", encoding="utf-8")
    with pytest.raises(EmptyTranscriptError):
        load_transcript_text(txt_file)


def test_load_transcript_text_srt(tmp_path: Path) -> None:
    srt_file = tmp_path / "subs.srt"
    srt_content = (
        "1\n"
        "00:00:01,000 --> 00:00:04,500\n"
        "Welcome to the session.\n\n"
        "2\n"
        "00:00:05,000 --> 00:00:08,200\n"
        "Today we discuss AI.\n"
    )
    srt_file.write_text(srt_content, encoding="utf-8")
    parsed = load_transcript_text(srt_file)
    assert "[00:00:01 → 00:00:04] Welcome to the session." in parsed
    assert "[00:00:05 → 00:00:08] Today we discuss AI." in parsed


def test_load_transcript_text_empty_srt(tmp_path: Path) -> None:
    srt_file = tmp_path / "empty.srt"
    srt_file.write_text("1\n00:00:01,000 --> 00:00:04,500\n\n", encoding="utf-8")
    with pytest.raises(EmptyTranscriptError):
        load_transcript_text(srt_file)


def test_chunk_text_under_limit() -> None:
    text = "Short text under limit"
    chunks = chunk_text(text, max_chars=100)
    assert chunks == [text]


def test_chunk_text_splits_by_lines() -> None:
    lines = [f"Line {i} content is here" for i in range(20)]
    full_text = "\n".join(lines)
    chunks = chunk_text(full_text, max_chars=80)
    assert len(chunks) > 1
    # Check that all content is preserved
    assert "".join(chunks).replace("\n", "") == full_text.replace("\n", "")


def test_chunk_text_long_single_line() -> None:
    long_line = "A" * 150
    chunks = chunk_text(long_line, max_chars=50)
    assert len(chunks) == 3
    assert "".join(chunks) == long_line


def test_detect_local_ollama_success() -> None:
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.__enter__.return_value = mock_resp
    with patch("urllib.request.urlopen", return_value=mock_resp):
        assert detect_local_ollama() is True


def test_detect_local_ollama_failure() -> None:
    with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
        assert detect_local_ollama() is False


def test_resolve_summary_config_prefers_cli() -> None:
    cfg = SummaryConfig(
        model="custom-model",
        base_url="https://custom.endpoint/v1",
        api_key="cli-key",
    )
    resolved = resolve_summary_config(cfg)
    assert resolved.model == "custom-model"
    assert resolved.base_url == "https://custom.endpoint/v1"
    assert resolved.api_key == "cli-key"


def test_resolve_summary_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://env.endpoint/v1")
    monkeypatch.setenv("OPENAI_MODEL", "env-model")

    cfg = SummaryConfig()
    resolved = resolve_summary_config(cfg)
    assert resolved.api_key == "env-key"
    assert resolved.base_url == "https://env.endpoint/v1"
    assert resolved.model == "env-model"


def test_load_dotenv(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# Comment\n"
        "OPENAI_API_KEY=test-from-dotenv\n"
        "SUMMARY_MODEL='custom-dotenv-model'\n"
        "INVALID_LINE_WITHOUT_EQUALS\n",
        encoding="utf-8",
    )
    fake_env: dict[str, str] = {}
    load_dotenv(env_file, env=fake_env)
    assert fake_env["OPENAI_API_KEY"] == "test-from-dotenv"
    assert fake_env["SUMMARY_MODEL"] == "custom-dotenv-model"


def test_resolve_summary_config_auto_detects_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("SUMMARY_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("SUMMARY_MODEL", raising=False)

    with patch("whisper_transcriber.summarizer.detect_local_ollama", return_value=True):
        cfg = SummaryConfig()
        resolved = resolve_summary_config(cfg)
        assert resolved.base_url == "http://localhost:11434/v1"
        assert resolved.model == "llama3.2"
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("SUMMARY_API_KEY", raising=False)

    with patch("whisper_transcriber.summarizer.detect_local_ollama", return_value=True):
        cfg = SummaryConfig()
        resolved = resolve_summary_config(cfg)
        assert resolved.base_url == "http://localhost:11434/v1"
        assert resolved.model == "llama3.2"


def test_summarize_text_single_chunk() -> None:
    cfg = SummaryConfig(api_key="test-key")
    fake_response = {"choices": [{"message": {"content": "# Executive Summary\nAll went well."}}]}
    mock_http_resp = MagicMock()
    mock_http_resp.read.return_value = json.dumps(fake_response).encode("utf-8")
    mock_http_resp.__enter__.return_value = mock_http_resp

    with patch("urllib.request.urlopen", return_value=mock_http_resp) as mock_urlopen:
        result = summarize_text("This is a transcript text.", cfg)
        assert "# Executive Summary" in result
        mock_urlopen.assert_called_once()


def test_summarize_text_multi_chunk() -> None:
    cfg = SummaryConfig(api_key="test-key", chunk_size=50)
    long_text = (
        "Line 1 is about introduction.\n"
        "Line 2 discusses methodology.\n"
        "Line 3 shows results.\n"
        "Line 4 concludes."
    )

    def fake_urlopen(req, timeout=120.0):
        mock_http_resp = MagicMock()
        mock_http_resp.read.return_value = json.dumps(
            {"choices": [{"message": {"content": "Summary piece"}}]}
        ).encode("utf-8")
        mock_http_resp.__enter__.return_value = mock_http_resp
        return mock_http_resp

    with patch("urllib.request.urlopen", side_effect=fake_urlopen) as mock_urlopen:
        progress_calls: list[tuple[int, int]] = []
        result = summarize_text(
            long_text,
            cfg,
            on_chunk_progress=lambda cur, tot: progress_calls.append((cur, tot)),
        )
        assert result == "Summary piece"
        assert len(progress_calls) > 1
        # Number of calls: len(chunks) for pieces + 1 for final synthesis
        assert mock_urlopen.call_count == len(progress_calls) + 1


def test_summarize_text_missing_api_key_raises_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("SUMMARY_API_KEY", raising=False)

    with patch("whisper_transcriber.summarizer.detect_local_ollama", return_value=False):
        cfg = SummaryConfig()
        with pytest.raises(SummaryConfigError, match="No AI API key found"):
            summarize_text("Sample text", cfg)


def test_summarize_text_api_http_error() -> None:
    cfg = SummaryConfig(api_key="test-key")
    error_content = json.dumps({"error": {"message": "Rate limit exceeded"}}).encode("utf-8")
    http_err = urllib.error.HTTPError(
        url="https://api.openai.com/v1/chat/completions",
        code=429,
        msg="Too Many Requests",
        hdrs=None,  # type: ignore[arg-type]
        fp=io.BytesIO(error_content),
    )

    with patch("urllib.request.urlopen", side_effect=http_err):
        with pytest.raises(SummaryAPIError, match="Rate limit exceeded"):
            summarize_text("Sample text", cfg)


def test_summarize_file_creates_output(tmp_path: Path) -> None:
    txt_file = tmp_path / "lecture.txt"
    txt_file.write_text("Spoken words during the lecture.", encoding="utf-8")

    cfg = SummaryConfig(api_key="test-key")
    fake_response = {"choices": [{"message": {"content": "# Lecture Summary\nKey points..."}}]}
    mock_http_resp = MagicMock()
    mock_http_resp.read.return_value = json.dumps(fake_response).encode("utf-8")
    mock_http_resp.__enter__.return_value = mock_http_resp

    with patch("urllib.request.urlopen", return_value=mock_http_resp):
        out_path, summary_text = summarize_file(txt_file, cfg)
        assert out_path == tmp_path / "lecture.summary.md"
        assert out_path.is_file()
        assert out_path.read_text(encoding="utf-8").startswith("# Lecture Summary")
        assert summary_text.startswith("# Lecture Summary")
