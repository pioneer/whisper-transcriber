"""Unit tests for path validation and the transcribe() orchestration.

The faster-whisper model is mocked; no GPU and no model download happen here.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from whisper_transcriber.config import TranscriptionConfig
from whisper_transcriber.transcriber import (
    MediaNotFoundError,
    Segment,
    UnsupportedMediaError,
    transcribe,
    validate_media_path,
)


def test_validate_media_path_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.mp4"
    with pytest.raises(MediaNotFoundError):
        validate_media_path(missing)


def test_validate_media_path_unsupported_extension(tmp_path: Path) -> None:
    bad_file = tmp_path / "notes.txt"
    bad_file.write_text("hello")
    with pytest.raises(UnsupportedMediaError):
        validate_media_path(bad_file)


@pytest.mark.parametrize("suffix", [".mp4", ".mkv", ".webm", ".mp3", ".wav", ".m4a"])
def test_validate_media_path_supported_extensions(tmp_path: Path, suffix: str) -> None:
    media_file = tmp_path / f"media{suffix}"
    media_file.write_bytes(b"fake data")
    validate_media_path(media_file)  # must not raise


def _mock_raw_segment(start: float, end: float, text: str) -> MagicMock:
    segment = MagicMock()
    segment.start = start
    segment.end = end
    segment.text = text
    return segment


def test_transcribe_yields_segments_and_info(tmp_path: Path) -> None:
    media_file = tmp_path / "sample.wav"
    media_file.write_bytes(b"fake data")

    raw_segments = [
        _mock_raw_segment(0.0, 3.0, "Привіт"),
        _mock_raw_segment(3.0, 6.0, "Світ"),
    ]
    raw_info = MagicMock()
    raw_info.language = "uk"
    raw_info.language_probability = 0.98
    raw_info.duration = 6.0

    mock_model = MagicMock()
    mock_model.transcribe.return_value = (iter(raw_segments), raw_info)

    config = TranscriptionConfig(language="uk")
    segments_iter, info = transcribe(media_file, config, model=mock_model)

    assert info.language == "uk"
    assert info.language_probability == pytest.approx(0.98)
    assert info.duration == pytest.approx(6.0)

    segments = list(segments_iter)
    assert segments == [
        Segment(start=0.0, end=3.0, text="Привіт"),
        Segment(start=3.0, end=6.0, text="Світ"),
    ]

    mock_model.transcribe.assert_called_once_with(
        str(media_file),
        language="uk",
        beam_size=config.beam_size,
        vad_filter=config.vad_filter,
        clip_timestamps="0",
    )


def test_transcribe_passes_start_time_as_clip_timestamps(tmp_path: Path) -> None:
    media_file = tmp_path / "sample.wav"
    media_file.write_bytes(b"fake data")

    raw_info = MagicMock()
    raw_info.language = "en"
    raw_info.language_probability = 0.9
    raw_info.duration = 10.0

    mock_model = MagicMock()
    mock_model.transcribe.return_value = (iter([]), raw_info)

    config = TranscriptionConfig()
    transcribe(media_file, config, model=mock_model, start_time=42.5)

    mock_model.transcribe.assert_called_once_with(
        str(media_file),
        language=config.language,
        beam_size=config.beam_size,
        vad_filter=config.vad_filter,
        clip_timestamps="42.5",
    )


def test_transcribe_rejects_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.mp4"
    config = TranscriptionConfig()
    with pytest.raises(MediaNotFoundError):
        transcribe(missing, config, model=MagicMock())
