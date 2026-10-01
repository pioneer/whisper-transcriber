"""Local video/audio transcription powered by faster-whisper."""

from .config import SummaryConfig, TranscriptionConfig
from .summarizer import (
    SummaryError,
    detect_text_language,
    resolve_language_name,
    summarize_file,
    summarize_text,
)
from .transcriber import transcribe

__all__ = [
    "SummaryConfig",
    "SummaryError",
    "TranscriptionConfig",
    "detect_text_language",
    "resolve_language_name",
    "summarize_file",
    "summarize_text",
    "transcribe",
]
