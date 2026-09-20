"""Core transcription logic built on top of faster-whisper.

This module intentionally knows nothing about the CLI or Invoke; it exposes
plain functions/generators that ``cli.py`` orchestrates.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import SUPPORTED_EXTENSIONS, TranscriptionConfig


class TranscriptionError(Exception):
    """Base class for user-facing, readable transcription errors."""


class MediaNotFoundError(TranscriptionError):
    """The requested media file does not exist."""


class UnsupportedMediaError(TranscriptionError):
    """The file extension is not one of the supported formats."""


class CudaUnavailableError(TranscriptionError):
    """CUDA was requested but is not usable on this machine."""


class CudaLibraryError(TranscriptionError):
    """CUDA is present but required libraries (cuBLAS/cuDNN) could not be loaded."""


class OutOfMemoryError(TranscriptionError):
    """The GPU ran out of VRAM."""


class ModelDownloadError(TranscriptionError):
    """The model could not be downloaded from the Hugging Face hub."""


class UnsupportedComputeTypeError(TranscriptionError):
    """The requested compute type is not supported on this device/backend."""


@dataclass(slots=True)
class Segment:
    """A single transcribed segment, decoupled from faster-whisper's type."""

    start: float
    end: float
    text: str


@dataclass(slots=True)
class TranscriptionInfo:
    """Metadata about the media/transcription, known before segments finish."""

    language: str
    language_probability: float
    duration: float


def validate_media_path(path: Path) -> None:
    """Raise a readable error if ``path`` is not a supported, existing file."""
    if not path.exists():
        raise MediaNotFoundError(f"File not found: {path}")
    if not path.is_file():
        raise MediaNotFoundError(f"Not a regular file: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise UnsupportedMediaError(
            f"Unsupported file extension '{path.suffix}'. Supported extensions: {supported}"
        )


def _classify_load_error(exc: Exception, device: str) -> TranscriptionError:
    message = str(exc).lower()

    if "compute type" in message and ("support" in message or "not support" in message):
        return UnsupportedComputeTypeError(
            "The requested --compute-type is not supported on this device/backend.\n"
            "Run `uv run inv diagnose` to see supported compute types, then retry with "
            "one of them (e.g. --compute-type=int8_float32 or --compute-type=int8).\n"
            f"Original error: {exc}"
        )

    if device == "cuda":
        if any(lib in message for lib in ("libcudnn", "libcublas", "cudnn", "cublas")):
            return CudaLibraryError(
                "Failed to load CUDA libraries (cuBLAS/cuDNN) required by CTranslate2.\n"
                "This usually means the required CUDA 12 shared libraries are not on "
                "LD_LIBRARY_PATH. Run `uv run inv cuda-env` to see the required export, "
                "or retry with --device=cpu. See README 'CUDA troubleshooting' section.\n"
                f"Original error: {exc}"
            )
        no_device = "no cuda-capable device" in message
        driver_issue = "cuda driver" in message or "cuda_error" in message
        if no_device or driver_issue:
            return CudaUnavailableError(
                "CUDA was requested but no usable CUDA device was found.\n"
                "Check `nvidia-smi` and `uv run inv diagnose`, or retry with --device=cpu.\n"
                f"Original error: {exc}"
            )
        if "out of memory" in message or "cuda_error_out_of_memory" in message:
            return OutOfMemoryError(
                "The GPU ran out of VRAM while loading the model.\n"
                "Try a smaller model (e.g. --model=small) or a lighter --compute-type "
                "(e.g. int8), or retry with --device=cpu.\n"
                f"Original error: {exc}"
            )

    download_error_terms = (
        "could not download",
        "connection",
        "huggingface",
        "resolve host",
        "timed out",
    )
    if any(term in message for term in download_error_terms):
        return ModelDownloadError(
            "Failed to download the model files (no cached copy found).\n"
            "Check your internet connection, then retry. Models are cached under "
            "~/.cache/huggingface once downloaded.\n"
            f"Original error: {exc}"
        )

    return TranscriptionError(f"Failed to load model: {exc}")


def load_model(config: TranscriptionConfig) -> Any:
    """Load and return a ``faster_whisper.WhisperModel``, raising readable errors."""
    from faster_whisper import WhisperModel

    try:
        return WhisperModel(
            config.model,
            device=config.device,
            compute_type=config.compute_type,
        )
    except Exception as exc:
        raise _classify_load_error(exc, config.device) from exc


def transcribe(
    path: Path,
    config: TranscriptionConfig,
    model: Any | None = None,
) -> tuple[Iterator[Segment], TranscriptionInfo]:
    """Transcribe ``path`` and return ``(segments_iterator, info)``.

    ``info`` (language, language_probability, duration) is available
    immediately; ``segments_iterator`` is a lazy generator that must be
    consumed to actually run inference, one segment at a time.
    """
    validate_media_path(path)

    if model is None:
        model = load_model(config)

    try:
        raw_segments, raw_info = model.transcribe(
            str(path),
            language=config.language,
            beam_size=config.beam_size,
            vad_filter=config.vad_filter,
        )
    except Exception as exc:
        raise _classify_load_error(exc, config.device) from exc

    info = TranscriptionInfo(
        language=raw_info.language,
        language_probability=raw_info.language_probability,
        duration=raw_info.duration,
    )

    def _iter_segments() -> Iterator[Segment]:
        try:
            for raw_segment in raw_segments:
                yield Segment(start=raw_segment.start, end=raw_segment.end, text=raw_segment.text)
        except Exception as exc:
            raise _classify_load_error(exc, config.device) from exc

    return _iter_segments(), info
