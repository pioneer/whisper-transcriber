"""Centralized configuration and defaults for whisper-transcriber."""

from __future__ import annotations

from dataclasses import dataclass

#: Supported media file extensions (video and audio).
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".mp4", ".mkv", ".webm", ".mp3", ".wav", ".m4a"})

# Available models: tiny, base, small, medium, large, large-v3
DEFAULT_MODEL = "large-v3"
DEFAULT_DEVICE = "cuda"
# int8_float16 needs compute capability >= 7.0 (Turing+). Pascal cards like the
# GTX 1050 Ti (6.1) only support float16-free compute types efficiently, so
# int8_float32 is the fastest type that works everywhere; override with
# --compute-type=int8_float16 on newer GPUs.
DEFAULT_COMPUTE_TYPE = "int8_float32"
DEFAULT_BEAM_SIZE = 5
DEFAULT_VAD_FILTER = True
# None means "automatic language detection".
DEFAULT_LANGUAGE: str | None = None

# Where videos downloaded from a URL are saved.
DEFAULT_VIDEO_DOWNLOAD_DIR = "~/Video"
# Command used to download a video URL. ``{url}`` and ``{output}`` are
# substituted with the source URL and the destination output template.
DEFAULT_VIDEO_DOWNLOAD_COMMAND = "yt-dlp -o {output} {url}"
# Whether to delete a downloaded video once it has been transcribed.
DEFAULT_DELETE_VIDEO_AFTER_TRANSCRIBE = False


@dataclass(frozen=True, slots=True)
class TranscriptionConfig:
    """All tunable parameters for a single transcription run."""

    model: str = DEFAULT_MODEL
    device: str = DEFAULT_DEVICE
    compute_type: str = DEFAULT_COMPUTE_TYPE
    beam_size: int = DEFAULT_BEAM_SIZE
    vad_filter: bool = DEFAULT_VAD_FILTER
    language: str | None = DEFAULT_LANGUAGE
