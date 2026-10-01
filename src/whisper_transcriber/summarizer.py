"""AI-powered summarization of transcribed audio/video text.

Uses OpenAI-compatible Chat Completions API (works with OpenAI, Ollama,
Groq, OpenRouter, vLLM, etc.) via standard library urllib with zero extra
dependencies.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from .config import (
    DEFAULT_SUMMARY_BASE_URL,
    DEFAULT_SUMMARY_CHUNK_SIZE,
    DEFAULT_SUMMARY_MODEL,
    SummaryConfig,
)
from .output import summary_output_path_for

SRT_TIMESTAMP_RE = re.compile(r"^(\d{2}:\d{2}:\d{2}),\d{3}\s+-->\s+(\d{2}:\d{2}:\d{2}),\d{3}$")

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert summarizer. Your task is to produce a well-structured, clear, "
    "and comprehensive summary of the provided audio/video transcript.\n\n"
    "Guidelines:\n"
    "1. Language: Write the summary in the same language as the transcript "
    "(unless specifically instructed otherwise).\n"
    "2. Structure in Markdown:\n"
    "   - # Summary\n"
    "   - ## Overview: A concise executive summary (2-4 sentences) capturing the core "
    "topic and purpose.\n"
    "   - ## Key Takeaways: Bullet points of the most important insights, decisions, "
    "or conclusions.\n"
    "   - ## Main Topics & Discussion: Detailed section-by-section breakdown of the subjects "
    "discussed. Include approximate timestamps where relevant if timestamps are present "
    "in the transcript.\n"
    "   - ## Action Items / Next Steps: Any follow-up actions, recommendations, or next steps "
    "mentioned (if applicable).\n"
    "3. Tone and Accuracy: Objective, faithful to the source, concise yet informative, "
    "avoiding filler and speculation."
)


class SummaryError(Exception):
    """Base exception for summarization errors."""


class TranscriptNotFoundError(SummaryError):
    """Raised when a transcript file cannot be found."""


class EmptyTranscriptError(SummaryError):
    """Raised when a transcript file contains no text."""


class SummaryConfigError(SummaryError):
    """Raised when configuration for AI summarization is invalid or missing."""


class SummaryAPIError(SummaryError):
    """Raised when the AI API call fails or returns an error."""


def resolve_transcript_path(path: Path) -> Path:
    """Find the transcript file for a given path (media file, .txt, or .srt)."""
    if path.is_file():
        if path.suffix in {".txt", ".srt"}:
            return path
        # If media file exists, look for corresponding .txt or .srt next to it
        txt_candidate = path.with_suffix(".txt")
        if txt_candidate.is_file():
            return txt_candidate
        srt_candidate = path.with_suffix(".srt")
        if srt_candidate.is_file():
            return srt_candidate
        raise TranscriptNotFoundError(
            f"No transcript found for '{path.name}'. Looked for '{txt_candidate.name}' "
            f"and '{srt_candidate.name}'. Transcribe the file first with 'inv transcribe'."
        )

    # Path doesn't exist directly; check if .txt or .srt candidate exists
    txt_candidate = path.with_suffix(".txt")
    if txt_candidate.is_file():
        return txt_candidate
    srt_candidate = path.with_suffix(".srt")
    if srt_candidate.is_file():
        return srt_candidate

    raise TranscriptNotFoundError(f"Transcript file not found: '{path}'")


def load_transcript_text(path: Path) -> str:
    """Load text content from a .txt or .srt transcript file."""
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise EmptyTranscriptError(f"Transcript file '{path}' is empty.")

    if path.suffix == ".srt":
        return _parse_srt_text(raw)

    return raw


def _parse_srt_text(srt_content: str) -> str:
    """Parse text from SRT subtitle content, retaining readable timestamps."""
    lines = srt_content.splitlines()
    output_lines: list[str] = []
    current_time_tag: str | None = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            current_time_tag = None
            continue
        if stripped.isdigit():
            continue
        match = SRT_TIMESTAMP_RE.match(stripped)
        if match:
            current_time_tag = f"[{match.group(1)} \u2192 {match.group(2)}]"
            continue
        if current_time_tag:
            output_lines.append(f"{current_time_tag} {stripped}")
            current_time_tag = None
        else:
            output_lines.append(stripped)

    result = "\n".join(output_lines).strip()
    if not result:
        raise EmptyTranscriptError("SRT file contains no subtitle text.")
    return result


def detect_local_ollama(timeout: float = 0.5) -> bool:
    """Check if a local Ollama server is running at http://localhost:11434."""
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def load_dotenv(
    path: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    """Load key-value pairs from a .env file into os.environ if not already present."""
    if path is None:
        candidates = [
            Path.cwd() / ".env",
            Path(__file__).resolve().parents[2] / ".env",
        ]
        for candidate in candidates:
            if candidate.is_file():
                path = candidate
                break
    if path is None or not path.is_file():
        return

    target_env = os.environ if env is None else env

    try:
        content = path.read_text(encoding="utf-8")
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip("'\"")
            if key and key not in target_env:
                target_env[key] = val
    except Exception:
        pass


def resolve_summary_config(config: SummaryConfig) -> SummaryConfig:
    """Resolve API key, base URL, and model from config and environment.

    Supports automatic detection of local Ollama when no API key is set.
    """
    load_dotenv()

    api_key = (
        config.api_key or os.environ.get("SUMMARY_API_KEY") or os.environ.get("OPENAI_API_KEY")
    )

    base_url = config.base_url
    if base_url == DEFAULT_SUMMARY_BASE_URL:
        env_url = os.environ.get("SUMMARY_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        if env_url:
            base_url = env_url

    model = config.model
    if model == DEFAULT_SUMMARY_MODEL:
        env_model = os.environ.get("SUMMARY_MODEL") or os.environ.get("OPENAI_MODEL")
        if env_model:
            model = env_model

    # Auto-detect local Ollama if no API key is provided and using default OpenAI endpoint
    if not api_key and base_url == DEFAULT_SUMMARY_BASE_URL:
        if detect_local_ollama():
            base_url = "http://localhost:11434/v1"
            if model == DEFAULT_SUMMARY_MODEL:
                model = "llama3.2"

    return SummaryConfig(
        model=model,
        base_url=base_url,
        api_key=api_key,
        system_prompt=config.system_prompt,
        chunk_size=config.chunk_size,
    )


def _build_chat_endpoint(base_url: str) -> str:
    """Construct standard Chat Completions endpoint URL from base URL."""
    clean = base_url.rstrip("/")
    if clean.endswith("/chat/completions") or clean.endswith("/api/chat"):
        return clean
    if clean.endswith("/v1"):
        return f"{clean}/chat/completions"
    return f"{clean}/v1/chat/completions"


def call_chat_completion(
    messages: list[dict[str, str]],
    config: SummaryConfig,
    timeout: float = 120.0,
) -> str:
    """Send a chat completion request to the OpenAI-compatible API endpoint."""
    resolved = resolve_summary_config(config)
    endpoint = _build_chat_endpoint(resolved.base_url)

    is_local = "://localhost" in resolved.base_url or "://127.0.0.1" in resolved.base_url

    if not resolved.api_key and not is_local:
        raise SummaryConfigError(
            "No AI API key found. Please set the OPENAI_API_KEY environment variable, "
            "pass --api-key, or run a local Ollama server (http://localhost:11434)."
        )

    headers = {
        "Content-Type": "application/json",
    }
    if resolved.api_key:
        headers["Authorization"] = f"Bearer {resolved.api_key}"
    elif is_local:
        headers["Authorization"] = "Bearer ollama"

    payload = {
        "model": resolved.model,
        "messages": messages,
        "temperature": 0.3,
    }

    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            data = json.loads(body)
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            raw_err = exc.read().decode("utf-8")
            err_json = json.loads(raw_err)
            if "error" in err_json:
                e = err_json["error"]
                detail = e.get("message", str(e)) if isinstance(e, dict) else str(e)
            elif "message" in err_json:
                detail = str(err_json["message"])
        except Exception:
            pass
        msg = f"AI API HTTP error {exc.code}: {exc.reason}"
        if detail:
            msg = f"{msg} - {detail}"
        raise SummaryAPIError(msg) from exc
    except urllib.error.URLError as exc:
        raise SummaryAPIError(f"Failed to connect to AI API at {endpoint}: {exc.reason}") from exc
    except Exception as exc:
        raise SummaryAPIError(f"AI API request failed: {exc}") from exc

    if "choices" in data and len(data["choices"]) > 0:
        content = data["choices"][0].get("message", {}).get("content")
        if content:
            return content.strip()
    if "message" in data and "content" in data["message"]:
        return data["message"]["content"].strip()
    if "response" in data:
        return data["response"].strip()

    raise SummaryAPIError(f"Unexpected response format from AI API: {data}")


def chunk_text(text: str, max_chars: int = DEFAULT_SUMMARY_CHUNK_SIZE) -> list[str]:
    """Split text into chunks of at most max_chars, splitting on line boundaries."""
    if len(text) <= max_chars:
        return [text]

    lines = text.splitlines(keepends=True)
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_len = 0

    for line in lines:
        # If a single line exceeds max_chars by itself
        if len(line) > max_chars:
            if current_chunk:
                chunks.append("".join(current_chunk))
                current_chunk = []
                current_len = 0
            for i in range(0, len(line), max_chars):
                chunks.append(line[i : i + max_chars])
            continue

        if current_len + len(line) > max_chars and current_chunk:
            chunks.append("".join(current_chunk))
            current_chunk = []
            current_len = 0

        current_chunk.append(line)
        current_len += len(line)

    if current_chunk:
        chunks.append("".join(current_chunk))

    return chunks


def summarize_text(
    text: str,
    config: SummaryConfig,
    on_chunk_progress: Callable[[int, int], None] | None = None,
) -> str:
    """Generate a summary of transcript text, handling multi-chunk map-reduce if needed."""
    system_prompt = config.system_prompt or DEFAULT_SYSTEM_PROMPT
    chunks = chunk_text(text, max_chars=config.chunk_size)

    if len(chunks) == 1:
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Please summarize the following transcript:\n\n{chunks[0]}",
            },
        ]
        return call_chat_completion(messages, config)

    # Multi-chunk map-reduce
    chunk_summaries: list[str] = []
    total_chunks = len(chunks)
    for i, chunk in enumerate(chunks, 1):
        if on_chunk_progress:
            on_chunk_progress(i, total_chunks)
        chunk_messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant summarizing a section of a long transcript. "
                    "Provide a detailed summary of key points, facts, and topics discussed in "
                    "this section, retaining any relevant timestamps."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Transcript section {i} of {total_chunks}:\n\n{chunk}\n\n"
                    "Summarize this section:"
                ),
            },
        ]
        summary_piece = call_chat_completion(chunk_messages, config)
        chunk_summaries.append(f"### Section {i} Summary\n{summary_piece}")

    combined = "\n\n".join(chunk_summaries)
    final_messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "The following are section-by-section summaries of a long transcript. "
                "Synthesize them into a single, cohesive, well-structured final summary:\n\n"
                f"{combined}"
            ),
        },
    ]
    return call_chat_completion(final_messages, config)


def summarize_file(
    input_path: Path,
    config: SummaryConfig,
    output_path: Path | None = None,
    on_chunk_progress: Callable[[int, int], None] | None = None,
) -> tuple[Path, str]:
    """Load transcript from input_path, generate summary, and save to output_path."""
    resolved_path = resolve_transcript_path(input_path)
    text = load_transcript_text(resolved_path)

    target_output = output_path if output_path is not None else summary_output_path_for(input_path)
    target_output.parent.mkdir(parents=True, exist_ok=True)

    summary = summarize_text(text, config, on_chunk_progress=on_chunk_progress)
    target_output.write_text(summary + "\n", encoding="utf-8")
    return target_output, summary
