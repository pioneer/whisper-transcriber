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
    DEFAULT_DISPLAY_SUMMARY,
    DEFAULT_SUMMARY_BASE_URL,
    DEFAULT_SUMMARY_CHUNK_SIZE,
    DEFAULT_SUMMARY_LANGUAGE,
    DEFAULT_SUMMARY_MODEL,
    SummaryConfig,
)
from .output import summary_output_path_for

SRT_TIMESTAMP_RE = re.compile(r"^(\d{2}:\d{2}:\d{2}),\d{3}\s+-->\s+(\d{2}:\d{2}:\d{2}),\d{3}$")

LANGUAGE_CODE_TO_NAME: dict[str, str] = {
    "af": "Afrikaans",
    "am": "Amharic",
    "ar": "Arabic",
    "as": "Assamese",
    "az": "Azerbaijani",
    "ba": "Bashkir",
    "be": "Belarusian",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "bo": "Tibetan",
    "br": "Breton",
    "bs": "Bosnian",
    "ca": "Catalan",
    "cs": "Czech",
    "cy": "Welsh",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "eu": "Basque",
    "fa": "Persian",
    "fi": "Finnish",
    "fo": "Faroese",
    "fr": "French",
    "gl": "Galician",
    "gu": "Gujarati",
    "ha": "Hausa",
    "haw": "Hawaiian",
    "he": "Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "ht": "Haitian Creole",
    "hu": "Hungarian",
    "hy": "Armenian",
    "id": "Indonesian",
    "is": "Icelandic",
    "it": "Italian",
    "ja": "Japanese",
    "jw": "Javanese",
    "ka": "Georgian",
    "kk": "Kazakh",
    "km": "Khmer",
    "kn": "Kannada",
    "ko": "Korean",
    "la": "Latin",
    "lb": "Luxembourgish",
    "ln": "Lingala",
    "lo": "Lao",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mg": "Malagasy",
    "mi": "Maori",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mn": "Mongolian",
    "mr": "Marathi",
    "ms": "Malay",
    "mt": "Maltese",
    "my": "Myanmar",
    "ne": "Nepali",
    "nl": "Dutch",
    "nn": "Nynorsk",
    "no": "Norwegian",
    "oc": "Occitan",
    "pa": "Punjabi",
    "pl": "Polish",
    "ps": "Pashto",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sa": "Sanskrit",
    "sd": "Sindhi",
    "si": "Sinhala",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sn": "Shona",
    "so": "Somali",
    "sq": "Albanian",
    "sr": "Serbian",
    "su": "Sundanese",
    "sv": "Swedish",
    "sw": "Swahili",
    "ta": "Tamil",
    "te": "Telugu",
    "tg": "Tajik",
    "th": "Thai",
    "tk": "Turkmen",
    "tl": "Tagalog",
    "tr": "Turkish",
    "tt": "Tatar",
    "ug": "Uyghur",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "uz": "Uzbek",
    "vi": "Vietnamese",
    "yi": "Yiddish",
    "yo": "Yoruba",
    "zh": "Chinese",
}


def resolve_language_name(code_or_name: str | None) -> str | None:
    """Resolve an ISO-639-1 code or language name to its canonical English name."""
    if not code_or_name:
        return None
    cleaned = code_or_name.strip()
    lower = cleaned.lower()
    if lower in LANGUAGE_CODE_TO_NAME:
        return LANGUAGE_CODE_TO_NAME[lower]
    for name in LANGUAGE_CODE_TO_NAME.values():
        if lower == name.lower():
            return name
    return cleaned


def detect_text_language(text: str) -> str | None:
    """Heuristically detect the primary language of transcript text."""
    if not text:
        return None

    # Strip timestamps
    cleaned = re.sub(r"\[\d{2}:\d{2}:\d{2}\s*→\s*\d{2}:\d{2}:\d{2}\]", " ", text)
    cleaned = re.sub(
        r"\d{2}:\d{2}:\d{2}[,\.]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[,\.]\d{3}",
        " ",
        cleaned,
    )

    # Check non-Latin scripts
    if re.search(r"[\u4e00-\u9fff]", cleaned):
        if re.search(r"[\u3040-\u30ff]", cleaned):
            return "Japanese"
        return "Chinese"
    if re.search(r"[\u3040-\u30ff]", cleaned):
        return "Japanese"
    if re.search(r"[\uac00-\ud7af]", cleaned):
        return "Korean"
    if re.search(r"[\u0600-\u06ff]", cleaned):
        return "Arabic"
    if re.search(r"[\u0590-\u05ff]", cleaned):
        return "Hebrew"
    if re.search(r"[\u0900-\u097f]", cleaned):
        return "Hindi"
    if re.search(r"[\u0370-\u03ff]", cleaned):
        return "Greek"

    # Cyrillic
    if re.search(r"[\u0400-\u04ff]", cleaned):
        lower = cleaned.lower()
        has_uk = any(c in "іїєґ" for c in lower)
        has_ru = any(c in "ыэъё" for c in lower)
        has_be = "ў" in lower
        if has_uk and not has_ru:
            return "Ukrainian"
        if has_ru and not has_uk:
            return "Russian"
        if has_be:
            return "Belarusian"

        words = re.findall(r"\b\w+\b", lower)
        word_counts: dict[str, int] = {}
        for w in words:
            word_counts[w] = word_counts.get(w, 0) + 1

        uk_stopwords = [
            "і",
            "в",
            "на",
            "не",
            "що",
            "як",
            "це",
            "до",
            "для",
            "та",
            "за",
            "з",
            "але",
            "ми",
            "ви",
            "дуже",
            "його",
            "її",
            "вони",
        ]
        ru_stopwords = [
            "и",
            "в",
            "на",
            "не",
            "что",
            "как",
            "это",
            "к",
            "для",
            "но",
            "за",
            "с",
            "мы",
            "вы",
            "очень",
            "его",
            "ее",
            "они",
        ]
        uk_score = sum(word_counts.get(w, 0) for w in uk_stopwords)
        ru_score = sum(word_counts.get(w, 0) for w in ru_stopwords)
        if uk_score > ru_score:
            return "Ukrainian"
        if ru_score > uk_score:
            return "Russian"
        return "Ukrainian" if has_uk else ("Russian" if has_ru else None)

    # Latin-script European languages
    lower = cleaned.lower()
    words = re.findall(r"\b\w+\b", lower)
    if not words:
        return None

    word_counts: dict[str, int] = {}
    for w in words:
        word_counts[w] = word_counts.get(w, 0) + 1

    stopwords = {
        "English": [
            "the",
            "and",
            "is",
            "of",
            "to",
            "in",
            "that",
            "it",
            "with",
            "for",
            "as",
            "was",
            "on",
            "are",
            "by",
            "this",
            "they",
            "at",
            "be",
            "from",
        ],
        "Spanish": [
            "el",
            "la",
            "de",
            "que",
            "y",
            "a",
            "en",
            "los",
            "las",
            "del",
            "por",
            "con",
            "para",
            "una",
            "uno",
            "es",
            "al",
            "lo",
            "como",
            "pero",
            "sus",
            "este",
        ],
        "German": [
            "der",
            "die",
            "das",
            "und",
            "in",
            "den",
            "von",
            "zu",
            "mit",
            "sich",
            "des",
            "auf",
            "ist",
            "im",
            "dem",
            "nicht",
            "eine",
            "als",
            "auch",
        ],
        "French": [
            "le",
            "la",
            "les",
            "de",
            "et",
            "un",
            "une",
            "du",
            "des",
            "en",
            "est",
            "que",
            "qui",
            "dans",
            "pour",
            "pas",
            "sur",
            "ce",
            "il",
        ],
        "Polish": [
            "i",
            "w",
            "na",
            "z",
            "do",
            "nie",
            "to",
            "się",
            "że",
            "o",
            "jak",
            "ale",
            "za",
            "od",
            "po",
            "tak",
            "jest",
            "co",
        ],
        "Italian": [
            "il",
            "la",
            "di",
            "che",
            "e",
            "un",
            "a",
            "in",
            "per",
            "una",
            "sono",
            "mi",
            "si",
            "ho",
            "con",
            "ti",
            "le",
            "ma",
            "da",
        ],
        "Portuguese": [
            "o",
            "a",
            "de",
            "que",
            "e",
            "do",
            "da",
            "em",
            "um",
            "para",
            "é",
            "com",
            "não",
            "uma",
            "os",
            "no",
            "se",
            "na",
            "por",
        ],
    }

    scores = {lang: sum(word_counts.get(w, 0) for w in sw) for lang, sw in stopwords.items()}
    best_lang, best_score = max(scores.items(), key=lambda x: x[1])
    if best_score >= 1:
        return best_lang

    return None


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


def build_system_prompt(config: SummaryConfig) -> str:
    """Build the system prompt, incorporating language preferences if specified."""
    if config.system_prompt:
        prompt = config.system_prompt
        if config.language:
            prompt = f"{prompt}\n\nPlease write the final summary in {config.language}."
        return prompt

    if config.language:
        lang_instruction = (
            f"1. Language: CRITICAL: Write the entire summary, including all headings, "
            f"bullet points, and text, in {config.language}."
        )
        heading_note = f"translate all heading titles to {config.language}"
    else:
        lang_instruction = (
            "1. Language: CRITICAL: Detect the language of the transcript and write the entire "
            "summary, including all headings, bullet points, and text, in the exact same language "
            "as the transcript. Do NOT default or translate to English unless the transcript "
            "itself is in English."
        )
        heading_note = "translate all heading titles to the transcript's language"

    return (
        "You are an expert summarizer. Your task is to produce a well-structured, clear, "
        "and comprehensive summary of the provided audio/video transcript.\n\n"
        "Guidelines:\n"
        f"{lang_instruction}\n"
        f"2. Structure in Markdown ({heading_note}):\n"
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

    language = config.language or os.environ.get("SUMMARY_LANGUAGE") or DEFAULT_SUMMARY_LANGUAGE

    display = config.display
    env_display = os.environ.get("SUMMARY_DISPLAY") or os.environ.get("DISPLAY_SUMMARY")
    if env_display is not None and config.display == DEFAULT_DISPLAY_SUMMARY:
        display = env_display.strip().lower() not in {"0", "false", "no", "off"}

    return SummaryConfig(
        model=model,
        base_url=base_url,
        api_key=api_key,
        system_prompt=config.system_prompt,
        chunk_size=config.chunk_size,
        language=language,
        display=display,
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
    resolved_config = resolve_summary_config(config)
    if not resolved_config.language:
        detected = detect_text_language(text)
        if detected:
            resolved_config = SummaryConfig(
                model=resolved_config.model,
                base_url=resolved_config.base_url,
                api_key=resolved_config.api_key,
                system_prompt=resolved_config.system_prompt,
                chunk_size=resolved_config.chunk_size,
                language=detected,
                display=resolved_config.display,
            )

    system_prompt = build_system_prompt(resolved_config)
    chunks = chunk_text(text, max_chars=resolved_config.chunk_size)

    lang_instruction = (
        f" in {resolved_config.language}"
        if resolved_config.language
        else " in the exact same language as the transcript (do NOT translate to English)"
    )

    if len(chunks) == 1:
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Please summarize the following transcript{lang_instruction}:\n\n{chunks[0]}"
                ),
            },
        ]
        return call_chat_completion(messages, resolved_config)

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
                    f"Provide a detailed summary of key points, facts, and topics discussed in "
                    f"this section{lang_instruction}, retaining any relevant timestamps."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Transcript section {i} of {total_chunks}:\n\n{chunk}\n\n"
                    f"Summarize this section{lang_instruction}:"
                ),
            },
        ]
        summary_piece = call_chat_completion(chunk_messages, resolved_config)
        chunk_summaries.append(f"### Section {i} Summary\n{summary_piece}")

    combined = "\n\n".join(chunk_summaries)
    final_messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "The following are section-by-section summaries of a long transcript. "
                "Synthesize them into a single, cohesive, well-structured final summary"
                f"{lang_instruction}:\n\n{combined}"
            ),
        },
    ]
    return call_chat_completion(final_messages, resolved_config)


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
