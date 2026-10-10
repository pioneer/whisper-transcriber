"""Gemini request/response tests; no network or credentials required."""

import io
import json
import urllib.error
from unittest.mock import MagicMock

import pytest

from whisper_transcriber import youtube
from whisper_transcriber.config import SummaryConfig
from whisper_transcriber.summarizer import SummaryAPIError, SummaryConfigError

VIDEO_ID = "9hE5-98ZeCg"
URL = f"https://www.youtube.com/watch?v={VIDEO_ID}"


@pytest.mark.parametrize(
    "url",
    [
        URL + "&t=30",
        f"https://youtu.be/{VIDEO_ID}?si=share",
        f"https://m.youtube.com/watch?v={VIDEO_ID}",
        f"https://music.youtube.com/watch?v={VIDEO_ID}",
        f"https://www.youtube.com/shorts/{VIDEO_ID}",
        f"https://www.youtube.com/live/{VIDEO_ID}",
        f"https://www.youtube-nocookie.com/embed/{VIDEO_ID}",
    ],
)
def test_youtube_video_id(url: str) -> None:
    assert youtube.youtube_video_id(url) == VIDEO_ID


@pytest.mark.parametrize(
    "url",
    [
        "video.mp4",
        "https://example.com/watch?v=" + VIDEO_ID,
        "https://youtube.com.evil.example/watch?v=" + VIDEO_ID,
        "https://youtube.com@evil.example/watch?v=" + VIDEO_ID,
        "https://youtube.com/playlist?list=abc",
        "https://youtube.com/watch?v=invalid",
        "https://youtu.be/" + VIDEO_ID + "/extra",
        "ftp://youtube.com/watch?v=" + VIDEO_ID,
    ],
)
def test_non_video_urls(url: str) -> None:
    assert youtube.youtube_video_id(url) is None


def _response(data: object) -> MagicMock:
    response = MagicMock()
    response.__enter__.return_value.read.return_value = json.dumps(data).encode()
    return response


def test_gemini_request_and_response(monkeypatch: pytest.MonkeyPatch) -> None:
    opener = MagicMock(
        return_value=_response(
            {
                "status": "completed",
                "steps": [
                    {"type": "thought", "content": [{"type": "text", "text": "Not the summary"}]},
                    {
                        "type": "model_output",
                        "content": [
                            {"type": "text", "text": " # Summary "},
                            {"type": "text", "text": " Main points. "},
                        ],
                    },
                ],
            }
        )
    )
    monkeypatch.setattr(youtube.urllib.request, "urlopen", opener)
    result = youtube.summarize_youtube(
        URL + "&t=100",
        SummaryConfig(language="Ukrainian", system_prompt="Be concise."),
        api_key="test-key",
        model="test-model",
        timeout=25,
    )
    assert result == "# Summary\n\nMain points."
    request = opener.call_args.args[0]
    payload = json.loads(request.data)
    assert request.full_url == "https://generativelanguage.googleapis.com/v1beta/interactions"
    assert request.get_header("X-goog-api-key") == "test-key"
    assert "test-key" not in request.full_url
    assert payload["model"] == "test-model"
    assert payload["store"] is False
    assert payload["input"][1] == {"type": "video", "uri": URL}
    assert "Be concise." in payload["input"][0]["text"]
    assert "Ukrainian" in payload["input"][0]["text"]
    assert "VIDEO_UNAVAILABLE" in payload["input"][0]["text"]
    assert opener.call_args.kwargs["timeout"] == 25


@pytest.mark.parametrize(
    "data",
    [
        None,
        [],
        {"status": "failed"},
        {"status": "in_progress"},
        {"status": "completed", "steps": None},
        {"status": "completed", "steps": []},
        {"status": "completed", "steps": [{"type": "model_output", "content": None}]},
        {
            "status": "completed",
            "steps": [
                {
                    "type": "model_output",
                    "content": [
                        {"type": "text", "text": "   "},
                    ],
                }
            ],
        },
        {
            "status": "completed",
            "steps": [
                {
                    "type": "model_output",
                    "content": [
                        {"type": "text", "text": "VIDEO_UNAVAILABLE"},
                    ],
                }
            ],
        },
    ],
)
def test_unusable_gemini_response(data: object, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(youtube.urllib.request, "urlopen", lambda *a, **kw: _response(data))
    with pytest.raises(SummaryAPIError):
        youtube.summarize_youtube(URL, SummaryConfig(), api_key="test-key")


@pytest.mark.parametrize(
    "error",
    [
        urllib.error.HTTPError(URL, 429, "Too many requests", {}, io.BytesIO(b"secret")),
        urllib.error.URLError("offline"),
        TimeoutError("timed out"),
        ValueError("invalid JSON"),
    ],
)
def test_gemini_errors(error: Exception, monkeypatch: pytest.MonkeyPatch) -> None:
    opener = MagicMock(side_effect=error)
    monkeypatch.setattr(youtube.urllib.request, "urlopen", opener)
    with pytest.raises(SummaryAPIError) as exc:
        youtube.summarize_youtube(URL, SummaryConfig(), api_key="test-key")
    assert "secret" not in str(exc.value)
    assert "test-key" not in str(exc.value)


@pytest.mark.parametrize(("url", "key"), [(URL, None), ("https://example.com/video", "key")])
def test_missing_config_does_not_make_request(
    url: str, key: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    opener = MagicMock()
    monkeypatch.setattr(youtube.urllib.request, "urlopen", opener)
    with pytest.raises(SummaryConfigError):
        youtube.summarize_youtube(url, SummaryConfig(), api_key=key)
    opener.assert_not_called()
