import json
from collections.abc import Callable

import httpx
import pytest

from fleetmind_rag.ollama import OllamaChatClient


def _build_chat_client(
    handler: Callable[[httpx.Request], httpx.Response],
) -> OllamaChatClient:
    return OllamaChatClient(
        "http://localhost:11434",
        "llama3.2:3b",
        timeout_seconds=1.0,
        transport=httpx.MockTransport(handler),
    )


def test_chat_returns_assistant_content() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url == httpx.URL("http://localhost:11434/api/chat")

        request_body = json.loads(request.content)

        assert request_body == {
            "model": "llama3.2:3b",
            "messages": [
                {
                    "role": "system",
                    "content": "Answer briefly.",
                },
                {
                    "role": "user",
                    "content": "What is FleetMind?",
                },
            ],
            "stream": False,
        }

        return httpx.Response(
            200,
            json={
                "model": "llama3.2:3b",
                "message": {
                    "role": "assistant",
                    "content": "FleetMind is a fleet operations copilot.",
                },
                "done": True,
            },
        )

    result = _build_chat_client(handler).chat(
        "What is FleetMind?",
        system_prompt="Answer briefly.",
    )

    assert result.succeeded is True
    assert result.content == "FleetMind is a fleet operations copilot."
    assert result.model == "llama3.2:3b"
    assert result.message == "The Ollama chat request succeeded."


def test_chat_uses_configured_model_when_response_omits_model() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "  FleetMind online.  ",
                }
            },
            request=request,
        )

    result = _build_chat_client(handler).chat("Hello")

    assert result.succeeded is True
    assert result.content == "FleetMind online."
    assert result.model == "llama3.2:3b"


def test_chat_handles_connection_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(
            "Connection refused",
            request=request,
        )

    result = _build_chat_client(handler).chat("Hello")

    assert result.succeeded is False
    assert result.content is None
    assert result.model is None
    assert result.message == "The Ollama API is unreachable."


def test_chat_handles_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout(
            "Request timed out",
            request=request,
        )

    result = _build_chat_client(handler).chat("Hello")

    assert result.succeeded is False
    assert result.content is None
    assert result.model is None
    assert result.message == "The Ollama chat request timed out."


def test_chat_handles_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            503,
            request=request,
        )

    result = _build_chat_client(handler).chat("Hello")

    assert result.succeeded is False
    assert result.content is None
    assert result.model is None
    assert result.message == "The Ollama API returned HTTP 503."


def test_chat_rejects_invalid_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"not-json",
            request=request,
        )

    result = _build_chat_client(handler).chat("Hello")

    assert result.succeeded is False
    assert result.content is None
    assert result.model is None
    assert result.message == "The Ollama API returned invalid JSON."


@pytest.mark.parametrize(
    ("response_body", "expected_message"),
    [
        (
            [],
            "The Ollama API returned an invalid chat response.",
        ),
        (
            {},
            "The Ollama API returned an invalid chat response.",
        ),
        (
            {"message": "not-an-object"},
            "The Ollama API returned an invalid chat response.",
        ),
        (
            {"message": {}},
            "The Ollama API returned an empty chat response.",
        ),
        (
            {"message": {"content": ""}},
            "The Ollama API returned an empty chat response.",
        ),
    ],
)
def test_chat_rejects_invalid_response(
    response_body: object,
    expected_message: str,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=response_body,
            request=request,
        )

    result = _build_chat_client(handler).chat("Hello")

    assert result.succeeded is False
    assert result.content is None
    assert result.model is None
    assert result.message == expected_message


@pytest.mark.parametrize(
    "model",
    [
        "",
        "   ",
    ],
)
def test_chat_rejects_empty_model(model: str) -> None:
    with pytest.raises(
        ValueError,
        match="chat model must not be empty",
    ):
        OllamaChatClient(
            "http://localhost:11434",
            model,
        )


@pytest.mark.parametrize(
    "prompt",
    [
        "",
        "   ",
    ],
)
def test_chat_rejects_empty_prompt(prompt: str) -> None:
    client = OllamaChatClient(
        "http://localhost:11434",
        "llama3.2:3b",
    )

    with pytest.raises(
        ValueError,
        match="chat prompt must not be empty",
    ):
        client.chat(prompt)
