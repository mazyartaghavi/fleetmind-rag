import json
from collections.abc import Callable

import httpx
import pytest

from fleetmind_rag.ollama import OllamaEmbeddingClient


def _build_embedding_client(
    handler: Callable[[httpx.Request], httpx.Response],
) -> OllamaEmbeddingClient:
    return OllamaEmbeddingClient(
        "http://localhost:11434",
        "embeddinggemma",
        timeout_seconds=1.0,
        transport=httpx.MockTransport(handler),
    )


def test_embed_returns_single_embedding() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url == httpx.URL("http://localhost:11434/api/embed")

        request_body = json.loads(request.content)

        assert request_body == {
            "model": "embeddinggemma",
            "input": "FleetMind monitors vehicle operations.",
        }

        return httpx.Response(
            200,
            json={
                "model": "embeddinggemma",
                "embeddings": [
                    [0.1, -0.2, 0.3],
                ],
            },
            request=request,
        )

    result = _build_embedding_client(handler).embed(
        "  FleetMind monitors vehicle operations.  "
    )

    assert result.succeeded is True
    assert result.embeddings == ((0.1, -0.2, 0.3),)
    assert result.model == "embeddinggemma"
    assert result.message == ("Generated 1 Ollama embedding(s) with dimension 3.")


@pytest.mark.parametrize(
    "response_model",
    [
        None,
        "   ",
    ],
)
def test_embed_returns_batch_and_uses_configured_model_fallback(
    response_model: str | None,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        request_body = json.loads(request.content)

        assert request_body == {
            "model": "embeddinggemma",
            "input": [
                "First fleet document.",
                "Second fleet document.",
            ],
        }

        response_body: dict[str, object] = {
            "embeddings": [
                [0.1, 0.2],
                [0.3, 0.4],
            ]
        }

        if response_model is not None:
            response_body["model"] = response_model

        return httpx.Response(
            200,
            json=response_body,
            request=request,
        )

    result = _build_embedding_client(handler).embed(
        [
            "  First fleet document.  ",
            "Second fleet document.",
        ]
    )

    assert result.succeeded is True
    assert result.embeddings == (
        (0.1, 0.2),
        (0.3, 0.4),
    )
    assert result.model == "embeddinggemma"
    assert result.message == ("Generated 2 Ollama embedding(s) with dimension 2.")


def test_embed_handles_connection_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(
            "Connection refused",
            request=request,
        )

    result = _build_embedding_client(handler).embed("Hello")

    assert result.succeeded is False
    assert result.embeddings == ()
    assert result.model is None
    assert result.message == "The Ollama API is unreachable."


def test_embed_handles_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout(
            "Request timed out",
            request=request,
        )

    result = _build_embedding_client(handler).embed("Hello")

    assert result.succeeded is False
    assert result.embeddings == ()
    assert result.model is None
    assert result.message == ("The Ollama embedding request timed out.")


def test_embed_handles_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            503,
            request=request,
        )

    result = _build_embedding_client(handler).embed("Hello")

    assert result.succeeded is False
    assert result.embeddings == ()
    assert result.model is None
    assert result.message == "The Ollama API returned HTTP 503."


def test_embed_rejects_invalid_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"not-json",
            request=request,
        )

    result = _build_embedding_client(handler).embed("Hello")

    assert result.succeeded is False
    assert result.embeddings == ()
    assert result.model is None
    assert result.message == "The Ollama API returned invalid JSON."


@pytest.mark.parametrize(
    "response_body",
    [
        [],
        {},
        {"embeddings": "not-a-list"},
        {"embeddings": []},
        {"embeddings": ["not-a-vector"]},
        {"embeddings": [[]]},
        {"embeddings": [["not-a-number"]]},
        {"embeddings": [[True]]},
        {
            "embeddings": [
                [0.1],
                [0.2, 0.3],
            ]
        },
    ],
)
def test_embed_rejects_invalid_response(
    response_body: object,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=response_body,
            request=request,
        )

    result = _build_embedding_client(handler).embed("Hello")

    assert result.succeeded is False
    assert result.embeddings == ()
    assert result.model is None
    assert result.message == ("The Ollama API returned an invalid embedding response.")


def test_embed_rejects_embedding_count_mismatch() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "embeddinggemma",
                "embeddings": [
                    [0.1, 0.2],
                ],
            },
            request=request,
        )

    result = _build_embedding_client(handler).embed(
        [
            "First input.",
            "Second input.",
        ]
    )

    assert result.succeeded is False
    assert result.embeddings == ()
    assert result.model is None
    assert result.message == ("The Ollama API returned an invalid embedding response.")


@pytest.mark.parametrize(
    "model",
    [
        "",
        "   ",
    ],
)
def test_embedding_client_rejects_empty_model(model: str) -> None:
    with pytest.raises(
        ValueError,
        match="embedding model must not be empty",
    ):
        OllamaEmbeddingClient(
            "http://localhost:11434",
            model,
        )


@pytest.mark.parametrize(
    "input_value",
    [
        "",
        "   ",
        [],
        (),
        ["Valid input.", "   "],
    ],
)
def test_embed_rejects_empty_input(
    input_value: str | list[str] | tuple[str, ...],
) -> None:
    client = OllamaEmbeddingClient(
        "http://localhost:11434",
        "embeddinggemma",
    )

    with pytest.raises(ValueError):
        client.embed(input_value)
