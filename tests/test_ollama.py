from collections.abc import Callable

import httpx
import pytest

from fleetmind_rag.ollama import (
    OllamaHealthClient,
    OllamaModelClient,
)


def _build_client(
    handler: Callable[[httpx.Request], httpx.Response],
) -> OllamaHealthClient:
    return OllamaHealthClient(
        "http://localhost:11434",
        timeout_seconds=1.0,
        transport=httpx.MockTransport(handler),
    )


def test_health_check_returns_server_version() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url == httpx.URL("http://localhost:11434/api/version")

        return httpx.Response(
            200,
            json={"version": "0.12.6"},
        )

    health = _build_client(handler).check()

    assert health.available is True
    assert health.version == "0.12.6"
    assert health.message == "The Ollama API is reachable."


def test_health_check_handles_connection_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(
            "Connection refused",
            request=request,
        )

    health = _build_client(handler).check()

    assert health.available is False
    assert health.version is None
    assert health.message == "The Ollama API is unreachable."


def test_health_check_handles_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout(
            "Request timed out",
            request=request,
        )

    health = _build_client(handler).check()

    assert health.available is False
    assert health.version is None
    assert health.message == "The Ollama health check timed out."


def test_health_check_handles_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            503,
            request=request,
        )

    health = _build_client(handler).check()

    assert health.available is False
    assert health.version is None
    assert health.message == "The Ollama API returned HTTP 503."


def test_health_check_rejects_invalid_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"not-json",
            request=request,
        )

    health = _build_client(handler).check()

    assert health.available is False
    assert health.version is None
    assert health.message == "The Ollama API returned invalid JSON."


def test_health_check_rejects_missing_version() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={},
            request=request,
        )

    health = _build_client(handler).check()

    assert health.available is False
    assert health.version is None
    assert health.message == ("The Ollama API response did not contain a version.")


def _build_model_client(
    handler: Callable[[httpx.Request], httpx.Response],
) -> OllamaModelClient:
    return OllamaModelClient(
        "http://localhost:11434",
        timeout_seconds=1.0,
        transport=httpx.MockTransport(handler),
    )


def test_model_discovery_returns_installed_models() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url == httpx.URL("http://localhost:11434/api/tags")

        return httpx.Response(
            200,
            json={
                "models": [
                    {"name": "llama3.2:3b"},
                    {"name": "embeddinggemma:latest"},
                ]
            },
        )

    result = _build_model_client(handler).list_models()

    assert result.succeeded is True
    assert tuple(model.name for model in result.models) == (
        "llama3.2:3b",
        "embeddinggemma:latest",
    )
    assert result.message == "Installed Ollama model count: 2."


def test_model_discovery_accepts_empty_model_list() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"models": []},
            request=request,
        )

    result = _build_model_client(handler).list_models()

    assert result.succeeded is True
    assert result.models == ()
    assert result.message == "Installed Ollama model count: 0."


def test_model_discovery_handles_connection_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(
            "Connection refused",
            request=request,
        )

    result = _build_model_client(handler).list_models()

    assert result.succeeded is False
    assert result.models == ()
    assert result.message == "The Ollama API is unreachable."


def test_model_discovery_handles_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout(
            "Request timed out",
            request=request,
        )

    result = _build_model_client(handler).list_models()

    assert result.succeeded is False
    assert result.models == ()
    assert result.message == "The Ollama model request timed out."


def test_model_discovery_handles_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            503,
            request=request,
        )

    result = _build_model_client(handler).list_models()

    assert result.succeeded is False
    assert result.models == ()
    assert result.message == "The Ollama API returned HTTP 503."


def test_model_discovery_rejects_invalid_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"not-json",
            request=request,
        )

    result = _build_model_client(handler).list_models()

    assert result.succeeded is False
    assert result.models == ()
    assert result.message == "The Ollama API returned invalid JSON."


def test_model_discovery_rejects_missing_model_list() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={},
            request=request,
        )

    result = _build_model_client(handler).list_models()

    assert result.succeeded is False
    assert result.models == ()
    assert result.message == ("The Ollama API response did not contain a model list.")


@pytest.mark.parametrize(
    "invalid_model",
    [
        "not-an-object",
        {},
        {"name": ""},
        {"name": 123},
    ],
)
def test_model_discovery_rejects_invalid_model_entry(
    invalid_model: object,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"models": [invalid_model]},
            request=request,
        )

    result = _build_model_client(handler).list_models()

    assert result.succeeded is False
    assert result.models == ()
    assert result.message == ("The Ollama API returned an invalid model entry.")
