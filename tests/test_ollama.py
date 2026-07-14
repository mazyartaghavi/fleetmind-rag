from collections.abc import Callable

import httpx

from fleetmind_rag.ollama import OllamaHealthClient


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
