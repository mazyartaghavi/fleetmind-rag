from pathlib import Path

import pytest
from pydantic_settings import SettingsConfigDict

from fleetmind_rag import main
from fleetmind_rag.app import (
    build_ollama_health_message,
    build_ollama_models_message,
    build_startup_message,
)
from fleetmind_rag.config import FleetMindSettings
from fleetmind_rag.ollama import (
    OllamaHealth,
    OllamaModel,
    OllamaModelListResult,
)


class _TestFleetMindSettings(FleetMindSettings):
    """Settings with dotenv loading disabled for isolated startup tests."""

    model_config = SettingsConfigDict(env_file=None)


def test_startup_message_reports_disabled_tracing() -> None:
    settings = _TestFleetMindSettings()

    message = build_startup_message(settings)

    assert "FleetMind-RAG configuration loaded successfully." in message
    assert "Environment: development" in message
    assert "LLM model: llama3.2:3b" in message
    assert "LangSmith tracing: disabled" in message


def test_main_prints_configuration_health_and_models(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class _FakeOllamaHealthClient:
        def __init__(self, base_url: str) -> None:
            assert base_url == "http://localhost:11434/"

        def check(self) -> OllamaHealth:
            return OllamaHealth(
                available=True,
                version="0.12.6",
                message="The Ollama API is reachable.",
            )

    class _FakeOllamaModelClient:
        def __init__(self, base_url: str) -> None:
            assert base_url == "http://localhost:11434/"

        def list_models(self) -> OllamaModelListResult:
            return OllamaModelListResult(
                succeeded=True,
                models=(
                    OllamaModel(name="llama3.2:3b"),
                    OllamaModel(name="embeddinggemma:latest"),
                ),
                message="Installed Ollama model count: 2.",
            )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FLEETMIND_ENVIRONMENT", "test")
    monkeypatch.setenv("FLEETMIND_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("LANGSMITH_API_KEY", "never-print-this-secret")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setattr(
        "fleetmind_rag.app.OllamaHealthClient",
        _FakeOllamaHealthClient,
    )
    monkeypatch.setattr(
        "fleetmind_rag.app.OllamaModelClient",
        _FakeOllamaModelClient,
    )

    main()

    captured = capsys.readouterr()

    assert "FleetMind-RAG configuration loaded successfully." in captured.out
    assert "Environment: test" in captured.out
    assert "Log level: DEBUG" in captured.out
    assert "LangSmith tracing: enabled" in captured.out
    assert "Ollama status: available (version 0.12.6)." in captured.out
    assert "Ollama models: llama3.2:3b, embeddinggemma:latest." in captured.out
    assert "never-print-this-secret" not in captured.out
    assert captured.err == ""


def test_ollama_health_message_reports_unavailable_server() -> None:
    health = OllamaHealth(
        available=False,
        version=None,
        message="The Ollama API is unreachable.",
    )

    message = build_ollama_health_message(health)

    assert message == ("Ollama status: unavailable. The Ollama API is unreachable.")


def test_ollama_models_message_reports_empty_list() -> None:
    result = OllamaModelListResult(
        succeeded=True,
        models=(),
        message="Installed Ollama model count: 0.",
    )

    message = build_ollama_models_message(result)

    assert message == "Ollama models: none installed."


def test_ollama_models_message_reports_discovery_failure() -> None:
    result = OllamaModelListResult(
        succeeded=False,
        models=(),
        message="The Ollama model request timed out.",
    )

    message = build_ollama_models_message(result)

    assert message == (
        "Ollama models: unavailable. The Ollama model request timed out."
    )


def test_main_skips_model_discovery_when_ollama_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class _FakeUnavailableHealthClient:
        def __init__(self, base_url: str) -> None:
            assert base_url == "http://localhost:11434/"

        def check(self) -> OllamaHealth:
            return OllamaHealth(
                available=False,
                version=None,
                message="The Ollama API is unreachable.",
            )

    class _UnexpectedModelClient:
        def __init__(self, base_url: str) -> None:
            raise AssertionError(
                "Model discovery must not run when Ollama is unavailable."
            )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "fleetmind_rag.app.OllamaHealthClient",
        _FakeUnavailableHealthClient,
    )
    monkeypatch.setattr(
        "fleetmind_rag.app.OllamaModelClient",
        _UnexpectedModelClient,
    )

    main()

    captured = capsys.readouterr()

    assert "Ollama status: unavailable. The Ollama API is unreachable." in captured.out
    assert (
        "Ollama models: unavailable because the Ollama API "
        "is not reachable." in captured.out
    )
    assert captured.err == ""
