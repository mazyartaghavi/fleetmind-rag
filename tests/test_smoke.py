from pathlib import Path

import pytest
from pydantic_settings import SettingsConfigDict

from fleetmind_rag import main
from fleetmind_rag.app import (
    build_ollama_health_message,
    build_startup_message,
)
from fleetmind_rag.config import FleetMindSettings
from fleetmind_rag.ollama import OllamaHealth


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


def test_main_prints_safe_configuration_and_ollama_summary(
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

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FLEETMIND_ENVIRONMENT", "test")
    monkeypatch.setenv("FLEETMIND_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("LANGSMITH_API_KEY", "never-print-this-secret")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setattr(
        "fleetmind_rag.app.OllamaHealthClient",
        _FakeOllamaHealthClient,
    )

    main()

    captured = capsys.readouterr()

    assert "FleetMind-RAG configuration loaded successfully." in captured.out
    assert "Environment: test" in captured.out
    assert "Log level: DEBUG" in captured.out
    assert "LangSmith tracing: enabled" in captured.out
    assert "Ollama status: available (version 0.12.6)." in captured.out
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
