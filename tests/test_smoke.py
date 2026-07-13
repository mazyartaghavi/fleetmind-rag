from pathlib import Path

import pytest
from pydantic_settings import SettingsConfigDict

from fleetmind_rag import main
from fleetmind_rag.app import build_startup_message
from fleetmind_rag.config import FleetMindSettings


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


def test_main_prints_safe_configuration_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FLEETMIND_ENVIRONMENT", "test")
    monkeypatch.setenv("FLEETMIND_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("LANGSMITH_API_KEY", "never-print-this-secret")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")

    main()

    captured = capsys.readouterr()

    assert "FleetMind-RAG configuration loaded successfully." in captured.out
    assert "Environment: test" in captured.out
    assert "Log level: DEBUG" in captured.out
    assert "LangSmith tracing: enabled" in captured.out
    assert "never-print-this-secret" not in captured.out
    assert captured.err == ""
