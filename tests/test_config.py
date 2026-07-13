import pytest
from pydantic import ValidationError
from pydantic_settings import SettingsConfigDict

from fleetmind_rag.config import FleetMindSettings


class _TestFleetMindSettings(FleetMindSettings):
    """FleetMind settings with dotenv loading disabled for isolated tests."""

    model_config = SettingsConfigDict(env_file=None)


def test_settings_use_expected_defaults() -> None:
    settings = _TestFleetMindSettings()

    assert settings.environment == "development"
    assert settings.log_level == "INFO"

    assert settings.llm_provider == "ollama"
    assert settings.llm_model == "llama3.2:3b"
    assert str(settings.llm_base_url) == "http://localhost:11434/"

    assert settings.embedding_provider == "ollama"
    assert settings.embedding_model == "embeddinggemma"

    assert str(settings.qdrant_url) == "http://localhost:6333/"
    assert settings.qdrant_collection == "fleetmind_documents"

    assert settings.langsmith_api_key is None
    assert settings.langsmith_tracing is False


def test_settings_load_environment_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FLEETMIND_ENVIRONMENT", "production")
    monkeypatch.setenv("FLEETMIND_LOG_LEVEL", "WARNING")
    monkeypatch.setenv("FLEETMIND_LLM_MODEL", "llama3.2:8b")
    monkeypatch.setenv(
        "FLEETMIND_LLM_BASE_URL",
        "http://ollama.internal:11434",
    )
    monkeypatch.setenv(
        "FLEETMIND_QDRANT_URL",
        "http://qdrant.internal:6333",
    )
    monkeypatch.setenv(
        "FLEETMIND_QDRANT_COLLECTION",
        "production_documents",
    )
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-secret")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")

    settings = _TestFleetMindSettings()

    assert settings.environment == "production"
    assert settings.log_level == "WARNING"
    assert settings.llm_model == "llama3.2:8b"
    assert str(settings.llm_base_url) == "http://ollama.internal:11434/"
    assert str(settings.qdrant_url) == "http://qdrant.internal:6333/"
    assert settings.qdrant_collection == "production_documents"

    assert settings.langsmith_api_key is not None
    assert settings.langsmith_api_key.get_secret_value() == "test-secret"
    assert settings.langsmith_tracing is True


def test_settings_reject_invalid_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FLEETMIND_ENVIRONMENT", "invalid")

    with pytest.raises(ValidationError):
        _TestFleetMindSettings()
