from pathlib import Path

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
    assert settings.ollama_timeout_seconds == 120.0

    assert settings.embedding_provider == "ollama"
    assert settings.embedding_model == "embeddinggemma"

    assert str(settings.qdrant_url) == "http://localhost:6333/"
    assert settings.qdrant_path == Path("data/qdrant_local")
    assert settings.qdrant_collection == "fleetmind_documents"

    assert settings.chunk_size_words == 180
    assert settings.chunk_overlap_words == 30
    assert settings.retrieval_limit == 5
    assert settings.minimum_grounding_score == 0.5
    assert settings.max_context_chars == 6000

    assert settings.langsmith_api_key is None
    assert settings.langsmith_tracing is False


def test_settings_load_environment_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FLEETMIND_ENVIRONMENT", "production")
    monkeypatch.setenv("FLEETMIND_LOG_LEVEL", "WARNING")
    monkeypatch.setenv("FLEETMIND_LLM_MODEL", "llama3.2:8b")
    monkeypatch.setenv("FLEETMIND_LLM_BASE_URL", "http://ollama.internal:11434")
    monkeypatch.setenv("FLEETMIND_OLLAMA_TIMEOUT_SECONDS", "45")
    monkeypatch.setenv("FLEETMIND_QDRANT_URL", "http://qdrant.internal:6333")
    monkeypatch.setenv("FLEETMIND_QDRANT_PATH", "runtime/qdrant")
    monkeypatch.setenv("FLEETMIND_QDRANT_COLLECTION", "production_documents")
    monkeypatch.setenv("FLEETMIND_CHUNK_SIZE_WORDS", "240")
    monkeypatch.setenv("FLEETMIND_CHUNK_OVERLAP_WORDS", "40")
    monkeypatch.setenv("FLEETMIND_RETRIEVAL_LIMIT", "8")
    monkeypatch.setenv("FLEETMIND_MINIMUM_GROUNDING_SCORE", "0.65")
    monkeypatch.setenv("FLEETMIND_MAX_CONTEXT_CHARS", "9000")
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-secret")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")

    settings = _TestFleetMindSettings()

    assert settings.environment == "production"
    assert settings.log_level == "WARNING"
    assert settings.llm_model == "llama3.2:8b"
    assert str(settings.llm_base_url) == "http://ollama.internal:11434/"
    assert settings.ollama_timeout_seconds == 45.0
    assert str(settings.qdrant_url) == "http://qdrant.internal:6333/"
    assert settings.qdrant_path == Path("runtime/qdrant")
    assert settings.qdrant_collection == "production_documents"
    assert settings.chunk_size_words == 240
    assert settings.chunk_overlap_words == 40
    assert settings.retrieval_limit == 8
    assert settings.minimum_grounding_score == 0.65
    assert settings.max_context_chars == 9000

    assert settings.langsmith_api_key is not None
    assert settings.langsmith_api_key.get_secret_value() == "test-secret"
    assert settings.langsmith_tracing is True


def test_settings_reject_invalid_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FLEETMIND_ENVIRONMENT", "invalid")

    with pytest.raises(ValidationError):
        _TestFleetMindSettings()


def test_settings_reject_overlap_equal_to_chunk_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FLEETMIND_CHUNK_SIZE_WORDS", "30")
    monkeypatch.setenv("FLEETMIND_CHUNK_OVERLAP_WORDS", "30")

    with pytest.raises(ValidationError, match="overlap"):
        _TestFleetMindSettings()


def test_settings_reject_invalid_grounding_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FLEETMIND_MINIMUM_GROUNDING_SCORE", "1.5")

    with pytest.raises(ValidationError):
        _TestFleetMindSettings()
