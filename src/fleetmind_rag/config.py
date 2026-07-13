from typing import Literal

from pydantic import Field, HttpUrl, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class FleetMindSettings(BaseSettings):
    """Validated application settings loaded from the environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="FLEETMIND_",
        env_ignore_empty=True,
        case_sensitive=False,
        extra="ignore",
        str_strip_whitespace=True,
        validate_default=True,
    )

    environment: Literal["development", "test", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    llm_provider: Literal["ollama"] = "ollama"
    llm_model: str = Field(default="llama3.2:3b", min_length=1)
    llm_base_url: HttpUrl = HttpUrl("http://localhost:11434")

    embedding_provider: Literal["ollama"] = "ollama"
    embedding_model: str = Field(default="embeddinggemma", min_length=1)

    qdrant_url: HttpUrl = HttpUrl("http://localhost:6333")
    qdrant_collection: str = Field(
        default="fleetmind_documents",
        min_length=1,
    )

    langsmith_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="LANGSMITH_API_KEY",
    )
    langsmith_tracing: bool = Field(
        default=False,
        validation_alias="LANGSMITH_TRACING",
    )
