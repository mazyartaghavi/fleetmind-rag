from pathlib import Path
from typing import Literal, Self

from pydantic import Field, HttpUrl, SecretStr, model_validator
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
    ollama_timeout_seconds: float = Field(default=120.0, gt=0.0)

    embedding_provider: Literal["ollama"] = "ollama"
    embedding_model: str = Field(default="embeddinggemma", min_length=1)

    qdrant_url: HttpUrl = HttpUrl("http://localhost:6333")
    qdrant_path: Path = Path("data/qdrant_local")
    qdrant_collection: str = Field(
        default="fleetmind_documents",
        min_length=1,
    )

    chunk_size_words: int = Field(default=180, gt=0)
    chunk_overlap_words: int = Field(default=30, ge=0)
    retrieval_limit: int = Field(default=5, ge=1, le=50)
    minimum_grounding_score: float = Field(default=0.5, ge=-1.0, le=1.0)
    max_context_chars: int = Field(default=6000, ge=256)

    langsmith_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="LANGSMITH_API_KEY",
    )
    langsmith_tracing: bool = Field(
        default=False,
        validation_alias="LANGSMITH_TRACING",
    )

    @model_validator(mode="after")
    def validate_chunk_configuration(self) -> Self:
        """Reject chunk overlap that would prevent forward progress."""

        if self.chunk_overlap_words >= self.chunk_size_words:
            raise ValueError("The chunk overlap must be smaller than the chunk size.")

        return self
