from fleetmind_rag.config import FleetMindSettings


def build_startup_message(settings: FleetMindSettings) -> str:
    """Build a safe summary of the active application configuration."""

    tracing_status = "enabled" if settings.langsmith_tracing else "disabled"

    return "\n".join(
        (
            "FleetMind-RAG configuration loaded successfully.",
            f"Environment: {settings.environment}",
            f"Log level: {settings.log_level}",
            f"LLM provider: {settings.llm_provider}",
            f"LLM model: {settings.llm_model}",
            f"LLM base URL: {settings.llm_base_url}",
            f"Embedding provider: {settings.embedding_provider}",
            f"Embedding model: {settings.embedding_model}",
            f"Qdrant URL: {settings.qdrant_url}",
            f"Qdrant collection: {settings.qdrant_collection}",
            f"LangSmith tracing: {tracing_status}",
        )
    )


def main() -> None:
    """Start FleetMind-RAG using validated application settings."""

    settings = FleetMindSettings()
    print(build_startup_message(settings))
