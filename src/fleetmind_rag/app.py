from fleetmind_rag.config import FleetMindSettings
from fleetmind_rag.ollama import (
    OllamaHealth,
    OllamaHealthClient,
    OllamaModelClient,
    OllamaModelListResult,
)


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


def build_ollama_health_message(health: OllamaHealth) -> str:
    """Build a concise summary of the Ollama API health."""

    if health.available and health.version is not None:
        return f"Ollama status: available (version {health.version})."

    return f"Ollama status: unavailable. {health.message}"


def build_ollama_models_message(
    result: OllamaModelListResult,
) -> str:
    """Build a concise summary of the models available through Ollama."""

    if not result.succeeded:
        return f"Ollama models: unavailable. {result.message}"

    if not result.models:
        return "Ollama models: none installed."

    model_names = ", ".join(model.name for model in result.models)

    return f"Ollama models: {model_names}."


def main() -> None:
    """Start FleetMind-RAG using validated application settings."""

    settings = FleetMindSettings()
    base_url = str(settings.llm_base_url)

    print(build_startup_message(settings))

    health = OllamaHealthClient(base_url).check()

    print(build_ollama_health_message(health))

    if not health.available:
        print("Ollama models: unavailable because the Ollama API is not reachable.")
        return

    models_result = OllamaModelClient(base_url).list_models()

    print(build_ollama_models_message(models_result))
