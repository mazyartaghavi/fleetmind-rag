import argparse
import sys
from collections.abc import Sequence
from typing import cast

from fleetmind_rag.config import FleetMindSettings
from fleetmind_rag.ollama import (
    OllamaChatClient,
    OllamaHealth,
    OllamaHealthClient,
    OllamaModelClient,
    OllamaModelListResult,
)

DEFAULT_SYSTEM_PROMPT = (
    "You are FleetMind-RAG, a local copilot being developed for intelligent "
    "fleet operations. The project uses local language models and is intended "
    "to support retrieval-augmented answers about fleet documents and "
    "operational data. Do not invent customers, deployments, organizations, "
    "affiliations, data sources, or implemented capabilities. Base answers "
    "only on information provided in the conversation or retrieved context. "
    "When required information is missing, say that it is not available."
)

DEFAULT_SYSTEM_PROMPT = (
    "You are FleetMind-RAG, a local copilot being developed for intelligent "
    "fleet operations. The project uses local language models and is intended "
    "to support retrieval-augmented answers about fleet documents and "
    "operational data. Do not invent customers, deployments, organizations, "
    "affiliations, data sources, or implemented capabilities. Base answers "
    "only on information provided in the conversation or retrieved context. "
    "When required information is missing, say that it is not available."
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


def build_cli_parser() -> argparse.ArgumentParser:
    """Build the FleetMind-RAG command-line argument parser."""

    parser = argparse.ArgumentParser(
        prog="fleetmind-rag",
        description="FleetMind-RAG local fleet operations copilot.",
    )

    subparsers = parser.add_subparsers(dest="command")

    chat_parser = subparsers.add_parser(
        "chat",
        help="Send one prompt to the configured Ollama chat model.",
    )
    chat_parser.add_argument(
        "prompt",
        help="The user prompt to send to the configured model.",
    )
    chat_parser.add_argument(
        "--system",
        dest="system_prompt",
        default=None,
        help="Optional system instruction for the model.",
    )

    return parser


def run_status(settings: FleetMindSettings) -> int:
    """Print the active configuration and Ollama service status."""

    base_url = str(settings.llm_base_url)

    print(build_startup_message(settings))

    health = OllamaHealthClient(base_url).check()

    print(build_ollama_health_message(health))

    if not health.available:
        print("Ollama models: unavailable because the Ollama API is not reachable.")
        return 0

    models_result = OllamaModelClient(base_url).list_models()

    print(build_ollama_models_message(models_result))

    return 0


def build_system_prompt(
    user_system_prompt: str | None,
) -> str:
    """Combine the project guardrail with an optional user instruction."""

    if user_system_prompt is None or not user_system_prompt.strip():
        return DEFAULT_SYSTEM_PROMPT

    return (
        f"{DEFAULT_SYSTEM_PROMPT}\n\n"
        f"Additional user instruction: {user_system_prompt.strip()}"
    )


def run_chat(
    settings: FleetMindSettings,
    prompt: str,
    *,
    system_prompt: str | None = None,
) -> int:
    """Send one prompt to the configured Ollama model."""

    effective_system_prompt = build_system_prompt(system_prompt)

    result = OllamaChatClient(
        str(settings.llm_base_url),
        settings.llm_model,
    ).chat(
        prompt,
        system_prompt=effective_system_prompt,
    )

    if result.succeeded and result.content is not None:
        print(result.content)
        return 0

    print(
        f"FleetMind chat failed: {result.message}",
        file=sys.stderr,
    )
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    """Run the FleetMind-RAG command-line application."""

    args = build_cli_parser().parse_args(argv)
    settings = FleetMindSettings()

    command = cast(str | None, args.command)

    if command == "chat":
        prompt = cast(str, args.prompt)
        system_prompt = cast(str | None, args.system_prompt)

        return run_chat(
            settings,
            prompt,
            system_prompt=system_prompt,
        )

    return run_status(settings)
