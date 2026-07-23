from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from fleetmind_rag.adaptive_grounded_rag import AdaptiveGroundedAnswerResult
from fleetmind_rag.adaptive_retrieval import AdaptiveRetrievalConfig
from fleetmind_rag.config import FleetMindSettings
from fleetmind_rag.grounded_rag import GroundedAnswerResult
from fleetmind_rag.ollama import (
    OllamaChatClient,
    OllamaHealth,
    OllamaHealthClient,
    OllamaModelClient,
    OllamaModelListResult,
)
from fleetmind_rag.retrieval import DocumentIndexResult
from fleetmind_rag.runtime import FleetMindRAGRuntime

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
            f"Qdrant local path: {settings.qdrant_path}",
            f"Qdrant collection: {settings.qdrant_collection}",
            f"Chunk size: {settings.chunk_size_words} words",
            f"Chunk overlap: {settings.chunk_overlap_words} words",
            f"Retrieval limit: {settings.retrieval_limit}",
            f"Minimum grounding score: {settings.minimum_grounding_score}",
            f"Maximum context size: {settings.max_context_chars} characters",
            f"LangSmith tracing: {tracing_status}",
        )
    )


def build_ollama_health_message(health: OllamaHealth) -> str:
    """Build a concise summary of the Ollama API health."""

    if health.available and health.version is not None:
        return f"Ollama status: available (version {health.version})."

    return f"Ollama status: unavailable. {health.message}"


def build_ollama_models_message(result: OllamaModelListResult) -> str:
    """Build a concise summary of the models available through Ollama."""

    if not result.succeeded:
        return f"Ollama models: unavailable. {result.message}"

    if not result.models:
        return "Ollama models: none installed."

    model_names = ", ".join(model.name for model in result.models)
    return f"Ollama models: {model_names}."


def build_document_index_message(
    result: DocumentIndexResult,
    *,
    collection_name: str,
) -> str:
    """Build a readable summary of one successful indexing operation."""

    return "\n".join(
        (
            "Fleet document indexed successfully.",
            f"Source: {result.source_name}",
            f"Document ID: {result.document_id}",
            f"Sections: {result.section_count}",
            f"Chunks: {result.chunk_count}",
            f"Stored vectors: {result.stored_count}",
            f"Embedding model: {result.embedding_model}",
            f"Vector size: {result.vector_size}",
            f"Qdrant collection: {collection_name}",
        )
    )


def build_grounded_answer_message(result: GroundedAnswerResult) -> str:
    """Build terminal output for a grounded answer or safe abstention."""

    if result.answer is None:
        return result.message

    lines = [result.answer]

    if result.citations:
        lines.extend(("", "Sources:"))
        lines.extend(
            (
                f"[{citation.label}] {citation.section_title} "
                f"(score {citation.score:.4f}, chunk {citation.chunk_id})"
            )
            for citation in result.citations
        )

    if result.abstained and result.top_score is not None:
        lines.extend(("", f"Top retrieval score: {result.top_score:.4f}"))

    return "\n".join(lines)


def build_adaptive_grounded_answer_message(
    result: AdaptiveGroundedAnswerResult,
) -> str:
    """Build grounded output plus a concise adaptive-retrieval trace."""

    outcome = result.retrieval_outcome
    latest_result = outcome.state.latest_result
    final_strategy = (
        "none" if latest_result is None else latest_result.decision.strategy
    )
    lines = [
        build_grounded_answer_message(result.grounded_answer),
        "",
        "Adaptive retrieval:",
        f"Status: {outcome.state.status}",
        f"Attempts: {result.attempt_count}",
        f"Rewrites: {len(outcome.rewrites)}",
        f"Initial strategy: {result.initial_routing.strategy}",
        f"Final strategy: {final_strategy}",
        f"Feedback observations: {len(result.feedback_history.observations)}",
    ]
    return "\n".join(lines)


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

    index_parser = subparsers.add_parser(
        "index",
        help="Ingest and index one UTF-8 fleet document.",
    )
    index_parser.add_argument(
        "document",
        type=Path,
        help="Path to the text or Markdown fleet document.",
    )
    index_parser.add_argument(
        "--title",
        default=None,
        help="Fallback title when the document has no heading.",
    )
    index_parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Maximum words per chunk; defaults to the configured value.",
    )
    index_parser.add_argument(
        "--overlap",
        type=int,
        default=None,
        help="Words shared by adjacent chunks; defaults to the configured value.",
    )
    index_parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the Qdrant collection before indexing.",
    )

    ask_parser = subparsers.add_parser(
        "ask",
        help="Ask a citation-grounded question about indexed documents.",
    )
    ask_parser.add_argument(
        "question",
        help="Fleet-operations question to answer from indexed evidence.",
    )
    ask_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum retrieved chunks; defaults to the configured value.",
    )
    ask_parser.add_argument(
        "--adaptive",
        action="store_true",
        help=(
            "Use feedback-aware routing, quality checking, and bounded "
            "query-rewrite retries."
        ),
    )
    ask_parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Maximum adaptive retrieval attempts; used with --adaptive.",
    )
    ask_parser.add_argument(
        "--candidate-limit",
        type=int,
        default=None,
        help=(
            "Adaptive hybrid candidate pool; defaults to at least 20 and "
            "never less than --limit."
        ),
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


def build_system_prompt(user_system_prompt: str | None) -> str:
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
        timeout_seconds=settings.ollama_timeout_seconds,
    ).chat(
        prompt,
        system_prompt=effective_system_prompt,
    )

    if result.succeeded and result.content is not None:
        print(result.content)
        return 0

    print(f"FleetMind chat failed: {result.message}", file=sys.stderr)
    return 1


def create_rag_runtime(settings: FleetMindSettings) -> FleetMindRAGRuntime:
    """Create the configured local RAG runtime."""

    return FleetMindRAGRuntime.from_settings(settings)


def run_index(
    settings: FleetMindSettings,
    document: Path,
    *,
    title: str | None = None,
    chunk_size_words: int | None = None,
    overlap_words: int | None = None,
    recreate_collection: bool = False,
) -> int:
    """Index one fleet document in the configured local Qdrant collection."""

    effective_chunk_size = (
        settings.chunk_size_words if chunk_size_words is None else chunk_size_words
    )
    effective_overlap = (
        settings.chunk_overlap_words if overlap_words is None else overlap_words
    )

    try:
        with create_rag_runtime(settings) as runtime:
            result = runtime.retrieval_service.index_text_document(
                document,
                default_title=title,
                chunk_size_words=effective_chunk_size,
                overlap_words=effective_overlap,
                recreate_collection=recreate_collection,
            )
    except (OSError, UnicodeError, ValueError, RuntimeError) as error:
        print(f"FleetMind indexing failed: {error}", file=sys.stderr)
        return 1

    print(
        build_document_index_message(
            result,
            collection_name=settings.qdrant_collection,
        )
    )
    return 0


def run_ask(
    settings: FleetMindSettings,
    question: str,
    *,
    limit: int | None = None,
    adaptive: bool = False,
    max_attempts: int = 3,
    candidate_limit: int | None = None,
) -> int:
    """Answer one question from the configured indexed fleet documents."""

    effective_limit = settings.retrieval_limit if limit is None else limit
    effective_candidate_limit = (
        max(20, effective_limit) if candidate_limit is None else candidate_limit
    )
    adaptive_result: AdaptiveGroundedAnswerResult | None = None

    try:
        with create_rag_runtime(settings) as runtime:
            if adaptive:
                adaptive_result = runtime.adaptive_grounded_answer_service.answer(
                    question,
                    config=AdaptiveRetrievalConfig(
                        max_attempts=max_attempts,
                        limit=effective_limit,
                        candidate_limit=effective_candidate_limit,
                    ),
                )
                result = adaptive_result.grounded_answer
            else:
                result = runtime.grounded_answer_service.answer(
                    question,
                    limit=effective_limit,
                )
    except (OSError, ValueError, RuntimeError) as error:
        print(f"FleetMind grounded answer failed: {error}", file=sys.stderr)
        return 1

    if not result.succeeded:
        print(f"FleetMind grounded answer failed: {result.message}", file=sys.stderr)
        return 1

    if adaptive_result is None:
        print(build_grounded_answer_message(result))
    else:
        print(build_adaptive_grounded_answer_message(adaptive_result))

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the FleetMind-RAG command-line application."""

    args = build_cli_parser().parse_args(argv)
    settings = FleetMindSettings()
    command = cast(str | None, args.command)

    if command == "chat":
        return run_chat(
            settings,
            cast(str, args.prompt),
            system_prompt=cast(str | None, args.system_prompt),
        )

    if command == "index":
        return run_index(
            settings,
            cast(Path, args.document),
            title=cast(str | None, args.title),
            chunk_size_words=cast(int | None, args.chunk_size),
            overlap_words=cast(int | None, args.overlap),
            recreate_collection=cast(bool, args.recreate),
        )

    if command == "ask":
        return run_ask(
            settings,
            cast(str, args.question),
            limit=cast(int | None, args.limit),
            adaptive=cast(bool, args.adaptive),
            max_attempts=cast(int, args.max_attempts),
            candidate_limit=cast(int | None, args.candidate_limit),
        )

    return run_status(settings)
