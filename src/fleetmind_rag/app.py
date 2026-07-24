from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from fleetmind_rag.adaptive_grounded_rag import AdaptiveGroundedAnswerResult
from fleetmind_rag.adaptive_retrieval import AdaptiveRetrievalConfig
from fleetmind_rag.config import FleetMindSettings
from fleetmind_rag.feedback_analytics import (
    RoutingFeedbackAnalyzer,
    RoutingFeedbackReport,
)
from fleetmind_rag.feedback_gates import (
    FeedbackRegressionGate,
    FeedbackRegressionGateResult,
    GateEnforcement,
)
from fleetmind_rag.feedback_store import JsonRoutingFeedbackStore
from fleetmind_rag.feedback_trends import (
    FeedbackTrendPolicy,
    RoutingFeedbackTrendAnalyzer,
    RoutingFeedbackTrendReport,
)
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
    *,
    feedback_revision: int | None = None,
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

    if feedback_revision is not None:
        lines.append(f"Feedback revision: {feedback_revision}")

    return "\n".join(lines)


def build_feedback_report_message(
    report: RoutingFeedbackReport,
    *,
    path: Path,
    schema_version: int,
) -> str:
    """Build a deterministic terminal report for persisted routing feedback."""

    lines = [
        "FleetMind routing feedback report",
        f"Path: {path}",
        f"Schema version: {schema_version}",
        f"Revision: {report.revision}",
        f"Observations: {report.observation_count}",
        f"Accepted: {report.accepted_count}",
        f"Rewrites: {report.rewrite_count}",
        f"Acceptance rate: {_format_percentage(report.acceptance_rate)}",
        f"Rewrite rate: {_format_percentage(report.rewrite_rate)}",
        f"Average quality: {_format_decimal(report.average_quality_score)}",
        f"Average attempt: {_format_decimal(report.average_attempt_number)}",
        f"Retry-attempt rate: {_format_percentage(report.retry_rate)}",
        "",
        "Strategy performance:",
        (
            "strategy   observations  accepted  rewrites  "
            "acceptance  quality  retry-rate"
        ),
    ]

    lines.extend(
        (
            f"{metrics.strategy:10}"
            f"{metrics.observation_count:12}"
            f"{metrics.accepted_count:10}"
            f"{metrics.rewrite_count:10}  "
            f"{_format_percentage(metrics.acceptance_rate):>10}  "
            f"{_format_decimal(metrics.average_quality_score):>7}  "
            f"{_format_percentage(metrics.retry_rate):>10}"
        )
        for metrics in report.strategies
    )

    observed_features = tuple(
        metrics for metrics in report.features if metrics.observation_count > 0
    )
    lines.extend(("", "Feature performance:"))

    if not observed_features:
        lines.append("No query-signal features have been observed.")
    else:
        lines.append(
            "feature           observations  accepted  rewrites  "
            "acceptance  quality  retry-rate"
        )
        lines.extend(
            (
                f"{metrics.feature:18}"
                f"{metrics.observation_count:12}"
                f"{metrics.accepted_count:10}"
                f"{metrics.rewrite_count:10}  "
                f"{_format_percentage(metrics.acceptance_rate):>10}  "
                f"{_format_decimal(metrics.average_quality_score):>7}  "
                f"{_format_percentage(metrics.retry_rate):>10}"
            )
            for metrics in observed_features
        )

    return "\n".join(lines)


def _format_percentage(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2%}"


def _format_decimal(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def build_feedback_trend_message(
    report: RoutingFeedbackTrendReport,
    *,
    path: Path,
    schema_version: int,
) -> str:
    """Build a deterministic previous-versus-recent feedback trend report."""

    lines = [
        "FleetMind routing feedback trend report",
        f"Path: {path}",
        f"Schema version: {schema_version}",
        f"Revision: {report.revision}",
        f"Total observations: {report.total_observations}",
        f"Window size: {report.policy.window_size}",
        (f"Minimum utility change: {report.policy.minimum_utility_change:.4f}"),
        (
            "Previous window: "
            f"{
                _format_position_range(
                    report.previous_start_position,
                    report.previous_end_position,
                )
            }"
        ),
        (
            "Recent window: "
            f"{
                _format_position_range(
                    report.recent_start_position,
                    report.recent_end_position,
                )
            }"
        ),
        "",
        f"Overall trend: {report.overall.direction}",
        (f"Previous utility: {_format_decimal(report.overall.previous.utility_score)}"),
        (f"Recent utility: {_format_decimal(report.overall.recent.utility_score)}"),
        f"Utility delta: {_format_signed_decimal(report.overall.utility_delta)}",
        (
            "Acceptance delta: "
            f"{_format_signed_percentage(report.overall.acceptance_delta)}"
        ),
        (f"Quality delta: {_format_signed_decimal(report.overall.quality_delta)}"),
        (
            "Rewrite-rate delta: "
            f"{_format_signed_percentage(report.overall.rewrite_delta)}"
        ),
        (f"Retry-rate delta: {_format_signed_percentage(report.overall.retry_delta)}"),
        f"Reason: {report.overall.reason}",
        "",
        "Strategy trends:",
        "strategy   previous  recent  direction          utility-delta",
    ]
    lines.extend(
        (
            f"{comparison.strategy:10}"
            f"{comparison.previous.observation_count:9}"
            f"{comparison.recent.observation_count:8}  "
            f"{comparison.direction:18}"
            f"{_format_signed_decimal(comparison.utility_delta):>13}"
        )
        for comparison in report.strategies
    )
    return "\n".join(lines)


def _format_position_range(
    start: int | None,
    end: int | None,
) -> str:
    if start is None or end is None:
        return "none"

    return f"{start}-{end}"


def _format_signed_decimal(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.4f}"


def _format_signed_percentage(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.2%}"


def build_feedback_gate_message(
    result: FeedbackRegressionGateResult,
    *,
    path: Path,
    schema_version: int,
    enforcement: GateEnforcement,
    process_exit_code: int,
) -> str:
    """Build a human-readable routing-feedback regression gate result."""

    return "\n".join(
        (
            "FleetMind routing feedback regression gate",
            f"Path: {path}",
            f"Schema version: {schema_version}",
            f"Revision: {result.trend_report.revision}",
            f"Status: {result.status}",
            f"Overall trend: {result.overall_direction}",
            (
                "Regressing strategies: "
                f"{_format_strategy_list(result.regressing_strategies)}"
            ),
            (
                "Insufficient strategies: "
                f"{_format_strategy_list(result.insufficient_strategies)}"
            ),
            f"Recommended exit code: {result.recommended_exit_code}",
            f"Enforcement: {enforcement}",
            f"Process exit code: {process_exit_code}",
            "Reasons:",
            *(f"- {reason}" for reason in result.reasons),
        )
    )


def build_feedback_gate_json(
    result: FeedbackRegressionGateResult,
    *,
    path: Path,
    schema_version: int,
    enforcement: GateEnforcement,
    process_exit_code: int,
) -> str:
    """Build deterministic machine-readable regression gate output."""

    payload = {
        "enforcement": enforcement,
        "insufficient_strategies": list(result.insufficient_strategies),
        "output_schema_version": 1,
        "overall_direction": result.overall_direction,
        "path": str(path),
        "process_exit_code": process_exit_code,
        "recommended_exit_code": result.recommended_exit_code,
        "reasons": list(result.reasons),
        "regressing_strategies": list(result.regressing_strategies),
        "revision": result.trend_report.revision,
        "status": result.status,
        "store_schema_version": schema_version,
        "trend": {
            "minimum_utility_change": (
                result.trend_report.policy.minimum_utility_change
            ),
            "recent_end_position": (result.trend_report.recent_end_position),
            "recent_start_position": (result.trend_report.recent_start_position),
            "total_observations": result.trend_report.total_observations,
            "utility_delta": result.trend_report.overall.utility_delta,
            "window_size": result.trend_report.policy.window_size,
        },
    }
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)


def _format_strategy_list(
    strategies: tuple[str, ...],
) -> str:
    return "none" if not strategies else ", ".join(strategies)


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

    subparsers.add_parser(
        "feedback-report",
        help="Summarize persisted adaptive-routing feedback.",
    )
    trend_parser = subparsers.add_parser(
        "feedback-trend",
        help="Compare the latest two routing-feedback windows.",
    )
    trend_parser.add_argument(
        "--window-size",
        type=int,
        default=10,
        help="Observations per chronological comparison window.",
    )
    trend_parser.add_argument(
        "--minimum-change",
        type=float,
        default=0.05,
        help="Minimum absolute utility change for improvement or regression.",
    )
    trend_parser.add_argument(
        "--minimum-strategy-observations",
        type=int,
        default=2,
        help="Required strategy observations in each comparison window.",
    )
    gate_parser = subparsers.add_parser(
        "feedback-gate",
        help="Evaluate routing-feedback trends as an operational gate.",
    )
    gate_parser.add_argument(
        "--window-size",
        type=int,
        default=10,
        help="Observations per chronological comparison window.",
    )
    gate_parser.add_argument(
        "--minimum-change",
        type=float,
        default=0.05,
        help="Minimum absolute utility change for improvement or regression.",
    )
    gate_parser.add_argument(
        "--minimum-strategy-observations",
        type=int,
        default=2,
        help="Required strategy observations in each comparison window.",
    )
    gate_parser.add_argument(
        "--feedback-path",
        type=Path,
        default=None,
        help=(
            "Explicit routing-feedback JSON snapshot; defaults to the "
            "configured runtime store."
        ),
    )
    gate_parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format for people or automation.",
    )
    gate_parser.add_argument(
        "--fail-on",
        choices=("warn", "fail", "never"),
        default="fail",
        help="Lowest gate status that produces a non-zero exit code.",
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
    feedback_revision: int | None = None

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
                feedback_snapshot = runtime.persist_feedback(
                    adaptive_result.feedback_history
                )
                feedback_revision = feedback_snapshot.revision
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
        print(
            build_adaptive_grounded_answer_message(
                adaptive_result,
                feedback_revision=feedback_revision,
            )
        )

    return 0


def run_feedback_report(settings: FleetMindSettings) -> int:
    """Load and summarize the configured persistent routing feedback."""

    path = settings.qdrant_path / "routing_feedback.json"

    try:
        snapshot = JsonRoutingFeedbackStore(path).load()
        report = RoutingFeedbackAnalyzer().analyze(
            snapshot.history,
            revision=snapshot.revision,
        )
    except (OSError, ValueError, RuntimeError) as error:
        print(f"FleetMind feedback report failed: {error}", file=sys.stderr)
        return 1

    print(
        build_feedback_report_message(
            report,
            path=path,
            schema_version=snapshot.schema_version,
        )
    )
    return 0


def run_feedback_trend(
    settings: FleetMindSettings,
    *,
    window_size: int = 10,
    minimum_utility_change: float = 0.05,
    minimum_strategy_observations: int = 2,
) -> int:
    """Load persisted feedback and compare adjacent chronological windows."""

    path = settings.qdrant_path / "routing_feedback.json"

    try:
        snapshot = JsonRoutingFeedbackStore(path).load()
        report = RoutingFeedbackTrendAnalyzer(
            FeedbackTrendPolicy(
                window_size=window_size,
                minimum_utility_change=minimum_utility_change,
                minimum_strategy_observations=minimum_strategy_observations,
            )
        ).analyze(
            snapshot.history,
            revision=snapshot.revision,
        )
    except (OSError, ValueError, RuntimeError) as error:
        print(f"FleetMind feedback trend failed: {error}", file=sys.stderr)
        return 1

    print(
        build_feedback_trend_message(
            report,
            path=path,
            schema_version=snapshot.schema_version,
        )
    )
    return 0


def run_feedback_gate(
    settings: FleetMindSettings,
    *,
    window_size: int = 10,
    minimum_utility_change: float = 0.05,
    minimum_strategy_observations: int = 2,
    feedback_path: Path | None = None,
    output_format: str = "text",
    enforcement: GateEnforcement = "fail",
) -> int:
    """Evaluate the persisted trend report as an automation-friendly gate."""

    if output_format not in {"text", "json"}:
        print(
            f"FleetMind feedback gate failed: unsupported output format: "
            f"{output_format!r}",
            file=sys.stderr,
        )
        return 1

    path = (
        feedback_path.expanduser()
        if feedback_path is not None
        else settings.qdrant_path / "routing_feedback.json"
    )

    try:
        snapshot = JsonRoutingFeedbackStore(path).load()
        trend_report = RoutingFeedbackTrendAnalyzer(
            FeedbackTrendPolicy(
                window_size=window_size,
                minimum_utility_change=minimum_utility_change,
                minimum_strategy_observations=minimum_strategy_observations,
            )
        ).analyze(
            snapshot.history,
            revision=snapshot.revision,
        )
        result = FeedbackRegressionGate().evaluate(trend_report)
        exit_code = result.process_exit_code(enforcement)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"FleetMind feedback gate failed: {error}", file=sys.stderr)
        return 1

    if output_format == "json":
        print(
            build_feedback_gate_json(
                result,
                path=path,
                schema_version=snapshot.schema_version,
                enforcement=enforcement,
                process_exit_code=exit_code,
            )
        )
    else:
        print(
            build_feedback_gate_message(
                result,
                path=path,
                schema_version=snapshot.schema_version,
                enforcement=enforcement,
                process_exit_code=exit_code,
            )
        )

    return exit_code


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

    if command == "feedback-report":
        return run_feedback_report(settings)

    if command == "feedback-trend":
        return run_feedback_trend(
            settings,
            window_size=cast(int, args.window_size),
            minimum_utility_change=cast(float, args.minimum_change),
            minimum_strategy_observations=cast(
                int,
                args.minimum_strategy_observations,
            ),
        )

    if command == "feedback-gate":
        return run_feedback_gate(
            settings,
            window_size=cast(int, args.window_size),
            minimum_utility_change=cast(float, args.minimum_change),
            minimum_strategy_observations=cast(
                int,
                args.minimum_strategy_observations,
            ),
            feedback_path=cast(Path | None, args.feedback_path),
            output_format=cast(str, args.format),
            enforcement=cast(GateEnforcement, args.fail_on),
        )

    return run_status(settings)
