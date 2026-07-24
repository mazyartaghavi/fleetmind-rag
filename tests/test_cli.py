from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any

import pytest
from pydantic_settings import SettingsConfigDict

from fleetmind_rag.app import (
    DEFAULT_SYSTEM_PROMPT,
    build_adaptive_grounded_answer_message,
    build_feedback_gate_json,
    build_feedback_gate_message,
    build_feedback_report_message,
    build_feedback_trend_message,
    build_system_prompt,
    main,
    run_ask,
    run_feedback_gate,
    run_feedback_report,
    run_feedback_trend,
    run_index,
)
from fleetmind_rag.config import FleetMindSettings
from fleetmind_rag.feedback_analytics import RoutingFeedbackAnalyzer
from fleetmind_rag.feedback_gates import FeedbackRegressionGate
from fleetmind_rag.feedback_routing import (
    RoutingFeedbackHistory,
    RoutingFeedbackObservation,
)
from fleetmind_rag.feedback_store import FeedbackStoreSnapshot
from fleetmind_rag.feedback_trends import (
    FeedbackTrendPolicy,
    RoutingFeedbackTrendAnalyzer,
)
from fleetmind_rag.grounded_rag import (
    GroundedAnswerResult,
    GroundedCitation,
)
from fleetmind_rag.ollama import OllamaChatResult
from fleetmind_rag.retrieval import DocumentIndexResult


class _TestFleetMindSettings(FleetMindSettings):
    model_config = SettingsConfigDict(env_file=None)


@pytest.mark.parametrize("user_system_prompt", [None, "", "   "])
def test_build_system_prompt_uses_default_for_missing_instruction(
    user_system_prompt: str | None,
) -> None:
    result = build_system_prompt(user_system_prompt)

    assert result == DEFAULT_SYSTEM_PROMPT


def test_build_system_prompt_appends_user_instruction() -> None:
    result = build_system_prompt("  Answer briefly.  ")

    assert result == (
        f"{DEFAULT_SYSTEM_PROMPT}\n\nAdditional user instruction: Answer briefly."
    )


def test_chat_command_prints_model_response(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class _FakeOllamaChatClient:
        def __init__(
            self,
            base_url: str,
            model: str,
            *,
            timeout_seconds: float,
        ) -> None:
            assert base_url == "http://localhost:11434/"
            assert model == "llama3.2:3b"
            assert timeout_seconds == 120.0

        def chat(
            self,
            prompt: str,
            *,
            system_prompt: str | None = None,
        ) -> OllamaChatResult:
            assert prompt == "Summarize fleet status."
            assert system_prompt == (
                f"{DEFAULT_SYSTEM_PROMPT}\n\n"
                "Additional user instruction: Answer briefly."
            )

            return OllamaChatResult(
                succeeded=True,
                content="All monitored vehicles are operational.",
                model="llama3.2:3b",
                message="The Ollama chat request succeeded.",
            )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "fleetmind_rag.app.OllamaChatClient",
        _FakeOllamaChatClient,
    )

    exit_code = main(
        [
            "chat",
            "Summarize fleet status.",
            "--system",
            "Answer briefly.",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out == "All monitored vehicles are operational.\n"
    assert captured.err == ""


def test_chat_command_reports_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class _FakeFailedOllamaChatClient:
        def __init__(
            self,
            base_url: str,
            model: str,
            *,
            timeout_seconds: float,
        ) -> None:
            assert base_url == "http://localhost:11434/"
            assert model == "llama3.2:3b"
            assert timeout_seconds == 120.0

        def chat(
            self,
            prompt: str,
            *,
            system_prompt: str | None = None,
        ) -> OllamaChatResult:
            assert prompt == "Hello"
            assert system_prompt == DEFAULT_SYSTEM_PROMPT

            return OllamaChatResult(
                succeeded=False,
                content=None,
                model=None,
                message="The Ollama API is unreachable.",
            )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "fleetmind_rag.app.OllamaChatClient",
        _FakeFailedOllamaChatClient,
    )

    exit_code = main(["chat", "Hello"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    assert captured.err == "FleetMind chat failed: The Ollama API is unreachable.\n"


@dataclass
class _FakeRetrievalService:
    index_result: DocumentIndexResult
    index_calls: list[tuple[Path, str | None, int, int, bool]]

    def index_text_document(
        self,
        path: Path,
        *,
        default_title: str | None,
        chunk_size_words: int,
        overlap_words: int,
        recreate_collection: bool,
    ) -> DocumentIndexResult:
        self.index_calls.append(
            (
                path,
                default_title,
                chunk_size_words,
                overlap_words,
                recreate_collection,
            )
        )
        return self.index_result


@dataclass
class _FakeGroundedService:
    result: GroundedAnswerResult
    calls: list[tuple[str, int]]

    def answer(self, question: str, *, limit: int) -> GroundedAnswerResult:
        self.calls.append((question, limit))
        return self.result


@dataclass
class _FakeAdaptiveGroundedService:
    result: Any
    calls: list[tuple[str, Any]]

    def answer(self, question: str, *, config: Any) -> Any:
        self.calls.append((question, config))
        return self.result


@dataclass
class _FakeRuntime:
    retrieval_service: _FakeRetrievalService
    grounded_answer_service: _FakeGroundedService
    adaptive_grounded_answer_service: _FakeAdaptiveGroundedService
    persisted_histories: list[RoutingFeedbackHistory]
    feedback_revision: int = 0
    persist_error: RuntimeError | None = None
    closed: bool = False

    def __enter__(self) -> _FakeRuntime:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.closed = True

    def persist_feedback(
        self,
        history: RoutingFeedbackHistory,
    ) -> FeedbackStoreSnapshot:
        if self.persist_error is not None:
            raise self.persist_error

        self.persisted_histories.append(history)
        self.feedback_revision += 1
        return FeedbackStoreSnapshot(
            history=history,
            revision=self.feedback_revision,
        )


def _index_result() -> DocumentIndexResult:
    return DocumentIndexResult(
        document_id="doc-123",
        source_name="manual.md",
        section_count=2,
        chunk_count=3,
        stored_count=3,
        embedding_model="embeddinggemma",
        vector_size=768,
    )


def _grounded_result(*, abstained: bool = False) -> GroundedAnswerResult:
    citations: tuple[GroundedCitation, ...] = ()
    answer = "I do not have enough grounded evidence."
    top_score = 0.32

    if not abstained:
        citations = (
            GroundedCitation(
                label="S1",
                chunk_id="chunk-1",
                document_id="doc-123",
                section_id="section-1",
                section_title="Engine warning procedure",
                text="Stop the vehicle safely.",
                score=0.91,
            ),
        )
        answer = "Stop the vehicle safely [S1]."
        top_score = 0.91

    return GroundedAnswerResult(
        succeeded=True,
        abstained=abstained,
        question="What should I do?",
        answer=answer,
        citations=citations,
        retrieval_model="embeddinggemma",
        generation_model=None if abstained else "llama3.2:3b",
        top_score=top_score,
        message="completed",
    )


def _adaptive_result(
    grounded_result: GroundedAnswerResult,
    *,
    status: str = "completed",
    attempts: int = 2,
    rewrites: int = 1,
    initial_strategy: str = "dense",
    final_strategy: str | None = "hybrid",
    feedback_observations: int = 2,
) -> Any:
    resolved_status = status
    resolved_rewrites = tuple(object() for _ in range(rewrites))
    resolved_observations = tuple(
        RoutingFeedbackObservation(
            query=f"feedback query {index}",
            strategy="dense",
            verdict="accept",
            quality_score=1.0,
            attempt_number=index + 1,
            features=("general",),
        )
        for index in range(feedback_observations)
    )

    class _Decision:
        strategy = final_strategy

    class _LatestResult:
        decision = _Decision()

    class _State:
        status = resolved_status
        latest_result = None if final_strategy is None else _LatestResult()

    class _Outcome:
        state = _State()
        rewrites = resolved_rewrites

    class _InitialRouting:
        strategy = initial_strategy

    class _Result:
        grounded_answer = grounded_result
        retrieval_outcome = _Outcome()
        initial_routing = _InitialRouting()
        feedback_history = RoutingFeedbackHistory(resolved_observations)
        attempt_count = attempts

    return _Result()


def _fake_runtime(
    *,
    grounded_result: GroundedAnswerResult | None = None,
    adaptive_result: Any | None = None,
) -> _FakeRuntime:
    resolved_grounded = grounded_result or _grounded_result()
    resolved_adaptive = adaptive_result or _adaptive_result(resolved_grounded)
    return _FakeRuntime(
        _FakeRetrievalService(_index_result(), []),
        _FakeGroundedService(resolved_grounded, []),
        _FakeAdaptiveGroundedService(resolved_adaptive, []),
        [],
    )


def test_run_index_uses_configured_defaults_and_prints_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = _TestFleetMindSettings()
    retrieval = _FakeRetrievalService(_index_result(), [])
    runtime = _FakeRuntime(
        retrieval,
        _FakeGroundedService(_grounded_result(), []),
        _FakeAdaptiveGroundedService(
            _adaptive_result(_grounded_result()),
            [],
        ),
        [],
    )
    monkeypatch.setattr(
        "fleetmind_rag.app.create_rag_runtime", lambda settings: runtime
    )

    exit_code = run_index(settings, Path("manual.md"))
    captured = capsys.readouterr()

    assert exit_code == 0
    assert retrieval.index_calls == [(Path("manual.md"), None, 180, 30, False)]
    assert "Fleet document indexed successfully." in captured.out
    assert "Stored vectors: 3" in captured.out
    assert "Qdrant collection: fleetmind_documents" in captured.out
    assert captured.err == ""
    assert runtime.closed


def test_index_command_forwards_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    retrieval = _FakeRetrievalService(_index_result(), [])
    runtime = _FakeRuntime(
        retrieval,
        _FakeGroundedService(_grounded_result(), []),
        _FakeAdaptiveGroundedService(
            _adaptive_result(_grounded_result()),
            [],
        ),
        [],
    )
    monkeypatch.setattr(
        "fleetmind_rag.app.create_rag_runtime", lambda settings: runtime
    )

    exit_code = main(
        [
            "index",
            "manual.md",
            "--title",
            "Fleet Manual",
            "--chunk-size",
            "120",
            "--overlap",
            "20",
            "--recreate",
        ]
    )

    assert exit_code == 0
    assert retrieval.index_calls == [(Path("manual.md"), "Fleet Manual", 120, 20, True)]


def test_run_index_reports_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = _TestFleetMindSettings()

    def _raise_runtime_error(settings: FleetMindSettings) -> _FakeRuntime:
        del settings
        raise RuntimeError("embedding service unavailable")

    monkeypatch.setattr("fleetmind_rag.app.create_rag_runtime", _raise_runtime_error)

    exit_code = run_index(settings, Path("manual.md"))
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    assert captured.err == (
        "FleetMind indexing failed: embedding service unavailable\n"
    )


def test_run_ask_prints_answer_and_sources(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = _TestFleetMindSettings()
    grounded = _FakeGroundedService(_grounded_result(), [])
    runtime = _fake_runtime()
    grounded = runtime.grounded_answer_service
    monkeypatch.setattr(
        "fleetmind_rag.app.create_rag_runtime", lambda settings: runtime
    )

    exit_code = run_ask(settings, "What should I do?")
    captured = capsys.readouterr()

    assert exit_code == 0
    assert grounded.calls == [("What should I do?", 5)]
    assert "Stop the vehicle safely [S1]." in captured.out
    assert "Sources:" in captured.out
    assert "[S1] Engine warning procedure" in captured.out
    assert captured.err == ""
    assert runtime.closed


def test_ask_command_prints_safe_abstention(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    grounded = _FakeGroundedService(_grounded_result(abstained=True), [])
    runtime = _fake_runtime(grounded_result=grounded.result)
    grounded = runtime.grounded_answer_service
    monkeypatch.setattr(
        "fleetmind_rag.app.create_rag_runtime", lambda settings: runtime
    )

    exit_code = main(["ask", "What should I do?", "--limit", "7"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert grounded.calls == [("What should I do?", 7)]
    assert "I do not have enough grounded evidence." in captured.out
    assert "Top retrieval score: 0.3200" in captured.out
    assert captured.err == ""


def test_run_ask_reports_unsuccessful_result(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = _TestFleetMindSettings()
    failed_result = GroundedAnswerResult(
        succeeded=False,
        abstained=False,
        question="Question",
        answer=None,
        citations=(),
        retrieval_model="embeddinggemma",
        generation_model=None,
        top_score=0.9,
        message="generation failed",
    )
    runtime = _fake_runtime(grounded_result=failed_result)
    monkeypatch.setattr(
        "fleetmind_rag.app.create_rag_runtime", lambda settings: runtime
    )

    exit_code = run_ask(settings, "Question")
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    assert captured.err == "FleetMind grounded answer failed: generation failed\n"


def test_build_adaptive_grounded_answer_message_includes_trace() -> None:
    result = _adaptive_result(_grounded_result())

    message = build_adaptive_grounded_answer_message(result)

    assert "Stop the vehicle safely [S1]." in message
    assert "Adaptive retrieval:" in message
    assert "Status: completed" in message
    assert "Attempts: 2" in message
    assert "Rewrites: 1" in message
    assert "Initial strategy: dense" in message
    assert "Final strategy: hybrid" in message
    assert "Feedback observations: 2" in message


def test_run_ask_uses_adaptive_service_and_prints_trace(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = _TestFleetMindSettings()
    runtime = _fake_runtime()
    monkeypatch.setattr(
        "fleetmind_rag.app.create_rag_runtime", lambda settings: runtime
    )

    exit_code = run_ask(
        settings,
        "What should I do?",
        limit=7,
        adaptive=True,
        max_attempts=4,
        candidate_limit=30,
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert runtime.grounded_answer_service.calls == []
    assert len(runtime.adaptive_grounded_answer_service.calls) == 1
    question, config = runtime.adaptive_grounded_answer_service.calls[0]
    assert question == "What should I do?"
    assert config.max_attempts == 4
    assert config.limit == 7
    assert config.candidate_limit == 30
    assert runtime.persisted_histories == [
        runtime.adaptive_grounded_answer_service.result.feedback_history
    ]
    assert "Adaptive retrieval:" in captured.out
    assert "Feedback revision: 1" in captured.out
    assert captured.err == ""
    assert runtime.closed


def test_adaptive_ask_uses_safe_candidate_limit_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _TestFleetMindSettings()
    runtime = _fake_runtime()
    monkeypatch.setattr(
        "fleetmind_rag.app.create_rag_runtime", lambda settings: runtime
    )

    exit_code = run_ask(
        settings,
        "What should I do?",
        limit=25,
        adaptive=True,
    )

    assert exit_code == 0
    _, config = runtime.adaptive_grounded_answer_service.calls[0]
    assert config.limit == 25
    assert config.candidate_limit == 25


def test_adaptive_ask_rejects_invalid_retry_configuration(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = _TestFleetMindSettings()
    runtime = _fake_runtime()
    monkeypatch.setattr(
        "fleetmind_rag.app.create_rag_runtime", lambda settings: runtime
    )

    exit_code = run_ask(
        settings,
        "What should I do?",
        adaptive=True,
        max_attempts=0,
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert runtime.adaptive_grounded_answer_service.calls == []
    assert captured.out == ""
    assert "max_attempts must be greater than zero" in captured.err
    assert runtime.closed


def test_adaptive_ask_command_forwards_cli_controls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    runtime = _fake_runtime()
    monkeypatch.setattr(
        "fleetmind_rag.app.create_rag_runtime", lambda settings: runtime
    )

    exit_code = main(
        [
            "ask",
            "What should I do?",
            "--adaptive",
            "--limit",
            "6",
            "--max-attempts",
            "4",
            "--candidate-limit",
            "24",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    _, config = runtime.adaptive_grounded_answer_service.calls[0]
    assert config.limit == 6
    assert config.max_attempts == 4
    assert config.candidate_limit == 24
    assert "Adaptive retrieval:" in captured.out
    assert "Feedback revision: 1" in captured.out
    assert captured.err == ""


def test_legacy_ask_does_not_persist_feedback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = _TestFleetMindSettings()
    runtime = _fake_runtime()
    monkeypatch.setattr(
        "fleetmind_rag.app.create_rag_runtime", lambda settings: runtime
    )

    exit_code = run_ask(settings, "What should I do?")
    captured = capsys.readouterr()

    assert exit_code == 0
    assert runtime.persisted_histories == []
    assert "Feedback revision:" not in captured.out


def test_adaptive_ask_reports_feedback_persistence_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = _TestFleetMindSettings()
    runtime = _fake_runtime()
    runtime.persist_error = RuntimeError("feedback revision conflict")
    monkeypatch.setattr(
        "fleetmind_rag.app.create_rag_runtime", lambda settings: runtime
    )

    exit_code = run_ask(
        settings,
        "What should I do?",
        adaptive=True,
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    assert "feedback revision conflict" in captured.err
    assert runtime.closed


def test_cli_parser_lists_rag_commands(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(["--help"])

    captured = capsys.readouterr()

    assert error.value.code == 0
    assert (
        "{chat,index,ask,feedback-report,feedback-trend,feedback-gate}" in captured.out
    )
    assert "citation-grounded" in captured.out
    assert "persisted adaptive-routing feedback" in captured.out
    assert captured.err == ""


def test_ask_help_lists_adaptive_controls(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        main(["ask", "--help"])

    captured = capsys.readouterr()

    assert error.value.code == 0
    assert "--adaptive" in captured.out
    assert "--max-attempts" in captured.out
    assert "--candidate-limit" in captured.out
    assert captured.err == ""


def _feedback_history() -> RoutingFeedbackHistory:
    return RoutingFeedbackHistory(
        (
            RoutingFeedbackObservation(
                query="What does overheating mean?",
                strategy="dense",
                verdict="accept",
                quality_score=0.9,
                attempt_number=1,
                features=("conceptual",),
            ),
            RoutingFeedbackObservation(
                query="What does overheating mean?",
                strategy="hybrid",
                verdict="rewrite",
                quality_score=0.3,
                attempt_number=2,
                features=("conceptual",),
            ),
        )
    )


def test_build_feedback_report_message_formats_metrics() -> None:
    report = RoutingFeedbackAnalyzer().analyze(
        _feedback_history(),
        revision=7,
    )

    message = build_feedback_report_message(
        report,
        path=Path("data/qdrant_local/routing_feedback.json"),
        schema_version=1,
    )

    assert "FleetMind routing feedback report" in message
    assert "Revision: 7" in message
    assert "Observations: 2" in message
    assert "Accepted: 1" in message
    assert "Rewrites: 1" in message
    assert "Acceptance rate: 50.00%" in message
    assert "Rewrite rate: 50.00%" in message
    assert "Average quality: 0.6000" in message
    assert "Average attempt: 1.5000" in message
    assert "Retry-attempt rate: 50.00%" in message
    assert "dense" in message
    assert "hybrid" in message
    assert "conceptual" in message


def test_build_feedback_report_message_handles_empty_history() -> None:
    report = RoutingFeedbackAnalyzer().analyze(RoutingFeedbackHistory())

    message = build_feedback_report_message(
        report,
        path=Path("feedback.json"),
        schema_version=1,
    )

    assert "Observations: 0" in message
    assert "Acceptance rate: n/a" in message
    assert "Average quality: n/a" in message
    assert "No query-signal features have been observed." in message


def test_run_feedback_report_loads_configured_store_and_prints_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = _TestFleetMindSettings(qdrant_path=tmp_path / "qdrant")
    expected_path = tmp_path / "qdrant" / "routing_feedback.json"
    snapshot = FeedbackStoreSnapshot(
        history=_feedback_history(),
        revision=11,
    )
    created_paths: list[Path] = []

    class _FakeStore:
        def __init__(self, path: Path) -> None:
            created_paths.append(path)

        def load(self) -> FeedbackStoreSnapshot:
            return snapshot

    monkeypatch.setattr(
        "fleetmind_rag.app.JsonRoutingFeedbackStore",
        _FakeStore,
    )

    exit_code = run_feedback_report(settings)
    captured = capsys.readouterr()

    assert exit_code == 0
    assert created_paths == [expected_path]
    assert f"Path: {expected_path}" in captured.out
    assert "Schema version: 1" in captured.out
    assert "Revision: 11" in captured.out
    assert "Observations: 2" in captured.out
    assert captured.err == ""


def test_feedback_report_command_routes_through_main(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    snapshot = FeedbackStoreSnapshot(
        history=_feedback_history(),
        revision=3,
    )

    class _FakeStore:
        def __init__(self, path: Path) -> None:
            assert path == Path("data/qdrant_local/routing_feedback.json")

        def load(self) -> FeedbackStoreSnapshot:
            return snapshot

    monkeypatch.setattr(
        "fleetmind_rag.app.JsonRoutingFeedbackStore",
        _FakeStore,
    )

    exit_code = main(["feedback-report"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Revision: 3" in captured.out
    assert "Strategy performance:" in captured.out
    assert "Feature performance:" in captured.out
    assert captured.err == ""


def test_feedback_report_returns_failure_for_invalid_store(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = _TestFleetMindSettings()

    class _FailedStore:
        def __init__(self, path: Path) -> None:
            del path

        def load(self) -> FeedbackStoreSnapshot:
            raise RuntimeError("feedback JSON is malformed")

    monkeypatch.setattr(
        "fleetmind_rag.app.JsonRoutingFeedbackStore",
        _FailedStore,
    )

    exit_code = run_feedback_report(settings)
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    assert captured.err == (
        "FleetMind feedback report failed: feedback JSON is malformed\n"
    )


def _trend_history() -> RoutingFeedbackHistory:
    return RoutingFeedbackHistory(
        (
            RoutingFeedbackObservation(
                query="What does overheating mean?",
                strategy="dense",
                verdict="rewrite",
                quality_score=0.0,
                attempt_number=1,
                features=("conceptual",),
            ),
            RoutingFeedbackObservation(
                query="What does overheating mean?",
                strategy="dense",
                verdict="rewrite",
                quality_score=0.0,
                attempt_number=2,
                features=("conceptual",),
            ),
            RoutingFeedbackObservation(
                query="What does overheating mean?",
                strategy="dense",
                verdict="accept",
                quality_score=1.0,
                attempt_number=1,
                features=("conceptual",),
            ),
            RoutingFeedbackObservation(
                query="What does overheating mean?",
                strategy="dense",
                verdict="accept",
                quality_score=1.0,
                attempt_number=1,
                features=("conceptual",),
            ),
        )
    )


def test_build_feedback_trend_message_formats_improving_report() -> None:
    report = RoutingFeedbackTrendAnalyzer(
        FeedbackTrendPolicy(
            window_size=2,
            minimum_strategy_observations=1,
        )
    ).analyze(_trend_history(), revision=9)

    message = build_feedback_trend_message(
        report,
        path=Path("data/qdrant_local/routing_feedback.json"),
        schema_version=1,
    )

    assert "FleetMind routing feedback trend report" in message
    assert "Revision: 9" in message
    assert "Total observations: 4" in message
    assert "Previous window: 1-2" in message
    assert "Recent window: 3-4" in message
    assert "Overall trend: improving" in message
    assert "Previous utility: 0.0000" in message
    assert "Recent utility: 1.0000" in message
    assert "Utility delta: +1.0000" in message
    assert "Acceptance delta: +100.00%" in message
    assert "dense" in message


def test_run_feedback_trend_loads_store_and_prints_comparison(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = _TestFleetMindSettings(qdrant_path=tmp_path / "qdrant")
    expected_path = tmp_path / "qdrant" / "routing_feedback.json"
    snapshot = FeedbackStoreSnapshot(
        history=_trend_history(),
        revision=12,
    )
    created_paths: list[Path] = []

    class _FakeStore:
        def __init__(self, path: Path) -> None:
            created_paths.append(path)

        def load(self) -> FeedbackStoreSnapshot:
            return snapshot

    monkeypatch.setattr(
        "fleetmind_rag.app.JsonRoutingFeedbackStore",
        _FakeStore,
    )

    exit_code = run_feedback_trend(
        settings,
        window_size=2,
        minimum_strategy_observations=1,
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert created_paths == [expected_path]
    assert "Revision: 12" in captured.out
    assert "Overall trend: improving" in captured.out
    assert captured.err == ""


def test_feedback_trend_command_forwards_cli_controls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    snapshot = FeedbackStoreSnapshot(
        history=_trend_history(),
        revision=6,
    )

    class _FakeStore:
        def __init__(self, path: Path) -> None:
            assert path == Path("data/qdrant_local/routing_feedback.json")

        def load(self) -> FeedbackStoreSnapshot:
            return snapshot

    monkeypatch.setattr(
        "fleetmind_rag.app.JsonRoutingFeedbackStore",
        _FakeStore,
    )

    exit_code = main(
        [
            "feedback-trend",
            "--window-size",
            "2",
            "--minimum-change",
            "0.10",
            "--minimum-strategy-observations",
            "1",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Window size: 2" in captured.out
    assert "Minimum utility change: 0.1000" in captured.out
    assert "Overall trend: improving" in captured.out
    assert captured.err == ""


def test_feedback_trend_reports_invalid_controls(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = _TestFleetMindSettings()

    class _FakeStore:
        def __init__(self, path: Path) -> None:
            del path

        def load(self) -> FeedbackStoreSnapshot:
            return FeedbackStoreSnapshot(
                history=RoutingFeedbackHistory(),
                revision=0,
            )

    monkeypatch.setattr(
        "fleetmind_rag.app.JsonRoutingFeedbackStore",
        _FakeStore,
    )

    exit_code = run_feedback_trend(settings, window_size=0)
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    assert captured.err == (
        "FleetMind feedback trend failed: window_size must be greater than zero\n"
    )


def test_feedback_trend_help_lists_comparison_controls(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        main(["feedback-trend", "--help"])

    captured = capsys.readouterr()

    assert error.value.code == 0
    assert "--window-size" in captured.out
    assert "--minimum-change" in captured.out
    assert "--minimum-strategy-observations" in captured.out
    assert captured.err == ""


def _gate_history(*, regressing: bool = False) -> RoutingFeedbackHistory:
    if regressing:
        return RoutingFeedbackHistory(
            (
                RoutingFeedbackObservation(
                    query="What does overheating mean?",
                    strategy="dense",
                    verdict="accept",
                    quality_score=1.0,
                    attempt_number=1,
                    features=("conceptual",),
                ),
                RoutingFeedbackObservation(
                    query="What does overheating mean?",
                    strategy="dense",
                    verdict="accept",
                    quality_score=1.0,
                    attempt_number=1,
                    features=("conceptual",),
                ),
                RoutingFeedbackObservation(
                    query="What does overheating mean?",
                    strategy="dense",
                    verdict="rewrite",
                    quality_score=0.0,
                    attempt_number=2,
                    features=("conceptual",),
                ),
                RoutingFeedbackObservation(
                    query="What does overheating mean?",
                    strategy="dense",
                    verdict="rewrite",
                    quality_score=0.0,
                    attempt_number=2,
                    features=("conceptual",),
                ),
            )
        )

    return RoutingFeedbackHistory(
        (
            RoutingFeedbackObservation(
                query="What does overheating mean?",
                strategy="dense",
                verdict="accept",
                quality_score=1.0,
                attempt_number=1,
                features=("conceptual",),
            ),
        )
    )


def _gate_result(*, regressing: bool = False) -> Any:
    trend = RoutingFeedbackTrendAnalyzer(
        FeedbackTrendPolicy(
            window_size=2,
            minimum_strategy_observations=1,
        )
    ).analyze(_gate_history(regressing=regressing), revision=8)
    return FeedbackRegressionGate().evaluate(trend)


def test_build_feedback_gate_message_formats_warning() -> None:
    result = _gate_result()

    message = build_feedback_gate_message(
        result,
        path=Path("data/qdrant_local/routing_feedback.json"),
        schema_version=1,
        enforcement="fail",
        process_exit_code=0,
    )

    assert "FleetMind routing feedback regression gate" in message
    assert "Revision: 8" in message
    assert "Status: warn" in message
    assert "Overall trend: insufficient_data" in message
    assert "Regressing strategies: none" in message
    assert "Insufficient strategies: dense" in message
    assert "Recommended exit code: 2" in message
    assert "Enforcement: fail" in message
    assert "Process exit code: 0" in message
    assert "Reasons:" in message


def test_build_feedback_gate_json_is_machine_readable() -> None:
    result = _gate_result(regressing=True)

    message = build_feedback_gate_json(
        result,
        path=Path("feedback.json"),
        schema_version=1,
        enforcement="fail",
        process_exit_code=3,
    )
    payload = json.loads(message)

    assert payload["output_schema_version"] == 1
    assert payload["store_schema_version"] == 1
    assert payload["revision"] == 8
    assert payload["status"] == "fail"
    assert payload["overall_direction"] == "regressing"
    assert payload["regressing_strategies"] == ["dense"]
    assert payload["recommended_exit_code"] == 3
    assert payload["process_exit_code"] == 3
    assert payload["enforcement"] == "fail"
    assert payload["trend"]["window_size"] == 2
    assert payload["trend"]["utility_delta"] == pytest.approx(-1.0)


def test_run_feedback_gate_warns_without_failing_by_default(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = _TestFleetMindSettings()

    class _FakeStore:
        def __init__(self, path: Path) -> None:
            assert path == Path("data/qdrant_local/routing_feedback.json")

        def load(self) -> FeedbackStoreSnapshot:
            return FeedbackStoreSnapshot(
                history=RoutingFeedbackHistory(
                    (
                        RoutingFeedbackObservation(
                            query="What does overheating mean?",
                            strategy="dense",
                            verdict="accept",
                            quality_score=1.0,
                            attempt_number=1,
                            features=("conceptual",),
                        ),
                    )
                ),
                revision=2,
            )

    monkeypatch.setattr(
        "fleetmind_rag.app.JsonRoutingFeedbackStore",
        _FakeStore,
    )

    exit_code = run_feedback_gate(
        settings,
        window_size=2,
        minimum_strategy_observations=1,
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Status: warn" in captured.out
    assert "Recommended exit code: 2" in captured.out
    assert "Process exit code: 0" in captured.out
    assert captured.err == ""


def test_run_feedback_gate_can_enforce_warning(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = _TestFleetMindSettings()

    class _FakeStore:
        def __init__(self, path: Path) -> None:
            del path

        def load(self) -> FeedbackStoreSnapshot:
            return FeedbackStoreSnapshot(
                history=RoutingFeedbackHistory(),
                revision=0,
            )

    monkeypatch.setattr(
        "fleetmind_rag.app.JsonRoutingFeedbackStore",
        _FakeStore,
    )

    exit_code = run_feedback_gate(
        settings,
        window_size=2,
        minimum_strategy_observations=1,
        enforcement="warn",
    )
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "Status: warn" in captured.out
    assert "Process exit code: 2" in captured.out


def test_run_feedback_gate_returns_failure_exit_code(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = _TestFleetMindSettings()

    class _FakeStore:
        def __init__(self, path: Path) -> None:
            del path

        def load(self) -> FeedbackStoreSnapshot:
            return FeedbackStoreSnapshot(
                history=_gate_history(regressing=True),
                revision=8,
            )

    monkeypatch.setattr(
        "fleetmind_rag.app.JsonRoutingFeedbackStore",
        _FakeStore,
    )

    exit_code = run_feedback_gate(
        settings,
        window_size=2,
        minimum_strategy_observations=1,
    )
    captured = capsys.readouterr()

    assert exit_code == 3
    assert "Status: fail" in captured.out
    assert "Process exit code: 3" in captured.out


def test_feedback_gate_command_outputs_json_and_forwards_controls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    feedback_path = Path("evaluation/data/routing_feedback_ci.json")

    class _FakeStore:
        def __init__(self, path: Path) -> None:
            assert path == feedback_path

        def load(self) -> FeedbackStoreSnapshot:
            return FeedbackStoreSnapshot(
                history=RoutingFeedbackHistory(),
                revision=4,
            )

    monkeypatch.setattr(
        "fleetmind_rag.app.JsonRoutingFeedbackStore",
        _FakeStore,
    )

    exit_code = main(
        [
            "feedback-gate",
            "--window-size",
            "3",
            "--minimum-change",
            "0.10",
            "--minimum-strategy-observations",
            "1",
            "--feedback-path",
            str(feedback_path),
            "--format",
            "json",
            "--fail-on",
            "never",
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["revision"] == 4
    assert payload["status"] == "warn"
    assert payload["enforcement"] == "never"
    assert payload["process_exit_code"] == 0
    assert payload["trend"]["window_size"] == 3
    assert payload["trend"]["minimum_utility_change"] == pytest.approx(0.1)
    assert captured.err == ""


def test_run_feedback_gate_rejects_unknown_output_format(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = run_feedback_gate(
        _TestFleetMindSettings(),
        output_format="xml",
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    assert captured.err == (
        "FleetMind feedback gate failed: unsupported output format: 'xml'\n"
    )


def test_feedback_gate_help_lists_automation_controls(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        main(["feedback-gate", "--help"])

    captured = capsys.readouterr()

    assert error.value.code == 0
    assert "--window-size" in captured.out
    assert "--minimum-change" in captured.out
    assert "--minimum-strategy-observations" in captured.out
    assert "--feedback-path FEEDBACK_PATH" in captured.out
    assert "--format {text,json}" in captured.out
    assert "--fail-on {warn,fail,never}" in captured.out
    assert captured.err == ""
