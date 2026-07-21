from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

import pytest
from pydantic_settings import SettingsConfigDict

from fleetmind_rag.app import (
    DEFAULT_SYSTEM_PROMPT,
    build_system_prompt,
    main,
    run_ask,
    run_index,
)
from fleetmind_rag.config import FleetMindSettings
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
class _FakeRuntime:
    retrieval_service: _FakeRetrievalService
    grounded_answer_service: _FakeGroundedService
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


def test_run_index_uses_configured_defaults_and_prints_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = _TestFleetMindSettings()
    retrieval = _FakeRetrievalService(_index_result(), [])
    runtime = _FakeRuntime(retrieval, _FakeGroundedService(_grounded_result(), []))
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
    runtime = _FakeRuntime(retrieval, _FakeGroundedService(_grounded_result(), []))
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
    runtime = _FakeRuntime(_FakeRetrievalService(_index_result(), []), grounded)
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
    runtime = _FakeRuntime(_FakeRetrievalService(_index_result(), []), grounded)
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
    grounded = _FakeGroundedService(failed_result, [])
    runtime = _FakeRuntime(_FakeRetrievalService(_index_result(), []), grounded)
    monkeypatch.setattr(
        "fleetmind_rag.app.create_rag_runtime", lambda settings: runtime
    )

    exit_code = run_ask(settings, "Question")
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    assert captured.err == "FleetMind grounded answer failed: generation failed\n"


def test_cli_parser_lists_rag_commands(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(["--help"])

    captured = capsys.readouterr()

    assert error.value.code == 0
    assert "{chat,index,ask}" in captured.out
    assert "citation-grounded" in captured.out
    assert captured.err == ""
