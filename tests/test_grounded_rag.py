from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from fleetmind_rag.grounded_rag import (
    ABSTENTION_ANSWER,
    DEFAULT_GROUNDED_SYSTEM_PROMPT,
    GroundedAnswerService,
)
from fleetmind_rag.ollama import OllamaChatResult
from fleetmind_rag.retrieval import RetrievalResponse
from fleetmind_rag.vector_store import VectorSearchResult


@dataclass
class FakeRetrievalClient:
    response: RetrievalResponse
    calls: list[tuple[str, int, float | None]] = field(default_factory=list)

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        score_threshold: float | None = None,
    ) -> RetrievalResponse:
        self.calls.append((query, limit, score_threshold))
        return self.response


@dataclass
class FakeChatClient:
    result: OllamaChatResult
    prompts: list[str] = field(default_factory=list)
    system_prompts: list[str | None] = field(default_factory=list)

    def chat(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> OllamaChatResult:
        self.prompts.append(prompt)
        self.system_prompts.append(system_prompt)
        return self.result


def make_match(
    *,
    ordinal: int = 1,
    score: float = 0.9,
    text: str = "Stop the vehicle and inspect the engine warning indicator.",
) -> VectorSearchResult:
    return VectorSearchResult(
        chunk_id=f"chunk-{ordinal}",
        document_id="doc-1",
        section_id=f"section-{ordinal}",
        section_title=f"Warning Procedure {ordinal}",
        ordinal=ordinal,
        text=text,
        word_count=len(text.split()),
        start_word=0,
        end_word=len(text.split()),
        score=score,
    )


def make_retrieval_response(
    *matches: VectorSearchResult,
) -> RetrievalResponse:
    return RetrievalResponse(
        query="engine warning",
        embedding_model="embeddinggemma",
        matches=tuple(matches),
    )


def make_chat_result(
    content: str | None = "Stop and inspect the warning indicator [S1].",
    *,
    succeeded: bool = True,
    model: str | None = "llama3.2:3b",
    message: str = "The Ollama chat request succeeded.",
) -> OllamaChatResult:
    return OllamaChatResult(
        succeeded=succeeded,
        content=content,
        model=model,
        message=message,
    )


def make_service(
    retrieval_response: RetrievalResponse,
    chat_result: OllamaChatResult | None = None,
    *,
    minimum_score: float = 0.5,
    max_context_chars: int = 6000,
    system_prompt: str = DEFAULT_GROUNDED_SYSTEM_PROMPT,
) -> tuple[GroundedAnswerService, FakeRetrievalClient, FakeChatClient]:
    retrieval_client = FakeRetrievalClient(retrieval_response)
    chat_client = FakeChatClient(chat_result or make_chat_result())
    service = GroundedAnswerService(
        retrieval_client,
        chat_client,
        minimum_score=minimum_score,
        max_context_chars=max_context_chars,
        system_prompt=system_prompt,
    )
    return service, retrieval_client, chat_client


def test_rejects_non_finite_minimum_score() -> None:
    with pytest.raises(ValueError, match="finite"):
        make_service(make_retrieval_response(), minimum_score=float("nan"))


def test_rejects_context_budget_below_minimum() -> None:
    with pytest.raises(ValueError, match="at least 256"):
        make_service(make_retrieval_response(), max_context_chars=255)


def test_rejects_blank_system_prompt() -> None:
    with pytest.raises(ValueError, match="system prompt"):
        make_service(make_retrieval_response(), system_prompt="   ")


def test_rejects_blank_question() -> None:
    service, _, _ = make_service(make_retrieval_response())

    with pytest.raises(ValueError, match="question"):
        service.answer("   ")


def test_rejects_non_positive_limit() -> None:
    service, _, _ = make_service(make_retrieval_response())

    with pytest.raises(ValueError, match="limit"):
        service.answer("engine warning", limit=0)


def test_no_matches_abstains_without_generation() -> None:
    service, retrieval_client, chat_client = make_service(make_retrieval_response())

    result = service.answer("  engine warning  ", limit=3)

    assert result.succeeded
    assert result.abstained
    assert result.answer == ABSTENTION_ANSWER
    assert result.citations == ()
    assert result.top_score is None
    assert retrieval_client.calls == [("engine warning", 3, None)]
    assert chat_client.prompts == []


def test_below_threshold_abstains_and_preserves_top_score() -> None:
    service, _, chat_client = make_service(
        make_retrieval_response(make_match(score=0.49)),
        minimum_score=0.5,
    )

    result = service.answer("engine warning")

    assert result.abstained
    assert result.top_score == pytest.approx(0.49)
    assert chat_client.prompts == []


def test_exact_threshold_is_accepted() -> None:
    service, _, chat_client = make_service(
        make_retrieval_response(make_match(score=0.5)),
        minimum_score=0.5,
    )

    result = service.answer("engine warning")

    assert result.succeeded
    assert not result.abstained
    assert result.citations[0].score == pytest.approx(0.5)
    assert len(chat_client.prompts) == 1


def test_below_threshold_sources_are_excluded_from_prompt() -> None:
    service, _, chat_client = make_service(
        make_retrieval_response(
            make_match(ordinal=1, score=0.9, text="Relevant warning procedure."),
            make_match(ordinal=2, score=0.2, text="Irrelevant low-score source."),
        ),
        minimum_score=0.5,
    )

    service.answer("engine warning")

    assert "Relevant warning procedure." in chat_client.prompts[0]
    assert "Irrelevant low-score source." not in chat_client.prompts[0]
    assert "[S2]" not in chat_client.prompts[0]


def test_custom_system_prompt_is_forwarded() -> None:
    service, _, chat_client = make_service(
        make_retrieval_response(make_match()),
        system_prompt="Use only verified fleet evidence.",
    )

    service.answer("engine warning")

    assert chat_client.system_prompts == ["Use only verified fleet evidence."]


def test_chat_failure_returns_typed_failure() -> None:
    service, _, _ = make_service(
        make_retrieval_response(make_match()),
        make_chat_result(
            None,
            succeeded=False,
            model=None,
            message="The Ollama API is unreachable.",
        ),
    )

    result = service.answer("engine warning")

    assert not result.succeeded
    assert not result.abstained
    assert result.answer is None
    assert "unreachable" in result.message


def test_successful_chat_without_content_returns_failure() -> None:
    service, _, _ = make_service(
        make_retrieval_response(make_match()),
        make_chat_result(None),
    )

    result = service.answer("engine warning")

    assert not result.succeeded
    assert result.answer is None
    assert "no answer text" in result.message


def test_successful_chat_without_model_returns_failure() -> None:
    service, _, _ = make_service(
        make_retrieval_response(make_match()),
        make_chat_result(model=None),
    )

    result = service.answer("engine warning")

    assert not result.succeeded
    assert result.generation_model is None
    assert "no model name" in result.message


def test_model_insufficient_context_response_abstains() -> None:
    service, _, _ = make_service(
        make_retrieval_response(make_match()),
        make_chat_result("INSUFFICIENT_CONTEXT"),
    )

    result = service.answer("engine warning")

    assert result.succeeded
    assert result.abstained
    assert result.answer == ABSTENTION_ANSWER
    assert result.citations == ()
    assert result.generation_model == "llama3.2:3b"


def test_valid_citation_subset_is_returned() -> None:
    service, _, _ = make_service(
        make_retrieval_response(make_match(ordinal=1), make_match(ordinal=2)),
        make_chat_result("Use the second documented procedure [S2]."),
    )

    result = service.answer("engine warning")

    assert result.answer == "Use the second documented procedure [S2]."
    assert tuple(citation.label for citation in result.citations) == ("S2",)
    assert result.citations[0].chunk_id == "chunk-2"


def test_repeated_citation_labels_are_deduplicated() -> None:
    service, _, _ = make_service(
        make_retrieval_response(make_match()),
        make_chat_result("Stop the vehicle [S1], then inspect it [S1]."),
    )

    result = service.answer("engine warning")

    assert tuple(citation.label for citation in result.citations) == ("S1",)


def test_missing_citations_returns_extractive_fallback() -> None:
    match = make_match(text="Stop the vehicle before inspecting the engine warning.")
    service, _, _ = make_service(
        make_retrieval_response(match),
        make_chat_result("Stop the vehicle before inspection."),
    )

    result = service.answer("engine warning")

    assert result.succeeded
    assert not result.abstained
    assert result.answer == f"{match.text} [S1]"
    assert tuple(citation.label for citation in result.citations) == ("S1",)
    assert "extractive fallback" in result.message


def test_unknown_citation_returns_extractive_fallback() -> None:
    match = make_match()
    service, _, _ = make_service(
        make_retrieval_response(match),
        make_chat_result("Inspect the warning indicator [S99]."),
    )

    result = service.answer("engine warning")

    assert result.answer == f"{match.text} [S1]"
    assert tuple(citation.label for citation in result.citations) == ("S1",)


def test_context_budget_truncates_first_source() -> None:
    long_text = "inspection " * 200
    service, _, chat_client = make_service(
        make_retrieval_response(make_match(text=long_text)),
        max_context_chars=300,
    )

    result = service.answer("engine warning")

    assert len(result.citations[0].text) < len(long_text.strip())
    assert result.citations[0].text.endswith("...")
    assert "Content: " in chat_client.prompts[0]


def test_non_finite_retrieval_score_raises() -> None:
    service, _, _ = make_service(
        make_retrieval_response(make_match(score=float("inf"))),
    )

    with pytest.raises(RuntimeError, match="non-finite"):
        service.answer("engine warning")
