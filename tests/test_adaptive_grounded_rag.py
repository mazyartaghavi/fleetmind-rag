from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import cast

import pytest

from fleetmind_rag.adaptive_grounded_rag import (
    AcceptedEvidenceRetrievalAdapter,
    AdaptiveGroundedAnswerResult,
    AdaptiveGroundedAnswerService,
)
from fleetmind_rag.adaptive_retrieval import AdaptiveRetrievalConfig
from fleetmind_rag.feedback_routing import (
    RoutingFeedbackHistory,
    RoutingFeedbackObservation,
    query_signal_profile,
)
from fleetmind_rag.ollama import OllamaChatResult
from fleetmind_rag.retrieval import (
    HybridRetrievalResponse,
    RerankedRetrievalResponse,
    RerankedSearchResult,
    RetrievalResponse,
    SparseRetrievalResponse,
)
from fleetmind_rag.routed_retrieval import RoutedRetrievalResult
from fleetmind_rag.routing import RetrievalStrategyRouter
from fleetmind_rag.vector_store import (
    ChunkMetadataFilter,
    VectorSearchResult,
)

_DENSE_QUERY = "What does overheating mean?"
_SPARSE_QUERY = "error code P0420"
_HYBRID_QUERY = "battery warning smoke smell"
_RERANKED_QUERY = (
    "If the battery warning is accompanied by smoke, "
    "may the driver continue or must they stop safely?"
)


def _vector_match(
    *,
    text: str = (
        "Overheating means excessive engine temperature. "
        "Stop safely when smoke accompanies a battery warning. "
        "Error code P0420 identifies a catalyst efficiency fault."
    ),
    score: float = 0.8,
) -> VectorSearchResult:
    return VectorSearchResult(
        chunk_id="chunk-1",
        document_id="document-1",
        section_id="section-1",
        section_title="Fleet safety",
        ordinal=0,
        text=text,
        word_count=len(text.split()),
        start_word=0,
        end_word=len(text.split()),
        score=score,
    )


def _reranked_match(
    *,
    text: str = (
        "If a battery warning is accompanied by smoke, "
        "the driver must stop safely and must not continue."
    ),
    score: float = 0.9,
) -> RerankedSearchResult:
    return RerankedSearchResult(
        chunk_id="chunk-reranked",
        document_id="document-1",
        section_id="section-safety",
        section_title="Battery warning safety",
        ordinal=1,
        text=text,
        word_count=len(text.split()),
        start_word=20,
        end_word=20 + len(text.split()),
        score=score,
        hybrid_score=0.03,
        original_rank=2,
        lexical_coverage=0.90,
        section_title_coverage=0.50,
        exact_phrase_match=False,
    )


def _routed_result(
    query: str,
    response: (
        RetrievalResponse
        | SparseRetrievalResponse
        | HybridRetrievalResponse
        | RerankedRetrievalResponse
    ),
) -> RoutedRetrievalResult:
    return RoutedRetrievalResult(
        decision=RetrievalStrategyRouter().route(query),
        response=response,
    )


@dataclass
class PlannedRetrievalService:
    """Return relevant or empty evidence from every retrieval strategy."""

    failures_before_success: int = 0
    score: float = 0.8
    calls: list[tuple[str, str, int, int | None]] = field(default_factory=list)

    def _matches(
        self,
        query: str,
    ) -> tuple[VectorSearchResult, ...]:
        call_number = len(self.calls)

        if call_number <= self.failures_before_success:
            return ()

        return (
            _vector_match(
                text=(
                    f"{query}. Overheating means excessive engine temperature. "
                    "A battery warning with smoke requires the driver to stop "
                    "safely and not continue. Error code P0420 identifies a "
                    "catalyst efficiency fault."
                ),
                score=self.score,
            ),
        )

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        score_threshold: float | None = None,
        metadata_filter: ChunkMetadataFilter | None = None,
    ) -> RetrievalResponse:
        del score_threshold, metadata_filter
        self.calls.append(("dense", query, limit, None))
        return RetrievalResponse(
            query=query,
            embedding_model="test-embedding",
            matches=self._matches(query),
        )

    def search_sparse(
        self,
        query: str,
        *,
        limit: int = 5,
        metadata_filter: ChunkMetadataFilter | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> SparseRetrievalResponse:
        del metadata_filter, k1, b
        self.calls.append(("sparse", query, limit, None))
        return SparseRetrievalResponse(
            query=query,
            algorithm="bm25",
            matches=self._matches(query),
        )

    def search_hybrid(
        self,
        query: str,
        *,
        limit: int = 5,
        candidate_limit: int = 20,
        score_threshold: float | None = None,
        metadata_filter: ChunkMetadataFilter | None = None,
        rrf_k: float = 60.0,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> HybridRetrievalResponse:
        del (
            score_threshold,
            metadata_filter,
            rrf_k,
            dense_weight,
            sparse_weight,
            k1,
            b,
        )
        self.calls.append(("hybrid", query, limit, candidate_limit))
        return HybridRetrievalResponse(
            query=query,
            algorithm="rrf",
            embedding_model="test-embedding",
            dense_match_count=1,
            sparse_match_count=1,
            matches=self._matches(query),
        )

    def search_hybrid_reranked(
        self,
        query: str,
        *,
        limit: int = 5,
        candidate_limit: int = 20,
        score_threshold: float | None = None,
        metadata_filter: ChunkMetadataFilter | None = None,
        rrf_k: float = 60.0,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        k1: float = 1.5,
        b: float = 0.75,
        hybrid_score_weight: float = 0.45,
        lexical_coverage_weight: float = 0.35,
        section_title_weight: float = 0.15,
        exact_phrase_weight: float = 0.05,
    ) -> RerankedRetrievalResponse:
        del (
            score_threshold,
            metadata_filter,
            rrf_k,
            dense_weight,
            sparse_weight,
            k1,
            b,
            hybrid_score_weight,
            lexical_coverage_weight,
            section_title_weight,
            exact_phrase_weight,
        )
        self.calls.append(("reranked", query, limit, candidate_limit))
        vector_matches = self._matches(query)
        matches = (
            ()
            if not vector_matches
            else (
                _reranked_match(
                    text=vector_matches[0].text,
                    score=self.score,
                ),
            )
        )
        return RerankedRetrievalResponse(
            query=query,
            algorithm="transparent-reranking",
            embedding_model="test-embedding",
            dense_match_count=len(matches),
            sparse_match_count=len(matches),
            candidate_count=len(matches),
            matches=matches,
        )

    def count(self) -> int:
        return 1


@dataclass
class RecordingChatClient:
    succeeded: bool = True
    content: str | None = "Grounded fleet answer [S1]"
    model: str | None = "test-chat"
    calls: list[tuple[str, str | None]] = field(default_factory=list)

    def chat(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> OllamaChatResult:
        self.calls.append((prompt, system_prompt))
        return cast(
            OllamaChatResult,
            SimpleNamespace(
                succeeded=self.succeeded,
                model=self.model,
                content=self.content,
                message="synthetic chat result",
            ),
        )


def test_dense_adapter_preserves_embedding_model() -> None:
    adapter = AcceptedEvidenceRetrievalAdapter(
        _routed_result(
            _DENSE_QUERY,
            RetrievalResponse(
                query=_DENSE_QUERY,
                embedding_model="nomic-embed-text",
                matches=(_vector_match(),),
            ),
        )
    )

    assert adapter.retrieval_model == "nomic-embed-text"


def test_sparse_adapter_uses_algorithm_as_provenance() -> None:
    adapter = AcceptedEvidenceRetrievalAdapter(
        _routed_result(
            _SPARSE_QUERY,
            SparseRetrievalResponse(
                query=_SPARSE_QUERY,
                algorithm="bm25",
                matches=(_vector_match(),),
            ),
        )
    )

    assert adapter.retrieval_model == "bm25"


def test_hybrid_adapter_combines_algorithm_and_model() -> None:
    adapter = AcceptedEvidenceRetrievalAdapter(
        _routed_result(
            _HYBRID_QUERY,
            HybridRetrievalResponse(
                query=_HYBRID_QUERY,
                algorithm="rrf",
                embedding_model="nomic-embed-text",
                dense_match_count=1,
                sparse_match_count=1,
                matches=(_vector_match(),),
            ),
        )
    )

    assert adapter.retrieval_model == "rrf (nomic-embed-text)"


def test_reranked_adapter_combines_algorithm_and_model() -> None:
    adapter = AcceptedEvidenceRetrievalAdapter(
        _routed_result(
            _RERANKED_QUERY,
            RerankedRetrievalResponse(
                query=_RERANKED_QUERY,
                algorithm="transparent-reranking",
                embedding_model="nomic-embed-text",
                dense_match_count=1,
                sparse_match_count=1,
                candidate_count=1,
                matches=(_reranked_match(),),
            ),
        )
    )

    assert adapter.retrieval_model == "transparent-reranking (nomic-embed-text)"


def test_adapter_limits_matches() -> None:
    result = _routed_result(
        _DENSE_QUERY,
        RetrievalResponse(
            query=_DENSE_QUERY,
            embedding_model="test-embedding",
            matches=(_vector_match(), _vector_match()),
        ),
    )

    assert (
        len(
            AcceptedEvidenceRetrievalAdapter(result).search("question", limit=1).matches
        )
        == 1
    )


def test_adapter_converts_reranked_match() -> None:
    result = _routed_result(
        _RERANKED_QUERY,
        RerankedRetrievalResponse(
            query=_RERANKED_QUERY,
            algorithm="transparent-reranking",
            embedding_model="test-embedding",
            dense_match_count=1,
            sparse_match_count=1,
            candidate_count=1,
            matches=(_reranked_match(),),
        ),
    )
    match = AcceptedEvidenceRetrievalAdapter(result).search("question").matches[0]

    assert isinstance(match, VectorSearchResult)
    assert match.chunk_id == "chunk-reranked"


def test_adapter_rejects_blank_query() -> None:
    result = _routed_result(
        _DENSE_QUERY,
        RetrievalResponse(
            query=_DENSE_QUERY,
            embedding_model="test",
            matches=(_vector_match(),),
        ),
    )

    with pytest.raises(ValueError, match="query must not be blank"):
        AcceptedEvidenceRetrievalAdapter(result).search(" ")


def test_adapter_rejects_non_positive_limit() -> None:
    result = _routed_result(
        _DENSE_QUERY,
        RetrievalResponse(
            query=_DENSE_QUERY,
            embedding_model="test",
            matches=(_vector_match(),),
        ),
    )

    with pytest.raises(ValueError, match="limit must be greater than zero"):
        AcceptedEvidenceRetrievalAdapter(result).search("question", limit=0)


def test_adapter_rejects_cross_strategy_threshold() -> None:
    result = _routed_result(
        _DENSE_QUERY,
        RetrievalResponse(
            query=_DENSE_QUERY,
            embedding_model="test",
            matches=(_vector_match(),),
        ),
    )

    with pytest.raises(ValueError, match="does not support a score threshold"):
        AcceptedEvidenceRetrievalAdapter(result).search(
            "question",
            score_threshold=0.5,
        )


def test_service_rejects_small_context_budget() -> None:
    with pytest.raises(ValueError, match="at least 256"):
        AdaptiveGroundedAnswerService(
            PlannedRetrievalService(),
            RecordingChatClient(),
            max_context_chars=255,
        )


def test_service_rejects_blank_system_prompt() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        AdaptiveGroundedAnswerService(
            PlannedRetrievalService(),
            RecordingChatClient(),
            system_prompt=" ",
        )


def test_service_rejects_blank_question() -> None:
    service = AdaptiveGroundedAnswerService(
        PlannedRetrievalService(),
        RecordingChatClient(),
    )

    with pytest.raises(ValueError, match="question must not be blank"):
        service.answer(" ")


@pytest.mark.parametrize(
    ("query", "expected_strategy"),
    [
        (_DENSE_QUERY, "dense"),
        (_SPARSE_QUERY, "sparse"),
        (_HYBRID_QUERY, "hybrid"),
        (_RERANKED_QUERY, "reranked"),
    ],
)
def test_service_generates_after_each_accepted_strategy(
    query: str,
    expected_strategy: str,
) -> None:
    retrieval_service = PlannedRetrievalService()
    chat_client = RecordingChatClient()
    service = AdaptiveGroundedAnswerService(
        retrieval_service,
        chat_client,
    )

    result = service.answer(query)

    assert result.succeeded is True
    assert result.abstained is False
    assert result.answer is not None
    assert "[S1]" in result.answer
    assert result.grounded_answer.citations
    assert result.retrieval_outcome.succeeded is True
    assert result.retrieval_outcome.final_result is not None
    assert result.retrieval_outcome.final_result.decision.strategy == expected_strategy
    assert retrieval_service.calls[0][0] == expected_strategy
    assert len(chat_client.calls) == 1


def test_service_preserves_original_question_after_rewrite() -> None:
    service = AdaptiveGroundedAnswerService(
        PlannedRetrievalService(failures_before_success=1),
        RecordingChatClient(),
    )

    result = service.answer(
        _DENSE_QUERY,
        config=AdaptiveRetrievalConfig(max_attempts=2),
    )

    assert result.question == _DENSE_QUERY
    assert result.grounded_answer.question == _DENSE_QUERY
    assert result.attempt_count == 2
    assert len(result.retrieval_outcome.rewrites) == 1


def test_service_abstains_when_attempt_budget_is_exhausted() -> None:
    chat_client = RecordingChatClient()
    service = AdaptiveGroundedAnswerService(
        PlannedRetrievalService(failures_before_success=10),
        chat_client,
    )

    result = service.answer(
        _DENSE_QUERY,
        config=AdaptiveRetrievalConfig(max_attempts=2),
    )

    assert result.succeeded is True
    assert result.abstained is True
    assert result.retrieval_outcome.succeeded is False
    assert result.attempt_count == 2
    assert "generation was skipped" in result.grounded_answer.message
    assert chat_client.calls == []


def test_failed_retrieval_records_every_attempt_as_feedback() -> None:
    service = AdaptiveGroundedAnswerService(
        PlannedRetrievalService(failures_before_success=10),
        RecordingChatClient(),
    )

    result = service.answer(
        _DENSE_QUERY,
        config=AdaptiveRetrievalConfig(max_attempts=3),
    )

    assert len(result.feedback_history.observations) == 3
    assert len(service.history.observations) == 3


def test_service_history_grows_across_calls() -> None:
    service = AdaptiveGroundedAnswerService(
        PlannedRetrievalService(),
        RecordingChatClient(),
    )

    first = service.answer(_DENSE_QUERY)
    second = service.answer(_DENSE_QUERY)

    assert len(first.feedback_history.observations) == 1
    assert len(second.feedback_history.observations) == 2
    assert service.history == second.feedback_history


def test_result_properties_delegate_to_grounded_and_retrieval_results() -> None:
    result = AdaptiveGroundedAnswerService(
        PlannedRetrievalService(),
        RecordingChatClient(),
    ).answer(_DENSE_QUERY)

    assert isinstance(result, AdaptiveGroundedAnswerResult)
    assert result.succeeded == result.grounded_answer.succeeded
    assert result.abstained == result.grounded_answer.abstained
    assert result.answer == result.grounded_answer.answer
    assert result.attempt_count == result.retrieval_outcome.attempt_count


def test_result_exposes_initial_feedback_routing() -> None:
    result = AdaptiveGroundedAnswerService(
        PlannedRetrievalService(),
        RecordingChatClient(),
    ).answer(_DENSE_QUERY)

    assert result.initial_routing.base_decision.strategy == "dense"
    assert result.initial_routing.decision.strategy == "dense"


def test_seeded_feedback_changes_strategy_used_by_executor() -> None:
    base_decision = RetrievalStrategyRouter().route(_DENSE_QUERY)
    features = query_signal_profile(base_decision.signals)
    observations = tuple(
        [
            *(
                RoutingFeedbackObservation(
                    query=_DENSE_QUERY,
                    strategy="dense",
                    verdict="rewrite",
                    quality_score=0.0,
                    attempt_number=index,
                    features=features,
                )
                for index in range(1, 4)
            ),
            *(
                RoutingFeedbackObservation(
                    query=_DENSE_QUERY,
                    strategy="hybrid",
                    verdict="accept",
                    quality_score=1.0,
                    attempt_number=index,
                    features=features,
                )
                for index in range(1, 4)
            ),
        ]
    )
    retrieval_service = PlannedRetrievalService()
    service = AdaptiveGroundedAnswerService(
        retrieval_service,
        RecordingChatClient(),
        history=RoutingFeedbackHistory(observations),
    )

    result = service.answer(_DENSE_QUERY)

    assert result.initial_routing.strategy == "hybrid"
    assert retrieval_service.calls[0][0] == "hybrid"


def test_chat_failure_is_preserved_in_grounded_result() -> None:
    service = AdaptiveGroundedAnswerService(
        PlannedRetrievalService(),
        RecordingChatClient(
            succeeded=False,
            content=None,
            model=None,
        ),
    )

    result = service.answer(_DENSE_QUERY)

    assert result.succeeded is False
    assert result.abstained is False


def test_negative_but_finite_accepted_score_reaches_generation() -> None:
    chat_client = RecordingChatClient()
    service = AdaptiveGroundedAnswerService(
        PlannedRetrievalService(score=-0.1),
        chat_client,
    )

    result = service.answer(_DENSE_QUERY)

    assert result.retrieval_outcome.succeeded is True
    assert result.abstained is False
    assert len(chat_client.calls) == 1


def test_retrieval_config_forwards_limits() -> None:
    retrieval_service = PlannedRetrievalService()
    service = AdaptiveGroundedAnswerService(
        retrieval_service,
        RecordingChatClient(),
    )

    service.answer(
        _HYBRID_QUERY,
        config=AdaptiveRetrievalConfig(
            limit=2,
            candidate_limit=7,
        ),
    )

    assert retrieval_service.calls[0] == (
        "hybrid",
        _HYBRID_QUERY,
        2,
        7,
    )
