from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from fleetmind_rag.retrieval import (
    HybridRetrievalResponse,
    RerankedRetrievalResponse,
    RetrievalResponse,
    SparseRetrievalResponse,
)
from fleetmind_rag.routed_retrieval import (
    RoutedRetrievalExecutor,
    RoutedRetrievalRequest,
)
from fleetmind_rag.vector_store import ChunkMetadataFilter


@dataclass(slots=True)
class RecordingRetrievalService:
    """Small deterministic retrieval double that records every call."""

    calls: list[tuple[str, str, dict[str, object]]] = field(default_factory=list)

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        score_threshold: float | None = None,
        metadata_filter: ChunkMetadataFilter | None = None,
    ) -> RetrievalResponse:
        self.calls.append(
            (
                "dense",
                query,
                {
                    "limit": limit,
                    "score_threshold": score_threshold,
                    "metadata_filter": metadata_filter,
                },
            )
        )
        return RetrievalResponse(
            query=query,
            embedding_model="test-embedding",
            matches=(),
        )

    def search_sparse(
        self,
        query: str,
        *,
        limit: int = 5,
        metadata_filter: ChunkMetadataFilter | None = None,
    ) -> SparseRetrievalResponse:
        self.calls.append(
            (
                "sparse",
                query,
                {
                    "limit": limit,
                    "metadata_filter": metadata_filter,
                },
            )
        )
        return SparseRetrievalResponse(
            query=query,
            algorithm="bm25",
            matches=(),
        )

    def search_hybrid(
        self,
        query: str,
        *,
        limit: int = 5,
        candidate_limit: int = 20,
        score_threshold: float | None = None,
        metadata_filter: ChunkMetadataFilter | None = None,
    ) -> HybridRetrievalResponse:
        self.calls.append(
            (
                "hybrid",
                query,
                {
                    "limit": limit,
                    "candidate_limit": candidate_limit,
                    "score_threshold": score_threshold,
                    "metadata_filter": metadata_filter,
                },
            )
        )
        return HybridRetrievalResponse(
            query=query,
            algorithm="weighted-rrf",
            embedding_model="test-embedding",
            dense_match_count=0,
            sparse_match_count=0,
            matches=(),
        )

    def search_hybrid_reranked(
        self,
        query: str,
        *,
        limit: int = 5,
        candidate_limit: int = 20,
        score_threshold: float | None = None,
        metadata_filter: ChunkMetadataFilter | None = None,
    ) -> RerankedRetrievalResponse:
        self.calls.append(
            (
                "reranked",
                query,
                {
                    "limit": limit,
                    "candidate_limit": candidate_limit,
                    "score_threshold": score_threshold,
                    "metadata_filter": metadata_filter,
                },
            )
        )
        return RerankedRetrievalResponse(
            query=query,
            algorithm="transparent-reranking",
            embedding_model="test-embedding",
            dense_match_count=0,
            sparse_match_count=0,
            candidate_count=0,
            matches=(),
        )

    def only_call(self) -> tuple[str, str, dict[str, object]]:
        assert len(self.calls) == 1
        return self.calls[0]


@pytest.mark.parametrize(
    ("query", "expected_strategy", "response_type"),
    [
        (
            "What does overheating mean?",
            "dense",
            RetrievalResponse,
        ),
        (
            "error code P0420",
            "sparse",
            SparseRetrievalResponse,
        ),
        (
            "battery warning smoke smell",
            "hybrid",
            HybridRetrievalResponse,
        ),
        (
            (
                "If the battery warning is accompanied by smoke, "
                "may the driver continue the trip or must they stop safely?"
            ),
            "reranked",
            RerankedRetrievalResponse,
        ),
    ],
)
def test_executor_dispatches_to_selected_strategy(
    query: str,
    expected_strategy: str,
    response_type: type[object],
) -> None:
    service = RecordingRetrievalService()
    executor = RoutedRetrievalExecutor(service)

    result = executor.execute(RoutedRetrievalRequest(query=query))

    strategy, forwarded_query, _ = service.only_call()
    assert strategy == expected_strategy
    assert forwarded_query == result.decision.query
    assert result.decision.strategy == expected_strategy
    assert isinstance(result.response, response_type)


def test_executor_forwards_common_dense_controls() -> None:
    service = RecordingRetrievalService()
    executor = RoutedRetrievalExecutor(service)

    result = executor.execute(
        RoutedRetrievalRequest(
            query="  What   does overheating mean?  ",
            limit=3,
            candidate_limit=9,
            score_threshold=0.42,
        )
    )

    strategy, query, arguments = service.only_call()
    assert strategy == "dense"
    assert query == "What does overheating mean?"
    assert arguments == {
        "limit": 3,
        "score_threshold": 0.42,
        "metadata_filter": None,
    }
    assert result.response.query == result.decision.query


def test_executor_forwards_sparse_controls_without_dense_threshold() -> None:
    service = RecordingRetrievalService()
    executor = RoutedRetrievalExecutor(service)

    executor.execute(
        RoutedRetrievalRequest(
            query="error code P0420",
            limit=2,
            candidate_limit=7,
            score_threshold=0.91,
        )
    )

    strategy, _, arguments = service.only_call()
    assert strategy == "sparse"
    assert arguments == {
        "limit": 2,
        "metadata_filter": None,
    }


@pytest.mark.parametrize(
    ("query", "expected_strategy"),
    [
        ("battery warning smoke smell", "hybrid"),
        (
            (
                "If the battery warning is accompanied by smoke, "
                "may the driver continue the trip or must they stop safely?"
            ),
            "reranked",
        ),
    ],
)
def test_executor_forwards_candidate_controls(
    query: str,
    expected_strategy: str,
) -> None:
    service = RecordingRetrievalService()
    executor = RoutedRetrievalExecutor(service)

    executor.execute(
        RoutedRetrievalRequest(
            query=query,
            limit=4,
            candidate_limit=12,
            score_threshold=0.25,
        )
    )

    strategy, _, arguments = service.only_call()
    assert strategy == expected_strategy
    assert arguments == {
        "limit": 4,
        "candidate_limit": 12,
        "score_threshold": 0.25,
        "metadata_filter": None,
    }


def test_result_preserves_routing_evidence() -> None:
    service = RecordingRetrievalService()
    executor = RoutedRetrievalExecutor(service)

    result = executor.execute(RoutedRetrievalRequest(query="error code P0420"))

    assert result.decision.strategy == "sparse"
    assert result.decision.signals.exact_identifiers == ("P0420",)
    assert result.decision.selected_score > 0
    assert result.decision.reason


def test_result_reports_empty_match_count() -> None:
    service = RecordingRetrievalService()
    executor = RoutedRetrievalExecutor(service)

    result = executor.execute(
        RoutedRetrievalRequest(query="battery warning smoke smell")
    )

    assert result.match_count == 0


@pytest.mark.parametrize("limit", [0, -1])
def test_request_rejects_non_positive_limit(limit: int) -> None:
    with pytest.raises(
        ValueError,
        match="limit must be greater than zero",
    ):
        RoutedRetrievalRequest(
            query="What does overheating mean?",
            limit=limit,
        )


@pytest.mark.parametrize("candidate_limit", [0, -1])
def test_request_rejects_non_positive_candidate_limit(
    candidate_limit: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="candidate_limit must be greater than zero",
    ):
        RoutedRetrievalRequest(
            query="What does overheating mean?",
            candidate_limit=candidate_limit,
        )


@pytest.mark.parametrize(
    "query",
    [
        "battery warning smoke smell",
        (
            "If the battery warning is accompanied by smoke, "
            "may the driver continue the trip or must they stop safely?"
        ),
    ],
)
def test_executor_rejects_small_candidate_pool_for_candidate_routes(
    query: str,
) -> None:
    service = RecordingRetrievalService()
    executor = RoutedRetrievalExecutor(service)

    with pytest.raises(
        ValueError,
        match=(
            "candidate_limit must be greater than or equal to limit "
            "for hybrid and reranked retrieval"
        ),
    ):
        executor.execute(
            RoutedRetrievalRequest(
                query=query,
                limit=6,
                candidate_limit=5,
            )
        )

    assert service.calls == []


def test_dense_route_does_not_apply_candidate_pool_relationship() -> None:
    service = RecordingRetrievalService()
    executor = RoutedRetrievalExecutor(service)

    result = executor.execute(
        RoutedRetrievalRequest(
            query="What does overheating mean?",
            limit=6,
            candidate_limit=5,
        )
    )

    assert result.decision.strategy == "dense"
    assert service.only_call()[0] == "dense"


@pytest.mark.parametrize(
    "score_threshold",
    [None, -0.5, 0.0, 0.5, 1.0],
)
def test_request_preserves_score_threshold_for_retrieval_service(
    score_threshold: float | None,
) -> None:
    request = RoutedRetrievalRequest(
        query="What does overheating mean?",
        score_threshold=score_threshold,
    )

    assert request.score_threshold == score_threshold


def test_blank_query_is_rejected_by_router_before_retrieval() -> None:
    service = RecordingRetrievalService()
    executor = RoutedRetrievalExecutor(service)

    with pytest.raises(ValueError):
        executor.execute(RoutedRetrievalRequest(query="   "))

    assert service.calls == []


def test_executor_is_deterministic_across_repeated_calls() -> None:
    service = RecordingRetrievalService()
    executor = RoutedRetrievalExecutor(service)
    request = RoutedRetrievalRequest(query="error code P0420")

    first = executor.execute(request)
    second = executor.execute(request)

    assert first.decision == second.decision
    assert first.response == second.response
    assert [call[0] for call in service.calls] == ["sparse", "sparse"]
