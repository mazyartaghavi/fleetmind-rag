from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from fleetmind_rag.retrieval import (
    HybridRetrievalResponse,
    RerankedRetrievalResponse,
    RetrievalResponse,
    SparseRetrievalResponse,
)
from fleetmind_rag.routing import (
    RetrievalStrategyRouter,
    RoutingDecision,
)
from fleetmind_rag.vector_store import ChunkMetadataFilter

RetrievalExecutionResponse = (
    RetrievalResponse
    | SparseRetrievalResponse
    | HybridRetrievalResponse
    | RerankedRetrievalResponse
)


class RetrievalService(Protocol):
    """Retrieval operations required by the routed executor."""

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        score_threshold: float | None = None,
        metadata_filter: ChunkMetadataFilter | None = None,
    ) -> RetrievalResponse:
        """Run dense vector retrieval."""

    def search_sparse(
        self,
        query: str,
        *,
        limit: int = 5,
        metadata_filter: ChunkMetadataFilter | None = None,
    ) -> SparseRetrievalResponse:
        """Run sparse lexical retrieval."""

    def search_hybrid(
        self,
        query: str,
        *,
        limit: int = 5,
        candidate_limit: int = 20,
        score_threshold: float | None = None,
        metadata_filter: ChunkMetadataFilter | None = None,
    ) -> HybridRetrievalResponse:
        """Run hybrid reciprocal-rank-fusion retrieval."""

    def search_hybrid_reranked(
        self,
        query: str,
        *,
        limit: int = 5,
        candidate_limit: int = 20,
        score_threshold: float | None = None,
        metadata_filter: ChunkMetadataFilter | None = None,
    ) -> RerankedRetrievalResponse:
        """Run hybrid retrieval followed by transparent reranking."""


@dataclass(frozen=True, slots=True)
class RoutedRetrievalRequest:
    """Validated controls for one routed retrieval operation."""

    query: str
    limit: int = 5
    candidate_limit: int = 20
    score_threshold: float | None = None
    metadata_filter: ChunkMetadataFilter | None = None

    def __post_init__(self) -> None:
        """Reject invalid retrieval limits and score thresholds."""

        if self.limit <= 0:
            raise ValueError("limit must be greater than zero")

        if self.candidate_limit <= 0:
            raise ValueError("candidate_limit must be greater than zero")


@dataclass(frozen=True, slots=True)
class RoutedRetrievalResult:
    """Auditable routing decision paired with its retrieval response."""

    decision: RoutingDecision
    response: RetrievalExecutionResponse

    @property
    def match_count(self) -> int:
        """Return the number of matches produced by the selected strategy."""

        return len(self.response.matches)


class RoutedRetrievalExecutor:
    """Select and execute one retrieval strategy for a query."""

    def __init__(
        self,
        retrieval_service: RetrievalService,
        *,
        router: RetrievalStrategyRouter | None = None,
    ) -> None:
        """Initialize the executor with retrieval and routing dependencies."""

        self._retrieval_service = retrieval_service
        self._router = router or RetrievalStrategyRouter()

    def execute(
        self,
        request: RoutedRetrievalRequest,
    ) -> RoutedRetrievalResult:
        """Route the query and execute exactly one retrieval operation."""

        decision = self._router.route(request.query)
        query = decision.query

        if decision.strategy == "dense":
            response: RetrievalExecutionResponse = self._retrieval_service.search(
                query,
                limit=request.limit,
                score_threshold=request.score_threshold,
                metadata_filter=request.metadata_filter,
            )
        elif decision.strategy == "sparse":
            response = self._retrieval_service.search_sparse(
                query,
                limit=request.limit,
                metadata_filter=request.metadata_filter,
            )
        elif decision.strategy == "hybrid":
            self._validate_candidate_limit(request)
            response = self._retrieval_service.search_hybrid(
                query,
                limit=request.limit,
                candidate_limit=request.candidate_limit,
                score_threshold=request.score_threshold,
                metadata_filter=request.metadata_filter,
            )
        elif decision.strategy == "reranked":
            self._validate_candidate_limit(request)
            response = self._retrieval_service.search_hybrid_reranked(
                query,
                limit=request.limit,
                candidate_limit=request.candidate_limit,
                score_threshold=request.score_threshold,
                metadata_filter=request.metadata_filter,
            )
        else:
            raise RuntimeError(f"Unsupported retrieval strategy: {decision.strategy}")

        return RoutedRetrievalResult(
            decision=decision,
            response=response,
        )

    @staticmethod
    def _validate_candidate_limit(
        request: RoutedRetrievalRequest,
    ) -> None:
        if request.candidate_limit < request.limit:
            raise ValueError(
                "candidate_limit must be greater than or equal to limit "
                "for hybrid and reranked retrieval"
            )
