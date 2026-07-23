from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from fleetmind_rag.adaptive_retrieval import (
    AdaptiveRetrievalAgent,
    AdaptiveRetrievalConfig,
    DeterministicQueryRewriter,
    RoutedRetrievalExecutorLike,
)
from fleetmind_rag.retrieval import (
    HybridRetrievalResponse,
    RerankedRetrievalResponse,
    RerankedSearchResult,
    RetrievalResponse,
    SparseRetrievalResponse,
)
from fleetmind_rag.retrieval_quality import RetrievalQualityChecker
from fleetmind_rag.routed_retrieval import (
    RetrievalExecutionResponse,
    RoutedRetrievalRequest,
    RoutedRetrievalResult,
)
from fleetmind_rag.routing import RetrievalStrategyRouter
from fleetmind_rag.vector_store import (
    ChunkMetadataFilter,
    VectorSearchResult,
)


def vector_match(
    text: str,
    *,
    score: float = 0.8,
) -> VectorSearchResult:
    words = text.split()
    return VectorSearchResult(
        chunk_id="chunk-1",
        document_id="document-1",
        section_id="section-1",
        section_title="Fleet guidance",
        ordinal=0,
        text=text,
        word_count=len(words),
        start_word=0,
        end_word=len(words),
        score=score,
    )


def reranked_match(
    text: str,
) -> RerankedSearchResult:
    words = text.split()
    return RerankedSearchResult(
        chunk_id="chunk-1",
        document_id="document-1",
        section_id="section-1",
        section_title="Safety procedure",
        ordinal=0,
        text=text,
        word_count=len(words),
        start_word=0,
        end_word=len(words),
        score=0.9,
        hybrid_score=0.03,
        original_rank=1,
        lexical_coverage=0.8,
        section_title_coverage=0.5,
        exact_phrase_match=False,
    )


def make_result(
    query: str,
    *,
    evidence: str | None = None,
) -> RoutedRetrievalResult:
    decision = RetrievalStrategyRouter().route(query)
    response: RetrievalExecutionResponse

    if decision.strategy == "reranked":
        reranked_matches = () if evidence is None else (reranked_match(evidence),)
        response = RerankedRetrievalResponse(
            query=decision.query,
            algorithm="transparent-reranking",
            embedding_model="test-embedding",
            dense_match_count=len(reranked_matches),
            sparse_match_count=len(reranked_matches),
            candidate_count=len(reranked_matches),
            matches=reranked_matches,
        )
    else:
        vector_matches = () if evidence is None else (vector_match(evidence),)

        if decision.strategy == "dense":
            response = RetrievalResponse(
                query=decision.query,
                embedding_model="test-embedding",
                matches=vector_matches,
            )
        elif decision.strategy == "sparse":
            response = SparseRetrievalResponse(
                query=decision.query,
                algorithm="bm25",
                matches=vector_matches,
            )
        else:
            response = HybridRetrievalResponse(
                query=decision.query,
                algorithm="weighted-rrf",
                embedding_model="test-embedding",
                dense_match_count=len(vector_matches),
                sparse_match_count=len(vector_matches),
                matches=vector_matches,
            )

    return RoutedRetrievalResult(
        decision=decision,
        response=response,
    )


def rewrite_for(
    query: str,
) -> str:
    result = make_result(query)
    assessment = RetrievalQualityChecker().assess(result)
    rewrite = DeterministicQueryRewriter().rewrite(
        result,
        assessment,
        after_attempt=1,
    )
    return rewrite.rewritten_query


@dataclass(slots=True)
class PlannedExecutor:
    """Return evidence plans in order while recording every request."""

    evidence_plan: tuple[str | None, ...]
    requests: list[RoutedRetrievalRequest] = field(default_factory=list)

    def execute(
        self,
        request: RoutedRetrievalRequest,
    ) -> RoutedRetrievalResult:
        index = len(self.requests)
        self.requests.append(request)
        evidence = self.evidence_plan[index]
        return make_result(request.query, evidence=evidence)


def assert_executor_protocol(
    executor: RoutedRetrievalExecutorLike,
) -> None:
    """Statically verify that the planned executor satisfies the protocol."""


def test_planned_executor_satisfies_executor_protocol() -> None:
    assert_executor_protocol(PlannedExecutor((None,)))


def test_dense_rewrite_simplifies_conceptual_query() -> None:
    assert rewrite_for("What does overheating mean?") == "Explain overheating."


def test_sparse_rewrite_preserves_exact_identifier() -> None:
    result = make_result("error code P0420")
    assessment = RetrievalQualityChecker().assess(result)
    rewrite = DeterministicQueryRewriter().rewrite(
        result,
        assessment,
        after_attempt=1,
    )

    assert rewrite.rewritten_query == "P0420 error code"
    assert rewrite.preserved_identifiers == ("P0420",)
    assert "P0420" in rewrite.rewritten_query


def test_sparse_rewrite_preserves_quoted_phrase() -> None:
    query = 'Find the exact phrase "sidewall bulge".'
    result = make_result(query)
    assessment = RetrievalQualityChecker().assess(result)
    rewrite = DeterministicQueryRewriter().rewrite(
        result,
        assessment,
        after_attempt=1,
    )

    assert '"sidewall bulge"' in rewrite.rewritten_query
    assert rewrite.preserved_quoted_phrases == ("sidewall bulge",)


def test_hybrid_rewrite_adds_connector_without_losing_terms() -> None:
    rewritten = rewrite_for("battery warning smoke smell")

    assert rewritten == "battery warning smoke and smell"
    assert {"battery", "warning", "smoke", "smell"} <= set(rewritten.split())


def test_reranked_rewrite_simplifies_conditional_wording() -> None:
    query = (
        "If the battery warning is accompanied by smoke, "
        "may the driver continue the trip or must they stop safely?"
    )
    rewritten = rewrite_for(query)

    assert "has smoke" in rewritten
    assert "can the driver" in rewritten
    assert "must the driver" in rewritten


def test_rewrite_records_failed_assessment_reasons() -> None:
    result = make_result("What does overheating mean?")
    assessment = RetrievalQualityChecker().assess(result)
    rewrite = DeterministicQueryRewriter().rewrite(
        result,
        assessment,
        after_attempt=2,
    )

    assert rewrite.source_query == result.decision.query
    assert rewrite.strategy == "dense"
    assert rewrite.after_attempt == 2
    assert rewrite.reasons == assessment.reasons


def test_rewriter_rejects_accept_assessment() -> None:
    result = make_result(
        "What does overheating mean?",
        evidence="Overheating means excessive engine temperature.",
    )
    assessment = RetrievalQualityChecker().assess(result)

    assert assessment.verdict == "accept"

    with pytest.raises(
        ValueError,
        match="requires a rewrite quality verdict",
    ):
        DeterministicQueryRewriter().rewrite(
            result,
            assessment,
            after_attempt=1,
        )


def test_rewriter_rejects_non_positive_attempt_number() -> None:
    result = make_result("What does overheating mean?")
    assessment = RetrievalQualityChecker().assess(result)

    with pytest.raises(
        ValueError,
        match="after_attempt must be greater than zero",
    ):
        DeterministicQueryRewriter().rewrite(
            result,
            assessment,
            after_attempt=0,
        )


def test_rewriter_is_deterministic() -> None:
    result = make_result("battery warning smoke smell")
    assessment = RetrievalQualityChecker().assess(result)
    rewriter = DeterministicQueryRewriter()

    first = rewriter.rewrite(
        result,
        assessment,
        after_attempt=1,
    )
    second = rewriter.rewrite(
        result,
        assessment,
        after_attempt=1,
    )

    assert first == second


@pytest.mark.parametrize("max_attempts", [0, -1])
def test_config_rejects_non_positive_max_attempts(
    max_attempts: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="max_attempts must be greater than zero",
    ):
        AdaptiveRetrievalConfig(max_attempts=max_attempts)


@pytest.mark.parametrize("limit", [0, -1])
def test_config_rejects_non_positive_limit(limit: int) -> None:
    with pytest.raises(
        ValueError,
        match="limit must be greater than zero",
    ):
        AdaptiveRetrievalConfig(limit=limit)


@pytest.mark.parametrize("candidate_limit", [0, -1])
def test_config_rejects_non_positive_candidate_limit(
    candidate_limit: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="candidate_limit must be greater than zero",
    ):
        AdaptiveRetrievalConfig(candidate_limit=candidate_limit)


def test_config_rejects_candidate_limit_below_limit() -> None:
    with pytest.raises(
        ValueError,
        match="candidate_limit must be greater than or equal to limit",
    ):
        AdaptiveRetrievalConfig(limit=6, candidate_limit=5)


def test_agent_accepts_good_first_attempt() -> None:
    executor = PlannedExecutor(("Overheating means excessive engine temperature.",))
    outcome = AdaptiveRetrievalAgent(executor).run("What does overheating mean?")

    assert outcome.succeeded
    assert outcome.state.status == "completed"
    assert outcome.attempt_count == 1
    assert len(outcome.assessments) == 1
    assert outcome.assessments[0].should_accept
    assert outcome.rewrites == ()
    assert outcome.final_result is outcome.state.final_result
    assert len(executor.requests) == 1


def test_agent_rewrites_then_accepts_second_attempt() -> None:
    executor = PlannedExecutor(
        (
            "Tire pressure should be checked before each shift.",
            "Overheating means excessive engine temperature.",
        )
    )
    outcome = AdaptiveRetrievalAgent(executor).run("What does overheating mean?")

    assert outcome.succeeded
    assert outcome.attempt_count == 2
    assert [item.verdict for item in outcome.assessments] == [
        "rewrite",
        "accept",
    ]
    assert len(outcome.rewrites) == 1
    assert outcome.rewrites[0].rewritten_query == "Explain overheating."
    assert executor.requests[1].query == "Explain overheating."
    assert [attempt.number for attempt in outcome.state.attempts] == [
        1,
        2,
    ]


def test_agent_fails_after_exhausting_attempt_budget() -> None:
    executor = PlannedExecutor(
        (
            "Tire pressure guidance.",
            "Battery charging guidance.",
        )
    )
    outcome = AdaptiveRetrievalAgent(executor).run(
        "What does overheating mean?",
        config=AdaptiveRetrievalConfig(max_attempts=2),
    )

    assert not outcome.succeeded
    assert outcome.state.status == "failed"
    assert outcome.attempt_count == 2
    assert len(outcome.assessments) == 2
    assert len(outcome.rewrites) == 1
    assert outcome.final_result is None
    assert outcome.state.termination_reason is not None
    assert "after 2 attempts" in outcome.state.termination_reason
    assert len(executor.requests) == 2


def test_agent_never_exceeds_configured_attempt_budget() -> None:
    executor = PlannedExecutor((None, None, None))
    outcome = AdaptiveRetrievalAgent(executor).run(
        "What does overheating mean?",
        config=AdaptiveRetrievalConfig(max_attempts=3),
    )

    assert outcome.attempt_count == 3
    assert len(executor.requests) == 3
    assert len(outcome.rewrites) == 2


def test_agent_forwards_retrieval_configuration() -> None:
    metadata_filter = ChunkMetadataFilter(document_ids=("document-1",))
    config = AdaptiveRetrievalConfig(
        max_attempts=1,
        limit=3,
        candidate_limit=9,
        score_threshold=0.42,
        metadata_filter=metadata_filter,
    )
    executor = PlannedExecutor(("Overheating means excessive engine temperature.",))

    AdaptiveRetrievalAgent(executor).run(
        "What does overheating mean?",
        config=config,
    )

    request = executor.requests[0]
    assert request.limit == 3
    assert request.candidate_limit == 9
    assert request.score_threshold == 0.42
    assert request.metadata_filter is metadata_filter


def test_agent_transition_history_is_complete() -> None:
    executor = PlannedExecutor(
        (
            "Tire pressure guidance.",
            "Overheating means excessive engine temperature.",
        )
    )
    outcome = AdaptiveRetrievalAgent(executor).run("What does overheating mean?")

    assert [transition.action for transition in outcome.state.transitions] == [
        "begin_retrieval",
        "record_retrieval",
        "request_rewrite",
        "apply_rewrite",
        "begin_retrieval",
        "record_retrieval",
        "complete",
    ]


def test_agent_run_is_deterministic() -> None:
    def run_once() -> object:
        executor = PlannedExecutor(
            (
                "Tire pressure guidance.",
                "Overheating means excessive engine temperature.",
            )
        )
        return AdaptiveRetrievalAgent(executor).run("What does overheating mean?")

    assert run_once() == run_once()


def test_blank_query_is_rejected_before_executor_call() -> None:
    executor = PlannedExecutor((None,))

    with pytest.raises(ValueError, match="query must not be blank"):
        AdaptiveRetrievalAgent(executor).run("   ")

    assert executor.requests == []
