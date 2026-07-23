from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from langgraph.graph.state import CompiledStateGraph

from fleetmind_rag.adaptive_retrieval import (
    AdaptiveRetrievalConfig,
    RoutedRetrievalExecutorLike,
)
from fleetmind_rag.langgraph_workflow import (
    LangGraphAdaptiveRetrievalWorkflow,
)
from fleetmind_rag.retrieval import (
    HybridRetrievalResponse,
    RerankedRetrievalResponse,
    RerankedSearchResult,
    RetrievalResponse,
    SparseRetrievalResponse,
)
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


def vector_match(text: str) -> VectorSearchResult:
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
        score=0.8,
    )


def reranked_match(text: str) -> RerankedSearchResult:
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
    evidence: str | None,
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


@dataclass(slots=True)
class PlannedExecutor:
    evidence_plan: tuple[str | None, ...]
    requests: list[RoutedRetrievalRequest] = field(default_factory=list)

    def execute(
        self,
        request: RoutedRetrievalRequest,
    ) -> RoutedRetrievalResult:
        index = len(self.requests)
        self.requests.append(request)
        return make_result(
            request.query,
            evidence=self.evidence_plan[index],
        )


def accept_executor_protocol(
    executor: RoutedRetrievalExecutorLike,
) -> None:
    """Statically verify the test executor's structural interface."""


def test_planned_executor_satisfies_protocol() -> None:
    accept_executor_protocol(PlannedExecutor((None,)))


def test_workflow_exposes_compiled_langgraph() -> None:
    workflow = LangGraphAdaptiveRetrievalWorkflow(PlannedExecutor((None,)))

    assert isinstance(workflow.graph, CompiledStateGraph)
    assert workflow.graph.name == "fleetmind-adaptive-retrieval"


def test_compiled_graph_contains_expected_nodes() -> None:
    workflow = LangGraphAdaptiveRetrievalWorkflow(PlannedExecutor((None,)))
    node_names = set(workflow.graph.get_graph().nodes)

    assert {
        "__start__",
        "retrieve",
        "assess",
        "rewrite",
        "complete",
        "fail",
        "__end__",
    } <= node_names


def test_mermaid_topology_contains_graph_routes() -> None:
    workflow = LangGraphAdaptiveRetrievalWorkflow(PlannedExecutor((None,)))
    mermaid = workflow.draw_mermaid()

    assert "retrieve" in mermaid
    assert "assess" in mermaid
    assert "rewrite" in mermaid
    assert "complete" in mermaid
    assert "fail" in mermaid


def test_graph_accepts_good_first_attempt() -> None:
    executor = PlannedExecutor(("Overheating means excessive engine temperature.",))
    outcome = LangGraphAdaptiveRetrievalWorkflow(executor).run(
        "What does overheating mean?"
    )

    assert outcome.succeeded
    assert outcome.state.status == "completed"
    assert outcome.attempt_count == 1
    assert len(outcome.assessments) == 1
    assert outcome.assessments[0].should_accept
    assert outcome.rewrites == ()
    assert len(executor.requests) == 1


def test_graph_rewrites_then_accepts() -> None:
    executor = PlannedExecutor(
        (
            "Tire pressure guidance.",
            "Overheating means excessive engine temperature.",
        )
    )
    outcome = LangGraphAdaptiveRetrievalWorkflow(executor).run(
        "What does overheating mean?"
    )

    assert outcome.succeeded
    assert outcome.attempt_count == 2
    assert [item.verdict for item in outcome.assessments] == [
        "rewrite",
        "accept",
    ]
    assert len(outcome.rewrites) == 1
    assert outcome.rewrites[0].rewritten_query == "Explain overheating."
    assert executor.requests[1].query == "Explain overheating."


def test_graph_fails_when_attempts_are_exhausted() -> None:
    executor = PlannedExecutor(
        (
            "Tire pressure guidance.",
            "Battery charging guidance.",
        )
    )
    outcome = LangGraphAdaptiveRetrievalWorkflow(executor).run(
        "What does overheating mean?",
        config=AdaptiveRetrievalConfig(max_attempts=2),
    )

    assert not outcome.succeeded
    assert outcome.state.status == "failed"
    assert outcome.attempt_count == 2
    assert len(outcome.assessments) == 2
    assert len(outcome.rewrites) == 1
    assert outcome.state.termination_reason is not None
    assert "after 2 attempts" in outcome.state.termination_reason


def test_graph_preserves_domain_transition_history() -> None:
    executor = PlannedExecutor(
        (
            "Tire pressure guidance.",
            "Overheating means excessive engine temperature.",
        )
    )
    outcome = LangGraphAdaptiveRetrievalWorkflow(executor).run(
        "What does overheating mean?"
    )

    assert [transition.action for transition in outcome.state.transitions] == [
        "begin_retrieval",
        "record_retrieval",
        "request_rewrite",
        "apply_rewrite",
        "begin_retrieval",
        "record_retrieval",
        "complete",
    ]


def test_graph_forwards_retrieval_configuration() -> None:
    metadata_filter = ChunkMetadataFilter(document_ids=("document-1",))
    config = AdaptiveRetrievalConfig(
        max_attempts=1,
        limit=3,
        candidate_limit=9,
        score_threshold=0.42,
        metadata_filter=metadata_filter,
    )
    executor = PlannedExecutor(("Overheating means excessive engine temperature.",))

    LangGraphAdaptiveRetrievalWorkflow(executor).run(
        "What does overheating mean?",
        config=config,
    )

    request = executor.requests[0]
    assert request.limit == 3
    assert request.candidate_limit == 9
    assert request.score_threshold == 0.42
    assert request.metadata_filter is metadata_filter


def test_graph_supports_attempt_budget_above_default_recursion_limit() -> None:
    attempts = 9
    executor = PlannedExecutor(tuple(None for _ in range(attempts)))
    outcome = LangGraphAdaptiveRetrievalWorkflow(executor).run(
        "What does overheating mean?",
        config=AdaptiveRetrievalConfig(max_attempts=attempts),
    )

    assert outcome.state.status == "failed"
    assert outcome.attempt_count == attempts
    assert len(executor.requests) == attempts


def test_graph_run_is_deterministic() -> None:
    def run_once() -> object:
        executor = PlannedExecutor(
            (
                "Tire pressure guidance.",
                "Overheating means excessive engine temperature.",
            )
        )
        return LangGraphAdaptiveRetrievalWorkflow(executor).run(
            "What does overheating mean?"
        )

    assert run_once() == run_once()


def test_blank_query_is_rejected_before_graph_execution() -> None:
    executor = PlannedExecutor((None,))
    workflow = LangGraphAdaptiveRetrievalWorkflow(executor)

    with pytest.raises(ValueError, match="query must not be blank"):
        workflow.run("   ")

    assert executor.requests == []
