from __future__ import annotations

import math

import pytest

from fleetmind_rag.retrieval import (
    HybridRetrievalResponse,
    RerankedRetrievalResponse,
    RerankedSearchResult,
    RetrievalResponse,
    SparseRetrievalResponse,
)
from fleetmind_rag.retrieval_quality import (
    RetrievalQualityChecker,
    RetrievalQualityPolicy,
)
from fleetmind_rag.routed_retrieval import (
    RetrievalExecutionResponse,
    RoutedRetrievalResult,
)
from fleetmind_rag.routing import RetrievalStrategyRouter
from fleetmind_rag.vector_store import VectorSearchResult


def vector_match(
    text: str,
    *,
    score: float = 0.8,
    chunk_id: str = "chunk-1",
    section_title: str = "Fleet guidance",
) -> VectorSearchResult:
    words = text.split()
    return VectorSearchResult(
        chunk_id=chunk_id,
        document_id="document-1",
        section_id="section-1",
        section_title=section_title,
        ordinal=0,
        text=text,
        word_count=len(words),
        start_word=0,
        end_word=len(words),
        score=score,
    )


def reranked_match(
    text: str,
    *,
    score: float = 0.9,
    lexical_coverage: float = 0.6,
) -> RerankedSearchResult:
    words = text.split()
    return RerankedSearchResult(
        chunk_id="chunk-1",
        document_id="document-1",
        section_id="section-1",
        section_title="Battery warning",
        ordinal=0,
        text=text,
        word_count=len(words),
        start_word=0,
        end_word=len(words),
        score=score,
        hybrid_score=0.03,
        original_rank=1,
        lexical_coverage=lexical_coverage,
        section_title_coverage=0.5,
        exact_phrase_match=False,
    )


def routed_result(
    query: str,
    *,
    texts: tuple[str, ...] = (),
    scores: tuple[float, ...] | None = None,
    reranked_lexical_coverage: float = 0.6,
) -> RoutedRetrievalResult:
    decision = RetrievalStrategyRouter().route(query)
    resolved_scores = scores or tuple(0.8 for _ in texts)

    if len(resolved_scores) != len(texts):
        raise ValueError("scores and texts must have equal lengths")

    response: RetrievalExecutionResponse

    if decision.strategy == "reranked":
        matches = tuple(
            reranked_match(
                text,
                score=score,
                lexical_coverage=reranked_lexical_coverage,
            )
            for text, score in zip(texts, resolved_scores, strict=True)
        )
        response = RerankedRetrievalResponse(
            query=decision.query,
            algorithm="transparent-reranking",
            embedding_model="test-embedding",
            dense_match_count=len(matches),
            sparse_match_count=len(matches),
            candidate_count=len(matches),
            matches=matches,
        )
    else:
        vector_matches = tuple(
            vector_match(
                text,
                score=score,
                chunk_id=f"chunk-{index}",
            )
            for index, (text, score) in enumerate(
                zip(texts, resolved_scores, strict=True),
                start=1,
            )
        )

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


@pytest.mark.parametrize(
    ("query", "evidence", "expected_strategy"),
    [
        (
            "What does overheating mean?",
            ("Overheating means the engine temperature is dangerously high."),
            "dense",
        ),
        (
            "error code P0420",
            "Diagnostic error P0420 indicates catalyst system efficiency.",
            "sparse",
        ),
        (
            "battery warning smoke smell",
            (
                "A battery warning with smoke or a burning smell requires "
                "the vehicle to stop."
            ),
            "hybrid",
        ),
        (
            (
                "If the battery warning is accompanied by smoke, "
                "may the driver continue the trip or must they stop safely?"
            ),
            (
                "If a battery warning is accompanied by smoke, the driver "
                "must stop safely and must not continue the trip."
            ),
            "reranked",
        ),
    ],
)
def test_relevant_evidence_is_accepted_for_every_strategy(
    query: str,
    evidence: str,
    expected_strategy: str,
) -> None:
    assessment = RetrievalQualityChecker().assess(
        routed_result(query, texts=(evidence,))
    )

    assert assessment.strategy == expected_strategy
    assert assessment.verdict == "accept"
    assert assessment.should_accept
    assert not assessment.should_rewrite
    assert assessment.quality_score == 1.0
    assert all(signal.passed for signal in assessment.signals)


def test_empty_results_request_rewrite() -> None:
    assessment = RetrievalQualityChecker().assess(
        routed_result("What does overheating mean?")
    )

    assert assessment.verdict == "rewrite"
    assert assessment.should_rewrite
    assert not assessment.should_accept
    assert assessment.match_count == 0
    assert assessment.top_score is None
    assert any(
        signal.name == "minimum_matches" and not signal.passed
        for signal in assessment.signals
    )


def test_configured_minimum_match_count_is_enforced() -> None:
    checker = RetrievalQualityChecker(RetrievalQualityPolicy(minimum_matches=2))
    result = routed_result(
        "What does overheating mean?",
        texts=("Overheating means excessive engine temperature.",),
    )

    assessment = checker.assess(result)

    assert assessment.verdict == "rewrite"
    assert assessment.match_count == 1
    assert assessment.signals[0].required == ">= 2"


def test_low_query_token_coverage_requests_rewrite() -> None:
    assessment = RetrievalQualityChecker().assess(
        routed_result(
            "What does overheating mean?",
            texts=("Tire pressure should be checked before every shift.",),
        )
    )

    assert assessment.verdict == "rewrite"
    assert assessment.query_token_coverage == 0.0
    assert any(
        signal.name == "query_token_coverage" and not signal.passed
        for signal in assessment.signals
    )


def test_missing_exact_identifier_requests_rewrite() -> None:
    assessment = RetrievalQualityChecker().assess(
        routed_result(
            "error code P0420",
            texts=("The diagnostic system recorded a catalyst warning.",),
        )
    )

    assert assessment.verdict == "rewrite"
    assert assessment.exact_identifier_coverage == 0.0
    assert any(
        signal.name == "exact_identifier_coverage" and not signal.passed
        for signal in assessment.signals
    )


def test_exact_identifier_matching_is_case_insensitive() -> None:
    assessment = RetrievalQualityChecker().assess(
        routed_result(
            "error code P0420",
            texts=("The stored diagnostic identifier is p0420.",),
        )
    )

    assert assessment.verdict == "accept"
    assert assessment.exact_identifier_coverage == 1.0


def test_missing_quoted_phrase_requests_rewrite() -> None:
    assessment = RetrievalQualityChecker().assess(
        routed_result(
            'Find the exact phrase "sidewall bulge".',
            texts=("The tire has visible sidewall damage.",),
        )
    )

    assert assessment.verdict == "rewrite"
    assert assessment.quoted_phrase_coverage == 0.0


def test_quoted_phrase_matching_normalizes_case_and_whitespace() -> None:
    assessment = RetrievalQualityChecker().assess(
        routed_result(
            'Find the exact phrase "sidewall bulge".',
            texts=("Remove a tire with a SIDEWALL   BULGE from service.",),
        )
    )

    assert assessment.verdict == "accept"
    assert assessment.quoted_phrase_coverage == 1.0


@pytest.mark.parametrize("score", [math.nan, math.inf, -math.inf])
def test_non_finite_score_requests_rewrite(score: float) -> None:
    assessment = RetrievalQualityChecker().assess(
        routed_result(
            "What does overheating mean?",
            texts=("Overheating means excessive engine temperature.",),
            scores=(score,),
        )
    )

    assert assessment.verdict == "rewrite"
    assert any(
        signal.name == "finite_scores" and not signal.passed
        for signal in assessment.signals
    )


def test_finite_negative_score_is_not_compared_across_strategies() -> None:
    assessment = RetrievalQualityChecker().assess(
        routed_result(
            "What does overheating mean?",
            texts=("Overheating means excessive engine temperature.",),
            scores=(-0.25,),
        )
    )

    assert assessment.verdict == "accept"
    assert assessment.top_score == -0.25


def test_low_reranked_lexical_coverage_requests_rewrite() -> None:
    query = (
        "If the battery warning is accompanied by smoke, "
        "may the driver continue the trip or must they stop safely?"
    )
    result = routed_result(
        query,
        texts=("A battery warning with smoke means the driver must stop safely.",),
        reranked_lexical_coverage=0.1,
    )

    assessment = RetrievalQualityChecker().assess(result)

    assert assessment.strategy == "reranked"
    assert assessment.verdict == "rewrite"
    assert any(
        signal.name == "reranked_lexical_coverage" and not signal.passed
        for signal in assessment.signals
    )


def test_evidence_limit_controls_coverage_window() -> None:
    result = routed_result(
        "What does overheating mean?",
        texts=(
            "Tire pressure guidance.",
            "Overheating means excessive engine temperature.",
        ),
    )

    first_only = RetrievalQualityChecker(
        RetrievalQualityPolicy(evidence_limit=1)
    ).assess(result)
    first_two = RetrievalQualityChecker(
        RetrievalQualityPolicy(evidence_limit=2)
    ).assess(result)

    assert first_only.verdict == "rewrite"
    assert first_two.verdict == "accept"


def test_failed_reasons_include_only_failed_signal_details() -> None:
    assessment = RetrievalQualityChecker().assess(
        routed_result("What does overheating mean?")
    )
    failed_details = tuple(
        signal.detail for signal in assessment.signals if not signal.passed
    )

    assert assessment.reasons == failed_details


def test_accepted_reason_is_human_readable() -> None:
    assessment = RetrievalQualityChecker().assess(
        routed_result(
            "What does overheating mean?",
            texts=("Overheating means excessive engine temperature.",),
        )
    )

    assert assessment.reasons == (
        "Retrieved evidence passed every configured quality check.",
    )


def test_quality_score_is_fraction_of_passed_signals() -> None:
    assessment = RetrievalQualityChecker().assess(
        routed_result("What does overheating mean?")
    )
    passed = sum(signal.passed for signal in assessment.signals)

    assert assessment.quality_score == round(
        passed / len(assessment.signals),
        4,
    )


def test_assessment_is_deterministic() -> None:
    checker = RetrievalQualityChecker()
    result = routed_result(
        "battery warning smoke smell",
        texts=("A battery warning with smoke and smell requires action.",),
    )

    assert checker.assess(result) == checker.assess(result)


def test_checker_exposes_immutable_policy() -> None:
    policy = RetrievalQualityPolicy(
        minimum_matches=2,
        minimum_query_token_coverage=0.5,
    )
    checker = RetrievalQualityChecker(policy)

    assert checker.policy is policy


@pytest.mark.parametrize("minimum_matches", [0, -1])
def test_policy_rejects_non_positive_minimum_matches(
    minimum_matches: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="minimum_matches must be greater than zero",
    ):
        RetrievalQualityPolicy(minimum_matches=minimum_matches)


@pytest.mark.parametrize(
    "coverage",
    [-0.01, 1.01],
)
def test_policy_rejects_invalid_query_coverage(
    coverage: float,
) -> None:
    with pytest.raises(
        ValueError,
        match=("minimum_query_token_coverage must be between 0.0 and 1.0"),
    ):
        RetrievalQualityPolicy(minimum_query_token_coverage=coverage)


@pytest.mark.parametrize(
    "coverage",
    [-0.01, 1.01],
)
def test_policy_rejects_invalid_reranked_coverage(
    coverage: float,
) -> None:
    with pytest.raises(
        ValueError,
        match=("minimum_reranked_lexical_coverage must be between 0.0 and 1.0"),
    ):
        RetrievalQualityPolicy(minimum_reranked_lexical_coverage=coverage)


@pytest.mark.parametrize("evidence_limit", [0, -1])
def test_policy_rejects_non_positive_evidence_limit(
    evidence_limit: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="evidence_limit must be greater than zero",
    ):
        RetrievalQualityPolicy(evidence_limit=evidence_limit)


def test_response_query_must_match_routing_query() -> None:
    result = routed_result(
        "What does overheating mean?",
        texts=("Overheating means excessive engine temperature.",),
    )
    wrong_response = RetrievalResponse(
        query="different query",
        embedding_model="test-embedding",
        matches=(vector_match("Overheating means excessive engine temperature."),),
    )

    with pytest.raises(
        ValueError,
        match=("retrieval response query must match routing decision query"),
    ):
        RetrievalQualityChecker().assess(
            RoutedRetrievalResult(
                decision=result.decision,
                response=wrong_response,
            )
        )


@pytest.mark.parametrize(
    "query",
    [
        "What does overheating mean?",
        "error code P0420",
        "battery warning smoke smell",
        (
            "If the battery warning is accompanied by smoke, "
            "may the driver continue the trip or must they stop safely?"
        ),
    ],
)
def test_response_type_must_match_selected_strategy(
    query: str,
) -> None:
    result = routed_result(query)
    wrong_response = RetrievalResponse(
        query=result.decision.query,
        embedding_model="test-embedding",
        matches=(),
    )

    if result.decision.strategy == "dense":
        wrong: RetrievalExecutionResponse = SparseRetrievalResponse(
            query=result.decision.query,
            algorithm="bm25",
            matches=(),
        )
    else:
        wrong = wrong_response

    with pytest.raises(
        ValueError,
        match="response type does not match",
    ):
        RetrievalQualityChecker().assess(
            RoutedRetrievalResult(
                decision=result.decision,
                response=wrong,
            )
        )
