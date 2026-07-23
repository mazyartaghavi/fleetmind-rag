from __future__ import annotations

from dataclasses import replace
from typing import Literal, cast

import pytest

from fleetmind_rag.adaptive_retrieval import AdaptiveRetrievalOutcome
from fleetmind_rag.agent_state import RetrievalAgentState
from fleetmind_rag.feedback_routing import (
    FeedbackDrivenRetrievalRouter,
    FeedbackRoutingPolicy,
    RoutingFeedbackHistory,
    RoutingFeedbackObservation,
    query_signal_profile,
)
from fleetmind_rag.retrieval import RetrievalResponse
from fleetmind_rag.retrieval_quality import RetrievalQualityAssessment
from fleetmind_rag.routed_retrieval import RoutedRetrievalResult
from fleetmind_rag.routing import (
    RetrievalStrategy,
    RetrievalStrategyRouter,
    RoutingDecision,
)

_CONCEPTUAL_QUERY = "What does overheating mean?"
_EXACT_QUERY = "What does error code P0420 mean?"
_QUOTED_QUERY = 'Find the exact phrase "sidewall bulge".'
_SAFETY_QUERY = (
    "If the battery warning is accompanied by smoke, "
    "may the driver continue or must they stop safely?"
)


class MisroutingRouter(RetrievalStrategyRouter):
    """Base router used to prove hard guardrails win."""

    def __init__(self, forced_strategy: RetrievalStrategy) -> None:
        self._forced_strategy = forced_strategy

    def route(self, query: str) -> RoutingDecision:
        decision = super().route(query)
        forced_score = next(
            score.score
            for score in decision.scores
            if score.strategy == self._forced_strategy
        )
        return replace(
            decision,
            strategy=self._forced_strategy,
            selected_score=forced_score,
        )


def _observation(
    query: str,
    strategy: RetrievalStrategy,
    *,
    verdict: str,
    quality_score: float,
    attempt_number: int = 1,
) -> RoutingFeedbackObservation:
    decision = RetrievalStrategyRouter().route(query)
    return RoutingFeedbackObservation(
        query=query,
        strategy=strategy,
        verdict=cast("Literal['accept', 'rewrite']", verdict),
        quality_score=quality_score,
        attempt_number=attempt_number,
        features=query_signal_profile(decision.signals),
    )


def _history(
    query: str,
    strategy: RetrievalStrategy,
    *,
    verdict: str,
    quality_score: float,
    count: int = 3,
) -> RoutingFeedbackHistory:
    history = RoutingFeedbackHistory()

    for attempt_number in range(1, count + 1):
        history = history.record(
            _observation(
                query,
                strategy,
                verdict=verdict,
                quality_score=quality_score,
                attempt_number=attempt_number,
            )
        )

    return history


def _combined_history(
    *histories: RoutingFeedbackHistory,
) -> RoutingFeedbackHistory:
    return RoutingFeedbackHistory(
        tuple(
            observation for history in histories for observation in history.observations
        )
    )


def _outcome(
    query: str,
    *,
    verdict: str = "accept",
    quality_score: float = 0.90,
) -> AdaptiveRetrievalOutcome:
    decision = RetrievalStrategyRouter().route(query)
    result = RoutedRetrievalResult(
        decision=decision,
        response=RetrievalResponse(
            query=decision.query,
            embedding_model="test-embedding",
            matches=(),
        ),
    )
    state = RetrievalAgentState.start(query, max_attempts=1)
    state = state.begin_retrieval().record_retrieval(result)
    assessment = RetrievalQualityAssessment(
        query=decision.query,
        strategy=decision.strategy,
        verdict=cast("Literal['accept', 'rewrite']", verdict),
        quality_score=quality_score,
        match_count=0,
        top_score=None,
        query_token_coverage=quality_score,
        exact_identifier_coverage=quality_score,
        quoted_phrase_coverage=quality_score,
        signals=(),
        reasons=("Synthetic quality evidence.",),
    )
    state = (
        state.complete("Accepted synthetic evidence.")
        if verdict == "accept"
        else state.fail("Synthetic evidence was insufficient.")
    )
    return AdaptiveRetrievalOutcome(
        state=state,
        assessments=(assessment,),
        rewrites=(),
    )


def test_general_query_has_general_profile() -> None:
    decision = RetrievalStrategyRouter().route("hello world")

    assert query_signal_profile(decision.signals) == ("general",)


def test_exact_query_profile_contains_identifier() -> None:
    decision = RetrievalStrategyRouter().route(_EXACT_QUERY)

    assert "exact_identifier" in query_signal_profile(decision.signals)


def test_safety_query_profile_contains_operational_signals() -> None:
    decision = RetrievalStrategyRouter().route(_SAFETY_QUERY)
    profile = query_signal_profile(decision.signals)

    assert "conditional" in profile
    assert "action" in profile
    assert "safety" in profile


def test_observation_normalizes_query() -> None:
    observation = RoutingFeedbackObservation(
        query="  What   does overheating mean?  ",
        strategy="dense",
        verdict="accept",
        quality_score=0.9,
        attempt_number=1,
        features=("conceptual",),
    )

    assert observation.query == _CONCEPTUAL_QUERY


def test_observation_reports_acceptance() -> None:
    accepted = _observation(
        _CONCEPTUAL_QUERY,
        "dense",
        verdict="accept",
        quality_score=0.9,
    )
    rewritten = _observation(
        _CONCEPTUAL_QUERY,
        "dense",
        verdict="rewrite",
        quality_score=0.2,
    )

    assert accepted.accepted is True
    assert rewritten.accepted is False


def test_observation_rejects_blank_query() -> None:
    with pytest.raises(ValueError, match="query must not be blank"):
        RoutingFeedbackObservation(
            query=" ",
            strategy="dense",
            verdict="accept",
            quality_score=0.9,
            attempt_number=1,
            features=("conceptual",),
        )


@pytest.mark.parametrize("quality_score", [-0.1, 1.1])
def test_observation_rejects_out_of_range_quality(
    quality_score: float,
) -> None:
    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        RoutingFeedbackObservation(
            query=_CONCEPTUAL_QUERY,
            strategy="dense",
            verdict="accept",
            quality_score=quality_score,
            attempt_number=1,
            features=("conceptual",),
        )


@pytest.mark.parametrize("quality_score", [float("nan"), float("inf")])
def test_observation_rejects_non_finite_quality(
    quality_score: float,
) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        RoutingFeedbackObservation(
            query=_CONCEPTUAL_QUERY,
            strategy="dense",
            verdict="accept",
            quality_score=quality_score,
            attempt_number=1,
            features=("conceptual",),
        )


def test_observation_rejects_non_positive_attempt_number() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        _observation(
            _CONCEPTUAL_QUERY,
            "dense",
            verdict="accept",
            quality_score=0.9,
            attempt_number=0,
        )


def test_features_are_deduplicated_and_ordered() -> None:
    observation = RoutingFeedbackObservation(
        query=_CONCEPTUAL_QUERY,
        strategy="dense",
        verdict="accept",
        quality_score=0.9,
        attempt_number=1,
        features=("domain", "conceptual", "domain"),
    )

    assert observation.features == ("conceptual", "domain")


def test_general_cannot_be_combined_with_specific_features() -> None:
    with pytest.raises(ValueError, match="general cannot be combined"):
        RoutingFeedbackObservation(
            query=_CONCEPTUAL_QUERY,
            strategy="dense",
            verdict="accept",
            quality_score=0.9,
            attempt_number=1,
            features=("general", "conceptual"),
        )


def test_history_record_is_immutable() -> None:
    original = RoutingFeedbackHistory()
    updated = original.record(
        _observation(
            _CONCEPTUAL_QUERY,
            "dense",
            verdict="accept",
            quality_score=0.9,
        )
    )

    assert original.observations == ()
    assert len(updated.observations) == 1


def test_history_matching_uses_complete_profile() -> None:
    conceptual = _observation(
        _CONCEPTUAL_QUERY,
        "dense",
        verdict="accept",
        quality_score=0.9,
    )
    exact = _observation(
        _EXACT_QUERY,
        "sparse",
        verdict="accept",
        quality_score=0.9,
    )
    history = RoutingFeedbackHistory((conceptual, exact))

    assert history.matching(conceptual.features) == (conceptual,)


def test_default_policy_is_bounded() -> None:
    policy = FeedbackRoutingPolicy()

    assert policy.minimum_observations == 3
    assert policy.maximum_score_adjustment == 4


def test_policy_rejects_non_positive_minimum_observations() -> None:
    with pytest.raises(ValueError, match="minimum_observations"):
        FeedbackRoutingPolicy(minimum_observations=0)


def test_policy_rejects_non_positive_maximum_adjustment() -> None:
    with pytest.raises(ValueError, match="maximum_score_adjustment"):
        FeedbackRoutingPolicy(maximum_score_adjustment=0)


def test_policy_rejects_out_of_range_threshold() -> None:
    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        FeedbackRoutingPolicy(positive_threshold=1.1)


def test_policy_rejects_misordered_thresholds() -> None:
    with pytest.raises(ValueError, match="must be ordered"):
        FeedbackRoutingPolicy(
            negative_threshold=0.75,
            positive_threshold=0.50,
        )


def test_empty_history_preserves_base_decision() -> None:
    router = FeedbackDrivenRetrievalRouter()
    explained = router.explain(_CONCEPTUAL_QUERY)

    assert explained.strategy == explained.base_decision.strategy
    assert explained.feedback_applied is False
    assert explained.strategy_changed is False


def test_insufficient_feedback_has_zero_adjustment() -> None:
    history = _history(
        _CONCEPTUAL_QUERY,
        "dense",
        verdict="accept",
        quality_score=1.0,
        count=2,
    )
    explained = FeedbackDrivenRetrievalRouter(history).explain(_CONCEPTUAL_QUERY)
    dense = next(
        adjustment
        for adjustment in explained.adjustments
        if adjustment.strategy == "dense"
    )

    assert dense.adjustment == 0
    assert "2 observed, 3 required" in dense.reasons[0]


def test_strong_positive_feedback_receives_maximum_reward() -> None:
    history = _history(
        _CONCEPTUAL_QUERY,
        "hybrid",
        verdict="accept",
        quality_score=1.0,
    )
    explained = FeedbackDrivenRetrievalRouter(history).explain(_CONCEPTUAL_QUERY)
    hybrid = next(
        adjustment
        for adjustment in explained.adjustments
        if adjustment.strategy == "hybrid"
    )

    assert hybrid.adjustment == 4
    assert hybrid.statistics.acceptance_rate == 1.0
    assert hybrid.statistics.average_quality_score == 1.0


def test_strong_negative_feedback_receives_maximum_penalty() -> None:
    history = _history(
        _CONCEPTUAL_QUERY,
        "dense",
        verdict="rewrite",
        quality_score=0.0,
    )
    explained = FeedbackDrivenRetrievalRouter(history).explain(_CONCEPTUAL_QUERY)
    dense = next(
        adjustment
        for adjustment in explained.adjustments
        if adjustment.strategy == "dense"
    )

    assert dense.adjustment == -4
    assert dense.statistics.acceptance_rate == 0.0


def test_moderate_positive_feedback_receives_bounded_reward() -> None:
    history = _history(
        _CONCEPTUAL_QUERY,
        "hybrid",
        verdict="accept",
        quality_score=0.20,
    )
    explained = FeedbackDrivenRetrievalRouter(history).explain(_CONCEPTUAL_QUERY)
    hybrid = next(
        adjustment
        for adjustment in explained.adjustments
        if adjustment.strategy == "hybrid"
    )

    assert hybrid.statistics.utility_score == pytest.approx(0.68)
    assert hybrid.adjustment == 2


def test_neutral_feedback_does_not_adjust_score() -> None:
    history = _combined_history(
        _history(
            _CONCEPTUAL_QUERY,
            "hybrid",
            verdict="accept",
            quality_score=0.0,
            count=2,
        ),
        _history(
            _CONCEPTUAL_QUERY,
            "hybrid",
            verdict="rewrite",
            quality_score=0.5,
            count=2,
        ),
    )
    explained = FeedbackDrivenRetrievalRouter(history).explain(_CONCEPTUAL_QUERY)
    hybrid = next(
        adjustment
        for adjustment in explained.adjustments
        if adjustment.strategy == "hybrid"
    )

    assert hybrid.statistics.utility_score == pytest.approx(0.4)
    assert hybrid.adjustment == 0


def test_feedback_can_change_conceptual_route_to_hybrid() -> None:
    history = _combined_history(
        _history(
            _CONCEPTUAL_QUERY,
            "dense",
            verdict="rewrite",
            quality_score=0.0,
        ),
        _history(
            _CONCEPTUAL_QUERY,
            "hybrid",
            verdict="accept",
            quality_score=1.0,
        ),
    )
    explained = FeedbackDrivenRetrievalRouter(history).explain(_CONCEPTUAL_QUERY)

    assert explained.base_decision.strategy == "dense"
    assert explained.strategy == "hybrid"
    assert explained.strategy_changed is True
    assert explained.decision.confidence == "medium"


def test_base_strategy_wins_an_adjusted_score_tie() -> None:
    history = _history(
        _CONCEPTUAL_QUERY,
        "dense",
        verdict="rewrite",
        quality_score=0.0,
    )
    policy = FeedbackRoutingPolicy(maximum_score_adjustment=6)
    explained = FeedbackDrivenRetrievalRouter(
        history,
        policy=policy,
    ).explain(_CONCEPTUAL_QUERY)

    assert explained.base_decision.strategy == "dense"
    assert explained.strategy == "dense"


def test_identifier_guardrail_forces_sparse() -> None:
    router = FeedbackDrivenRetrievalRouter(base_router=MisroutingRouter("dense"))
    explained = router.explain(_EXACT_QUERY)

    assert explained.guardrail == "exact_identifier"
    assert explained.strategy == "sparse"
    assert explained.decision.confidence == "high"


def test_quoted_phrase_guardrail_forces_sparse() -> None:
    router = FeedbackDrivenRetrievalRouter(base_router=MisroutingRouter("dense"))
    explained = router.explain(_QUOTED_QUERY)

    assert explained.guardrail == "quoted_phrase"
    assert explained.strategy == "sparse"


def test_safety_guardrail_forces_reranked() -> None:
    router = FeedbackDrivenRetrievalRouter(base_router=MisroutingRouter("dense"))
    explained = router.explain(_SAFETY_QUERY)

    assert explained.guardrail == "safety_sensitive"
    assert explained.strategy == "reranked"


def test_guardrail_score_is_strictly_highest() -> None:
    explained = FeedbackDrivenRetrievalRouter(
        base_router=MisroutingRouter("dense")
    ).explain(_SAFETY_QUERY)
    selected_score = next(
        score.score
        for score in explained.decision.scores
        if score.strategy == explained.strategy
    )
    other_scores = tuple(
        score.score
        for score in explained.decision.scores
        if score.strategy != explained.strategy
    )

    assert selected_score > max(other_scores)
    assert explained.decision.selected_score == selected_score


def test_route_returns_standard_routing_decision() -> None:
    decision = FeedbackDrivenRetrievalRouter().route(_CONCEPTUAL_QUERY)

    assert isinstance(decision, RoutingDecision)


def test_feedback_router_is_compatible_with_base_router_type() -> None:
    router = FeedbackDrivenRetrievalRouter()

    assert isinstance(router, RetrievalStrategyRouter)


def test_feedback_reasons_are_appended_to_every_score() -> None:
    explained = FeedbackDrivenRetrievalRouter().explain(_CONCEPTUAL_QUERY)

    assert all(
        "Insufficient matching feedback" in score.reasons[-1]
        for score in explained.decision.scores
    )


def test_record_outcome_creates_one_observation_per_attempt() -> None:
    outcome = _outcome(_CONCEPTUAL_QUERY)
    history = RoutingFeedbackHistory().record_outcome(outcome)
    observation = history.observations[0]

    assert len(history.observations) == 1
    assert observation.query == _CONCEPTUAL_QUERY
    assert observation.verdict == "accept"
    assert observation.quality_score == 0.90


def test_record_outcome_requires_one_assessment_per_attempt() -> None:
    outcome = replace(_outcome(_CONCEPTUAL_QUERY), assessments=())

    with pytest.raises(ValueError, match="one quality assessment per attempt"):
        RoutingFeedbackHistory().record_outcome(outcome)


def test_record_outcome_rejects_mismatched_assessment_query() -> None:
    outcome = _outcome(_CONCEPTUAL_QUERY)
    assessment = replace(
        outcome.assessments[0],
        query="different query",
    )
    mismatched = replace(outcome, assessments=(assessment,))

    with pytest.raises(ValueError, match="query must match"):
        RoutingFeedbackHistory().record_outcome(mismatched)


def test_record_outcome_rejects_mismatched_assessment_strategy() -> None:
    outcome = _outcome(_CONCEPTUAL_QUERY)
    assessment = replace(
        outcome.assessments[0],
        strategy="sparse",
    )
    mismatched = replace(outcome, assessments=(assessment,))

    with pytest.raises(ValueError, match="strategy must match"):
        RoutingFeedbackHistory().record_outcome(mismatched)


def test_with_outcome_returns_new_router_with_preserved_original() -> None:
    original = FeedbackDrivenRetrievalRouter()
    updated = original.with_outcome(_outcome(_CONCEPTUAL_QUERY))

    assert original.history.observations == ()
    assert len(updated.history.observations) == 1
    assert updated.policy is original.policy


def test_unrelated_feedback_does_not_affect_query() -> None:
    history = _history(
        _EXACT_QUERY,
        "sparse",
        verdict="accept",
        quality_score=1.0,
    )
    explained = FeedbackDrivenRetrievalRouter(history).explain(_CONCEPTUAL_QUERY)

    assert explained.feedback_applied is False


def test_feedback_routing_is_deterministic() -> None:
    history = _combined_history(
        _history(
            _CONCEPTUAL_QUERY,
            "dense",
            verdict="rewrite",
            quality_score=0.0,
        ),
        _history(
            _CONCEPTUAL_QUERY,
            "hybrid",
            verdict="accept",
            quality_score=1.0,
        ),
    )
    router = FeedbackDrivenRetrievalRouter(history)

    assert router.explain(_CONCEPTUAL_QUERY) == router.explain(_CONCEPTUAL_QUERY)


def test_selected_score_matches_selected_strategy_score() -> None:
    explained = FeedbackDrivenRetrievalRouter().explain(_CONCEPTUAL_QUERY)
    selected_score = next(
        score.score
        for score in explained.decision.scores
        if score.strategy == explained.strategy
    )

    assert explained.decision.selected_score == selected_score
