from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from fleetmind_rag.feedback_analytics import (
    RoutingFeedbackAnalyzer,
)
from fleetmind_rag.feedback_routing import (
    FeedbackFeature,
    RoutingFeedbackHistory,
    RoutingFeedbackObservation,
)
from fleetmind_rag.routing import RetrievalStrategy


def _observation(
    *,
    strategy: RetrievalStrategy = "dense",
    verdict: str = "accept",
    quality_score: float = 0.8,
    attempt_number: int = 1,
    features: tuple[FeedbackFeature, ...] = ("conceptual",),
) -> RoutingFeedbackObservation:
    return RoutingFeedbackObservation(
        query="What does overheating mean?",
        strategy=strategy,
        verdict=verdict,  # type: ignore[arg-type]
        quality_score=quality_score,
        attempt_number=attempt_number,
        features=features,
    )


def test_empty_history_produces_zero_counts_and_undefined_rates() -> None:
    report = RoutingFeedbackAnalyzer().analyze(RoutingFeedbackHistory())

    assert report.revision == 0
    assert report.observation_count == 0
    assert report.accepted_count == 0
    assert report.rewrite_count == 0
    assert report.retry_attempt_count == 0
    assert report.acceptance_rate is None
    assert report.rewrite_rate is None
    assert report.average_quality_score is None
    assert report.average_attempt_number is None
    assert report.retry_rate is None
    assert report.is_empty


def test_revision_is_preserved_in_report() -> None:
    report = RoutingFeedbackAnalyzer().analyze(
        RoutingFeedbackHistory(),
        revision=17,
    )

    assert report.revision == 17


@pytest.mark.parametrize("revision", [True, False, 1.5, "1", None])
def test_revision_must_be_an_integer(revision: object) -> None:
    with pytest.raises(TypeError, match="revision must be an integer"):
        RoutingFeedbackAnalyzer().analyze(
            RoutingFeedbackHistory(),
            revision=revision,  # type: ignore[arg-type]
        )


def test_revision_must_not_be_negative() -> None:
    with pytest.raises(ValueError, match="revision must not be negative"):
        RoutingFeedbackAnalyzer().analyze(
            RoutingFeedbackHistory(),
            revision=-1,
        )


def test_strategy_metrics_have_stable_order() -> None:
    report = RoutingFeedbackAnalyzer().analyze(RoutingFeedbackHistory())

    assert tuple(metrics.strategy for metrics in report.strategies) == (
        "dense",
        "sparse",
        "hybrid",
        "reranked",
    )


def test_feature_metrics_have_stable_order() -> None:
    report = RoutingFeedbackAnalyzer().analyze(RoutingFeedbackHistory())

    assert tuple(metrics.feature for metrics in report.features) == (
        "exact_identifier",
        "quoted_phrase",
        "conceptual",
        "conditional",
        "action",
        "safety",
        "domain",
        "complex",
        "general",
    )


def test_overall_metrics_aggregate_accepts_rewrites_and_quality() -> None:
    history = RoutingFeedbackHistory(
        (
            _observation(quality_score=0.9),
            _observation(
                strategy="hybrid",
                verdict="rewrite",
                quality_score=0.3,
                attempt_number=2,
            ),
            _observation(
                strategy="hybrid",
                quality_score=0.6,
                attempt_number=3,
            ),
        )
    )

    report = RoutingFeedbackAnalyzer().analyze(history, revision=4)

    assert report.revision == 4
    assert report.observation_count == 3
    assert report.accepted_count == 2
    assert report.rewrite_count == 1
    assert report.retry_attempt_count == 2
    assert report.acceptance_rate == pytest.approx(2 / 3)
    assert report.rewrite_rate == pytest.approx(1 / 3)
    assert report.average_quality_score == pytest.approx(0.6)
    assert report.average_attempt_number == pytest.approx(2.0)
    assert report.retry_rate == pytest.approx(2 / 3)
    assert not report.is_empty


def test_strategy_metrics_only_include_matching_strategy() -> None:
    history = RoutingFeedbackHistory(
        (
            _observation(strategy="dense", quality_score=1.0),
            _observation(
                strategy="dense",
                verdict="rewrite",
                quality_score=0.2,
                attempt_number=2,
            ),
            _observation(strategy="sparse", quality_score=0.8),
        )
    )

    report = RoutingFeedbackAnalyzer().analyze(history)
    dense = report.strategy("dense")
    sparse = report.strategy("sparse")

    assert dense.observation_count == 2
    assert dense.accepted_count == 1
    assert dense.rewrite_count == 1
    assert dense.retry_attempt_count == 1
    assert dense.acceptance_rate == pytest.approx(0.5)
    assert dense.rewrite_rate == pytest.approx(0.5)
    assert dense.average_quality_score == pytest.approx(0.6)
    assert dense.average_attempt_number == pytest.approx(1.5)
    assert dense.retry_rate == pytest.approx(0.5)
    assert sparse.observation_count == 1
    assert sparse.acceptance_rate == pytest.approx(1.0)


@pytest.mark.parametrize(
    "strategy",
    ["dense", "sparse", "hybrid", "reranked"],
)
def test_strategy_accessor_returns_requested_metrics(
    strategy: RetrievalStrategy,
) -> None:
    report = RoutingFeedbackAnalyzer().analyze(RoutingFeedbackHistory())

    assert report.strategy(strategy).strategy == strategy


def test_unobserved_strategy_has_zero_counts_and_undefined_rates() -> None:
    report = RoutingFeedbackAnalyzer().analyze(
        RoutingFeedbackHistory((_observation(strategy="dense"),))
    )
    reranked = report.strategy("reranked")

    assert reranked.observation_count == 0
    assert reranked.accepted_count == 0
    assert reranked.rewrite_count == 0
    assert reranked.retry_attempt_count == 0
    assert reranked.acceptance_rate is None
    assert reranked.rewrite_rate is None
    assert reranked.average_quality_score is None
    assert reranked.average_attempt_number is None
    assert reranked.retry_rate is None


def test_one_observation_contributes_to_each_of_its_features() -> None:
    history = RoutingFeedbackHistory(
        (
            _observation(
                strategy="reranked",
                quality_score=0.75,
                features=("conditional", "action", "safety", "complex"),
            ),
        )
    )

    report = RoutingFeedbackAnalyzer().analyze(history)

    for feature in ("conditional", "action", "safety", "complex"):
        metrics = report.feature(feature)
        assert metrics.observation_count == 1
        assert metrics.accepted_count == 1
        assert metrics.acceptance_rate == pytest.approx(1.0)
        assert metrics.average_quality_score == pytest.approx(0.75)

    assert report.feature("conceptual").observation_count == 0


def test_feature_metrics_combine_strategies_with_same_feature() -> None:
    history = RoutingFeedbackHistory(
        (
            _observation(strategy="dense", quality_score=0.9),
            _observation(
                strategy="hybrid",
                verdict="rewrite",
                quality_score=0.1,
                attempt_number=2,
            ),
        )
    )

    conceptual = RoutingFeedbackAnalyzer().analyze(history).feature("conceptual")

    assert conceptual.observation_count == 2
    assert conceptual.accepted_count == 1
    assert conceptual.rewrite_count == 1
    assert conceptual.retry_attempt_count == 1
    assert conceptual.acceptance_rate == pytest.approx(0.5)
    assert conceptual.rewrite_rate == pytest.approx(0.5)
    assert conceptual.average_quality_score == pytest.approx(0.5)
    assert conceptual.average_attempt_number == pytest.approx(1.5)
    assert conceptual.retry_rate == pytest.approx(0.5)


@pytest.mark.parametrize(
    "feature",
    [
        "exact_identifier",
        "quoted_phrase",
        "conceptual",
        "conditional",
        "action",
        "safety",
        "domain",
        "complex",
        "general",
    ],
)
def test_feature_accessor_returns_requested_metrics(
    feature: FeedbackFeature,
) -> None:
    report = RoutingFeedbackAnalyzer().analyze(RoutingFeedbackHistory())

    assert report.feature(feature).feature == feature


@pytest.mark.parametrize("attempt_number, expected", [(1, 0), (2, 1), (5, 1)])
def test_only_attempts_after_first_are_counted_as_retries(
    attempt_number: int,
    expected: int,
) -> None:
    report = RoutingFeedbackAnalyzer().analyze(
        RoutingFeedbackHistory((_observation(attempt_number=attempt_number),))
    )

    assert report.retry_attempt_count == expected
    assert report.retry_rate == pytest.approx(float(expected))


def test_report_and_nested_metrics_are_immutable() -> None:
    report = RoutingFeedbackAnalyzer().analyze(
        RoutingFeedbackHistory((_observation(),))
    )

    with pytest.raises(FrozenInstanceError):
        report.revision = 2  # type: ignore[misc]

    with pytest.raises(FrozenInstanceError):
        report.strategies[0].observation_count = 99  # type: ignore[misc]


def test_repeated_analysis_is_deterministic() -> None:
    history = RoutingFeedbackHistory(
        (
            _observation(strategy="dense", quality_score=0.9),
            _observation(
                strategy="hybrid",
                verdict="rewrite",
                quality_score=0.2,
                attempt_number=2,
            ),
        )
    )
    analyzer = RoutingFeedbackAnalyzer()

    assert analyzer.analyze(history, revision=8) == analyzer.analyze(
        history,
        revision=8,
    )
