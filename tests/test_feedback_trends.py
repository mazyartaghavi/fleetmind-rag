from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from fleetmind_rag.feedback_routing import (
    RoutingFeedbackHistory,
    RoutingFeedbackObservation,
)
from fleetmind_rag.feedback_trends import (
    FeedbackTrendPolicy,
    RoutingFeedbackTrendAnalyzer,
)
from fleetmind_rag.routing import RetrievalStrategy


def _observation(
    *,
    strategy: RetrievalStrategy = "dense",
    accepted: bool = True,
    quality_score: float = 1.0,
    attempt_number: int = 1,
) -> RoutingFeedbackObservation:
    return RoutingFeedbackObservation(
        query="What does overheating mean?",
        strategy=strategy,
        verdict="accept" if accepted else "rewrite",
        quality_score=quality_score,
        attempt_number=attempt_number,
        features=("conceptual",),
    )


def _history(
    observations: list[RoutingFeedbackObservation],
) -> RoutingFeedbackHistory:
    return RoutingFeedbackHistory(tuple(observations))


@pytest.mark.parametrize("window_size", [True, False, 1.5, "2", None])
def test_window_size_must_be_an_integer(window_size: object) -> None:
    with pytest.raises(TypeError, match="window_size must be an integer"):
        FeedbackTrendPolicy(window_size=window_size)  # type: ignore[arg-type]


@pytest.mark.parametrize("window_size", [0, -1])
def test_window_size_must_be_positive(window_size: int) -> None:
    with pytest.raises(ValueError, match="window_size must be greater than zero"):
        FeedbackTrendPolicy(window_size=window_size)


@pytest.mark.parametrize("minimum", [True, False, 1.5, "2", None])
def test_minimum_strategy_observations_must_be_an_integer(
    minimum: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="minimum_strategy_observations must be an integer",
    ):
        FeedbackTrendPolicy(
            minimum_strategy_observations=minimum,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("minimum", [0, -1])
def test_minimum_strategy_observations_must_be_positive(minimum: int) -> None:
    with pytest.raises(
        ValueError,
        match="minimum_strategy_observations must be greater than zero",
    ):
        FeedbackTrendPolicy(minimum_strategy_observations=minimum)


def test_minimum_strategy_observations_cannot_exceed_window() -> None:
    with pytest.raises(
        ValueError,
        match="minimum_strategy_observations must not exceed window_size",
    ):
        FeedbackTrendPolicy(
            window_size=2,
            minimum_strategy_observations=3,
        )


@pytest.mark.parametrize(
    "minimum_change",
    [float("nan"), float("inf"), float("-inf")],
)
def test_minimum_utility_change_must_be_finite(minimum_change: float) -> None:
    with pytest.raises(
        ValueError,
        match="minimum_utility_change must be finite",
    ):
        FeedbackTrendPolicy(minimum_utility_change=minimum_change)


@pytest.mark.parametrize("minimum_change", [0.0, -0.1, 1.1])
def test_minimum_utility_change_must_be_in_range(
    minimum_change: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="minimum_utility_change must be greater than zero and at most one",
    ):
        FeedbackTrendPolicy(minimum_utility_change=minimum_change)


def test_analyzer_exposes_configured_policy() -> None:
    policy = FeedbackTrendPolicy(
        window_size=3,
        minimum_utility_change=0.1,
        minimum_strategy_observations=1,
    )

    assert RoutingFeedbackTrendAnalyzer(policy).policy is policy


def test_empty_history_has_no_positions_and_insufficient_data() -> None:
    report = RoutingFeedbackTrendAnalyzer(FeedbackTrendPolicy(window_size=2)).analyze(
        RoutingFeedbackHistory(), revision=4
    )

    assert report.revision == 4
    assert report.total_observations == 0
    assert report.previous_start_position is None
    assert report.previous_end_position is None
    assert report.recent_start_position is None
    assert report.recent_end_position is None
    assert report.overall.direction == "insufficient_data"
    assert not report.overall.sufficient_data
    assert report.overall.utility_delta is None


def test_partial_history_is_exposed_as_recent_but_not_compared() -> None:
    report = RoutingFeedbackTrendAnalyzer(FeedbackTrendPolicy(window_size=3)).analyze(
        _history([_observation(), _observation()]),
    )

    assert report.total_observations == 2
    assert report.previous_start_position is None
    assert report.previous_end_position is None
    assert report.recent_start_position == 1
    assert report.recent_end_position == 2
    assert report.overall.previous.observation_count == 0
    assert report.overall.recent.observation_count == 2
    assert report.overall.direction == "insufficient_data"
    assert "previous 0, recent 2" in report.overall.reason


def test_exactly_two_windows_report_one_based_positions() -> None:
    report = RoutingFeedbackTrendAnalyzer(FeedbackTrendPolicy(window_size=2)).analyze(
        _history([_observation() for _ in range(4)])
    )

    assert report.previous_start_position == 1
    assert report.previous_end_position == 2
    assert report.recent_start_position == 3
    assert report.recent_end_position == 4


def test_only_latest_two_windows_are_compared() -> None:
    observations = [_observation(accepted=False, quality_score=0.0) for _ in range(3)]
    observations.extend(_observation() for _ in range(4))
    report = RoutingFeedbackTrendAnalyzer(FeedbackTrendPolicy(window_size=2)).analyze(
        _history(observations)
    )

    assert report.total_observations == 7
    assert report.previous_start_position == 4
    assert report.previous_end_position == 5
    assert report.recent_start_position == 6
    assert report.recent_end_position == 7
    assert report.overall.direction == "stable"


def test_improving_utility_is_detected() -> None:
    observations = [
        _observation(accepted=False, quality_score=0.0),
        _observation(accepted=False, quality_score=0.0),
        _observation(quality_score=1.0),
        _observation(quality_score=1.0),
    ]
    report = RoutingFeedbackTrendAnalyzer(FeedbackTrendPolicy(window_size=2)).analyze(
        _history(observations)
    )

    assert report.overall.direction == "improving"
    assert report.overall.sufficient_data
    assert report.overall.previous.utility_score == pytest.approx(0.0)
    assert report.overall.recent.utility_score == pytest.approx(1.0)
    assert report.overall.utility_delta == pytest.approx(1.0)
    assert report.overall.acceptance_delta == pytest.approx(1.0)
    assert report.overall.quality_delta == pytest.approx(1.0)
    assert report.overall.rewrite_delta == pytest.approx(-1.0)


def test_regressing_utility_is_detected() -> None:
    observations = [
        _observation(quality_score=1.0),
        _observation(quality_score=1.0),
        _observation(accepted=False, quality_score=0.0, attempt_number=2),
        _observation(accepted=False, quality_score=0.0, attempt_number=2),
    ]
    report = RoutingFeedbackTrendAnalyzer(FeedbackTrendPolicy(window_size=2)).analyze(
        _history(observations)
    )

    assert report.overall.direction == "regressing"
    assert report.overall.utility_delta == pytest.approx(-1.0)
    assert report.overall.retry_delta == pytest.approx(1.0)


def test_change_below_threshold_is_stable() -> None:
    observations = [
        _observation(quality_score=0.8),
        _observation(quality_score=0.8),
        _observation(quality_score=0.9),
        _observation(quality_score=0.9),
    ]
    report = RoutingFeedbackTrendAnalyzer(
        FeedbackTrendPolicy(
            window_size=2,
            minimum_utility_change=0.05,
        )
    ).analyze(_history(observations))

    assert report.overall.utility_delta == pytest.approx(0.04)
    assert report.overall.direction == "stable"


@pytest.mark.parametrize(
    "recent_quality, expected_direction",
    [(0.625, "improving"), (0.375, "regressing")],
)
def test_change_equal_to_threshold_is_directional(
    recent_quality: float,
    expected_direction: str,
) -> None:
    observations = [
        _observation(quality_score=0.5),
        _observation(quality_score=0.5),
        _observation(quality_score=recent_quality),
        _observation(quality_score=recent_quality),
    ]
    report = RoutingFeedbackTrendAnalyzer(
        FeedbackTrendPolicy(
            window_size=2,
            minimum_utility_change=0.05,
        )
    ).analyze(_history(observations))

    assert report.overall.utility_delta == pytest.approx(
        0.05 if recent_quality > 0.5 else -0.05
    )
    assert report.overall.direction == expected_direction


def test_window_metrics_include_attempt_and_retry_statistics() -> None:
    observations = [
        _observation(attempt_number=1),
        _observation(attempt_number=2),
        _observation(attempt_number=2),
        _observation(attempt_number=3),
    ]
    report = RoutingFeedbackTrendAnalyzer(FeedbackTrendPolicy(window_size=2)).analyze(
        _history(observations)
    )

    assert report.overall.previous.average_attempt_number == pytest.approx(1.5)
    assert report.overall.previous.retry_rate == pytest.approx(0.5)
    assert report.overall.recent.average_attempt_number == pytest.approx(2.5)
    assert report.overall.recent.retry_rate == pytest.approx(1.0)
    assert report.overall.retry_delta == pytest.approx(0.5)


def test_strategy_comparisons_have_stable_order() -> None:
    report = RoutingFeedbackTrendAnalyzer(FeedbackTrendPolicy(window_size=2)).analyze(
        RoutingFeedbackHistory()
    )

    assert tuple(comparison.strategy for comparison in report.strategies) == (
        "dense",
        "sparse",
        "hybrid",
        "reranked",
    )


@pytest.mark.parametrize(
    "strategy",
    ["dense", "sparse", "hybrid", "reranked"],
)
def test_strategy_accessor_returns_requested_comparison(
    strategy: RetrievalStrategy,
) -> None:
    report = RoutingFeedbackTrendAnalyzer(FeedbackTrendPolicy(window_size=2)).analyze(
        RoutingFeedbackHistory()
    )

    assert report.strategy(strategy).strategy == strategy


def test_strategy_requires_minimum_evidence_in_each_global_window() -> None:
    observations = [
        _observation(strategy="dense"),
        _observation(strategy="sparse"),
        _observation(strategy="dense"),
        _observation(strategy="sparse"),
    ]
    report = RoutingFeedbackTrendAnalyzer(
        FeedbackTrendPolicy(
            window_size=2,
            minimum_strategy_observations=2,
        )
    ).analyze(_history(observations))

    assert report.overall.direction == "stable"
    assert report.strategy("dense").direction == "insufficient_data"
    assert report.strategy("dense").previous.observation_count == 1
    assert report.strategy("dense").recent.observation_count == 1


def test_strategy_trend_uses_only_matching_observations() -> None:
    observations = [
        _observation(
            strategy="dense",
            accepted=False,
            quality_score=0.0,
        ),
        _observation(strategy="sparse"),
        _observation(strategy="dense"),
        _observation(strategy="sparse"),
    ]
    report = RoutingFeedbackTrendAnalyzer(
        FeedbackTrendPolicy(
            window_size=2,
            minimum_strategy_observations=1,
        )
    ).analyze(_history(observations))

    dense = report.strategy("dense")
    sparse_metrics = report.strategy("sparse")

    assert dense.direction == "improving"
    assert dense.utility_delta == pytest.approx(1.0)
    assert sparse_metrics.direction == "stable"
    assert sparse_metrics.utility_delta == pytest.approx(0.0)


def test_revision_validation_is_delegated_to_feedback_analytics() -> None:
    with pytest.raises(ValueError, match="revision must not be negative"):
        RoutingFeedbackTrendAnalyzer().analyze(
            RoutingFeedbackHistory(),
            revision=-1,
        )


def test_report_models_are_immutable() -> None:
    report = RoutingFeedbackTrendAnalyzer(FeedbackTrendPolicy(window_size=2)).analyze(
        _history([_observation() for _ in range(4)])
    )

    with pytest.raises(FrozenInstanceError):
        report.revision = 2  # type: ignore[misc]

    with pytest.raises(FrozenInstanceError):
        report.overall.direction = "regressing"  # type: ignore[misc]


def test_repeated_analysis_is_deterministic() -> None:
    history = _history(
        [
            _observation(accepted=False, quality_score=0.2),
            _observation(accepted=False, quality_score=0.3),
            _observation(quality_score=0.8),
            _observation(quality_score=0.9),
        ]
    )
    analyzer = RoutingFeedbackTrendAnalyzer(FeedbackTrendPolicy(window_size=2))

    assert analyzer.analyze(history, revision=5) == analyzer.analyze(
        history,
        revision=5,
    )
