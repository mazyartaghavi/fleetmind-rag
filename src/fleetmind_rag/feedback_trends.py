from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from fleetmind_rag.feedback_analytics import RoutingFeedbackAnalyzer
from fleetmind_rag.feedback_routing import (
    RoutingFeedbackHistory,
    RoutingFeedbackObservation,
)
from fleetmind_rag.routing import RetrievalStrategy

TrendDirection = Literal[
    "improving",
    "stable",
    "regressing",
    "insufficient_data",
]

_STRATEGIES: tuple[RetrievalStrategy, ...] = (
    "dense",
    "sparse",
    "hybrid",
    "reranked",
)


@dataclass(frozen=True, slots=True)
class FeedbackTrendPolicy:
    """Controls for comparing adjacent chronological feedback windows."""

    window_size: int = 10
    minimum_utility_change: float = 0.05
    minimum_strategy_observations: int = 2

    def __post_init__(self) -> None:
        """Validate deterministic trend thresholds and evidence requirements."""

        if isinstance(self.window_size, bool) or not isinstance(
            self.window_size,
            int,
        ):
            raise TypeError("window_size must be an integer")

        if self.window_size <= 0:
            raise ValueError("window_size must be greater than zero")

        if isinstance(self.minimum_strategy_observations, bool) or not isinstance(
            self.minimum_strategy_observations,
            int,
        ):
            raise TypeError("minimum_strategy_observations must be an integer")

        if self.minimum_strategy_observations <= 0:
            raise ValueError("minimum_strategy_observations must be greater than zero")

        if self.minimum_strategy_observations > self.window_size:
            raise ValueError(
                "minimum_strategy_observations must not exceed window_size"
            )

        if not math.isfinite(self.minimum_utility_change):
            raise ValueError("minimum_utility_change must be finite")

        if not 0.0 < self.minimum_utility_change <= 1.0:
            raise ValueError(
                "minimum_utility_change must be greater than zero and at most one"
            )


@dataclass(frozen=True, slots=True)
class TrendWindowMetrics:
    """Quality and workflow metrics for one chronological observation window."""

    observation_count: int
    accepted_count: int
    rewrite_count: int
    retry_attempt_count: int
    acceptance_rate: float | None
    rewrite_rate: float | None
    average_quality_score: float | None
    average_attempt_number: float | None
    retry_rate: float | None
    utility_score: float | None


@dataclass(frozen=True, slots=True)
class FeedbackTrendComparison:
    """Previous-versus-recent evidence and one explainable trend verdict."""

    strategy: RetrievalStrategy | None
    direction: TrendDirection
    previous: TrendWindowMetrics
    recent: TrendWindowMetrics
    utility_delta: float | None
    acceptance_delta: float | None
    quality_delta: float | None
    rewrite_delta: float | None
    retry_delta: float | None
    reason: str

    @property
    def sufficient_data(self) -> bool:
        """Return whether the comparison produced a measured trend."""

        return self.direction != "insufficient_data"


@dataclass(frozen=True, slots=True)
class RoutingFeedbackTrendReport:
    """Immutable chronological trend report for one feedback snapshot."""

    revision: int
    total_observations: int
    policy: FeedbackTrendPolicy
    previous_start_position: int | None
    previous_end_position: int | None
    recent_start_position: int | None
    recent_end_position: int | None
    overall: FeedbackTrendComparison
    strategies: tuple[FeedbackTrendComparison, ...]

    def strategy(
        self,
        strategy: RetrievalStrategy,
    ) -> FeedbackTrendComparison:
        """Return the comparison for one retrieval strategy."""

        return next(
            comparison
            for comparison in self.strategies
            if comparison.strategy == strategy
        )


class RoutingFeedbackTrendAnalyzer:
    """Compare the latest two ordered feedback windows deterministically."""

    def __init__(
        self,
        policy: FeedbackTrendPolicy | None = None,
    ) -> None:
        """Initialize the analyzer with immutable comparison controls."""

        self._policy = policy or FeedbackTrendPolicy()
        self._analytics = RoutingFeedbackAnalyzer()

    @property
    def policy(self) -> FeedbackTrendPolicy:
        """Return the immutable trend policy."""

        return self._policy

    def analyze(
        self,
        history: RoutingFeedbackHistory,
        *,
        revision: int = 0,
    ) -> RoutingFeedbackTrendReport:
        """Compare recent observations with the immediately previous window."""

        self._analytics.analyze(RoutingFeedbackHistory(), revision=revision)
        observations = history.observations
        window_size = self._policy.window_size
        previous_observations = observations[-2 * window_size : -window_size]
        recent_observations = observations[-window_size:]
        total_observations = len(observations)
        previous_positions = _positions(
            total_observations,
            len(previous_observations),
            offset_from_end=len(recent_observations),
        )
        recent_positions = _positions(
            total_observations,
            len(recent_observations),
            offset_from_end=0,
        )
        overall = self._compare(
            previous_observations,
            recent_observations,
            strategy=None,
            required_observations=window_size,
        )
        strategies = tuple(
            self._compare(
                tuple(
                    observation
                    for observation in previous_observations
                    if observation.strategy == strategy
                ),
                tuple(
                    observation
                    for observation in recent_observations
                    if observation.strategy == strategy
                ),
                strategy=strategy,
                required_observations=self._policy.minimum_strategy_observations,
            )
            for strategy in _STRATEGIES
        )

        return RoutingFeedbackTrendReport(
            revision=revision,
            total_observations=total_observations,
            policy=self._policy,
            previous_start_position=previous_positions[0],
            previous_end_position=previous_positions[1],
            recent_start_position=recent_positions[0],
            recent_end_position=recent_positions[1],
            overall=overall,
            strategies=strategies,
        )

    def _compare(
        self,
        previous_observations: tuple[RoutingFeedbackObservation, ...],
        recent_observations: tuple[RoutingFeedbackObservation, ...],
        *,
        strategy: RetrievalStrategy | None,
        required_observations: int,
    ) -> FeedbackTrendComparison:
        previous = _window_metrics(previous_observations)
        recent = _window_metrics(recent_observations)
        scope = "overall history" if strategy is None else f"{strategy} retrieval"

        if (
            previous.observation_count < required_observations
            or recent.observation_count < required_observations
        ):
            return FeedbackTrendComparison(
                strategy=strategy,
                direction="insufficient_data",
                previous=previous,
                recent=recent,
                utility_delta=None,
                acceptance_delta=None,
                quality_delta=None,
                rewrite_delta=None,
                retry_delta=None,
                reason=(
                    f"Insufficient {scope} evidence: previous "
                    f"{previous.observation_count}, recent "
                    f"{recent.observation_count}, at least "
                    f"{required_observations} required in each window."
                ),
            )

        utility_delta = _required_delta(
            previous.utility_score,
            recent.utility_score,
        )
        acceptance_delta = _required_delta(
            previous.acceptance_rate,
            recent.acceptance_rate,
        )
        quality_delta = _required_delta(
            previous.average_quality_score,
            recent.average_quality_score,
        )
        rewrite_delta = _required_delta(
            previous.rewrite_rate,
            recent.rewrite_rate,
        )
        retry_delta = _required_delta(
            previous.retry_rate,
            recent.retry_rate,
        )
        threshold = self._policy.minimum_utility_change

        reaches_positive_threshold = utility_delta > threshold or math.isclose(
            utility_delta,
            threshold,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        reaches_negative_threshold = utility_delta < -threshold or math.isclose(
            utility_delta,
            -threshold,
            rel_tol=0.0,
            abs_tol=1e-12,
        )

        if reaches_positive_threshold:
            direction: TrendDirection = "improving"
        elif reaches_negative_threshold:
            direction = "regressing"
        else:
            direction = "stable"

        return FeedbackTrendComparison(
            strategy=strategy,
            direction=direction,
            previous=previous,
            recent=recent,
            utility_delta=utility_delta,
            acceptance_delta=acceptance_delta,
            quality_delta=quality_delta,
            rewrite_delta=rewrite_delta,
            retry_delta=retry_delta,
            reason=(
                f"{scope.capitalize()} utility changed by "
                f"{utility_delta:+.4f}; the configured trend threshold is "
                f"{threshold:.4f}."
            ),
        )


def _window_metrics(
    observations: tuple[RoutingFeedbackObservation, ...],
) -> TrendWindowMetrics:
    report = RoutingFeedbackAnalyzer().analyze(RoutingFeedbackHistory(observations))
    utility_score = (
        None
        if report.acceptance_rate is None or report.average_quality_score is None
        else 0.60 * report.acceptance_rate + 0.40 * report.average_quality_score
    )
    return TrendWindowMetrics(
        observation_count=report.observation_count,
        accepted_count=report.accepted_count,
        rewrite_count=report.rewrite_count,
        retry_attempt_count=report.retry_attempt_count,
        acceptance_rate=report.acceptance_rate,
        rewrite_rate=report.rewrite_rate,
        average_quality_score=report.average_quality_score,
        average_attempt_number=report.average_attempt_number,
        retry_rate=report.retry_rate,
        utility_score=utility_score,
    )


def _positions(
    total_observations: int,
    observation_count: int,
    *,
    offset_from_end: int,
) -> tuple[int | None, int | None]:
    if observation_count == 0:
        return None, None

    end = total_observations - offset_from_end
    start = end - observation_count + 1
    return start, end


def _required_delta(previous: float | None, recent: float | None) -> float:
    if previous is None or recent is None:
        raise RuntimeError("sufficient windows must contain defined metrics")

    return recent - previous
