from __future__ import annotations

from dataclasses import dataclass

from fleetmind_rag.feedback_routing import (
    FeedbackFeature,
    RoutingFeedbackHistory,
    RoutingFeedbackObservation,
)
from fleetmind_rag.routing import RetrievalStrategy

_STRATEGY_ORDER: tuple[RetrievalStrategy, ...] = (
    "dense",
    "sparse",
    "hybrid",
    "reranked",
)
_FEATURE_ORDER: tuple[FeedbackFeature, ...] = (
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


@dataclass(frozen=True, slots=True)
class StrategyFeedbackAnalytics:
    """Deterministic outcome metrics for one retrieval strategy."""

    strategy: RetrievalStrategy
    observation_count: int
    accepted_count: int
    rewrite_count: int
    retry_attempt_count: int
    acceptance_rate: float | None
    rewrite_rate: float | None
    average_quality_score: float | None
    average_attempt_number: float | None
    retry_rate: float | None


@dataclass(frozen=True, slots=True)
class FeatureFeedbackAnalytics:
    """Deterministic outcome metrics for one query-signal feature."""

    feature: FeedbackFeature
    observation_count: int
    accepted_count: int
    rewrite_count: int
    retry_attempt_count: int
    acceptance_rate: float | None
    rewrite_rate: float | None
    average_quality_score: float | None
    average_attempt_number: float | None
    retry_rate: float | None


@dataclass(frozen=True, slots=True)
class RoutingFeedbackReport:
    """Immutable aggregate report for one persisted feedback snapshot."""

    revision: int
    observation_count: int
    accepted_count: int
    rewrite_count: int
    retry_attempt_count: int
    acceptance_rate: float | None
    rewrite_rate: float | None
    average_quality_score: float | None
    average_attempt_number: float | None
    retry_rate: float | None
    strategies: tuple[StrategyFeedbackAnalytics, ...]
    features: tuple[FeatureFeedbackAnalytics, ...]

    @property
    def is_empty(self) -> bool:
        """Return whether the analyzed history contained no observations."""

        return self.observation_count == 0

    def strategy(
        self,
        strategy: RetrievalStrategy,
    ) -> StrategyFeedbackAnalytics:
        """Return metrics for one known retrieval strategy."""

        return next(
            metrics for metrics in self.strategies if metrics.strategy == strategy
        )

    def feature(
        self,
        feature: FeedbackFeature,
    ) -> FeatureFeedbackAnalytics:
        """Return metrics for one known query-signal feature."""

        return next(metrics for metrics in self.features if metrics.feature == feature)


class RoutingFeedbackAnalyzer:
    """Aggregate immutable routing observations without external services."""

    def analyze(
        self,
        history: RoutingFeedbackHistory,
        *,
        revision: int = 0,
    ) -> RoutingFeedbackReport:
        """Build one deterministic report from an immutable history."""

        if isinstance(revision, bool) or not isinstance(revision, int):
            raise TypeError("revision must be an integer")

        if revision < 0:
            raise ValueError("revision must not be negative")

        observations = history.observations
        overall = _aggregate(observations)
        strategies = tuple(
            _strategy_analytics(
                strategy,
                tuple(
                    observation
                    for observation in observations
                    if observation.strategy == strategy
                ),
            )
            for strategy in _STRATEGY_ORDER
        )
        features = tuple(
            _feature_analytics(
                feature,
                tuple(
                    observation
                    for observation in observations
                    if feature in observation.features
                ),
            )
            for feature in _FEATURE_ORDER
        )

        return RoutingFeedbackReport(
            revision=revision,
            observation_count=overall.observation_count,
            accepted_count=overall.accepted_count,
            rewrite_count=overall.rewrite_count,
            retry_attempt_count=overall.retry_attempt_count,
            acceptance_rate=overall.acceptance_rate,
            rewrite_rate=overall.rewrite_rate,
            average_quality_score=overall.average_quality_score,
            average_attempt_number=overall.average_attempt_number,
            retry_rate=overall.retry_rate,
            strategies=strategies,
            features=features,
        )


@dataclass(frozen=True, slots=True)
class _Aggregate:
    observation_count: int
    accepted_count: int
    rewrite_count: int
    retry_attempt_count: int
    acceptance_rate: float | None
    rewrite_rate: float | None
    average_quality_score: float | None
    average_attempt_number: float | None
    retry_rate: float | None


def _aggregate(
    observations: tuple[RoutingFeedbackObservation, ...],
) -> _Aggregate:
    observation_count = len(observations)

    if observation_count == 0:
        return _Aggregate(
            observation_count=0,
            accepted_count=0,
            rewrite_count=0,
            retry_attempt_count=0,
            acceptance_rate=None,
            rewrite_rate=None,
            average_quality_score=None,
            average_attempt_number=None,
            retry_rate=None,
        )

    accepted_count = sum(observation.accepted for observation in observations)
    rewrite_count = observation_count - accepted_count
    retry_attempt_count = sum(
        observation.attempt_number > 1 for observation in observations
    )

    return _Aggregate(
        observation_count=observation_count,
        accepted_count=accepted_count,
        rewrite_count=rewrite_count,
        retry_attempt_count=retry_attempt_count,
        acceptance_rate=accepted_count / observation_count,
        rewrite_rate=rewrite_count / observation_count,
        average_quality_score=(
            sum(observation.quality_score for observation in observations)
            / observation_count
        ),
        average_attempt_number=(
            sum(observation.attempt_number for observation in observations)
            / observation_count
        ),
        retry_rate=retry_attempt_count / observation_count,
    )


def _strategy_analytics(
    strategy: RetrievalStrategy,
    observations: tuple[RoutingFeedbackObservation, ...],
) -> StrategyFeedbackAnalytics:
    aggregate = _aggregate(observations)
    return StrategyFeedbackAnalytics(
        strategy=strategy,
        observation_count=aggregate.observation_count,
        accepted_count=aggregate.accepted_count,
        rewrite_count=aggregate.rewrite_count,
        retry_attempt_count=aggregate.retry_attempt_count,
        acceptance_rate=aggregate.acceptance_rate,
        rewrite_rate=aggregate.rewrite_rate,
        average_quality_score=aggregate.average_quality_score,
        average_attempt_number=aggregate.average_attempt_number,
        retry_rate=aggregate.retry_rate,
    )


def _feature_analytics(
    feature: FeedbackFeature,
    observations: tuple[RoutingFeedbackObservation, ...],
) -> FeatureFeedbackAnalytics:
    aggregate = _aggregate(observations)
    return FeatureFeedbackAnalytics(
        feature=feature,
        observation_count=aggregate.observation_count,
        accepted_count=aggregate.accepted_count,
        rewrite_count=aggregate.rewrite_count,
        retry_attempt_count=aggregate.retry_attempt_count,
        acceptance_rate=aggregate.acceptance_rate,
        rewrite_rate=aggregate.rewrite_rate,
        average_quality_score=aggregate.average_quality_score,
        average_attempt_number=aggregate.average_attempt_number,
        retry_rate=aggregate.retry_rate,
    )
