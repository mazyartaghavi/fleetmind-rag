from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal

from fleetmind_rag.adaptive_retrieval import AdaptiveRetrievalOutcome
from fleetmind_rag.routing import (
    QueryRoutingSignals,
    RetrievalStrategy,
    RetrievalStrategyRouter,
    RoutingConfidence,
    RoutingDecision,
    StrategyScore,
)

FeedbackFeature = Literal[
    "exact_identifier",
    "quoted_phrase",
    "conceptual",
    "conditional",
    "action",
    "safety",
    "domain",
    "complex",
    "general",
]
FeedbackGuardrail = Literal[
    "safety_sensitive",
    "exact_identifier",
    "quoted_phrase",
]

_STRATEGIES: tuple[RetrievalStrategy, ...] = (
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
class RoutingFeedbackObservation:
    """One quality-labelled routing attempt used as feedback evidence."""

    query: str
    strategy: RetrievalStrategy
    verdict: Literal["accept", "rewrite"]
    quality_score: float
    attempt_number: int
    features: tuple[FeedbackFeature, ...]

    def __post_init__(self) -> None:
        """Validate one immutable observation."""

        normalized_query = _normalize_required_text(self.query, field="query")
        object.__setattr__(self, "query", normalized_query)

        if self.strategy not in _STRATEGIES:
            raise ValueError(f"unsupported retrieval strategy: {self.strategy!r}")

        if self.verdict not in {"accept", "rewrite"}:
            raise ValueError(f"unsupported feedback verdict: {self.verdict!r}")

        if not math.isfinite(self.quality_score):
            raise ValueError("quality_score must be finite")

        if not 0.0 <= self.quality_score <= 1.0:
            raise ValueError("quality_score must be between 0.0 and 1.0")

        if self.attempt_number <= 0:
            raise ValueError("attempt_number must be greater than zero")

        normalized_features = _normalize_features(self.features)
        object.__setattr__(self, "features", normalized_features)

    @property
    def accepted(self) -> bool:
        """Return whether quality checking accepted this attempt."""

        return self.verdict == "accept"


@dataclass(frozen=True, slots=True)
class RoutingFeedbackHistory:
    """Immutable ordered collection of routing feedback observations."""

    observations: tuple[RoutingFeedbackObservation, ...] = ()

    def record(
        self,
        observation: RoutingFeedbackObservation,
    ) -> RoutingFeedbackHistory:
        """Return a new history containing one additional observation."""

        return RoutingFeedbackHistory((*self.observations, observation))

    def record_outcome(
        self,
        outcome: AdaptiveRetrievalOutcome,
    ) -> RoutingFeedbackHistory:
        """Return a new history containing every attempt in one outcome."""

        attempts = outcome.state.attempts
        assessments = outcome.assessments

        if len(attempts) != len(assessments):
            raise ValueError(
                "adaptive outcome must contain one quality assessment per attempt"
            )

        history = self

        for attempt, assessment in zip(attempts, assessments, strict=True):
            decision = attempt.result.decision

            if assessment.query != decision.query:
                raise ValueError(
                    "quality assessment query must match its routing decision"
                )

            if assessment.strategy != decision.strategy:
                raise ValueError(
                    "quality assessment strategy must match its routing decision"
                )

            history = history.record(
                RoutingFeedbackObservation(
                    query=decision.query,
                    strategy=decision.strategy,
                    verdict=assessment.verdict,
                    quality_score=assessment.quality_score,
                    attempt_number=attempt.number,
                    features=query_signal_profile(decision.signals),
                )
            )

        return history

    def matching(
        self,
        features: tuple[FeedbackFeature, ...],
    ) -> tuple[RoutingFeedbackObservation, ...]:
        """Return observations with the same normalized signal profile."""

        normalized_features = _normalize_features(features)
        return tuple(
            observation
            for observation in self.observations
            if observation.features == normalized_features
        )


@dataclass(frozen=True, slots=True)
class FeedbackRoutingPolicy:
    """Bounds and evidence thresholds for feedback adjustments."""

    minimum_observations: int = 3
    maximum_score_adjustment: int = 4
    strong_positive_threshold: float = 0.80
    positive_threshold: float = 0.65
    negative_threshold: float = 0.35
    strong_negative_threshold: float = 0.20

    def __post_init__(self) -> None:
        """Validate policy thresholds and adjustment bounds."""

        if self.minimum_observations <= 0:
            raise ValueError("minimum_observations must be greater than zero")

        if self.maximum_score_adjustment <= 0:
            raise ValueError("maximum_score_adjustment must be greater than zero")

        thresholds = (
            self.strong_negative_threshold,
            self.negative_threshold,
            self.positive_threshold,
            self.strong_positive_threshold,
        )

        if any(not 0.0 <= threshold <= 1.0 for threshold in thresholds):
            raise ValueError("feedback thresholds must be between 0.0 and 1.0")

        if thresholds != tuple(sorted(thresholds)):
            raise ValueError(
                "feedback thresholds must be ordered from negative to positive"
            )


@dataclass(frozen=True, slots=True)
class StrategyFeedbackStatistics:
    """Aggregated feedback evidence for one retrieval strategy."""

    strategy: RetrievalStrategy
    observation_count: int
    accepted_count: int
    acceptance_rate: float | None
    average_quality_score: float | None
    utility_score: float | None


@dataclass(frozen=True, slots=True)
class StrategyFeedbackAdjustment:
    """One bounded score adjustment with its supporting statistics."""

    strategy: RetrievalStrategy
    base_score: int
    adjustment: int
    adjusted_score: int
    statistics: StrategyFeedbackStatistics
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FeedbackRoutingDecision:
    """Base and feedback-adjusted decisions with complete evidence."""

    base_decision: RoutingDecision
    decision: RoutingDecision
    features: tuple[FeedbackFeature, ...]
    adjustments: tuple[StrategyFeedbackAdjustment, ...]
    feedback_applied: bool
    strategy_changed: bool
    guardrail: FeedbackGuardrail | None
    reason: str

    @property
    def strategy(self) -> RetrievalStrategy:
        """Return the final retrieval strategy."""

        return self.decision.strategy


class FeedbackDrivenRetrievalRouter(RetrievalStrategyRouter):
    """Apply bounded historical feedback to explainable base routing."""

    def __init__(
        self,
        history: RoutingFeedbackHistory | None = None,
        *,
        policy: FeedbackRoutingPolicy | None = None,
        base_router: RetrievalStrategyRouter | None = None,
    ) -> None:
        """Initialize the router with immutable history and policy."""

        self._history = history or RoutingFeedbackHistory()
        self._policy = policy or FeedbackRoutingPolicy()
        self._base_router = base_router or RetrievalStrategyRouter()

    @property
    def history(self) -> RoutingFeedbackHistory:
        """Return the immutable feedback history."""

        return self._history

    @property
    def policy(self) -> FeedbackRoutingPolicy:
        """Return the immutable feedback policy."""

        return self._policy

    def with_outcome(
        self,
        outcome: AdaptiveRetrievalOutcome,
    ) -> FeedbackDrivenRetrievalRouter:
        """Return a new router trained on one additional outcome."""

        return FeedbackDrivenRetrievalRouter(
            self._history.record_outcome(outcome),
            policy=self._policy,
            base_router=self._base_router,
        )

    def route(self, query: str) -> RoutingDecision:
        """Return the standard routing decision used by retrieval executors."""

        return self.explain(query).decision

    def explain(self, query: str) -> FeedbackRoutingDecision:
        """Return a feedback-adjusted decision with transparent evidence."""

        base_decision = self._base_router.route(query)
        features = query_signal_profile(base_decision.signals)
        matching_observations = self._history.matching(features)
        base_scores = {score.strategy: score for score in base_decision.scores}
        adjustments = tuple(
            self._build_adjustment(
                strategy,
                base_scores[strategy],
                matching_observations,
            )
            for strategy in _STRATEGIES
        )
        feedback_applied = any(adjustment.adjustment != 0 for adjustment in adjustments)
        guardrail, guardrail_strategy = _guardrail_for(base_decision.signals)

        if guardrail_strategy is None:
            selected_adjustment = _select_adjustment(
                adjustments,
                base_strategy=base_decision.strategy,
            )
            selected_strategy = selected_adjustment.strategy
            final_scores = _feedback_scores(
                base_decision.scores,
                adjustments,
            )
            selected_score = selected_adjustment.adjusted_score
        else:
            selected_strategy = guardrail_strategy
            final_scores = _guardrail_scores(
                _feedback_scores(base_decision.scores, adjustments),
                guardrail_strategy,
                guardrail,
            )
            selected_score = next(
                score.score
                for score in final_scores
                if score.strategy == selected_strategy
            )

        strategy_changed = selected_strategy != base_decision.strategy
        reason = _feedback_reason(
            base_decision,
            selected_strategy=selected_strategy,
            matching_observation_count=len(matching_observations),
            feedback_applied=feedback_applied,
            guardrail=guardrail,
        )
        confidence: RoutingConfidence = (
            "high"
            if guardrail is not None
            else "medium"
            if strategy_changed
            else base_decision.confidence
        )
        decision = replace(
            base_decision,
            strategy=selected_strategy,
            confidence=confidence,
            selected_score=selected_score,
            reason=reason,
            scores=final_scores,
        )

        return FeedbackRoutingDecision(
            base_decision=base_decision,
            decision=decision,
            features=features,
            adjustments=adjustments,
            feedback_applied=feedback_applied,
            strategy_changed=strategy_changed,
            guardrail=guardrail,
            reason=reason,
        )

    def _build_adjustment(
        self,
        strategy: RetrievalStrategy,
        base_score: StrategyScore,
        observations: tuple[RoutingFeedbackObservation, ...],
    ) -> StrategyFeedbackAdjustment:
        strategy_observations = tuple(
            observation
            for observation in observations
            if observation.strategy == strategy
        )
        statistics = _strategy_statistics(strategy, strategy_observations)

        if statistics.observation_count < self._policy.minimum_observations:
            adjustment = 0
            reasons = (
                "Insufficient matching feedback: "
                f"{statistics.observation_count} observed, "
                f"{self._policy.minimum_observations} required.",
            )
        else:
            adjustment = _utility_adjustment(
                statistics.utility_score,
                self._policy,
            )
            reasons = (
                "Matching feedback produced utility "
                f"{statistics.utility_score:.4f} from "
                f"{statistics.observation_count} observations.",
            )

        return StrategyFeedbackAdjustment(
            strategy=strategy,
            base_score=base_score.score,
            adjustment=adjustment,
            adjusted_score=base_score.score + adjustment,
            statistics=statistics,
            reasons=reasons,
        )


def query_signal_profile(
    signals: QueryRoutingSignals,
) -> tuple[FeedbackFeature, ...]:
    """Convert detailed query signals into a stable feedback profile."""

    features: list[FeedbackFeature] = []

    if signals.exact_identifiers:
        features.append("exact_identifier")

    if signals.quoted_phrases:
        features.append("quoted_phrase")

    if signals.conceptual_cues:
        features.append("conceptual")

    if signals.conditional_cues:
        features.append("conditional")

    if signals.action_cues:
        features.append("action")

    if signals.safety_cues:
        features.append("safety")

    if signals.domain_cues:
        features.append("domain")

    if signals.is_complex:
        features.append("complex")

    if not features:
        features.append("general")

    return _normalize_features(tuple(features))


def _strategy_statistics(
    strategy: RetrievalStrategy,
    observations: tuple[RoutingFeedbackObservation, ...],
) -> StrategyFeedbackStatistics:
    observation_count = len(observations)

    if observation_count == 0:
        return StrategyFeedbackStatistics(
            strategy=strategy,
            observation_count=0,
            accepted_count=0,
            acceptance_rate=None,
            average_quality_score=None,
            utility_score=None,
        )

    accepted_count = sum(observation.accepted for observation in observations)
    acceptance_rate = accepted_count / observation_count
    average_quality_score = (
        sum(observation.quality_score for observation in observations)
        / observation_count
    )
    utility_score = 0.60 * acceptance_rate + 0.40 * average_quality_score

    return StrategyFeedbackStatistics(
        strategy=strategy,
        observation_count=observation_count,
        accepted_count=accepted_count,
        acceptance_rate=acceptance_rate,
        average_quality_score=average_quality_score,
        utility_score=utility_score,
    )


def _utility_adjustment(
    utility_score: float | None,
    policy: FeedbackRoutingPolicy,
) -> int:
    if utility_score is None:
        return 0

    maximum = policy.maximum_score_adjustment
    moderate = max(1, maximum // 2)

    if utility_score >= policy.strong_positive_threshold:
        return maximum

    if utility_score >= policy.positive_threshold:
        return moderate

    if utility_score <= policy.strong_negative_threshold:
        return -maximum

    if utility_score <= policy.negative_threshold:
        return -moderate

    return 0


def _select_adjustment(
    adjustments: tuple[StrategyFeedbackAdjustment, ...],
    *,
    base_strategy: RetrievalStrategy,
) -> StrategyFeedbackAdjustment:
    strategy_order = {strategy: index for index, strategy in enumerate(_STRATEGIES)}
    return max(
        adjustments,
        key=lambda adjustment: (
            adjustment.adjusted_score,
            adjustment.strategy == base_strategy,
            -strategy_order[adjustment.strategy],
        ),
    )


def _feedback_scores(
    base_scores: tuple[StrategyScore, ...],
    adjustments: tuple[StrategyFeedbackAdjustment, ...],
) -> tuple[StrategyScore, ...]:
    adjustment_by_strategy = {
        adjustment.strategy: adjustment for adjustment in adjustments
    }
    return tuple(
        StrategyScore(
            strategy=score.strategy,
            score=adjustment_by_strategy[score.strategy].adjusted_score,
            reasons=(
                *score.reasons,
                *adjustment_by_strategy[score.strategy].reasons,
            ),
        )
        for score in base_scores
    )


def _guardrail_scores(
    scores: tuple[StrategyScore, ...],
    guarded_strategy: RetrievalStrategy,
    guardrail: FeedbackGuardrail | None,
) -> tuple[StrategyScore, ...]:
    maximum_score = max(score.score for score in scores)
    guarded_score = maximum_score + 1
    guardrail_reason = (
        "Safety-sensitive routing requires reranked retrieval."
        if guardrail == "safety_sensitive"
        else "Exact constraints require sparse retrieval."
    )
    return tuple(
        StrategyScore(
            strategy=score.strategy,
            score=guarded_score if score.strategy == guarded_strategy else score.score,
            reasons=(
                *score.reasons,
                *((guardrail_reason,) if score.strategy == guarded_strategy else ()),
            ),
        )
        for score in scores
    )


def _guardrail_for(
    signals: QueryRoutingSignals,
) -> tuple[FeedbackGuardrail | None, RetrievalStrategy | None]:
    if signals.safety_cues and (
        signals.conditional_cues or signals.action_cues or signals.is_complex
    ):
        return "safety_sensitive", "reranked"

    if signals.exact_identifiers:
        return "exact_identifier", "sparse"

    if signals.quoted_phrases:
        return "quoted_phrase", "sparse"

    return None, None


def _feedback_reason(
    base_decision: RoutingDecision,
    *,
    selected_strategy: RetrievalStrategy,
    matching_observation_count: int,
    feedback_applied: bool,
    guardrail: FeedbackGuardrail | None,
) -> str:
    if guardrail == "safety_sensitive":
        detail = "Safety guardrail selected reranked retrieval."
    elif guardrail in {"exact_identifier", "quoted_phrase"}:
        detail = "Exact-constraint guardrail selected sparse retrieval."
    elif feedback_applied:
        detail = (
            f"Bounded feedback from {matching_observation_count} matching "
            f"observations selected {selected_strategy} retrieval."
        )
    else:
        detail = "No strategy had enough matching feedback to change the base decision."

    return f"{base_decision.reason} {detail}"


def _normalize_features(
    features: tuple[FeedbackFeature, ...],
) -> tuple[FeedbackFeature, ...]:
    if not features:
        raise ValueError("features must contain at least one feedback feature")

    unknown = set(features).difference(_FEATURE_ORDER)

    if unknown:
        raise ValueError(f"unsupported feedback features: {sorted(unknown)!r}")

    unique_features = tuple(
        feature for feature in _FEATURE_ORDER if feature in features
    )

    if "general" in unique_features and len(unique_features) > 1:
        raise ValueError("general cannot be combined with specialized features")

    return unique_features


def _normalize_required_text(value: str, *, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")

    normalized = " ".join(value.split())

    if not normalized:
        raise ValueError(f"{field} must not be blank")

    return normalized
