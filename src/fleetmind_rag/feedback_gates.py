from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from fleetmind_rag.feedback_trends import (
    RoutingFeedbackTrendReport,
    TrendDirection,
)
from fleetmind_rag.routing import RetrievalStrategy

GateStatus = Literal["pass", "warn", "fail"]
GateEnforcement = Literal["warn", "fail", "never"]

PASS_EXIT_CODE = 0
WARN_EXIT_CODE = 2
FAIL_EXIT_CODE = 3


@dataclass(frozen=True, slots=True)
class FeedbackRegressionGatePolicy:
    """Controls which measured trend conditions affect gate status."""

    fail_on_overall_regression: bool = True
    fail_on_strategy_regression: bool = True
    warn_on_insufficient_overall_data: bool = True
    warn_on_insufficient_strategy_data: bool = True

    def __post_init__(self) -> None:
        """Reject non-boolean policy values before evaluating evidence."""

        values = {
            "fail_on_overall_regression": self.fail_on_overall_regression,
            "fail_on_strategy_regression": self.fail_on_strategy_regression,
            "warn_on_insufficient_overall_data": (
                self.warn_on_insufficient_overall_data
            ),
            "warn_on_insufficient_strategy_data": (
                self.warn_on_insufficient_strategy_data
            ),
        }

        for name, value in values.items():
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a boolean")


@dataclass(frozen=True, slots=True)
class FeedbackRegressionGateResult:
    """One explainable operational decision derived from a trend report."""

    status: GateStatus
    recommended_exit_code: int
    overall_direction: TrendDirection
    regressing_strategies: tuple[RetrievalStrategy, ...]
    insufficient_strategies: tuple[RetrievalStrategy, ...]
    reasons: tuple[str, ...]
    trend_report: RoutingFeedbackTrendReport

    @property
    def passed(self) -> bool:
        """Return whether no warning or failure condition was found."""

        return self.status == "pass"

    def process_exit_code(
        self,
        enforcement: GateEnforcement = "fail",
    ) -> int:
        """Map gate status to a process exit code for one enforcement mode."""

        if enforcement not in {"warn", "fail", "never"}:
            raise ValueError(f"unsupported gate enforcement: {enforcement!r}")

        if enforcement == "never":
            return PASS_EXIT_CODE

        if self.status == "fail":
            return FAIL_EXIT_CODE

        if self.status == "warn" and enforcement == "warn":
            return WARN_EXIT_CODE

        return PASS_EXIT_CODE


class FeedbackRegressionGate:
    """Convert explainable trend evidence into pass, warn, or fail."""

    def __init__(
        self,
        policy: FeedbackRegressionGatePolicy | None = None,
    ) -> None:
        """Initialize the gate with immutable operational policy."""

        self._policy = policy or FeedbackRegressionGatePolicy()

    @property
    def policy(self) -> FeedbackRegressionGatePolicy:
        """Return the immutable gate policy."""

        return self._policy

    def evaluate(
        self,
        report: RoutingFeedbackTrendReport,
    ) -> FeedbackRegressionGateResult:
        """Evaluate overall and strategy trends without changing feedback."""

        regressing_strategies = tuple(
            comparison.strategy
            for comparison in report.strategies
            if comparison.strategy is not None and comparison.direction == "regressing"
        )
        insufficient_strategies = tuple(
            comparison.strategy
            for comparison in report.strategies
            if comparison.strategy is not None
            and comparison.direction == "insufficient_data"
            and (
                comparison.previous.observation_count > 0
                or comparison.recent.observation_count > 0
            )
        )
        failure_reasons: list[str] = []
        warning_reasons: list[str] = []

        if (
            self._policy.fail_on_overall_regression
            and report.overall.direction == "regressing"
        ):
            failure_reasons.append("Overall routing-feedback utility is regressing.")

        if self._policy.fail_on_strategy_regression and regressing_strategies:
            failure_reasons.append(
                f"Regressing retrieval strategies: {', '.join(regressing_strategies)}."
            )

        if (
            self._policy.warn_on_insufficient_overall_data
            and report.overall.direction == "insufficient_data"
        ):
            warning_reasons.append(
                "Overall routing-feedback history has insufficient data."
            )

        if self._policy.warn_on_insufficient_strategy_data and insufficient_strategies:
            warning_reasons.append(
                "Insufficient observed strategy evidence: "
                f"{', '.join(insufficient_strategies)}."
            )

        if failure_reasons:
            status: GateStatus = "fail"
            reasons = (*failure_reasons, *warning_reasons)
            recommended_exit_code = FAIL_EXIT_CODE
        elif warning_reasons:
            status = "warn"
            reasons = tuple(warning_reasons)
            recommended_exit_code = WARN_EXIT_CODE
        else:
            status = "pass"
            reasons = ("No configured routing-feedback regression was detected.",)
            recommended_exit_code = PASS_EXIT_CODE

        return FeedbackRegressionGateResult(
            status=status,
            recommended_exit_code=recommended_exit_code,
            overall_direction=report.overall.direction,
            regressing_strategies=regressing_strategies,
            insufficient_strategies=insufficient_strategies,
            reasons=reasons,
            trend_report=report,
        )
