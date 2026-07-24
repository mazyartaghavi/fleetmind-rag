from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from fleetmind_rag.feedback_gates import (
    FAIL_EXIT_CODE,
    PASS_EXIT_CODE,
    WARN_EXIT_CODE,
    FeedbackRegressionGate,
    FeedbackRegressionGatePolicy,
    GateEnforcement,
)
from fleetmind_rag.feedback_routing import (
    RoutingFeedbackHistory,
    RoutingFeedbackObservation,
)
from fleetmind_rag.feedback_trends import (
    FeedbackTrendPolicy,
    RoutingFeedbackTrendAnalyzer,
    RoutingFeedbackTrendReport,
)
from fleetmind_rag.routing import RetrievalStrategy


def _observation(
    *,
    strategy: RetrievalStrategy = "dense",
    accepted: bool = True,
    quality_score: float = 1.0,
) -> RoutingFeedbackObservation:
    return RoutingFeedbackObservation(
        query="What does overheating mean?",
        strategy=strategy,
        verdict="accept" if accepted else "rewrite",
        quality_score=quality_score,
        attempt_number=1,
        features=("conceptual",),
    )


def _report(
    observations: list[RoutingFeedbackObservation],
    *,
    window_size: int = 2,
    minimum_strategy_observations: int = 1,
) -> RoutingFeedbackTrendReport:
    return RoutingFeedbackTrendAnalyzer(
        FeedbackTrendPolicy(
            window_size=window_size,
            minimum_strategy_observations=minimum_strategy_observations,
        )
    ).analyze(RoutingFeedbackHistory(tuple(observations)), revision=5)


@pytest.mark.parametrize(
    "field",
    [
        "fail_on_overall_regression",
        "fail_on_strategy_regression",
        "warn_on_insufficient_overall_data",
        "warn_on_insufficient_strategy_data",
    ],
)
def test_policy_fields_must_be_boolean(field: str) -> None:
    values: dict[str, object] = {field: 1}

    with pytest.raises(TypeError, match=f"{field} must be a boolean"):
        FeedbackRegressionGatePolicy(**values)  # type: ignore[arg-type]


def test_gate_exposes_configured_policy() -> None:
    policy = FeedbackRegressionGatePolicy(
        fail_on_strategy_regression=False,
    )

    assert FeedbackRegressionGate(policy).policy is policy


def test_insufficient_overall_history_warns() -> None:
    result = FeedbackRegressionGate().evaluate(
        _report([_observation()]),
    )

    assert result.status == "warn"
    assert result.recommended_exit_code == WARN_EXIT_CODE
    assert result.overall_direction == "insufficient_data"
    assert result.regressing_strategies == ()
    assert result.insufficient_strategies == ("dense",)
    assert "insufficient data" in result.reasons[0]
    assert not result.passed


def test_unobserved_strategies_are_not_insufficient_warnings() -> None:
    result = FeedbackRegressionGate().evaluate(
        _report([_observation()]),
    )

    assert result.insufficient_strategies == ("dense",)
    assert "sparse" not in " ".join(result.reasons)
    assert "hybrid" not in " ".join(result.reasons)
    assert "reranked" not in " ".join(result.reasons)


def test_stable_history_passes() -> None:
    result = FeedbackRegressionGate().evaluate(
        _report([_observation() for _ in range(4)]),
    )

    assert result.status == "pass"
    assert result.recommended_exit_code == PASS_EXIT_CODE
    assert result.overall_direction == "stable"
    assert result.regressing_strategies == ()
    assert result.insufficient_strategies == ()
    assert result.reasons == (
        "No configured routing-feedback regression was detected.",
    )
    assert result.passed


def test_improving_history_passes() -> None:
    result = FeedbackRegressionGate().evaluate(
        _report(
            [
                _observation(accepted=False, quality_score=0.0),
                _observation(accepted=False, quality_score=0.0),
                _observation(),
                _observation(),
            ]
        ),
    )

    assert result.status == "pass"
    assert result.overall_direction == "improving"
    assert result.passed


def test_overall_regression_fails() -> None:
    result = FeedbackRegressionGate().evaluate(
        _report(
            [
                _observation(),
                _observation(),
                _observation(accepted=False, quality_score=0.0),
                _observation(accepted=False, quality_score=0.0),
            ]
        ),
    )

    assert result.status == "fail"
    assert result.recommended_exit_code == FAIL_EXIT_CODE
    assert result.overall_direction == "regressing"
    assert result.regressing_strategies == ("dense",)
    assert "Overall routing-feedback utility is regressing." in result.reasons
    assert "Regressing retrieval strategies: dense." in result.reasons


def test_strategy_regression_fails_when_overall_is_stable() -> None:
    observations = [
        _observation(strategy="dense"),
        _observation(strategy="dense"),
        _observation(strategy="sparse", accepted=False, quality_score=0.0),
        _observation(strategy="sparse", accepted=False, quality_score=0.0),
        _observation(strategy="dense", accepted=False, quality_score=0.0),
        _observation(strategy="dense", accepted=False, quality_score=0.0),
        _observation(strategy="sparse"),
        _observation(strategy="sparse"),
    ]
    result = FeedbackRegressionGate().evaluate(
        _report(
            observations,
            window_size=4,
            minimum_strategy_observations=2,
        )
    )

    assert result.overall_direction == "stable"
    assert result.regressing_strategies == ("dense",)
    assert result.status == "fail"
    assert result.reasons == ("Regressing retrieval strategies: dense.",)


def test_overall_regression_can_be_disabled() -> None:
    report = _report(
        [
            _observation(strategy="dense"),
            _observation(strategy="sparse"),
            _observation(
                strategy="dense",
                accepted=False,
                quality_score=0.0,
            ),
            _observation(
                strategy="sparse",
                accepted=False,
                quality_score=0.0,
            ),
        ],
        minimum_strategy_observations=2,
    )
    gate = FeedbackRegressionGate(
        FeedbackRegressionGatePolicy(
            fail_on_overall_regression=False,
        )
    )

    result = gate.evaluate(report)

    assert result.overall_direction == "regressing"
    assert result.regressing_strategies == ()
    assert result.status == "warn"


def test_strategy_regression_can_be_disabled() -> None:
    observations = [
        _observation(strategy="dense"),
        _observation(strategy="dense"),
        _observation(strategy="sparse", accepted=False, quality_score=0.0),
        _observation(strategy="sparse", accepted=False, quality_score=0.0),
        _observation(strategy="dense", accepted=False, quality_score=0.0),
        _observation(strategy="dense", accepted=False, quality_score=0.0),
        _observation(strategy="sparse"),
        _observation(strategy="sparse"),
    ]
    report = _report(
        observations,
        window_size=4,
        minimum_strategy_observations=2,
    )
    gate = FeedbackRegressionGate(
        FeedbackRegressionGatePolicy(
            fail_on_strategy_regression=False,
        )
    )

    result = gate.evaluate(report)

    assert result.overall_direction == "stable"
    assert result.regressing_strategies == ("dense",)
    assert result.status == "pass"


def test_insufficient_warnings_can_be_disabled() -> None:
    gate = FeedbackRegressionGate(
        FeedbackRegressionGatePolicy(
            warn_on_insufficient_overall_data=False,
            warn_on_insufficient_strategy_data=False,
        )
    )

    result = gate.evaluate(_report([_observation()]))

    assert result.status == "pass"
    assert result.passed


def test_failure_has_priority_over_insufficient_warning() -> None:
    observations = [
        _observation(strategy="dense"),
        _observation(strategy="dense"),
        _observation(strategy="dense", accepted=False, quality_score=0.0),
        _observation(strategy="sparse", accepted=False, quality_score=0.0),
    ]
    result = FeedbackRegressionGate().evaluate(
        _report(
            observations,
            minimum_strategy_observations=2,
        )
    )

    assert result.status == "fail"
    assert result.regressing_strategies == ()
    assert result.insufficient_strategies == ("dense", "sparse")
    assert result.reasons[0] == ("Overall routing-feedback utility is regressing.")
    assert "Insufficient observed strategy evidence" in result.reasons[1]


@pytest.mark.parametrize(
    ("status_report", "enforcement", "expected_exit_code"),
    [
        ("pass", "warn", PASS_EXIT_CODE),
        ("pass", "fail", PASS_EXIT_CODE),
        ("pass", "never", PASS_EXIT_CODE),
        ("warn", "warn", WARN_EXIT_CODE),
        ("warn", "fail", PASS_EXIT_CODE),
        ("warn", "never", PASS_EXIT_CODE),
        ("fail", "warn", FAIL_EXIT_CODE),
        ("fail", "fail", FAIL_EXIT_CODE),
        ("fail", "never", PASS_EXIT_CODE),
    ],
)
def test_process_exit_code_respects_enforcement(
    status_report: str,
    enforcement: GateEnforcement,
    expected_exit_code: int,
) -> None:
    if status_report == "pass":
        report = _report([_observation() for _ in range(4)])
    elif status_report == "warn":
        report = _report([_observation()])
    else:
        report = _report(
            [
                _observation(),
                _observation(),
                _observation(accepted=False, quality_score=0.0),
                _observation(accepted=False, quality_score=0.0),
            ]
        )
    result = FeedbackRegressionGate().evaluate(report)

    assert result.status == status_report
    assert result.process_exit_code(enforcement) == expected_exit_code


def test_process_exit_code_rejects_unknown_enforcement() -> None:
    result = FeedbackRegressionGate().evaluate(
        _report([_observation() for _ in range(4)])
    )

    with pytest.raises(ValueError, match="unsupported gate enforcement"):
        result.process_exit_code("sometimes")  # type: ignore[arg-type]


def test_result_and_policy_are_immutable() -> None:
    gate = FeedbackRegressionGate()
    result = gate.evaluate(_report([_observation() for _ in range(4)]))

    with pytest.raises(FrozenInstanceError):
        gate.policy.fail_on_overall_regression = False  # type: ignore[misc]

    with pytest.raises(FrozenInstanceError):
        result.status = "fail"  # type: ignore[misc]


def test_repeated_evaluation_is_deterministic() -> None:
    report = _report(
        [
            _observation(),
            _observation(),
            _observation(accepted=False, quality_score=0.0),
            _observation(accepted=False, quality_score=0.0),
        ]
    )
    gate = FeedbackRegressionGate()

    assert gate.evaluate(report) == gate.evaluate(report)
