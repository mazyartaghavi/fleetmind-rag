from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from fleetmind_rag.agent_state import (
    InvalidAgentStateTransition,
    RetrievalAgentState,
)
from fleetmind_rag.retrieval import (
    HybridRetrievalResponse,
    RerankedRetrievalResponse,
    RetrievalResponse,
    SparseRetrievalResponse,
)
from fleetmind_rag.routed_retrieval import (
    RetrievalExecutionResponse,
    RoutedRetrievalResult,
)
from fleetmind_rag.routing import RetrievalStrategyRouter


def make_result(query: str) -> RoutedRetrievalResult:
    decision = RetrievalStrategyRouter().route(query)
    response: RetrievalExecutionResponse

    if decision.strategy == "dense":
        response = RetrievalResponse(
            query=decision.query,
            embedding_model="test-embedding",
            matches=(),
        )
    elif decision.strategy == "sparse":
        response = SparseRetrievalResponse(
            query=decision.query,
            algorithm="bm25",
            matches=(),
        )
    elif decision.strategy == "hybrid":
        response = HybridRetrievalResponse(
            query=decision.query,
            algorithm="weighted-rrf",
            embedding_model="test-embedding",
            dense_match_count=0,
            sparse_match_count=0,
            matches=(),
        )
    else:
        response = RerankedRetrievalResponse(
            query=decision.query,
            algorithm="transparent-reranking",
            embedding_model="test-embedding",
            dense_match_count=0,
            sparse_match_count=0,
            candidate_count=0,
            matches=(),
        )

    return RoutedRetrievalResult(
        decision=decision,
        response=response,
    )


def state_with_result(
    query: str = "What does overheating mean?",
    *,
    max_attempts: int = 3,
) -> RetrievalAgentState:
    return (
        RetrievalAgentState.start(query, max_attempts=max_attempts)
        .begin_retrieval()
        .record_retrieval(make_result(query))
    )


def test_start_normalizes_query_and_sets_defaults() -> None:
    state = RetrievalAgentState.start("  What   does overheating mean?  ")

    assert state.original_query == "What does overheating mean?"
    assert state.current_query == "What does overheating mean?"
    assert state.status == "ready"
    assert state.max_attempts == 3
    assert state.attempts == ()
    assert state.transitions == ()
    assert state.final_result is None
    assert state.termination_reason is None


@pytest.mark.parametrize("query", ["", " ", "\n\t"])
def test_start_rejects_blank_query(query: str) -> None:
    with pytest.raises(ValueError, match="query must not be blank"):
        RetrievalAgentState.start(query)


def test_start_rejects_non_string_query() -> None:
    with pytest.raises(TypeError, match="query must be a string"):
        RetrievalAgentState.start(123)  # type: ignore[arg-type]


@pytest.mark.parametrize("max_attempts", [0, -1])
def test_start_rejects_non_positive_max_attempts(
    max_attempts: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="max_attempts must be greater than zero",
    ):
        RetrievalAgentState.start(
            "What does overheating mean?",
            max_attempts=max_attempts,
        )


def test_new_state_exposes_attempt_properties() -> None:
    state = RetrievalAgentState.start(
        "What does overheating mean?",
        max_attempts=2,
    )

    assert state.attempt_count == 0
    assert state.attempts_remaining == 2
    assert state.next_attempt_number == 1
    assert state.latest_attempt is None
    assert state.latest_result is None
    assert state.can_retry
    assert not state.is_terminal


def test_begin_retrieval_records_transition() -> None:
    state = RetrievalAgentState.start("What does overheating mean?").begin_retrieval()

    assert state.status == "retrieving"
    assert len(state.transitions) == 1
    transition = state.transitions[0]
    assert transition.sequence == 1
    assert transition.action == "begin_retrieval"
    assert transition.from_status == "ready"
    assert transition.to_status == "retrieving"
    assert transition.detail == "Retrieval attempt 1 started."


def test_record_retrieval_creates_attempt() -> None:
    result = make_result("What does overheating mean?")
    state = (
        RetrievalAgentState.start("What does overheating mean?")
        .begin_retrieval()
        .record_retrieval(result)
    )

    assert state.status == "evaluating"
    assert state.attempt_count == 1
    assert state.attempts_remaining == 2
    assert state.next_attempt_number == 2
    assert state.latest_result is result
    assert state.latest_attempt is not None
    assert state.latest_attempt.number == 1
    assert state.latest_attempt.query == result.decision.query
    assert state.latest_attempt.strategy == "dense"
    assert state.latest_attempt.match_count == 0
    assert state.transitions[-1].action == "record_retrieval"


def test_record_retrieval_rejects_mismatched_query() -> None:
    state = RetrievalAgentState.start("What does overheating mean?").begin_retrieval()

    with pytest.raises(
        ValueError,
        match="retrieval result query must match current_query",
    ):
        state.record_retrieval(make_result("error code P0420"))


def test_complete_preserves_latest_result() -> None:
    evaluating = state_with_result()
    completed = evaluating.complete("  Quality threshold passed.  ")

    assert completed.status == "completed"
    assert completed.is_terminal
    assert not completed.can_retry
    assert completed.final_result is evaluating.latest_result
    assert completed.termination_reason == "Quality threshold passed."
    assert completed.transitions[-1].action == "complete"
    assert completed.transitions[-1].to_status == "completed"


def test_complete_uses_default_reason() -> None:
    completed = state_with_result().complete()

    assert completed.termination_reason == "Retrieval result accepted."


def test_rewrite_cycle_preserves_original_query() -> None:
    evaluating = state_with_result()
    rewriting = evaluating.request_rewrite("  Retrieved evidence was insufficient.  ")
    ready = rewriting.apply_rewrite(
        "  Explain the dangerous engine overheating warning. "
    )

    assert rewriting.status == "rewriting"
    assert ready.status == "ready"
    assert ready.original_query == "What does overheating mean?"
    assert ready.current_query == "Explain the dangerous engine overheating warning."
    assert ready.attempt_count == 1
    assert ready.attempts_remaining == 2
    assert ready.transitions[-2].action == "request_rewrite"
    assert ready.transitions[-1].action == "apply_rewrite"


def test_second_attempt_is_numbered_sequentially() -> None:
    ready = (
        state_with_result()
        .request_rewrite("More specific evidence is required.")
        .apply_rewrite("Explain the dangerous engine overheating warning.")
    )
    rewritten_query = ready.current_query
    evaluating = ready.begin_retrieval().record_retrieval(make_result(rewritten_query))

    assert evaluating.attempt_count == 2
    assert evaluating.attempts[-1].number == 2
    assert evaluating.next_attempt_number == 3
    assert [item.sequence for item in evaluating.transitions] == [
        1,
        2,
        3,
        4,
        5,
        6,
    ]


def test_request_rewrite_rejects_exhausted_attempt_budget() -> None:
    state = state_with_result(max_attempts=1)

    assert state.attempts_remaining == 0
    assert not state.can_retry

    with pytest.raises(
        InvalidAgentStateTransition,
        match="no retrieval attempts remain",
    ):
        state.request_rewrite("Try a more specific query.")


def test_apply_rewrite_rejects_unchanged_query() -> None:
    state = state_with_result().request_rewrite("Try a more specific query.")

    with pytest.raises(
        ValueError,
        match="rewritten query must differ from current_query",
    ):
        state.apply_rewrite("  What does overheating mean?  ")


@pytest.mark.parametrize("query", ["", " ", "\n"])
def test_apply_rewrite_rejects_blank_query(query: str) -> None:
    state = state_with_result().request_rewrite("Try a more specific query.")

    with pytest.raises(ValueError, match="query must not be blank"):
        state.apply_rewrite(query)


@pytest.mark.parametrize(
    "reason",
    ["", " ", "\n"],
)
def test_request_rewrite_rejects_blank_reason(reason: str) -> None:
    with pytest.raises(ValueError, match="reason must not be blank"):
        state_with_result().request_rewrite(reason)


@pytest.mark.parametrize(
    "reason",
    ["", " ", "\n"],
)
def test_complete_rejects_blank_reason(reason: str) -> None:
    with pytest.raises(ValueError, match="reason must not be blank"):
        state_with_result().complete(reason)


@pytest.mark.parametrize(
    "state",
    [
        RetrievalAgentState.start("What does overheating mean?"),
        RetrievalAgentState.start("What does overheating mean?").begin_retrieval(),
        state_with_result(),
        state_with_result().request_rewrite("A more specific query is required."),
    ],
)
def test_fail_terminates_each_non_terminal_status(
    state: RetrievalAgentState,
) -> None:
    failed = state.fail("  Retrieval workflow stopped.  ")

    assert failed.status == "failed"
    assert failed.is_terminal
    assert not failed.can_retry
    assert failed.termination_reason == "Retrieval workflow stopped."
    assert failed.transitions[-1].action == "fail"
    assert failed.transitions[-1].from_status == state.status


def test_fail_rejects_blank_reason() -> None:
    with pytest.raises(ValueError, match="reason must not be blank"):
        RetrievalAgentState.start("What does overheating mean?").fail(" ")


@pytest.mark.parametrize("terminal_status", ["completed", "failed"])
def test_terminal_state_rejects_failure_transition(
    terminal_status: str,
) -> None:
    state = state_with_result()
    terminal = (
        state.complete() if terminal_status == "completed" else state.fail("Stopped.")
    )

    with pytest.raises(
        InvalidAgentStateTransition,
        match="terminal status",
    ):
        terminal.fail("Cannot fail twice.")


def test_begin_retrieval_requires_ready_status() -> None:
    with pytest.raises(
        InvalidAgentStateTransition,
        match="requires status 'ready'",
    ):
        state_with_result().begin_retrieval()


def test_record_retrieval_requires_retrieving_status() -> None:
    state = RetrievalAgentState.start("What does overheating mean?")

    with pytest.raises(
        InvalidAgentStateTransition,
        match="requires status 'retrieving'",
    ):
        state.record_retrieval(make_result("What does overheating mean?"))


def test_request_rewrite_requires_evaluating_status() -> None:
    with pytest.raises(
        InvalidAgentStateTransition,
        match="requires status 'evaluating'",
    ):
        RetrievalAgentState.start("What does overheating mean?").request_rewrite(
            "Rewrite required."
        )


def test_apply_rewrite_requires_rewriting_status() -> None:
    with pytest.raises(
        InvalidAgentStateTransition,
        match="requires status 'rewriting'",
    ):
        RetrievalAgentState.start("What does overheating mean?").apply_rewrite(
            "Explain engine overheating."
        )


def test_complete_requires_evaluating_status() -> None:
    with pytest.raises(
        InvalidAgentStateTransition,
        match="requires status 'evaluating'",
    ):
        RetrievalAgentState.start("What does overheating mean?").complete()


def test_state_and_nested_records_are_immutable() -> None:
    state = state_with_result()

    with pytest.raises(FrozenInstanceError):
        state.status = "completed"  # type: ignore[misc]

    with pytest.raises(FrozenInstanceError):
        state.attempts[0].number = 99  # type: ignore[misc]

    with pytest.raises(FrozenInstanceError):
        state.transitions[0].sequence = 99  # type: ignore[misc]


def test_transitions_do_not_mutate_previous_states() -> None:
    ready = RetrievalAgentState.start("What does overheating mean?")
    retrieving = ready.begin_retrieval()

    assert ready.status == "ready"
    assert ready.transitions == ()
    assert retrieving.status == "retrieving"
    assert len(retrieving.transitions) == 1


def test_same_workflow_produces_equal_deterministic_states() -> None:
    def run_workflow() -> RetrievalAgentState:
        return state_with_result().complete("Quality threshold passed.")

    assert run_workflow() == run_workflow()
