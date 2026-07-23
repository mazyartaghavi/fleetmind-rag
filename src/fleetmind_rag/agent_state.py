from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from fleetmind_rag.routed_retrieval import RoutedRetrievalResult
from fleetmind_rag.routing import RetrievalStrategy

RetrievalAgentStatus = Literal[
    "ready",
    "retrieving",
    "evaluating",
    "rewriting",
    "completed",
    "failed",
]

RetrievalAgentAction = Literal[
    "begin_retrieval",
    "record_retrieval",
    "request_rewrite",
    "apply_rewrite",
    "complete",
    "fail",
]


class InvalidAgentStateTransition(RuntimeError):
    """Raised when an operation violates the retrieval-agent lifecycle."""


@dataclass(frozen=True, slots=True)
class AgentStateTransition:
    """One deterministic and auditable retrieval-agent state transition."""

    sequence: int
    action: RetrievalAgentAction
    from_status: RetrievalAgentStatus
    to_status: RetrievalAgentStatus
    detail: str


@dataclass(frozen=True, slots=True)
class RetrievalAttempt:
    """One routed retrieval result recorded by the agent."""

    number: int
    query: str
    result: RoutedRetrievalResult

    @property
    def strategy(self) -> RetrievalStrategy:
        """Return the strategy selected for this attempt."""

        return self.result.decision.strategy

    @property
    def match_count(self) -> int:
        """Return the number of retrieved matches."""

        return self.result.match_count


@dataclass(frozen=True, slots=True)
class RetrievalAgentState:
    """Immutable state for deterministic adaptive-retrieval workflows."""

    original_query: str
    current_query: str
    status: RetrievalAgentStatus
    max_attempts: int
    attempts: tuple[RetrievalAttempt, ...] = ()
    transitions: tuple[AgentStateTransition, ...] = ()
    final_result: RoutedRetrievalResult | None = None
    termination_reason: str | None = None

    @classmethod
    def start(
        cls,
        query: str,
        *,
        max_attempts: int = 3,
    ) -> RetrievalAgentState:
        """Create a validated agent state ready for its first retrieval."""

        normalized_query = _normalize_required_text(query, field="query")

        if max_attempts <= 0:
            raise ValueError("max_attempts must be greater than zero")

        return cls(
            original_query=normalized_query,
            current_query=normalized_query,
            status="ready",
            max_attempts=max_attempts,
        )

    @property
    def attempt_count(self) -> int:
        """Return the number of completed retrieval attempts."""

        return len(self.attempts)

    @property
    def attempts_remaining(self) -> int:
        """Return how many additional retrieval attempts are permitted."""

        return self.max_attempts - self.attempt_count

    @property
    def next_attempt_number(self) -> int:
        """Return the one-based number of the next retrieval attempt."""

        return self.attempt_count + 1

    @property
    def latest_attempt(self) -> RetrievalAttempt | None:
        """Return the most recent attempt, if retrieval has occurred."""

        if not self.attempts:
            return None

        return self.attempts[-1]

    @property
    def latest_result(self) -> RoutedRetrievalResult | None:
        """Return the latest routed retrieval result, if one exists."""

        attempt = self.latest_attempt
        return None if attempt is None else attempt.result

    @property
    def is_terminal(self) -> bool:
        """Return whether the workflow has completed or failed."""

        return self.status in {"completed", "failed"}

    @property
    def can_retry(self) -> bool:
        """Return whether another retrieval attempt remains available."""

        return not self.is_terminal and self.attempts_remaining > 0

    def begin_retrieval(self) -> RetrievalAgentState:
        """Move a ready workflow into its retrieval stage."""

        self._require_status("ready")

        if self.attempts_remaining <= 0:
            raise InvalidAgentStateTransition("No retrieval attempts remain")

        detail = f"Retrieval attempt {self.next_attempt_number} started."
        return replace(
            self,
            status="retrieving",
            transitions=self._append_transition(
                action="begin_retrieval",
                to_status="retrieving",
                detail=detail,
            ),
        )

    def record_retrieval(
        self,
        result: RoutedRetrievalResult,
    ) -> RetrievalAgentState:
        """Record one routed result and enter quality evaluation."""

        self._require_status("retrieving")

        if result.decision.query != self.current_query:
            raise ValueError("retrieval result query must match current_query")

        attempt = RetrievalAttempt(
            number=self.next_attempt_number,
            query=self.current_query,
            result=result,
        )
        detail = (
            f"Retrieval attempt {attempt.number} recorded with "
            f"{attempt.strategy} strategy and {attempt.match_count} matches."
        )

        return replace(
            self,
            status="evaluating",
            attempts=(*self.attempts, attempt),
            transitions=self._append_transition(
                action="record_retrieval",
                to_status="evaluating",
                detail=detail,
            ),
        )

    def request_rewrite(self, reason: str) -> RetrievalAgentState:
        """Request a query rewrite after an insufficient retrieval result."""

        self._require_status("evaluating")
        normalized_reason = _normalize_required_text(
            reason,
            field="reason",
        )

        if self.attempts_remaining <= 0:
            raise InvalidAgentStateTransition(
                "Cannot rewrite because no retrieval attempts remain"
            )

        return replace(
            self,
            status="rewriting",
            transitions=self._append_transition(
                action="request_rewrite",
                to_status="rewriting",
                detail=normalized_reason,
            ),
        )

    def apply_rewrite(self, query: str) -> RetrievalAgentState:
        """Apply a rewritten query and make the workflow ready to retry."""

        self._require_status("rewriting")
        normalized_query = _normalize_required_text(query, field="query")

        if normalized_query == self.current_query:
            raise ValueError("rewritten query must differ from current_query")

        detail = f"Query rewritten from {self.current_query!r} to {normalized_query!r}."
        return replace(
            self,
            current_query=normalized_query,
            status="ready",
            transitions=self._append_transition(
                action="apply_rewrite",
                to_status="ready",
                detail=detail,
            ),
        )

    def complete(
        self,
        reason: str = "Retrieval result accepted.",
    ) -> RetrievalAgentState:
        """Complete the workflow with the latest retrieval result."""

        self._require_status("evaluating")
        normalized_reason = _normalize_required_text(
            reason,
            field="reason",
        )
        result = self.latest_result

        if result is None:
            raise InvalidAgentStateTransition(
                "Cannot complete without a retrieval result"
            )

        return replace(
            self,
            status="completed",
            final_result=result,
            termination_reason=normalized_reason,
            transitions=self._append_transition(
                action="complete",
                to_status="completed",
                detail=normalized_reason,
            ),
        )

    def fail(self, reason: str) -> RetrievalAgentState:
        """Terminate a non-terminal workflow with an explicit reason."""

        if self.is_terminal:
            raise InvalidAgentStateTransition(
                f"Cannot fail workflow from terminal status {self.status!r}"
            )

        normalized_reason = _normalize_required_text(
            reason,
            field="reason",
        )
        return replace(
            self,
            status="failed",
            termination_reason=normalized_reason,
            transitions=self._append_transition(
                action="fail",
                to_status="failed",
                detail=normalized_reason,
            ),
        )

    def _require_status(
        self,
        expected_status: RetrievalAgentStatus,
    ) -> None:
        if self.status != expected_status:
            raise InvalidAgentStateTransition(
                f"Operation requires status {expected_status!r}; "
                f"current status is {self.status!r}"
            )

    def _append_transition(
        self,
        *,
        action: RetrievalAgentAction,
        to_status: RetrievalAgentStatus,
        detail: str,
    ) -> tuple[AgentStateTransition, ...]:
        transition = AgentStateTransition(
            sequence=len(self.transitions) + 1,
            action=action,
            from_status=self.status,
            to_status=to_status,
            detail=detail,
        )
        return (*self.transitions, transition)


def _normalize_required_text(value: str, *, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")

    normalized = " ".join(value.split())

    if not normalized:
        raise ValueError(f"{field} must not be blank")

    return normalized
