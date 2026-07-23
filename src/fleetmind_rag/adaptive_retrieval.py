from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from fleetmind_rag.agent_state import RetrievalAgentState
from fleetmind_rag.retrieval_quality import (
    RetrievalQualityAssessment,
    RetrievalQualityChecker,
)
from fleetmind_rag.routed_retrieval import (
    RoutedRetrievalRequest,
    RoutedRetrievalResult,
)
from fleetmind_rag.routing import RetrievalStrategy
from fleetmind_rag.vector_store import ChunkMetadataFilter


class RoutedRetrievalExecutorLike(Protocol):
    """Executor interface required by the adaptive retrieval loop."""

    def execute(
        self,
        request: RoutedRetrievalRequest,
    ) -> RoutedRetrievalResult:
        """Execute one routed retrieval request."""


@dataclass(frozen=True, slots=True)
class QueryRewrite:
    """One deterministic query rewrite with preservation evidence."""

    source_query: str
    rewritten_query: str
    strategy: RetrievalStrategy
    after_attempt: int
    reasons: tuple[str, ...]
    preserved_identifiers: tuple[str, ...]
    preserved_quoted_phrases: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AdaptiveRetrievalConfig:
    """Controls for one bounded adaptive-retrieval run."""

    max_attempts: int = 3
    limit: int = 5
    candidate_limit: int = 20
    score_threshold: float | None = None
    metadata_filter: ChunkMetadataFilter | None = None

    def __post_init__(self) -> None:
        """Validate retry and retrieval limits."""

        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be greater than zero")

        if self.limit <= 0:
            raise ValueError("limit must be greater than zero")

        if self.candidate_limit <= 0:
            raise ValueError("candidate_limit must be greater than zero")

        if self.candidate_limit < self.limit:
            raise ValueError("candidate_limit must be greater than or equal to limit")


@dataclass(frozen=True, slots=True)
class AdaptiveRetrievalOutcome:
    """Final state plus every quality assessment and query rewrite."""

    state: RetrievalAgentState
    assessments: tuple[RetrievalQualityAssessment, ...]
    rewrites: tuple[QueryRewrite, ...]

    @property
    def succeeded(self) -> bool:
        """Return whether acceptable evidence completed the workflow."""

        return self.state.status == "completed"

    @property
    def attempt_count(self) -> int:
        """Return the number of retrieval attempts performed."""

        return self.state.attempt_count

    @property
    def final_result(self) -> RoutedRetrievalResult | None:
        """Return the accepted result, if the workflow succeeded."""

        return self.state.final_result


class DeterministicQueryRewriter:
    """Rewrite failed queries without an LLM or external service."""

    def rewrite(
        self,
        result: RoutedRetrievalResult,
        assessment: RetrievalQualityAssessment,
        *,
        after_attempt: int,
    ) -> QueryRewrite:
        """Produce a distinct query while preserving exact constraints."""

        if assessment.verdict != "rewrite":
            raise ValueError("query rewriting requires a rewrite quality verdict")

        if assessment.query != result.decision.query:
            raise ValueError("quality assessment query must match retrieval result")

        if assessment.strategy != result.decision.strategy:
            raise ValueError("quality assessment strategy must match retrieval result")

        if after_attempt <= 0:
            raise ValueError("after_attempt must be greater than zero")

        decision = result.decision
        signals = decision.signals
        source_query = decision.query

        if signals.exact_identifiers:
            candidate = self._rewrite_identifiers(signals.exact_identifiers)
        elif signals.quoted_phrases:
            candidate = self._rewrite_quoted_phrases(signals.quoted_phrases)
        elif decision.strategy == "dense":
            candidate = self._rewrite_dense(
                source_query,
                signals.conceptual_cues,
            )
        elif decision.strategy == "hybrid":
            candidate = self._rewrite_hybrid(source_query)
        else:
            candidate = self._rewrite_reranked(source_query)

        rewritten_query = _ensure_distinct_query(
            source_query,
            candidate,
        )
        self._validate_preserved_constraints(
            rewritten_query,
            identifiers=signals.exact_identifiers,
            quoted_phrases=signals.quoted_phrases,
        )

        return QueryRewrite(
            source_query=source_query,
            rewritten_query=rewritten_query,
            strategy=decision.strategy,
            after_attempt=after_attempt,
            reasons=assessment.reasons,
            preserved_identifiers=signals.exact_identifiers,
            preserved_quoted_phrases=signals.quoted_phrases,
        )

    @staticmethod
    def _rewrite_identifiers(
        identifiers: tuple[str, ...],
    ) -> str:
        identifier_text = " ".join(identifiers)
        return f"{identifier_text} error code"

    @staticmethod
    def _rewrite_quoted_phrases(
        phrases: tuple[str, ...],
    ) -> str:
        phrase_text = " ".join(f'"{phrase}"' for phrase in phrases)
        return f"{phrase_text} exact lookup"

    @staticmethod
    def _rewrite_dense(
        query: str,
        conceptual_cues: tuple[str, ...],
    ) -> str:
        simplified = query

        for cue in conceptual_cues:
            simplified = re.sub(
                re.escape(cue),
                " ",
                simplified,
                flags=re.IGNORECASE,
            )

        simplified = re.sub(
            r"\b(mean|means|meaning)\b",
            " ",
            simplified,
            flags=re.IGNORECASE,
        )
        simplified = _normalize_query(simplified.strip(" .?!,:;-"))

        if not simplified:
            simplified = query.strip(" .?!,:;-")

        return f"Explain {simplified}."

    @staticmethod
    def _rewrite_hybrid(query: str) -> str:
        words = query.strip(" .?!,:;-").split()

        if len(words) >= 2:
            return " ".join((*words[:-1], "and", words[-1]))

        return f"{query.strip()}?"

    @staticmethod
    def _rewrite_reranked(query: str) -> str:
        rewritten = query
        replacements = (
            (r"\bis accompanied by\b", "has"),
            (r"\bmay the driver\b", "can the driver"),
            (r"\bor must they\b", "or must the driver"),
            (r"\bmust they\b", "must the driver"),
        )

        for pattern, replacement in replacements:
            rewritten = re.sub(
                pattern,
                replacement,
                rewritten,
                flags=re.IGNORECASE,
            )

        return _normalize_query(rewritten)

    @staticmethod
    def _validate_preserved_constraints(
        rewritten_query: str,
        *,
        identifiers: tuple[str, ...],
        quoted_phrases: tuple[str, ...],
    ) -> None:
        lowered_query = rewritten_query.lower()

        if any(identifier.lower() not in lowered_query for identifier in identifiers):
            raise RuntimeError("query rewrite did not preserve every exact identifier")

        if any(f'"{phrase.lower()}"' not in lowered_query for phrase in quoted_phrases):
            raise RuntimeError("query rewrite did not preserve every quoted phrase")


class AdaptiveRetrievalAgent:
    """Run bounded route-retrieve-assess-rewrite workflows."""

    def __init__(
        self,
        executor: RoutedRetrievalExecutorLike,
        *,
        quality_checker: RetrievalQualityChecker | None = None,
        query_rewriter: DeterministicQueryRewriter | None = None,
    ) -> None:
        """Initialize the agent with injectable deterministic components."""

        self._executor = executor
        self._quality_checker = quality_checker or RetrievalQualityChecker()
        self._query_rewriter = query_rewriter or DeterministicQueryRewriter()

    def run(
        self,
        query: str,
        *,
        config: AdaptiveRetrievalConfig | None = None,
    ) -> AdaptiveRetrievalOutcome:
        """Run until evidence is accepted or attempts are exhausted."""

        resolved_config = config or AdaptiveRetrievalConfig()
        state = RetrievalAgentState.start(
            query,
            max_attempts=resolved_config.max_attempts,
        )
        assessments: list[RetrievalQualityAssessment] = []
        rewrites: list[QueryRewrite] = []

        while True:
            state = state.begin_retrieval()
            result = self._executor.execute(
                RoutedRetrievalRequest(
                    query=state.current_query,
                    limit=resolved_config.limit,
                    candidate_limit=resolved_config.candidate_limit,
                    score_threshold=resolved_config.score_threshold,
                    metadata_filter=resolved_config.metadata_filter,
                )
            )
            state = state.record_retrieval(result)
            assessment = self._quality_checker.assess(result)
            assessments.append(assessment)

            if assessment.should_accept:
                state = state.complete(
                    "Retrieval quality checks passed with score "
                    f"{assessment.quality_score:.4f}."
                )
                return AdaptiveRetrievalOutcome(
                    state=state,
                    assessments=tuple(assessments),
                    rewrites=tuple(rewrites),
                )

            if not state.can_retry:
                state = state.fail(
                    "Retrieval quality remained insufficient after "
                    f"{state.attempt_count} attempts: "
                    f"{'; '.join(assessment.reasons)}"
                )
                return AdaptiveRetrievalOutcome(
                    state=state,
                    assessments=tuple(assessments),
                    rewrites=tuple(rewrites),
                )

            state = state.request_rewrite("; ".join(assessment.reasons))
            rewrite = self._query_rewriter.rewrite(
                result,
                assessment,
                after_attempt=state.attempt_count,
            )
            rewrites.append(rewrite)
            state = state.apply_rewrite(rewrite.rewritten_query)


def _normalize_query(query: str) -> str:
    return " ".join(query.split())


def _ensure_distinct_query(
    source_query: str,
    candidate: str,
) -> str:
    normalized_source = _normalize_query(source_query)
    normalized_candidate = _normalize_query(candidate)

    if not normalized_candidate:
        raise RuntimeError("query rewriter produced a blank query")

    if normalized_candidate != normalized_source:
        return normalized_candidate

    if normalized_source.endswith("?"):
        return f"{normalized_source[:-1].rstrip()}."

    if normalized_source.endswith("."):
        return f"{normalized_source[:-1].rstrip()}?"

    return f"{normalized_source}?"
