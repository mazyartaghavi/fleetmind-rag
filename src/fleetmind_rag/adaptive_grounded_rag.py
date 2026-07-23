from __future__ import annotations

import math
import sys
from dataclasses import dataclass

from fleetmind_rag.adaptive_retrieval import (
    AdaptiveRetrievalConfig,
    AdaptiveRetrievalOutcome,
    DeterministicQueryRewriter,
)
from fleetmind_rag.feedback_routing import (
    FeedbackDrivenRetrievalRouter,
    FeedbackRoutingDecision,
    FeedbackRoutingPolicy,
    RoutingFeedbackHistory,
)
from fleetmind_rag.grounded_rag import (
    ABSTENTION_ANSWER,
    DEFAULT_GROUNDED_SYSTEM_PROMPT,
    ChatClient,
    GroundedAnswerResult,
    GroundedAnswerService,
)
from fleetmind_rag.langgraph_workflow import LangGraphAdaptiveRetrievalWorkflow
from fleetmind_rag.retrieval import (
    HybridRetrievalResponse,
    RerankedRetrievalResponse,
    RerankedSearchResult,
    RetrievalResponse,
    SparseRetrievalResponse,
)
from fleetmind_rag.retrieval_quality import RetrievalQualityChecker
from fleetmind_rag.routed_retrieval import (
    RetrievalService,
    RoutedRetrievalExecutor,
    RoutedRetrievalResult,
)
from fleetmind_rag.routing import RetrievalStrategyRouter
from fleetmind_rag.vector_store import VectorSearchResult

_ALL_FINITE_SCORES = -sys.float_info.max


@dataclass(frozen=True, slots=True)
class AdaptiveGroundedAnswerResult:
    """Grounded answer paired with its complete adaptive retrieval trace."""

    question: str
    grounded_answer: GroundedAnswerResult
    retrieval_outcome: AdaptiveRetrievalOutcome
    initial_routing: FeedbackRoutingDecision
    feedback_history: RoutingFeedbackHistory

    @property
    def succeeded(self) -> bool:
        """Return whether the grounded-answer operation succeeded."""

        return self.grounded_answer.succeeded

    @property
    def abstained(self) -> bool:
        """Return whether the operation safely declined to answer."""

        return self.grounded_answer.abstained

    @property
    def answer(self) -> str | None:
        """Return the answer text or safe abstention text."""

        return self.grounded_answer.answer

    @property
    def attempt_count(self) -> int:
        """Return the number of retrieval attempts performed."""

        return self.retrieval_outcome.attempt_count


class AcceptedEvidenceRetrievalAdapter:
    """Expose one accepted routed result through the dense retrieval protocol."""

    def __init__(self, result: RoutedRetrievalResult) -> None:
        """Capture one quality-approved routed retrieval result."""

        self._result = result

    @property
    def retrieval_model(self) -> str:
        """Return transparent retrieval provenance for grounded output."""

        response = self._result.response

        if isinstance(response, RetrievalResponse):
            return response.embedding_model

        if isinstance(response, SparseRetrievalResponse):
            return response.algorithm

        if isinstance(response, HybridRetrievalResponse):
            return f"{response.algorithm} ({response.embedding_model})"

        if isinstance(response, RerankedRetrievalResponse):
            return f"{response.algorithm} ({response.embedding_model})"

        raise RuntimeError(
            f"Unsupported routed retrieval response: {type(response).__name__}"
        )

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        score_threshold: float | None = None,
    ) -> RetrievalResponse:
        """Return accepted evidence without cross-strategy score filtering."""

        clean_query = _normalize_required_text(query, field="query")

        if limit <= 0:
            raise ValueError("limit must be greater than zero")

        if score_threshold is not None:
            raise ValueError(
                "accepted adaptive evidence does not support a score threshold"
            )

        response = self._result.response
        matches = (
            tuple(_as_vector_result(match) for match in response.matches)
            if isinstance(response, RerankedRetrievalResponse)
            else response.matches
        )
        return RetrievalResponse(
            query=clean_query,
            embedding_model=self.retrieval_model,
            matches=matches[:limit],
        )


class AdaptiveGroundedAnswerService:
    """Run feedback-aware adaptive retrieval before grounded generation."""

    def __init__(
        self,
        retrieval_service: RetrievalService,
        chat_client: ChatClient,
        *,
        history: RoutingFeedbackHistory | None = None,
        feedback_policy: FeedbackRoutingPolicy | None = None,
        base_router: RetrievalStrategyRouter | None = None,
        quality_checker: RetrievalQualityChecker | None = None,
        query_rewriter: DeterministicQueryRewriter | None = None,
        max_context_chars: int = 6000,
        system_prompt: str = DEFAULT_GROUNDED_SYSTEM_PROMPT,
    ) -> None:
        """Initialize adaptive retrieval, generation, and feedback dependencies."""

        if max_context_chars < 256:
            raise ValueError(
                "The maximum context size must be at least 256 characters."
            )

        clean_system_prompt = system_prompt.strip()

        if not clean_system_prompt:
            raise ValueError("The grounded-answer system prompt must not be empty.")

        self._retrieval_service = retrieval_service
        self._chat_client = chat_client
        self._history = history or RoutingFeedbackHistory()
        self._feedback_policy = feedback_policy or FeedbackRoutingPolicy()
        self._base_router = base_router or RetrievalStrategyRouter()
        self._quality_checker = quality_checker or RetrievalQualityChecker()
        self._query_rewriter = query_rewriter or DeterministicQueryRewriter()
        self._max_context_chars = max_context_chars
        self._system_prompt = clean_system_prompt

    @property
    def history(self) -> RoutingFeedbackHistory:
        """Return feedback accumulated from completed service calls."""

        return self._history

    def answer(
        self,
        question: str,
        *,
        config: AdaptiveRetrievalConfig | None = None,
    ) -> AdaptiveGroundedAnswerResult:
        """Retrieve adaptively, learn from the outcome, and answer safely."""

        clean_question = _normalize_required_text(question, field="question")
        resolved_config = config or AdaptiveRetrievalConfig()
        feedback_router = FeedbackDrivenRetrievalRouter(
            self._history,
            policy=self._feedback_policy,
            base_router=self._base_router,
        )
        initial_routing = feedback_router.explain(clean_question)
        executor = RoutedRetrievalExecutor(
            self._retrieval_service,
            router=feedback_router,
        )
        workflow = LangGraphAdaptiveRetrievalWorkflow(
            executor,
            quality_checker=self._quality_checker,
            query_rewriter=self._query_rewriter,
        )
        retrieval_outcome = workflow.run(
            clean_question,
            config=resolved_config,
        )
        updated_history = self._history.record_outcome(retrieval_outcome)
        self._history = updated_history

        if retrieval_outcome.succeeded:
            final_result = retrieval_outcome.final_result

            if final_result is None:
                raise RuntimeError(
                    "successful adaptive retrieval did not expose a final result"
                )

            grounded_answer = self._answer_from_accepted_evidence(
                clean_question,
                final_result,
                limit=resolved_config.limit,
            )
        else:
            grounded_answer = _adaptive_abstention(
                clean_question,
                retrieval_outcome,
            )

        return AdaptiveGroundedAnswerResult(
            question=clean_question,
            grounded_answer=grounded_answer,
            retrieval_outcome=retrieval_outcome,
            initial_routing=initial_routing,
            feedback_history=updated_history,
        )

    def _answer_from_accepted_evidence(
        self,
        question: str,
        result: RoutedRetrievalResult,
        *,
        limit: int,
    ) -> GroundedAnswerResult:
        adapter = AcceptedEvidenceRetrievalAdapter(result)
        grounded_service = GroundedAnswerService(
            adapter,
            self._chat_client,
            minimum_score=_ALL_FINITE_SCORES,
            max_context_chars=self._max_context_chars,
            system_prompt=self._system_prompt,
        )
        return grounded_service.answer(question, limit=limit)


def _adaptive_abstention(
    question: str,
    outcome: AdaptiveRetrievalOutcome,
) -> GroundedAnswerResult:
    latest_result = outcome.state.latest_result
    retrieval_model: str | None = None
    top_score: float | None = None

    if latest_result is not None:
        adapter = AcceptedEvidenceRetrievalAdapter(latest_result)
        retrieval_model = adapter.retrieval_model
        scores = tuple(match.score for match in latest_result.response.matches)

        if any(not math.isfinite(score) for score in scores):
            raise RuntimeError("A retrieval result contains a non-finite score.")

        top_score = max(scores) if scores else None

    termination_reason = outcome.state.termination_reason
    reason_suffix = "" if termination_reason is None else f" {termination_reason}"
    return GroundedAnswerResult(
        succeeded=True,
        abstained=True,
        question=question,
        answer=ABSTENTION_ANSWER,
        citations=(),
        retrieval_model=retrieval_model,
        generation_model=None,
        top_score=top_score,
        message=(
            "Adaptive retrieval did not produce acceptable evidence after "
            f"{outcome.attempt_count} attempts; generation was skipped."
            f"{reason_suffix}"
        ),
    )


def _as_vector_result(match: RerankedSearchResult) -> VectorSearchResult:
    return VectorSearchResult(
        chunk_id=match.chunk_id,
        document_id=match.document_id,
        section_id=match.section_id,
        section_title=match.section_title,
        ordinal=match.ordinal,
        text=match.text,
        word_count=match.word_count,
        start_word=match.start_word,
        end_word=match.end_word,
        score=match.score,
    )


def _normalize_required_text(value: str, *, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")

    normalized = " ".join(value.split())

    if not normalized:
        raise ValueError(f"{field} must not be blank")

    return normalized
