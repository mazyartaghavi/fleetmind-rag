from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Literal, Protocol

from fleetmind_rag.retrieval import (
    HybridRetrievalResponse,
    RerankedRetrievalResponse,
    RetrievalResponse,
    SparseRetrievalResponse,
)
from fleetmind_rag.routed_retrieval import RoutedRetrievalResult
from fleetmind_rag.routing import RetrievalStrategy

RetrievalQualityVerdict = Literal["accept", "rewrite"]

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_QUERY_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "code",
        "could",
        "did",
        "do",
        "does",
        "error",
        "explain",
        "fault",
        "for",
        "from",
        "how",
        "if",
        "in",
        "is",
        "it",
        "may",
        "mean",
        "meaning",
        "must",
        "number",
        "of",
        "or",
        "serial",
        "should",
        "the",
        "they",
        "to",
        "what",
        "when",
        "why",
        "with",
        "would",
    }
)


class _EvidenceMatch(Protocol):
    """Common evidence fields shared by all retrieval result models."""

    @property
    def section_title(self) -> str:
        """Return the source section title."""

    @property
    def text(self) -> str:
        """Return the retrieved chunk text."""

    @property
    def score(self) -> float:
        """Return the strategy-specific retrieval score."""


@dataclass(frozen=True, slots=True)
class RetrievalQualityPolicy:
    """Thresholds for deterministic retrieval-quality assessment."""

    minimum_matches: int = 1
    minimum_query_token_coverage: float = 0.40
    minimum_reranked_lexical_coverage: float = 0.20
    evidence_limit: int = 3

    def __post_init__(self) -> None:
        """Validate quality-policy thresholds."""

        if self.minimum_matches <= 0:
            raise ValueError("minimum_matches must be greater than zero")

        if not 0.0 <= self.minimum_query_token_coverage <= 1.0:
            raise ValueError("minimum_query_token_coverage must be between 0.0 and 1.0")

        if not 0.0 <= self.minimum_reranked_lexical_coverage <= 1.0:
            raise ValueError(
                "minimum_reranked_lexical_coverage must be between 0.0 and 1.0"
            )

        if self.evidence_limit <= 0:
            raise ValueError("evidence_limit must be greater than zero")


@dataclass(frozen=True, slots=True)
class RetrievalQualitySignal:
    """One transparent pass-or-fail quality check."""

    name: str
    passed: bool
    observed: str
    required: str
    detail: str


@dataclass(frozen=True, slots=True)
class RetrievalQualityAssessment:
    """Deterministic quality verdict with its supporting evidence."""

    query: str
    strategy: RetrievalStrategy
    verdict: RetrievalQualityVerdict
    quality_score: float
    match_count: int
    top_score: float | None
    query_token_coverage: float
    exact_identifier_coverage: float
    quoted_phrase_coverage: float
    signals: tuple[RetrievalQualitySignal, ...]
    reasons: tuple[str, ...]

    @property
    def should_accept(self) -> bool:
        """Return whether the retrieved evidence passed every check."""

        return self.verdict == "accept"

    @property
    def should_rewrite(self) -> bool:
        """Return whether a query rewrite should be attempted."""

        return self.verdict == "rewrite"


class RetrievalQualityChecker:
    """Assess routed retrieval results using strategy-independent evidence."""

    def __init__(
        self,
        policy: RetrievalQualityPolicy | None = None,
    ) -> None:
        """Initialize the checker with explicit or default thresholds."""

        self._policy = policy or RetrievalQualityPolicy()

    @property
    def policy(self) -> RetrievalQualityPolicy:
        """Return the immutable policy used by this checker."""

        return self._policy

    def assess(
        self,
        result: RoutedRetrievalResult,
    ) -> RetrievalQualityAssessment:
        """Return an explainable accept-or-rewrite quality verdict."""

        self._validate_result_contract(result)

        decision = result.decision
        matches = result.response.matches
        evidence_matches = matches[: self._policy.evidence_limit]
        evidence = _build_evidence_text(evidence_matches)
        match_count = len(matches)
        top_score = None if not matches else matches[0].score

        query_tokens = _meaningful_query_tokens(decision.query)
        query_token_coverage = _token_coverage(
            query_tokens,
            evidence,
        )
        exact_identifier_coverage = _phrase_coverage(
            decision.signals.exact_identifiers,
            evidence,
        )
        quoted_phrase_coverage = _phrase_coverage(
            decision.signals.quoted_phrases,
            evidence,
        )

        signals = [
            self._match_count_signal(match_count),
            self._finite_scores_signal(matches),
            self._query_coverage_signal(query_token_coverage),
            self._identifier_coverage_signal(
                decision.signals.exact_identifiers,
                exact_identifier_coverage,
            ),
            self._quoted_phrase_coverage_signal(
                decision.signals.quoted_phrases,
                quoted_phrase_coverage,
            ),
        ]

        if isinstance(result.response, RerankedRetrievalResponse):
            signals.append(self._reranked_lexical_signal(result.response))

        signal_tuple = tuple(signals)
        failed_signals = tuple(
            signal.detail for signal in signal_tuple if not signal.passed
        )
        verdict: RetrievalQualityVerdict = "accept" if not failed_signals else "rewrite"
        reasons = (
            ("Retrieved evidence passed every configured quality check.",)
            if verdict == "accept"
            else failed_signals
        )
        passed_count = sum(signal.passed for signal in signal_tuple)
        quality_score = round(passed_count / len(signal_tuple), 4)

        return RetrievalQualityAssessment(
            query=decision.query,
            strategy=decision.strategy,
            verdict=verdict,
            quality_score=quality_score,
            match_count=match_count,
            top_score=top_score,
            query_token_coverage=query_token_coverage,
            exact_identifier_coverage=exact_identifier_coverage,
            quoted_phrase_coverage=quoted_phrase_coverage,
            signals=signal_tuple,
            reasons=reasons,
        )

    def _match_count_signal(
        self,
        match_count: int,
    ) -> RetrievalQualitySignal:
        passed = match_count >= self._policy.minimum_matches
        return RetrievalQualitySignal(
            name="minimum_matches",
            passed=passed,
            observed=str(match_count),
            required=f">= {self._policy.minimum_matches}",
            detail=(
                f"Retrieved {match_count} matches; at least "
                f"{self._policy.minimum_matches} are required."
            ),
        )

    @staticmethod
    def _finite_scores_signal(
        matches: tuple[_EvidenceMatch, ...],
    ) -> RetrievalQualitySignal:
        scores = tuple(match.score for match in matches)
        passed = all(math.isfinite(score) for score in scores)
        return RetrievalQualitySignal(
            name="finite_scores",
            passed=passed,
            observed="all finite" if passed else "non-finite score detected",
            required="all scores must be finite",
            detail=(
                "Every retrieval score is finite."
                if passed
                else "At least one retrieval score is missing or non-finite."
            ),
        )

    def _query_coverage_signal(
        self,
        coverage: float,
    ) -> RetrievalQualitySignal:
        threshold = self._policy.minimum_query_token_coverage
        return RetrievalQualitySignal(
            name="query_token_coverage",
            passed=coverage >= threshold,
            observed=f"{coverage:.4f}",
            required=f">= {threshold:.4f}",
            detail=(
                f"Meaningful query-token coverage is {coverage:.4f}; "
                f"at least {threshold:.4f} is required."
            ),
        )

    @staticmethod
    def _identifier_coverage_signal(
        identifiers: tuple[str, ...],
        coverage: float,
    ) -> RetrievalQualitySignal:
        required = 1.0 if identifiers else 0.0
        return RetrievalQualitySignal(
            name="exact_identifier_coverage",
            passed=coverage >= required,
            observed=f"{coverage:.4f}",
            required=f">= {required:.4f}",
            detail=(
                f"Exact identifier coverage is {coverage:.4f}; "
                f"{required:.4f} is required."
            ),
        )

    @staticmethod
    def _quoted_phrase_coverage_signal(
        phrases: tuple[str, ...],
        coverage: float,
    ) -> RetrievalQualitySignal:
        required = 1.0 if phrases else 0.0
        return RetrievalQualitySignal(
            name="quoted_phrase_coverage",
            passed=coverage >= required,
            observed=f"{coverage:.4f}",
            required=f">= {required:.4f}",
            detail=(
                f"Quoted-phrase coverage is {coverage:.4f}; {required:.4f} is required."
            ),
        )

    def _reranked_lexical_signal(
        self,
        response: RerankedRetrievalResponse,
    ) -> RetrievalQualitySignal:
        coverage = max(
            (match.lexical_coverage for match in response.matches),
            default=0.0,
        )
        threshold = self._policy.minimum_reranked_lexical_coverage
        return RetrievalQualitySignal(
            name="reranked_lexical_coverage",
            passed=coverage >= threshold,
            observed=f"{coverage:.4f}",
            required=f">= {threshold:.4f}",
            detail=(
                f"Best reranked lexical coverage is {coverage:.4f}; "
                f"at least {threshold:.4f} is required."
            ),
        )

    @staticmethod
    def _validate_result_contract(
        result: RoutedRetrievalResult,
    ) -> None:
        decision = result.decision
        response = result.response

        if response.query != decision.query:
            raise ValueError(
                "retrieval response query must match routing decision query"
            )

        strategy = decision.strategy
        valid_type = (
            (strategy == "dense" and isinstance(response, RetrievalResponse))
            or (strategy == "sparse" and isinstance(response, SparseRetrievalResponse))
            or (strategy == "hybrid" and isinstance(response, HybridRetrievalResponse))
            or (
                strategy == "reranked"
                and isinstance(response, RerankedRetrievalResponse)
            )
        )

        if not valid_type:
            raise ValueError(f"response type does not match {strategy!r} strategy")


def _meaningful_query_tokens(query: str) -> tuple[str, ...]:
    tokens = (token.lower() for token in _TOKEN_PATTERN.findall(query))
    return tuple(
        dict.fromkeys(token for token in tokens if token not in _QUERY_STOP_WORDS)
    )


def _token_coverage(
    tokens: tuple[str, ...],
    evidence: str,
) -> float:
    if not tokens:
        return 1.0

    evidence_tokens = set(_TOKEN_PATTERN.findall(evidence.lower()))
    matched_count = sum(token in evidence_tokens for token in tokens)
    return round(matched_count / len(tokens), 4)


def _phrase_coverage(
    phrases: tuple[str, ...],
    evidence: str,
) -> float:
    if not phrases:
        return 1.0

    normalized_evidence = " ".join(evidence.lower().split())
    matched_count = sum(
        " ".join(phrase.lower().split()) in normalized_evidence for phrase in phrases
    )
    return round(matched_count / len(phrases), 4)


def _build_evidence_text(
    matches: tuple[_EvidenceMatch, ...],
) -> str:
    evidence_parts: list[str] = []

    for match in matches:
        evidence_parts.extend((match.section_title, match.text))

    return " ".join(evidence_parts)
