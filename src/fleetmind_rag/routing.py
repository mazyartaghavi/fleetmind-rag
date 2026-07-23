from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final, Literal

RetrievalStrategy = Literal["dense", "sparse", "hybrid", "reranked"]
RoutingConfidence = Literal["low", "medium", "high"]

STRATEGIES: Final[tuple[RetrievalStrategy, ...]] = (
    "dense",
    "sparse",
    "hybrid",
    "reranked",
)
_TIE_BREAK_ORDER: Final[tuple[RetrievalStrategy, ...]] = (
    "reranked",
    "hybrid",
    "sparse",
    "dense",
)

_TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"[a-z0-9]+(?:[-_.:/][a-z0-9]+)*",
    re.IGNORECASE,
)
_QUOTED_PHRASE_PATTERN: Final[re.Pattern[str]] = re.compile(r"[\"']([^\"']+)[\"']")
_CODE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b(?=[a-z0-9._:/-]{3,}\b)(?=[a-z0-9._:/-]*[a-z])"
    r"(?=[a-z0-9._:/-]*\d)[a-z0-9]+(?:[._:/-][a-z0-9]+)*\b",
    re.IGNORECASE,
)

_CONCEPTUAL_CUES: Final[tuple[str, ...]] = (
    "what does",
    "what is",
    "why does",
    "why is",
    "explain",
    "meaning of",
    "how does",
    "how is",
)
_EXACT_LOOKUP_CUES: Final[tuple[str, ...]] = (
    "error code",
    "fault code",
    "diagnostic code",
    "part number",
    "serial number",
    "vehicle identifier",
    "vin",
    "exact phrase",
    "find the section",
    "locate the section",
)
_CONDITIONAL_CUES: Final[tuple[str, ...]] = (
    "if",
    "when",
    "unless",
    "while",
    "before",
    "after",
    "accompanied by",
    "under the stated conditions",
)
_ACTION_CUES: Final[tuple[str, ...]] = (
    "what should",
    "what must",
    "should i",
    "must i",
    "can i",
    "may i",
    "continue",
    "stop",
    "switch off",
    "report",
    "notify",
    "contact",
    "remove from service",
)
_SAFETY_CUES: Final[tuple[str, ...]] = (
    "warning",
    "danger",
    "unsafe",
    "smoke",
    "burning smell",
    "serious burns",
    "fire",
    "collision",
    "prohibited",
    "must not",
    "stop safely",
    "remove from service",
    "electrical failure",
    "overheating condition",
)
_DOMAIN_CUES: Final[tuple[str, ...]] = (
    "vehicle",
    "fleet",
    "engine",
    "battery",
    "charging system",
    "tire",
    "coolant",
    "maintenance",
    "inspection",
    "dispatch",
    "warning",
    "defect",
    "driver",
)


@dataclass(frozen=True, slots=True)
class QueryRoutingSignals:
    """Transparent lexical and structural signals extracted from one query."""

    normalized_query: str
    tokens: tuple[str, ...]
    quoted_phrases: tuple[str, ...]
    exact_identifiers: tuple[str, ...]
    conceptual_cues: tuple[str, ...]
    exact_lookup_cues: tuple[str, ...]
    conditional_cues: tuple[str, ...]
    action_cues: tuple[str, ...]
    safety_cues: tuple[str, ...]
    domain_cues: tuple[str, ...]
    clause_count: int

    @property
    def token_count(self) -> int:
        """Return the number of normalized lexical tokens."""

        return len(self.tokens)

    @property
    def is_complex(self) -> bool:
        """Return whether the query contains multiple constraints or clauses."""

        return (
            self.token_count >= 10
            or self.clause_count >= 2
            or bool(self.conditional_cues and self.action_cues)
        )


@dataclass(frozen=True, slots=True)
class StrategyScore:
    """One retrieval strategy's deterministic score and supporting reasons."""

    strategy: RetrievalStrategy
    score: int
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    """Explainable retrieval strategy selected for one normalized query."""

    query: str
    strategy: RetrievalStrategy
    confidence: RoutingConfidence
    selected_score: int
    reason: str
    signals: QueryRoutingSignals
    scores: tuple[StrategyScore, ...]


class RetrievalStrategyRouter:
    """Choose a retrieval strategy with deterministic, inspectable heuristics."""

    def analyze(self, query: str) -> QueryRoutingSignals:
        """Normalize one query and extract signals used by the routing policy."""

        normalized_query = " ".join(query.strip().split())
        if not normalized_query:
            raise ValueError("The routing query must not be empty.")

        lowered = normalized_query.lower()
        tokens = tuple(
            match.group(0).lower() for match in _TOKEN_PATTERN.finditer(lowered)
        )
        if not tokens:
            raise ValueError(
                "The routing query must contain at least one lexical term."
            )

        quoted_phrases = self._unique_matches(
            match.group(1).strip().lower()
            for match in _QUOTED_PHRASE_PATTERN.finditer(normalized_query)
            if match.group(1).strip()
        )
        exact_identifiers = self._unique_matches(
            match.group(0).upper()
            for match in _CODE_PATTERN.finditer(normalized_query)
            if not match.group(0).isdigit()
        )

        return QueryRoutingSignals(
            normalized_query=normalized_query,
            tokens=tokens,
            quoted_phrases=quoted_phrases,
            exact_identifiers=exact_identifiers,
            conceptual_cues=self._matched_cues(lowered, _CONCEPTUAL_CUES),
            exact_lookup_cues=self._matched_cues(lowered, _EXACT_LOOKUP_CUES),
            conditional_cues=self._matched_cues(lowered, _CONDITIONAL_CUES),
            action_cues=self._matched_cues(lowered, _ACTION_CUES),
            safety_cues=self._matched_cues(lowered, _SAFETY_CUES),
            domain_cues=self._matched_cues(lowered, _DOMAIN_CUES),
            clause_count=self._clause_count(normalized_query),
        )

    def route(self, query: str) -> RoutingDecision:
        """Score every retrieval strategy and return the selected route."""

        signals = self.analyze(query)
        score_values: dict[RetrievalStrategy, int] = {
            strategy: 0 for strategy in STRATEGIES
        }
        reasons: dict[RetrievalStrategy, list[str]] = {
            strategy: [] for strategy in STRATEGIES
        }

        def award(strategy: RetrievalStrategy, points: int, reason: str) -> None:
            score_values[strategy] += points
            reasons[strategy].append(reason)

        if signals.conceptual_cues:
            award(
                "dense",
                8,
                "conceptual or explanatory wording favors semantic search",
            )
        if signals.token_count >= 5:
            award("dense", 1, "the query contains enough context for an embedding")

        if signals.exact_identifiers:
            award("sparse", 12, "code-like identifiers require exact lexical matching")
        if signals.quoted_phrases:
            award("sparse", 8, "quoted text indicates an exact phrase lookup")
        if signals.exact_lookup_cues:
            award("sparse", 7, "explicit lookup wording favors lexical retrieval")
        if (
            signals.token_count <= 3
            and not signals.conceptual_cues
            and (signals.domain_cues or signals.safety_cues)
        ):
            award("sparse", 2, "a short domain query benefits from exact matching")

        if signals.domain_cues:
            award(
                "hybrid",
                4,
                "fleet-domain terminology supports semantic and lexical search",
            )
        if signals.safety_cues:
            award(
                "hybrid",
                4,
                "safety terminology should be preserved while matching meaning",
            )
        if 4 <= signals.token_count <= 10:
            award(
                "hybrid",
                2,
                "a medium-length query benefits from complementary signals",
            )
        if signals.conceptual_cues and (
            signals.domain_cues or signals.exact_lookup_cues
        ):
            award("hybrid", 2, "the query mixes conceptual and exact-domain intent")

        if signals.safety_cues and signals.is_complex:
            award(
                "reranked",
                12,
                "a complex safety query benefits from second-stage reranking",
            )
        if signals.conditional_cues:
            award("reranked", 3, "conditional wording introduces decision constraints")
        if signals.action_cues:
            award(
                "reranked",
                3,
                "requested operational actions require precise ordering",
            )
        if signals.clause_count >= 2:
            award("reranked", 2, "multiple clauses increase ranking complexity")
        if signals.token_count >= 12:
            award("reranked", 2, "a long query contains multiple relevance signals")

        if not any(score_values.values()):
            award("hybrid", 1, "balanced hybrid retrieval is the safe default")

        selected_strategy = max(
            _TIE_BREAK_ORDER,
            key=lambda strategy: score_values[strategy],
        )
        ordered_scores = tuple(
            StrategyScore(
                strategy=strategy,
                score=score_values[strategy],
                reasons=tuple(reasons[strategy]),
            )
            for strategy in STRATEGIES
        )
        ranked_values = sorted(score_values.values(), reverse=True)
        margin = ranked_values[0] - ranked_values[1]
        confidence: RoutingConfidence
        if margin >= 5:
            confidence = "high"
        elif margin >= 2:
            confidence = "medium"
        else:
            confidence = "low"

        selected_reasons = reasons[selected_strategy]
        reason = (
            "; ".join(selected_reasons)
            if selected_reasons
            else "deterministic tie-breaking selected this strategy"
        )

        return RoutingDecision(
            query=signals.normalized_query,
            strategy=selected_strategy,
            confidence=confidence,
            selected_score=score_values[selected_strategy],
            reason=reason,
            signals=signals,
            scores=ordered_scores,
        )

    @staticmethod
    def _matched_cues(text: str, cues: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(
            cue for cue in cues if RetrievalStrategyRouter._contains_cue(text, cue)
        )

    @staticmethod
    def _contains_cue(text: str, cue: str) -> bool:
        if " " in cue:
            return cue in text
        return re.search(rf"\b{re.escape(cue)}\b", text) is not None

    @staticmethod
    def _unique_matches(values: Iterable[str]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(values))

    @staticmethod
    def _clause_count(query: str) -> int:
        separators = len(re.findall(r"[;]+|\b(?:and|or|but)\b", query.lower()))
        return max(1, separators + 1)
