from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol, cast

from fleetmind_rag.config import FleetMindSettings
from fleetmind_rag.ollama import OllamaEmbeddingClient
from fleetmind_rag.retrieval import (
    DocumentRetrievalService,
    HybridRetrievalResponse,
    RerankedRetrievalResponse,
    RetrievalResponse,
    SparseRetrievalResponse,
)
from fleetmind_rag.vector_store import QdrantChunkStore

RetrievalStrategy = Literal["dense", "sparse", "hybrid", "reranked"]
STRATEGIES: tuple[RetrievalStrategy, ...] = (
    "dense",
    "sparse",
    "hybrid",
    "reranked",
)


class RetrievalEvaluationClient(Protocol):
    """Structural interface required by the retrieval benchmark."""

    def search(self, query: str, *, limit: int = 5) -> RetrievalResponse:
        """Return dense embedding matches."""

    def search_sparse(self, query: str, *, limit: int = 5) -> SparseRetrievalResponse:
        """Return sparse BM25 matches."""

    def search_hybrid(
        self,
        query: str,
        *,
        limit: int = 5,
        candidate_limit: int = 20,
    ) -> HybridRetrievalResponse:
        """Return reciprocal-rank-fused dense and sparse matches."""

    def search_hybrid_reranked(
        self,
        query: str,
        *,
        limit: int = 5,
        candidate_limit: int = 20,
    ) -> RerankedRetrievalResponse:
        """Return transparently reranked hybrid matches."""


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationCase:
    """One frozen query and its expected section-level retrieval target."""

    case_id: str
    query: str
    expected_section: str
    expected_rank_at_most: int


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationCaseResult:
    """Measured outcome for one case under one retrieval strategy."""

    case_id: str
    query: str
    strategy: RetrievalStrategy
    expected_section: str
    expected_rank_at_most: int
    retrieved_sections: tuple[str, ...]
    retrieved_count: int
    top_section: str | None
    expected_rank: int | None
    top_1_correct: bool
    hit_at_k: bool
    reciprocal_rank: float
    passed: bool
    message: str


@dataclass(frozen=True, slots=True)
class RetrievalStrategyMetrics:
    """Aggregate section-level metrics for one retrieval strategy."""

    strategy: RetrievalStrategy
    total_cases: int
    passed_cases: int
    top_1_correct_cases: int
    hits_at_k: int
    empty_results: int
    errors: int
    mean_result_count: float
    pass_rate: float
    top_1_accuracy: float
    hit_rate_at_k: float
    mean_reciprocal_rank: float
    empty_result_rate: float
    error_rate: float


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationReport:
    """Serializable comparison of every configured retrieval strategy."""

    generated_at_utc: str
    retrieval_limit: int
    candidate_limit: int
    strategies: tuple[RetrievalStrategy, ...]
    metrics: tuple[RetrievalStrategyMetrics, ...]
    cases: tuple[RetrievalEvaluationCaseResult, ...]

    @property
    def all_quality_gates_passed(self) -> bool:
        """Return whether every strategy met every case rank expectation."""

        return all(
            metric.errors == 0 and metric.passed_cases == metric.total_cases
            for metric in self.metrics
        )

    @property
    def best_strategy(self) -> RetrievalStrategy:
        """Return the best strategy using deterministic metric tie-breaking."""

        if not self.metrics:
            raise RuntimeError(
                "The retrieval evaluation report has no strategy metrics."
            )

        strategy_order = {strategy: index for index, strategy in enumerate(STRATEGIES)}
        best = max(
            self.metrics,
            key=lambda metric: (
                metric.mean_reciprocal_rank,
                metric.top_1_accuracy,
                metric.hit_rate_at_k,
                metric.pass_rate,
                -metric.error_rate,
                -strategy_order[metric.strategy],
            ),
        )
        return best.strategy


class RetrievalEvaluator:
    """Benchmark dense, sparse, hybrid, and reranked retrieval consistently."""

    def __init__(
        self,
        retrieval_client: RetrievalEvaluationClient,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._retrieval_client = retrieval_client
        self._clock = clock or _utc_now

    def evaluate(
        self,
        cases: Sequence[RetrievalEvaluationCase],
        *,
        strategies: Sequence[RetrievalStrategy] = STRATEGIES,
        limit: int = 4,
        candidate_limit: int = 12,
    ) -> RetrievalEvaluationReport:
        """Run all cases through the same configured retrieval strategies."""

        self._validate_configuration(
            cases,
            strategies=strategies,
            limit=limit,
            candidate_limit=candidate_limit,
        )
        selected_strategies = tuple(strategies)
        results = tuple(
            self._evaluate_case(
                case,
                strategy=strategy,
                limit=limit,
                candidate_limit=candidate_limit,
            )
            for strategy in selected_strategies
            for case in cases
        )
        metrics = tuple(
            _calculate_strategy_metrics(
                strategy,
                tuple(result for result in results if result.strategy == strategy),
            )
            for strategy in selected_strategies
        )

        return RetrievalEvaluationReport(
            generated_at_utc=self._clock().astimezone(UTC).isoformat(),
            retrieval_limit=limit,
            candidate_limit=candidate_limit,
            strategies=selected_strategies,
            metrics=metrics,
            cases=results,
        )

    @staticmethod
    def _validate_configuration(
        cases: Sequence[RetrievalEvaluationCase],
        *,
        strategies: Sequence[RetrievalStrategy],
        limit: int,
        candidate_limit: int,
    ) -> None:
        if not cases:
            raise ValueError("At least one retrieval evaluation case is required.")

        case_ids = tuple(case.case_id for case in cases)
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("Retrieval evaluation case identifiers must be unique.")

        if not strategies:
            raise ValueError("At least one retrieval strategy is required.")

        if len(set(strategies)) != len(strategies):
            raise ValueError("Retrieval evaluation strategies must be unique.")

        unknown_strategies = set(strategies) - set(STRATEGIES)
        if unknown_strategies:
            unknown = ", ".join(sorted(unknown_strategies))
            raise ValueError(f"Unknown retrieval strategies: {unknown}.")

        if limit <= 0:
            raise ValueError("The retrieval evaluation limit must be positive.")

        if candidate_limit < limit:
            raise ValueError(
                "The retrieval evaluation candidate limit must be greater than or "
                "equal to the result limit."
            )

    def _evaluate_case(
        self,
        case: RetrievalEvaluationCase,
        *,
        strategy: RetrievalStrategy,
        limit: int,
        candidate_limit: int,
    ) -> RetrievalEvaluationCaseResult:
        try:
            retrieved_sections = self._retrieve_sections(
                case.query,
                strategy=strategy,
                limit=limit,
                candidate_limit=candidate_limit,
            )
        except (OSError, RuntimeError, ValueError) as error:
            return RetrievalEvaluationCaseResult(
                case_id=case.case_id,
                query=case.query,
                strategy=strategy,
                expected_section=case.expected_section,
                expected_rank_at_most=case.expected_rank_at_most,
                retrieved_sections=(),
                retrieved_count=0,
                top_section=None,
                expected_rank=None,
                top_1_correct=False,
                hit_at_k=False,
                reciprocal_rank=0.0,
                passed=False,
                message=f"Retrieval execution failed: {error}",
            )

        expected_rank = _find_expected_rank(
            retrieved_sections,
            case.expected_section,
        )
        top_1_correct = expected_rank == 1
        hit_at_k = expected_rank is not None
        reciprocal_rank = 1.0 / expected_rank if expected_rank is not None else 0.0
        passed = (
            expected_rank is not None and expected_rank <= case.expected_rank_at_most
        )

        if passed:
            message = "Expected section satisfied the configured rank gate."
        elif expected_rank is None:
            message = "Expected section was not retrieved."
        else:
            message = (
                f"Expected section ranked {expected_rank}, exceeding the maximum "
                f"allowed rank {case.expected_rank_at_most}."
            )

        return RetrievalEvaluationCaseResult(
            case_id=case.case_id,
            query=case.query,
            strategy=strategy,
            expected_section=case.expected_section,
            expected_rank_at_most=case.expected_rank_at_most,
            retrieved_sections=retrieved_sections,
            retrieved_count=len(retrieved_sections),
            top_section=retrieved_sections[0] if retrieved_sections else None,
            expected_rank=expected_rank,
            top_1_correct=top_1_correct,
            hit_at_k=hit_at_k,
            reciprocal_rank=reciprocal_rank,
            passed=passed,
            message=message,
        )

    def _retrieve_sections(
        self,
        query: str,
        *,
        strategy: RetrievalStrategy,
        limit: int,
        candidate_limit: int,
    ) -> tuple[str, ...]:
        if strategy == "dense":
            dense_response = self._retrieval_client.search(query, limit=limit)
            return tuple(match.section_title for match in dense_response.matches)

        if strategy == "sparse":
            sparse_response = self._retrieval_client.search_sparse(query, limit=limit)
            return tuple(match.section_title for match in sparse_response.matches)

        if strategy == "hybrid":
            hybrid_response = self._retrieval_client.search_hybrid(
                query,
                limit=limit,
                candidate_limit=candidate_limit,
            )
            return tuple(match.section_title for match in hybrid_response.matches)

        reranked_response = self._retrieval_client.search_hybrid_reranked(
            query,
            limit=limit,
            candidate_limit=candidate_limit,
        )
        return tuple(match.section_title for match in reranked_response.matches)


def load_retrieval_evaluation_cases(
    path: str | Path,
) -> tuple[RetrievalEvaluationCase, ...]:
    """Load and strictly validate frozen retrieval cases from JSON."""

    source_path = Path(path)
    try:
        raw_value: object = json.loads(source_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            "The retrieval evaluation JSON is invalid at "
            f"line {error.lineno}, column {error.colno}: {error.msg}"
        ) from error

    if not isinstance(raw_value, list):
        raise ValueError("The retrieval evaluation JSON root must be a list.")

    if not raw_value:
        raise ValueError(
            "The retrieval evaluation dataset must contain at least one case."
        )

    cases = tuple(
        _parse_case(raw_case, index=index)
        for index, raw_case in enumerate(raw_value, start=1)
    )
    case_ids = tuple(case.case_id for case in cases)
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("Retrieval evaluation case identifiers must be unique.")

    return cases


def write_retrieval_evaluation_report(
    report: RetrievalEvaluationReport,
    path: str | Path,
) -> Path:
    """Write one readable JSON benchmark report and return its path."""

    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(asdict(report), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report_path


def build_retrieval_evaluation_summary(
    report: RetrievalEvaluationReport,
    *,
    report_path: str | Path | None = None,
) -> str:
    """Build a concise terminal comparison for a retrieval report."""

    lines = [
        "FleetMind retrieval benchmark completed.",
        f"Retrieval limit: {report.retrieval_limit}",
        f"Candidate limit: {report.candidate_limit}",
        "",
        "Strategy   Passed   Top-1   Hit@K    MRR     Empty   Errors",
        "---------  -------  ------  ------  ------  ------  ------",
    ]

    for metric in report.metrics:
        lines.append(
            f"{metric.strategy:<9}  "
            f"{metric.passed_cases:>2}/{metric.total_cases:<2}    "
            f"{metric.top_1_accuracy:>6.2%}  "
            f"{metric.hit_rate_at_k:>6.2%}  "
            f"{metric.mean_reciprocal_rank:>6.3f}  "
            f"{metric.empty_results:>6}  "
            f"{metric.errors:>6}"
        )

    lines.extend(
        [
            "",
            f"Best strategy: {report.best_strategy}",
            ("Result: PASS" if report.all_quality_gates_passed else "Result: FAIL"),
        ]
    )

    if report_path is not None:
        lines.append(f"Report: {Path(report_path)}")

    return "\n".join(lines)


def run_live_retrieval_evaluation(
    settings: FleetMindSettings,
    *,
    manual_path: str | Path,
    cases_path: str | Path,
    report_path: str | Path,
    strategies: Sequence[RetrievalStrategy] = STRATEGIES,
    chunk_size_words: int | None = None,
    overlap_words: int | None = None,
    limit: int = 4,
    candidate_limit: int = 12,
) -> RetrievalEvaluationReport:
    """Index the frozen manual in memory and run the retrieval benchmark."""

    cases = load_retrieval_evaluation_cases(cases_path)
    effective_chunk_size = chunk_size_words or settings.chunk_size_words
    effective_overlap = (
        settings.chunk_overlap_words if overlap_words is None else overlap_words
    )
    embedding_client = OllamaEmbeddingClient(
        str(settings.llm_base_url),
        settings.embedding_model,
        timeout_seconds=settings.ollama_timeout_seconds,
    )

    with QdrantChunkStore.in_memory(
        collection_name=f"{settings.qdrant_collection}_retrieval_evaluation"
    ) as vector_store:
        retrieval_service = DocumentRetrievalService(
            embedding_client,
            vector_store,
        )
        retrieval_service.index_text_document(
            manual_path,
            chunk_size_words=effective_chunk_size,
            overlap_words=effective_overlap,
            recreate_collection=True,
        )
        report = RetrievalEvaluator(retrieval_service).evaluate(
            cases,
            strategies=strategies,
            limit=limit,
            candidate_limit=candidate_limit,
        )

    write_retrieval_evaluation_report(report, report_path)
    return report


def build_cli_parser() -> argparse.ArgumentParser:
    """Build the standalone retrieval benchmark command-line parser."""

    parser = argparse.ArgumentParser(
        prog="python -m fleetmind_rag.retrieval_evaluation",
        description=(
            "Compare FleetMind dense, sparse, hybrid, and reranked retrieval."
        ),
    )
    parser.add_argument(
        "--manual",
        type=Path,
        default=Path("evaluation/data/fleet_manual.md"),
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("evaluation/data/retrieval_cases.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("evaluation/reports/retrieval_evaluation.json"),
    )
    parser.add_argument(
        "--strategy",
        action="append",
        choices=STRATEGIES,
        dest="strategies",
        default=None,
        help="Strategy to benchmark; repeat to select multiple strategies.",
    )
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=None)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--candidate-limit", type=int, default=12)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the live retrieval benchmark and return a CI-friendly exit code."""

    args = build_cli_parser().parse_args(argv)
    selected_strategies = (
        tuple(cast(list[RetrievalStrategy], args.strategies))
        if args.strategies is not None
        else STRATEGIES
    )

    try:
        report = run_live_retrieval_evaluation(
            FleetMindSettings(),
            manual_path=cast(Path, args.manual),
            cases_path=cast(Path, args.cases),
            report_path=cast(Path, args.report),
            strategies=selected_strategies,
            chunk_size_words=cast(int | None, args.chunk_size),
            overlap_words=cast(int | None, args.overlap),
            limit=cast(int, args.limit),
            candidate_limit=cast(int, args.candidate_limit),
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"FleetMind retrieval benchmark failed: {error}", file=sys.stderr)
        return 1

    print(
        build_retrieval_evaluation_summary(
            report,
            report_path=cast(Path, args.report),
        )
    )
    return 0 if report.all_quality_gates_passed else 2


def _parse_case(raw_case: object, *, index: int) -> RetrievalEvaluationCase:
    if not isinstance(raw_case, dict):
        raise ValueError(f"Retrieval evaluation case {index} must be a JSON object.")

    mapping = cast(Mapping[str, object], raw_case)
    expected_keys = {
        "case_id",
        "query",
        "expected_section",
        "expected_rank_at_most",
    }
    missing_keys = expected_keys - set(mapping)
    unknown_keys = set(mapping) - expected_keys

    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"Retrieval evaluation case {index} is missing: {missing}.")

    if unknown_keys:
        unknown = ", ".join(sorted(unknown_keys))
        raise ValueError(
            f"Retrieval evaluation case {index} has unknown fields: {unknown}."
        )

    case_id = _require_string(mapping["case_id"], field="case_id", index=index)
    query = _require_string(mapping["query"], field="query", index=index)
    expected_section = _require_string(
        mapping["expected_section"],
        field="expected_section",
        index=index,
    )
    expected_rank_at_most = mapping["expected_rank_at_most"]
    if (
        isinstance(expected_rank_at_most, bool)
        or not isinstance(expected_rank_at_most, int)
        or expected_rank_at_most <= 0
    ):
        raise ValueError(
            "Retrieval evaluation case "
            f"{index} field 'expected_rank_at_most' must be a positive integer."
        )

    return RetrievalEvaluationCase(
        case_id=case_id,
        query=query,
        expected_section=expected_section,
        expected_rank_at_most=expected_rank_at_most,
    )


def _require_string(value: object, *, field: str, index: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            "Retrieval evaluation case "
            f"{index} field {field!r} must be a non-empty string."
        )

    return value.strip()


def _find_expected_rank(
    retrieved_sections: Sequence[str],
    expected_section: str,
) -> int | None:
    normalized_expected = _normalize_text(expected_section)
    for rank, section in enumerate(retrieved_sections, start=1):
        if _normalize_text(section) == normalized_expected:
            return rank

    return None


def _calculate_strategy_metrics(
    strategy: RetrievalStrategy,
    results: Sequence[RetrievalEvaluationCaseResult],
) -> RetrievalStrategyMetrics:
    total_cases = len(results)
    errors = sum(
        result.message.startswith("Retrieval execution failed:") for result in results
    )
    return RetrievalStrategyMetrics(
        strategy=strategy,
        total_cases=total_cases,
        passed_cases=sum(result.passed for result in results),
        top_1_correct_cases=sum(result.top_1_correct for result in results),
        hits_at_k=sum(result.hit_at_k for result in results),
        empty_results=sum(result.retrieved_count == 0 for result in results),
        errors=errors,
        mean_result_count=_mean(
            tuple(float(result.retrieved_count) for result in results)
        ),
        pass_rate=_rate(sum(result.passed for result in results), total_cases),
        top_1_accuracy=_rate(
            sum(result.top_1_correct for result in results),
            total_cases,
        ),
        hit_rate_at_k=_rate(sum(result.hit_at_k for result in results), total_cases),
        mean_reciprocal_rank=_mean(tuple(result.reciprocal_rank for result in results)),
        empty_result_rate=_rate(
            sum(result.retrieved_count == 0 for result in results),
            total_cases,
        ),
        error_rate=_rate(errors, total_cases),
    )


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(normalized.split())


def _rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 1.0

    return numerator / denominator


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0

    return sum(values) / len(values)


def _utc_now() -> datetime:
    return datetime.now(UTC)


if __name__ == "__main__":
    raise SystemExit(main())
