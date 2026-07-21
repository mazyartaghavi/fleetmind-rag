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
from fleetmind_rag.grounded_rag import GroundedAnswerResult, GroundedAnswerService
from fleetmind_rag.ollama import OllamaChatClient, OllamaEmbeddingClient
from fleetmind_rag.retrieval import DocumentRetrievalService
from fleetmind_rag.vector_store import QdrantChunkStore

ExpectedDecision = Literal["answer", "abstain"]
ActualDecision = Literal["answer", "abstain", "error"]


class GroundedAnswerClient(Protocol):
    """Structural interface required by the deterministic evaluator."""

    def answer(
        self,
        question: str,
        *,
        limit: int = 5,
    ) -> GroundedAnswerResult:
        """Return one grounded answer or abstention."""


@dataclass(frozen=True, slots=True)
class RAGEvaluationCase:
    """One deterministic expectation for a grounded RAG request."""

    case_id: str
    question: str
    expected_decision: ExpectedDecision
    expected_section: str | None
    required_terms: tuple[str, ...]
    forbidden_terms: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RAGEvaluationCaseResult:
    """Measured outcome of evaluating one RAG case."""

    case_id: str
    question: str
    expected_decision: ExpectedDecision
    actual_decision: ActualDecision
    decision_correct: bool
    expected_section: str | None
    cited_sections: tuple[str, ...]
    expected_section_found: bool
    required_terms_found: tuple[str, ...]
    required_terms_missing: tuple[str, ...]
    required_term_recall: float
    forbidden_terms_found: tuple[str, ...]
    citation_present: bool
    top_score: float | None
    answer: str | None
    passed: bool
    message: str


@dataclass(frozen=True, slots=True)
class RAGEvaluationMetrics:
    """Aggregate deterministic quality metrics for one evaluation run."""

    total_cases: int
    passed_cases: int
    answer_cases: int
    abstention_cases: int
    overall_pass_rate: float
    decision_accuracy: float
    answer_case_pass_rate: float
    abstention_accuracy: float
    expected_section_accuracy: float
    required_term_recall: float
    forbidden_claim_violation_rate: float
    citation_presence_rate: float


@dataclass(frozen=True, slots=True)
class RAGEvaluationReport:
    """Serializable report containing per-case and aggregate RAG results."""

    generated_at_utc: str
    retrieval_limit: int
    metrics: RAGEvaluationMetrics
    cases: tuple[RAGEvaluationCaseResult, ...]

    @property
    def all_cases_passed(self) -> bool:
        """Return whether every evaluation case passed."""

        return self.metrics.passed_cases == self.metrics.total_cases


class RAGEvaluator:
    """Evaluate grounded RAG behavior with deterministic expectations."""

    def __init__(
        self,
        answer_client: GroundedAnswerClient,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._answer_client = answer_client
        self._clock = clock or _utc_now

    def evaluate(
        self,
        cases: Sequence[RAGEvaluationCase],
        *,
        limit: int = 5,
    ) -> RAGEvaluationReport:
        """Run every case and calculate deterministic aggregate metrics."""

        if not cases:
            raise ValueError("At least one RAG evaluation case is required.")

        if limit <= 0:
            raise ValueError("The RAG evaluation retrieval limit must be positive.")

        case_ids = tuple(case.case_id for case in cases)
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("RAG evaluation case identifiers must be unique.")

        results = tuple(self._evaluate_case(case, limit=limit) for case in cases)

        return RAGEvaluationReport(
            generated_at_utc=self._clock().astimezone(UTC).isoformat(),
            retrieval_limit=limit,
            metrics=_calculate_metrics(cases, results),
            cases=results,
        )

    def _evaluate_case(
        self,
        case: RAGEvaluationCase,
        *,
        limit: int,
    ) -> RAGEvaluationCaseResult:
        try:
            response = self._answer_client.answer(case.question, limit=limit)
        except (OSError, RuntimeError, ValueError) as error:
            return _error_case_result(case, str(error))

        if not response.succeeded:
            return _error_case_result(case, response.message)

        actual_decision: ActualDecision = "abstain" if response.abstained else "answer"
        answer = response.answer or ""
        normalized_answer = _normalize_text(answer)
        cited_sections = tuple(
            dict.fromkeys(citation.section_title for citation in response.citations)
        )
        normalized_cited_sections = {
            _normalize_text(section) for section in cited_sections
        }

        expected_section_found = (
            case.expected_section is None
            or _normalize_text(case.expected_section) in normalized_cited_sections
        )
        required_terms_found = tuple(
            term
            for term in case.required_terms
            if _normalize_text(term) in normalized_answer
        )
        required_terms_missing = tuple(
            term for term in case.required_terms if term not in required_terms_found
        )
        forbidden_terms_found = tuple(
            term
            for term in case.forbidden_terms
            if _normalize_text(term) in normalized_answer
        )
        required_term_recall = _rate(
            len(required_terms_found),
            len(case.required_terms),
        )
        citation_present = bool(response.citations)
        decision_correct = actual_decision == case.expected_decision
        citation_expectation_met = (
            citation_present and expected_section_found
            if case.expected_decision == "answer"
            else not citation_present
        )
        passed = (
            decision_correct
            and citation_expectation_met
            and not required_terms_missing
            and not forbidden_terms_found
        )

        return RAGEvaluationCaseResult(
            case_id=case.case_id,
            question=case.question,
            expected_decision=case.expected_decision,
            actual_decision=actual_decision,
            decision_correct=decision_correct,
            expected_section=case.expected_section,
            cited_sections=cited_sections,
            expected_section_found=expected_section_found,
            required_terms_found=required_terms_found,
            required_terms_missing=required_terms_missing,
            required_term_recall=required_term_recall,
            forbidden_terms_found=forbidden_terms_found,
            citation_present=citation_present,
            top_score=response.top_score,
            answer=response.answer,
            passed=passed,
            message=_build_case_message(
                decision_correct=decision_correct,
                citation_expectation_met=citation_expectation_met,
                required_terms_missing=required_terms_missing,
                forbidden_terms_found=forbidden_terms_found,
            ),
        )


def load_evaluation_cases(path: str | Path) -> tuple[RAGEvaluationCase, ...]:
    """Load and strictly validate deterministic RAG cases from JSON."""

    source_path = Path(path)

    try:
        raw_value: object = json.loads(source_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"The RAG evaluation JSON is invalid at line {error.lineno}, "
            f"column {error.colno}: {error.msg}"
        ) from error

    if not isinstance(raw_value, list):
        raise ValueError("The RAG evaluation JSON root must be a list.")

    if not raw_value:
        raise ValueError("The RAG evaluation dataset must contain at least one case.")

    cases = tuple(
        _parse_case(raw_case, index=index)
        for index, raw_case in enumerate(raw_value, start=1)
    )
    case_ids = tuple(case.case_id for case in cases)

    if len(set(case_ids)) != len(case_ids):
        raise ValueError("RAG evaluation case identifiers must be unique.")

    return cases


def write_evaluation_report(
    report: RAGEvaluationReport,
    path: str | Path,
) -> Path:
    """Write one human-readable JSON evaluation report and return its path."""

    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(asdict(report), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report_path


def build_evaluation_summary(
    report: RAGEvaluationReport,
    *,
    report_path: str | Path | None = None,
) -> str:
    """Build concise terminal output for an evaluation report."""

    metrics = report.metrics
    lines = [
        "FleetMind RAG evaluation completed.",
        f"Cases passed: {metrics.passed_cases}/{metrics.total_cases}",
        f"Overall pass rate: {metrics.overall_pass_rate:.2%}",
        f"Decision accuracy: {metrics.decision_accuracy:.2%}",
        f"Answer-case pass rate: {metrics.answer_case_pass_rate:.2%}",
        f"Abstention accuracy: {metrics.abstention_accuracy:.2%}",
        f"Expected-section accuracy: {metrics.expected_section_accuracy:.2%}",
        f"Required-term recall: {metrics.required_term_recall:.2%}",
        (
            "Forbidden-claim violation rate: "
            f"{metrics.forbidden_claim_violation_rate:.2%}"
        ),
        f"Citation presence rate: {metrics.citation_presence_rate:.2%}",
        f"Result: {'PASS' if report.all_cases_passed else 'FAIL'}",
    ]

    if report_path is not None:
        lines.append(f"Report: {Path(report_path)}")

    return "\n".join(lines)


def run_live_evaluation(
    settings: FleetMindSettings,
    *,
    manual_path: str | Path,
    cases_path: str | Path,
    report_path: str | Path,
    chunk_size_words: int | None = None,
    overlap_words: int | None = None,
    limit: int | None = None,
) -> RAGEvaluationReport:
    """Run an isolated live Ollama evaluation with in-memory Qdrant storage."""

    cases = load_evaluation_cases(cases_path)
    effective_chunk_size = (
        settings.chunk_size_words if chunk_size_words is None else chunk_size_words
    )
    effective_overlap = (
        settings.chunk_overlap_words if overlap_words is None else overlap_words
    )
    effective_limit = settings.retrieval_limit if limit is None else limit
    base_url = str(settings.llm_base_url)

    with QdrantChunkStore.in_memory(
        collection_name=f"{settings.qdrant_collection}_evaluation",
    ) as vector_store:
        retrieval_service = DocumentRetrievalService(
            OllamaEmbeddingClient(
                base_url,
                settings.embedding_model,
                timeout_seconds=settings.ollama_timeout_seconds,
            ),
            vector_store,
        )
        retrieval_service.index_text_document(
            manual_path,
            chunk_size_words=effective_chunk_size,
            overlap_words=effective_overlap,
            recreate_collection=True,
        )
        grounded_service = GroundedAnswerService(
            retrieval_service,
            OllamaChatClient(
                base_url,
                settings.llm_model,
                timeout_seconds=settings.ollama_timeout_seconds,
            ),
            minimum_score=settings.minimum_grounding_score,
            max_context_chars=settings.max_context_chars,
        )
        report = RAGEvaluator(grounded_service).evaluate(
            cases,
            limit=effective_limit,
        )

    write_evaluation_report(report, report_path)
    return report


def build_cli_parser() -> argparse.ArgumentParser:
    """Build the standalone deterministic RAG evaluation parser."""

    parser = argparse.ArgumentParser(
        prog="python -m fleetmind_rag.evaluation",
        description="Evaluate FleetMind grounded RAG with deterministic cases.",
    )
    parser.add_argument(
        "--manual",
        type=Path,
        default=Path("evaluation/data/fleet_manual.md"),
        help="Fleet manual indexed in an isolated in-memory collection.",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("evaluation/data/rag_cases.json"),
        help="JSON file containing deterministic evaluation cases.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("evaluation/reports/rag_evaluation.json"),
        help="Destination JSON report path.",
    )
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the live deterministic FleetMind RAG evaluator."""

    args = build_cli_parser().parse_args(argv)

    try:
        report = run_live_evaluation(
            FleetMindSettings(),
            manual_path=cast(Path, args.manual),
            cases_path=cast(Path, args.cases),
            report_path=cast(Path, args.report),
            chunk_size_words=cast(int | None, args.chunk_size),
            overlap_words=cast(int | None, args.overlap),
            limit=cast(int | None, args.limit),
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"FleetMind RAG evaluation failed: {error}", file=sys.stderr)
        return 1

    print(
        build_evaluation_summary(
            report,
            report_path=cast(Path, args.report),
        )
    )
    return 0 if report.all_cases_passed else 2


def _parse_case(raw_case: object, *, index: int) -> RAGEvaluationCase:
    if not isinstance(raw_case, dict):
        raise ValueError(f"RAG evaluation case {index} must be a JSON object.")

    mapping = cast(Mapping[str, object], raw_case)
    expected_keys = {
        "case_id",
        "question",
        "expected_decision",
        "expected_section",
        "required_terms",
        "forbidden_terms",
    }
    unknown_keys = set(mapping) - expected_keys
    missing_keys = expected_keys - set(mapping)

    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"RAG evaluation case {index} is missing: {missing}.")

    if unknown_keys:
        unknown = ", ".join(sorted(unknown_keys))
        raise ValueError(f"RAG evaluation case {index} has unknown fields: {unknown}.")

    case_id = _require_string(mapping["case_id"], field="case_id", index=index)
    question = _require_string(mapping["question"], field="question", index=index)
    decision_value = _require_string(
        mapping["expected_decision"],
        field="expected_decision",
        index=index,
    )

    if decision_value not in {"answer", "abstain"}:
        raise ValueError(
            f"RAG evaluation case {index} expected_decision must be "
            "'answer' or 'abstain'."
        )

    expected_decision = cast(ExpectedDecision, decision_value)
    expected_section = _require_optional_string(
        mapping["expected_section"],
        field="expected_section",
        index=index,
    )
    required_terms = _require_string_list(
        mapping["required_terms"],
        field="required_terms",
        index=index,
    )
    forbidden_terms = _require_string_list(
        mapping["forbidden_terms"],
        field="forbidden_terms",
        index=index,
    )

    if expected_decision == "answer" and expected_section is None:
        raise ValueError(
            f"RAG evaluation case {index} must define expected_section for "
            "an answer case."
        )

    if expected_decision == "abstain" and expected_section is not None:
        raise ValueError(
            f"RAG evaluation case {index} must use null expected_section for "
            "an abstention case."
        )

    normalized_required = {_normalize_text(term) for term in required_terms}
    normalized_forbidden = {_normalize_text(term) for term in forbidden_terms}

    if normalized_required & normalized_forbidden:
        raise ValueError(
            f"RAG evaluation case {index} cannot require and forbid the same term."
        )

    return RAGEvaluationCase(
        case_id=case_id,
        question=question,
        expected_decision=expected_decision,
        expected_section=expected_section,
        required_terms=required_terms,
        forbidden_terms=forbidden_terms,
    )


def _require_string(value: object, *, field: str, index: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"RAG evaluation case {index} field {field!r} must be a non-empty string."
        )

    return value.strip()


def _require_optional_string(
    value: object,
    *,
    field: str,
    index: int,
) -> str | None:
    if value is None:
        return None

    return _require_string(value, field=field, index=index)


def _require_string_list(
    value: object,
    *,
    field: str,
    index: int,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"RAG evaluation case {index} field {field!r} must be a list.")

    terms = tuple(_require_string(term, field=field, index=index) for term in value)
    normalized_terms = tuple(_normalize_text(term) for term in terms)

    if len(set(normalized_terms)) != len(normalized_terms):
        raise ValueError(
            f"RAG evaluation case {index} field {field!r} contains duplicates."
        )

    return terms


def _calculate_metrics(
    cases: Sequence[RAGEvaluationCase],
    results: Sequence[RAGEvaluationCaseResult],
) -> RAGEvaluationMetrics:
    total_cases = len(cases)
    answer_results = tuple(
        result
        for case, result in zip(cases, results, strict=True)
        if case.expected_decision == "answer"
    )
    abstention_results = tuple(
        result
        for case, result in zip(cases, results, strict=True)
        if case.expected_decision == "abstain"
    )
    required_terms_total = sum(len(case.required_terms) for case in cases)
    required_terms_found = sum(len(result.required_terms_found) for result in results)

    return RAGEvaluationMetrics(
        total_cases=total_cases,
        passed_cases=sum(result.passed for result in results),
        answer_cases=len(answer_results),
        abstention_cases=len(abstention_results),
        overall_pass_rate=_rate(
            sum(result.passed for result in results),
            total_cases,
        ),
        decision_accuracy=_rate(
            sum(result.decision_correct for result in results),
            total_cases,
        ),
        answer_case_pass_rate=_rate(
            sum(result.passed for result in answer_results),
            len(answer_results),
        ),
        abstention_accuracy=_rate(
            sum(result.actual_decision == "abstain" for result in abstention_results),
            len(abstention_results),
        ),
        expected_section_accuracy=_rate(
            sum(result.expected_section_found for result in answer_results),
            len(answer_results),
        ),
        required_term_recall=_rate(
            required_terms_found,
            required_terms_total,
        ),
        forbidden_claim_violation_rate=_rate(
            sum(bool(result.forbidden_terms_found) for result in results),
            total_cases,
        ),
        citation_presence_rate=_rate(
            sum(result.citation_present for result in answer_results),
            len(answer_results),
        ),
    )


def _error_case_result(
    case: RAGEvaluationCase,
    message: str,
) -> RAGEvaluationCaseResult:
    return RAGEvaluationCaseResult(
        case_id=case.case_id,
        question=case.question,
        expected_decision=case.expected_decision,
        actual_decision="error",
        decision_correct=False,
        expected_section=case.expected_section,
        cited_sections=(),
        expected_section_found=False,
        required_terms_found=(),
        required_terms_missing=case.required_terms,
        required_term_recall=_rate(0, len(case.required_terms)),
        forbidden_terms_found=(),
        citation_present=False,
        top_score=None,
        answer=None,
        passed=False,
        message=f"Evaluation execution failed: {message}",
    )


def _build_case_message(
    *,
    decision_correct: bool,
    citation_expectation_met: bool,
    required_terms_missing: tuple[str, ...],
    forbidden_terms_found: tuple[str, ...],
) -> str:
    failures: list[str] = []

    if not decision_correct:
        failures.append("decision mismatch")
    if not citation_expectation_met:
        failures.append("citation expectation not met")
    if required_terms_missing:
        failures.append("missing required terms")
    if forbidden_terms_found:
        failures.append("forbidden claims found")

    if not failures:
        return "Evaluation case passed."

    return f"Evaluation case failed: {', '.join(failures)}."


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(normalized.split())


def _rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 1.0

    return numerator / denominator


def _utc_now() -> datetime:
    return datetime.now(UTC)


if __name__ == "__main__":
    raise SystemExit(main())
