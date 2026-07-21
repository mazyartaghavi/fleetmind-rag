from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from fleetmind_rag.evaluation import (
    RAGEvaluationCase,
    RAGEvaluationMetrics,
    RAGEvaluationReport,
    RAGEvaluator,
    build_evaluation_summary,
    load_evaluation_cases,
    main,
    write_evaluation_report,
)
from fleetmind_rag.grounded_rag import GroundedAnswerResult, GroundedCitation


class _FakeAnswerClient:
    def __init__(
        self,
        responses: dict[str, GroundedAnswerResult | Exception],
    ) -> None:
        self._responses = responses
        self.calls: list[tuple[str, int]] = []

    def answer(
        self,
        question: str,
        *,
        limit: int = 5,
    ) -> GroundedAnswerResult:
        self.calls.append((question, limit))
        response = self._responses[question]

        if isinstance(response, Exception):
            raise response

        return response


def _case(
    *,
    case_id: str = "case-1",
    question: str = "What should the driver do?",
    expected_decision: str = "answer",
    expected_section: str | None = "Safety",
    required_terms: tuple[str, ...] = ("stop safely",),
    forbidden_terms: tuple[str, ...] = (),
) -> RAGEvaluationCase:
    assert expected_decision in {"answer", "abstain"}
    return RAGEvaluationCase(
        case_id=case_id,
        question=question,
        expected_decision=expected_decision,  # type: ignore[arg-type]
        expected_section=expected_section,
        required_terms=required_terms,
        forbidden_terms=forbidden_terms,
    )


def _citation(
    *,
    section_title: str = "Safety",
    label: str = "S1",
) -> GroundedCitation:
    return GroundedCitation(
        label=label,
        chunk_id=f"chunk-{label}",
        document_id="document-1",
        section_id="section-1",
        section_title=section_title,
        text="The driver must stop safely.",
        score=0.91,
    )


def _answer_result(
    *,
    question: str = "What should the driver do?",
    answer: str = "The driver must stop safely [S1].",
    citations: tuple[GroundedCitation, ...] = (_citation(),),
    abstained: bool = False,
    succeeded: bool = True,
    message: str = "Generated a citation-grounded answer.",
) -> GroundedAnswerResult:
    return GroundedAnswerResult(
        succeeded=succeeded,
        abstained=abstained,
        question=question,
        answer=answer,
        citations=citations,
        retrieval_model="embeddinggemma",
        generation_model=None if abstained else "llama3.2:3b",
        top_score=0.91,
        message=message,
    )


def _report(*, passed_cases: int = 1, total_cases: int = 1) -> RAGEvaluationReport:
    metrics = RAGEvaluationMetrics(
        total_cases=total_cases,
        passed_cases=passed_cases,
        answer_cases=1,
        abstention_cases=0,
        overall_pass_rate=passed_cases / total_cases,
        decision_accuracy=1.0,
        answer_case_pass_rate=passed_cases / total_cases,
        abstention_accuracy=1.0,
        expected_section_accuracy=1.0,
        required_term_recall=1.0,
        forbidden_claim_violation_rate=0.0,
        citation_presence_rate=1.0,
    )
    return RAGEvaluationReport(
        generated_at_utc="2026-07-21T12:00:00+00:00",
        retrieval_limit=5,
        metrics=metrics,
        cases=(),
    )


def test_load_evaluation_cases_reads_valid_dataset(tmp_path: Path) -> None:
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps(
            [
                {
                    "case_id": "supported",
                    "question": "What is required?",
                    "expected_decision": "answer",
                    "expected_section": "Safety",
                    "required_terms": ["stop"],
                    "forbidden_terms": [],
                },
                {
                    "case_id": "unsupported",
                    "question": "What is the insurance cost?",
                    "expected_decision": "abstain",
                    "expected_section": None,
                    "required_terms": [],
                    "forbidden_terms": ["euro"],
                },
            ]
        ),
        encoding="utf-8",
    )

    cases = load_evaluation_cases(path)

    assert len(cases) == 2
    assert cases[0].case_id == "supported"
    assert cases[0].required_terms == ("stop",)
    assert cases[1].expected_decision == "abstain"


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({}, "root must be a list"),
        ([], "at least one case"),
        (["not-an-object"], "must be a JSON object"),
    ],
)
def test_load_evaluation_cases_rejects_invalid_roots(
    tmp_path: Path,
    payload: object,
    message: str,
) -> None:
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_evaluation_cases(path)


def test_load_evaluation_cases_reports_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "cases.json"
    path.write_text("[{", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid at line"):
        load_evaluation_cases(path)


def test_load_evaluation_cases_rejects_missing_and_unknown_fields(
    tmp_path: Path,
) -> None:
    missing_path = tmp_path / "missing.json"
    missing_path.write_text(json.dumps([{"case_id": "case-1"}]), encoding="utf-8")

    with pytest.raises(ValueError, match="is missing"):
        load_evaluation_cases(missing_path)

    unknown_path = tmp_path / "unknown.json"
    unknown_path.write_text(
        json.dumps(
            [
                {
                    "case_id": "case-1",
                    "question": "Question",
                    "expected_decision": "answer",
                    "expected_section": "Safety",
                    "required_terms": [],
                    "forbidden_terms": [],
                    "typo": True,
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown fields"):
        load_evaluation_cases(unknown_path)


@pytest.mark.parametrize(
    ("decision", "section", "message"),
    [
        ("invalid", "Safety", "must be 'answer' or 'abstain'"),
        ("answer", None, "must define expected_section"),
        ("abstain", "Safety", "must use null expected_section"),
    ],
)
def test_load_evaluation_cases_validates_decision_and_section(
    tmp_path: Path,
    decision: str,
    section: str | None,
    message: str,
) -> None:
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps(
            [
                {
                    "case_id": "case-1",
                    "question": "Question",
                    "expected_decision": decision,
                    "expected_section": section,
                    "required_terms": [],
                    "forbidden_terms": [],
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_evaluation_cases(path)


def test_load_evaluation_cases_rejects_duplicate_ids_and_terms(
    tmp_path: Path,
) -> None:
    duplicate_ids = [
        {
            "case_id": "same",
            "question": "Question one",
            "expected_decision": "answer",
            "expected_section": "Safety",
            "required_terms": [],
            "forbidden_terms": [],
        },
        {
            "case_id": "same",
            "question": "Question two",
            "expected_decision": "abstain",
            "expected_section": None,
            "required_terms": [],
            "forbidden_terms": [],
        },
    ]
    ids_path = tmp_path / "ids.json"
    ids_path.write_text(json.dumps(duplicate_ids), encoding="utf-8")

    with pytest.raises(ValueError, match="identifiers must be unique"):
        load_evaluation_cases(ids_path)

    duplicate_terms = [
        {
            "case_id": "case-1",
            "question": "Question",
            "expected_decision": "answer",
            "expected_section": "Safety",
            "required_terms": ["Stop", " stop "],
            "forbidden_terms": [],
        }
    ]
    terms_path = tmp_path / "terms.json"
    terms_path.write_text(json.dumps(duplicate_terms), encoding="utf-8")

    with pytest.raises(ValueError, match="contains duplicates"):
        load_evaluation_cases(terms_path)


def test_load_evaluation_cases_rejects_required_forbidden_overlap(
    tmp_path: Path,
) -> None:
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps(
            [
                {
                    "case_id": "case-1",
                    "question": "Question",
                    "expected_decision": "answer",
                    "expected_section": "Safety",
                    "required_terms": ["Stop"],
                    "forbidden_terms": [" stop "],
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="require and forbid the same term"):
        load_evaluation_cases(path)


def test_evaluator_passes_supported_answer() -> None:
    case = _case()
    client = _FakeAnswerClient({case.question: _answer_result()})
    evaluator = RAGEvaluator(
        client,
        clock=lambda: datetime(2026, 7, 21, 12, 0, tzinfo=UTC),
    )

    report = evaluator.evaluate([case], limit=3)
    result = report.cases[0]

    assert report.generated_at_utc == "2026-07-21T12:00:00+00:00"
    assert report.retrieval_limit == 3
    assert result.actual_decision == "answer"
    assert result.expected_section_found is True
    assert result.required_terms_found == ("stop safely",)
    assert result.passed is True
    assert report.all_cases_passed is True
    assert client.calls == [(case.question, 3)]


def test_evaluator_passes_safe_abstention() -> None:
    case = _case(
        expected_decision="abstain",
        expected_section=None,
        required_terms=(),
        forbidden_terms=("euro",),
    )
    response = _answer_result(
        answer="I do not have enough grounded evidence.",
        citations=(),
        abstained=True,
    )

    report = RAGEvaluator(_FakeAnswerClient({case.question: response})).evaluate([case])
    result = report.cases[0]

    assert result.actual_decision == "abstain"
    assert result.citation_present is False
    assert result.passed is True
    assert report.metrics.abstention_accuracy == 1.0


def test_evaluator_detects_decision_citation_term_and_forbidden_failures() -> None:
    case = _case(
        expected_section="Safety",
        required_terms=("stop safely", "parking brake"),
        forbidden_terms=("continue driving",),
    )
    response = _answer_result(
        answer="Continue driving [S1].",
        citations=(_citation(section_title="Unrelated"),),
    )

    result = (
        RAGEvaluator(_FakeAnswerClient({case.question: response}))
        .evaluate([case])
        .cases[0]
    )

    assert result.expected_section_found is False
    assert result.required_terms_missing == ("stop safely", "parking brake")
    assert result.forbidden_terms_found == ("continue driving",)
    assert result.required_term_recall == 0.0
    assert result.passed is False
    assert "citation expectation not met" in result.message


def test_evaluator_detects_wrong_decision_and_missing_citation() -> None:
    case = _case()
    response = _answer_result(
        answer="I do not have enough grounded evidence.",
        citations=(),
        abstained=True,
    )

    result = (
        RAGEvaluator(_FakeAnswerClient({case.question: response}))
        .evaluate([case])
        .cases[0]
    )

    assert result.decision_correct is False
    assert result.citation_present is False
    assert result.passed is False
    assert "decision mismatch" in result.message


def test_evaluator_converts_service_failure_and_exception_to_error_results() -> None:
    failed_case = _case(case_id="failed", question="Failed")
    raised_case = _case(case_id="raised", question="Raised")
    failed_response = _answer_result(
        question="Failed",
        answer="",
        citations=(),
        succeeded=False,
        message="Generation failed.",
    )
    client = _FakeAnswerClient(
        {
            "Failed": failed_response,
            "Raised": RuntimeError("Ollama unavailable"),
        }
    )

    report = RAGEvaluator(client).evaluate([failed_case, raised_case])

    assert tuple(result.actual_decision for result in report.cases) == (
        "error",
        "error",
    )
    assert report.metrics.passed_cases == 0
    assert "Generation failed" in report.cases[0].message
    assert "Ollama unavailable" in report.cases[1].message


def test_evaluator_validates_cases_and_limit() -> None:
    evaluator = RAGEvaluator(_FakeAnswerClient({}))

    with pytest.raises(ValueError, match="At least one"):
        evaluator.evaluate([])

    with pytest.raises(ValueError, match="limit must be positive"):
        evaluator.evaluate([_case()], limit=0)

    with pytest.raises(ValueError, match="identifiers must be unique"):
        evaluator.evaluate([_case(), _case(question="Another")])


def test_evaluator_calculates_mixed_aggregate_metrics() -> None:
    answer_case = _case(case_id="answer", question="Answer")
    abstain_case = _case(
        case_id="abstain",
        question="Abstain",
        expected_decision="abstain",
        expected_section=None,
        required_terms=(),
        forbidden_terms=("euro",),
    )
    client = _FakeAnswerClient(
        {
            "Answer": _answer_result(question="Answer"),
            "Abstain": _answer_result(
                question="Abstain",
                answer="The deductible is 50 euro.",
                citations=(),
                abstained=True,
            ),
        }
    )

    metrics = RAGEvaluator(client).evaluate([answer_case, abstain_case]).metrics

    assert metrics.total_cases == 2
    assert metrics.passed_cases == 1
    assert metrics.answer_cases == 1
    assert metrics.abstention_cases == 1
    assert metrics.overall_pass_rate == 0.5
    assert metrics.decision_accuracy == 1.0
    assert metrics.answer_case_pass_rate == 1.0
    assert metrics.abstention_accuracy == 1.0
    assert metrics.expected_section_accuracy == 1.0
    assert metrics.required_term_recall == 1.0
    assert metrics.forbidden_claim_violation_rate == 0.5
    assert metrics.citation_presence_rate == 1.0


def test_write_evaluation_report_creates_json_file(tmp_path: Path) -> None:
    report = _report()
    path = tmp_path / "nested" / "report.json"

    result_path = write_evaluation_report(report, path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert result_path == path
    assert payload["metrics"]["total_cases"] == 1
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_build_evaluation_summary_reports_metrics_and_path() -> None:
    summary = build_evaluation_summary(
        _report(),
        report_path="evaluation/reports/report.json",
    )

    assert "Cases passed: 1/1" in summary
    assert "Decision accuracy: 100.00%" in summary
    assert "Result: PASS" in summary
    assert "Report: evaluation\\reports\\report.json" in summary or (
        "Report: evaluation/reports/report.json" in summary
    )


def test_main_returns_success_or_quality_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "fleetmind_rag.evaluation.run_live_evaluation",
        lambda *args, **kwargs: _report(),
    )

    assert main([]) == 0
    assert "Result: PASS" in capsys.readouterr().out

    monkeypatch.setattr(
        "fleetmind_rag.evaluation.run_live_evaluation",
        lambda *args, **kwargs: _report(passed_cases=0),
    )

    assert main([]) == 2
    assert "Result: FAIL" in capsys.readouterr().out


def test_main_reports_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def _raise(*args: object, **kwargs: object) -> RAGEvaluationReport:
        del args, kwargs
        raise RuntimeError("Ollama unavailable")

    monkeypatch.setattr(
        "fleetmind_rag.evaluation.run_live_evaluation",
        _raise,
    )

    assert main([]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Ollama unavailable" in captured.err
