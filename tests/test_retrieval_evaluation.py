from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from fleetmind_rag.retrieval import (
    HybridRetrievalResponse,
    RerankedRetrievalResponse,
    RerankedSearchResult,
    RetrievalResponse,
    SparseRetrievalResponse,
)
from fleetmind_rag.retrieval_evaluation import (
    STRATEGIES,
    RetrievalEvaluationCase,
    RetrievalEvaluationReport,
    RetrievalEvaluator,
    RetrievalStrategy,
    RetrievalStrategyMetrics,
    build_retrieval_evaluation_summary,
    load_retrieval_evaluation_cases,
    main,
    write_retrieval_evaluation_report,
)
from fleetmind_rag.vector_store import VectorSearchResult


def _match(section: str, *, score: float = 1.0) -> VectorSearchResult:
    slug = section.lower().replace(" ", "-")
    return VectorSearchResult(
        chunk_id=f"chunk-{slug}",
        document_id="doc-manual",
        section_id=f"section-{slug}",
        section_title=section,
        ordinal=0,
        text=f"Text for {section}.",
        word_count=4,
        start_word=0,
        end_word=4,
        score=score,
    )


def _reranked_match(section: str, *, score: float = 1.0) -> RerankedSearchResult:
    base = _match(section, score=score)
    return RerankedSearchResult(
        chunk_id=base.chunk_id,
        document_id=base.document_id,
        section_id=base.section_id,
        section_title=base.section_title,
        ordinal=base.ordinal,
        text=base.text,
        word_count=base.word_count,
        start_word=base.start_word,
        end_word=base.end_word,
        score=score,
        hybrid_score=0.03,
        original_rank=1,
        lexical_coverage=1.0,
        section_title_coverage=1.0,
        exact_phrase_match=False,
    )


@dataclass
class FakeRetrievalClient:
    sections_by_strategy: dict[RetrievalStrategy, tuple[str, ...]] = field(
        default_factory=lambda: {
            "dense": ("Expected", "Other"),
            "sparse": ("Expected",),
            "hybrid": ("Expected", "Other"),
            "reranked": ("Expected", "Other"),
        }
    )
    fail_strategy: RetrievalStrategy | None = None
    calls: list[tuple[RetrievalStrategy, str, int, int | None]] = field(
        default_factory=list
    )

    def search(self, query: str, *, limit: int = 5) -> RetrievalResponse:
        self.calls.append(("dense", query, limit, None))
        self._maybe_fail("dense")
        return RetrievalResponse(
            query=query,
            embedding_model="embeddinggemma",
            matches=tuple(
                _match(section)
                for section in self.sections_by_strategy["dense"][:limit]
            ),
        )

    def search_sparse(
        self,
        query: str,
        *,
        limit: int = 5,
    ) -> SparseRetrievalResponse:
        self.calls.append(("sparse", query, limit, None))
        self._maybe_fail("sparse")
        return SparseRetrievalResponse(
            query=query,
            algorithm="bm25-local-v1",
            matches=tuple(
                _match(section)
                for section in self.sections_by_strategy["sparse"][:limit]
            ),
        )

    def search_hybrid(
        self,
        query: str,
        *,
        limit: int = 5,
        candidate_limit: int = 20,
    ) -> HybridRetrievalResponse:
        self.calls.append(("hybrid", query, limit, candidate_limit))
        self._maybe_fail("hybrid")
        sections = self.sections_by_strategy["hybrid"][:limit]
        return HybridRetrievalResponse(
            query=query,
            algorithm="rrf-dense-bm25-v1",
            embedding_model="embeddinggemma",
            dense_match_count=len(sections),
            sparse_match_count=len(sections),
            matches=tuple(_match(section) for section in sections),
        )

    def search_hybrid_reranked(
        self,
        query: str,
        *,
        limit: int = 5,
        candidate_limit: int = 20,
    ) -> RerankedRetrievalResponse:
        self.calls.append(("reranked", query, limit, candidate_limit))
        self._maybe_fail("reranked")
        sections = self.sections_by_strategy["reranked"][:limit]
        return RerankedRetrievalResponse(
            query=query,
            algorithm="hybrid-rrf-lexical-rerank-v1",
            embedding_model="embeddinggemma",
            dense_match_count=len(sections),
            sparse_match_count=len(sections),
            candidate_count=len(sections),
            matches=tuple(_reranked_match(section) for section in sections),
        )

    def _maybe_fail(self, strategy: RetrievalStrategy) -> None:
        if self.fail_strategy == strategy:
            raise RuntimeError(f"simulated {strategy} failure")


def _case(
    *,
    case_id: str = "case-1",
    expected_section: str = "Expected",
    expected_rank_at_most: int = 1,
) -> RetrievalEvaluationCase:
    return RetrievalEvaluationCase(
        case_id=case_id,
        query="example query",
        expected_section=expected_section,
        expected_rank_at_most=expected_rank_at_most,
    )


def _write_cases(tmp_path: Path, value: object) -> Path:
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _metrics(
    strategy: RetrievalStrategy,
    *,
    mrr: float,
    top_1: float,
    passed: int = 1,
) -> RetrievalStrategyMetrics:
    return RetrievalStrategyMetrics(
        strategy=strategy,
        total_cases=1,
        passed_cases=passed,
        top_1_correct_cases=int(top_1 == 1.0),
        hits_at_k=passed,
        empty_results=0,
        errors=0,
        mean_result_count=1.0,
        pass_rate=float(passed),
        top_1_accuracy=top_1,
        hit_rate_at_k=float(passed),
        mean_reciprocal_rank=mrr,
        empty_result_rate=0.0,
        error_rate=0.0,
    )


def _report(*metrics: RetrievalStrategyMetrics) -> RetrievalEvaluationReport:
    return RetrievalEvaluationReport(
        generated_at_utc="2026-07-21T20:00:00+00:00",
        retrieval_limit=4,
        candidate_limit=12,
        strategies=tuple(metric.strategy for metric in metrics),
        metrics=metrics,
        cases=(),
    )


def test_load_cases_accepts_and_strips_valid_values(tmp_path: Path) -> None:
    path = _write_cases(
        tmp_path,
        [
            {
                "case_id": " case-a ",
                "query": " warning query ",
                "expected_section": " Battery ",
                "expected_rank_at_most": 2,
            }
        ],
    )

    cases = load_retrieval_evaluation_cases(path)

    assert cases == (
        RetrievalEvaluationCase(
            case_id="case-a",
            query="warning query",
            expected_section="Battery",
            expected_rank_at_most=2,
        ),
    )


@pytest.mark.parametrize("value", [{}, "cases", 3, None])
def test_load_cases_requires_list_root(tmp_path: Path, value: object) -> None:
    with pytest.raises(ValueError, match="root must be a list"):
        load_retrieval_evaluation_cases(_write_cases(tmp_path, value))


def test_load_cases_rejects_empty_dataset(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one case"):
        load_retrieval_evaluation_cases(_write_cases(tmp_path, []))


def test_load_cases_rejects_non_object_case(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be a JSON object"):
        load_retrieval_evaluation_cases(_write_cases(tmp_path, ["case"]))


def test_load_cases_rejects_missing_fields(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="is missing"):
        load_retrieval_evaluation_cases(_write_cases(tmp_path, [{"case_id": "case-a"}]))


def test_load_cases_rejects_unknown_fields(tmp_path: Path) -> None:
    value = [
        {
            "case_id": "case-a",
            "query": "query",
            "expected_section": "Section",
            "expected_rank_at_most": 1,
            "unexpected": True,
        }
    ]
    with pytest.raises(ValueError, match="unknown fields"):
        load_retrieval_evaluation_cases(_write_cases(tmp_path, value))


@pytest.mark.parametrize("field", ["case_id", "query", "expected_section"])
def test_load_cases_rejects_blank_strings(tmp_path: Path, field: str) -> None:
    value = {
        "case_id": "case-a",
        "query": "query",
        "expected_section": "Section",
        "expected_rank_at_most": 1,
    }
    value[field] = "   "

    with pytest.raises(ValueError, match="non-empty string"):
        load_retrieval_evaluation_cases(_write_cases(tmp_path, [value]))


@pytest.mark.parametrize("rank", [0, -1, True, 1.5, "1"])
def test_load_cases_requires_positive_integer_rank(
    tmp_path: Path,
    rank: object,
) -> None:
    value = [
        {
            "case_id": "case-a",
            "query": "query",
            "expected_section": "Section",
            "expected_rank_at_most": rank,
        }
    ]
    with pytest.raises(ValueError, match="positive integer"):
        load_retrieval_evaluation_cases(_write_cases(tmp_path, value))


def test_load_cases_rejects_duplicate_identifiers(tmp_path: Path) -> None:
    value = [
        {
            "case_id": "duplicate",
            "query": "first",
            "expected_section": "Section",
            "expected_rank_at_most": 1,
        },
        {
            "case_id": "duplicate",
            "query": "second",
            "expected_section": "Section",
            "expected_rank_at_most": 1,
        },
    ]
    with pytest.raises(ValueError, match="identifiers must be unique"):
        load_retrieval_evaluation_cases(_write_cases(tmp_path, value))


def test_evaluate_requires_cases() -> None:
    with pytest.raises(ValueError, match="At least one"):
        RetrievalEvaluator(FakeRetrievalClient()).evaluate([])


def test_evaluate_rejects_duplicate_case_ids() -> None:
    with pytest.raises(ValueError, match="identifiers must be unique"):
        RetrievalEvaluator(FakeRetrievalClient()).evaluate([_case(), _case()])


def test_evaluate_requires_strategies() -> None:
    with pytest.raises(ValueError, match="strategy is required"):
        RetrievalEvaluator(FakeRetrievalClient()).evaluate([_case()], strategies=[])


def test_evaluate_rejects_duplicate_strategies() -> None:
    with pytest.raises(ValueError, match="strategies must be unique"):
        RetrievalEvaluator(FakeRetrievalClient()).evaluate(
            [_case()], strategies=["dense", "dense"]
        )


def test_evaluate_rejects_unknown_strategy() -> None:
    with pytest.raises(ValueError, match="Unknown retrieval strategies"):
        RetrievalEvaluator(FakeRetrievalClient()).evaluate(
            [_case()],
            strategies=cast(list[RetrievalStrategy], ["unknown"]),
        )


@pytest.mark.parametrize("limit", [0, -1])
def test_evaluate_requires_positive_limit(limit: int) -> None:
    with pytest.raises(ValueError, match="limit must be positive"):
        RetrievalEvaluator(FakeRetrievalClient()).evaluate([_case()], limit=limit)


def test_evaluate_requires_candidate_limit_at_least_result_limit() -> None:
    with pytest.raises(ValueError, match="candidate limit"):
        RetrievalEvaluator(FakeRetrievalClient()).evaluate(
            [_case()], limit=4, candidate_limit=3
        )


def test_evaluate_routes_all_strategies_and_forwards_limits() -> None:
    client = FakeRetrievalClient()

    report = RetrievalEvaluator(client).evaluate([_case()], limit=2, candidate_limit=7)

    assert report.strategies == STRATEGIES
    assert client.calls == [
        ("dense", "example query", 2, None),
        ("sparse", "example query", 2, None),
        ("hybrid", "example query", 2, 7),
        ("reranked", "example query", 2, 7),
    ]


def test_evaluate_calculates_rank_metrics() -> None:
    client = FakeRetrievalClient(
        sections_by_strategy={
            "dense": ("Other", "Expected"),
            "sparse": ("Expected",),
            "hybrid": ("Other", "Expected"),
            "reranked": ("Expected", "Other"),
        }
    )

    report = RetrievalEvaluator(client).evaluate(
        [_case(expected_rank_at_most=2)], limit=2, candidate_limit=2
    )

    dense_case = report.cases[0]
    assert dense_case.expected_rank == 2
    assert not dense_case.top_1_correct
    assert dense_case.hit_at_k
    assert dense_case.reciprocal_rank == pytest.approx(0.5)
    assert dense_case.passed
    assert report.metrics[0].mean_reciprocal_rank == pytest.approx(0.5)
    assert report.metrics[1].top_1_accuracy == pytest.approx(1.0)


def test_evaluate_matches_section_case_and_whitespace_insensitively() -> None:
    client = FakeRetrievalClient()
    client.sections_by_strategy["dense"] = ("  EXPECTED  ",)

    report = RetrievalEvaluator(client).evaluate(
        [_case(expected_section="expected")], strategies=["dense"]
    )

    assert report.cases[0].expected_rank == 1
    assert report.cases[0].passed


def test_evaluate_fails_case_when_rank_exceeds_gate() -> None:
    client = FakeRetrievalClient()
    client.sections_by_strategy["dense"] = ("Other", "Expected")

    result = (
        RetrievalEvaluator(client)
        .evaluate([_case(expected_rank_at_most=1)], strategies=["dense"], limit=2)
        .cases[0]
    )

    assert result.expected_rank == 2
    assert not result.passed
    assert "exceeding" in result.message


def test_evaluate_records_empty_results() -> None:
    client = FakeRetrievalClient()
    client.sections_by_strategy["sparse"] = ()

    report = RetrievalEvaluator(client).evaluate([_case()], strategies=["sparse"])

    result = report.cases[0]
    metric = report.metrics[0]
    assert result.retrieved_sections == ()
    assert result.top_section is None
    assert result.expected_rank is None
    assert not result.passed
    assert metric.empty_results == 1
    assert metric.empty_result_rate == pytest.approx(1.0)


def test_evaluate_records_strategy_error_without_stopping_other_strategies() -> None:
    client = FakeRetrievalClient(fail_strategy="sparse")

    report = RetrievalEvaluator(client).evaluate(
        [_case()], strategies=["dense", "sparse"]
    )

    assert report.cases[0].passed
    assert not report.cases[1].passed
    assert "simulated sparse failure" in report.cases[1].message
    assert report.metrics[1].errors == 1
    assert report.metrics[1].error_rate == pytest.approx(1.0)


def test_report_quality_gate_requires_every_strategy_case_to_pass() -> None:
    passing = _metrics("dense", mrr=1.0, top_1=1.0)
    failing = _metrics("sparse", mrr=0.0, top_1=0.0, passed=0)

    assert _report(passing).all_quality_gates_passed
    assert not _report(passing, failing).all_quality_gates_passed


def test_report_best_strategy_prefers_mrr_then_top1() -> None:
    report = _report(
        _metrics("dense", mrr=0.5, top_1=1.0),
        _metrics("sparse", mrr=0.75, top_1=0.0),
        _metrics("hybrid", mrr=0.75, top_1=1.0),
    )

    assert report.best_strategy == "hybrid"


def test_report_best_strategy_uses_stable_strategy_order_for_ties() -> None:
    report = _report(
        _metrics("dense", mrr=1.0, top_1=1.0),
        _metrics("reranked", mrr=1.0, top_1=1.0),
    )

    assert report.best_strategy == "dense"


def test_report_best_strategy_requires_metrics() -> None:
    with pytest.raises(RuntimeError, match="no strategy metrics"):
        _ = _report().best_strategy


def test_evaluate_uses_injected_clock() -> None:
    clock_value = datetime(2026, 7, 21, 20, 30, tzinfo=UTC)

    report = RetrievalEvaluator(
        FakeRetrievalClient(), clock=lambda: clock_value
    ).evaluate([_case()], strategies=["dense"])

    assert report.generated_at_utc == clock_value.isoformat()


def test_write_report_creates_parent_and_serializes_json(tmp_path: Path) -> None:
    report = _report(_metrics("dense", mrr=1.0, top_1=1.0))

    path = write_retrieval_evaluation_report(
        report, tmp_path / "nested" / "report.json"
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["metrics"][0]["strategy"] == "dense"
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_summary_contains_comparison_result_and_path(tmp_path: Path) -> None:
    report = _report(_metrics("dense", mrr=1.0, top_1=1.0))

    summary = build_retrieval_evaluation_summary(
        report, report_path=tmp_path / "report.json"
    )

    assert "Strategy" in summary
    assert "dense" in summary
    assert "Best strategy: dense" in summary
    assert "Result: PASS" in summary
    assert "report.json" in summary


def test_main_returns_zero_for_passing_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report = _report(_metrics("dense", mrr=1.0, top_1=1.0))
    monkeypatch.setattr(
        "fleetmind_rag.retrieval_evaluation.run_live_retrieval_evaluation",
        lambda *_args, **_kwargs: report,
    )

    assert main(["--report", str(tmp_path / "report.json")]) == 0


def test_main_returns_two_for_quality_gate_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _report(_metrics("dense", mrr=0.0, top_1=0.0, passed=0))
    monkeypatch.setattr(
        "fleetmind_rag.retrieval_evaluation.run_live_retrieval_evaluation",
        lambda *_args, **_kwargs: report,
    )

    assert main([]) == 2


def test_main_returns_one_for_execution_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(*_args: object, **_kwargs: object) -> RetrievalEvaluationReport:
        raise RuntimeError("benchmark unavailable")

    monkeypatch.setattr(
        "fleetmind_rag.retrieval_evaluation.run_live_retrieval_evaluation",
        fail,
    )

    assert main([]) == 1
    assert "benchmark unavailable" in capsys.readouterr().err
