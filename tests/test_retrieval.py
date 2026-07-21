from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from fleetmind_rag.documents import ingest_text_document
from fleetmind_rag.ollama import OllamaEmbeddingResult
from fleetmind_rag.retrieval import DocumentRetrievalService
from fleetmind_rag.vector_store import (
    ChunkMetadataFilter,
    QdrantChunkStore,
    VectorSearchResult,
)


@dataclass
class FakeEmbeddingClient:
    fail: bool = False
    forced_embeddings: tuple[tuple[float, ...], ...] | None = None
    model: str | None = "embeddinggemma"
    calls: list[str | list[str] | tuple[str, ...]] = field(default_factory=list)

    def embed(
        self,
        input_value: str | list[str] | tuple[str, ...],
    ) -> OllamaEmbeddingResult:
        self.calls.append(input_value)

        if self.fail:
            return OllamaEmbeddingResult(
                succeeded=False,
                embeddings=(),
                model=None,
                message="simulated embedding failure",
            )

        texts = (input_value,) if isinstance(input_value, str) else tuple(input_value)
        embeddings = self.forced_embeddings

        if embeddings is None:
            embeddings = tuple(_embedding_for_text(text) for text in texts)

        return OllamaEmbeddingResult(
            succeeded=True,
            embeddings=embeddings,
            model=self.model,
            message="generated test embeddings",
        )


def _embedding_for_text(text: str) -> tuple[float, ...]:
    lowered = text.lower()

    if "engine" in lowered or "oil" in lowered:
        return (1.0, 0.0, 0.0)

    if "tire" in lowered or "pressure" in lowered:
        return (0.0, 1.0, 0.0)

    return (0.0, 0.0, 1.0)


def _write_manual(tmp_path: Path, *, name: str = "manual.md") -> Path:
    path = tmp_path / name
    path.write_text(
        "# Engine\n"
        "Engine oil pressure warnings require the vehicle to stop safely.\n\n"
        "# Tires\n"
        "Tire pressure must be checked before long-distance operation.\n",
        encoding="utf-8",
    )
    return path


def test_index_text_document_returns_typed_summary(tmp_path: Path) -> None:
    client = FakeEmbeddingClient()

    with QdrantChunkStore.in_memory() as store:
        service = DocumentRetrievalService(client, store)
        result = service.index_text_document(_write_manual(tmp_path))

        assert result.source_name == "manual.md"
        assert result.section_count == 2
        assert result.chunk_count == 2
        assert result.stored_count == 2
        assert result.embedding_model == "embeddinggemma"
        assert result.vector_size == 3
        assert service.count() == 2


def test_index_text_document_sends_chunk_texts_as_one_batch(tmp_path: Path) -> None:
    client = FakeEmbeddingClient()

    with QdrantChunkStore.in_memory() as store:
        service = DocumentRetrievalService(client, store)
        service.index_text_document(_write_manual(tmp_path))

    assert len(client.calls) == 1
    call = client.calls[0]
    assert isinstance(call, list)
    assert len(call) == 2
    assert "Engine oil pressure" in call[0]
    assert "Tire pressure" in call[1]


def test_index_document_accepts_preingested_document(tmp_path: Path) -> None:
    ingested = ingest_text_document(_write_manual(tmp_path))

    with QdrantChunkStore.in_memory() as store:
        service = DocumentRetrievalService(FakeEmbeddingClient(), store)
        result = service.index_document(ingested)

        assert result.document_id == ingested.document.document_id
        assert result.stored_count == len(ingested.chunks)


def test_reindexing_same_document_replaces_existing_points(tmp_path: Path) -> None:
    path = _write_manual(tmp_path)

    with QdrantChunkStore.in_memory() as store:
        service = DocumentRetrievalService(FakeEmbeddingClient(), store)
        service.index_text_document(path)
        service.index_text_document(path)

        assert service.count() == 2


def test_recreate_collection_removes_previous_document(tmp_path: Path) -> None:
    first_path = _write_manual(tmp_path, name="first.md")
    second_path = tmp_path / "second.md"
    second_path.write_text(
        "# Battery\nBattery voltage must be monitored.",
        encoding="utf-8",
    )

    with QdrantChunkStore.in_memory() as store:
        service = DocumentRetrievalService(FakeEmbeddingClient(), store)
        service.index_text_document(first_path)
        result = service.index_text_document(second_path, recreate_collection=True)

        assert result.chunk_count == 1
        assert service.count() == 1


def test_custom_chunk_configuration_is_forwarded(tmp_path: Path) -> None:
    path = tmp_path / "long.txt"
    path.write_text(" ".join(f"word-{index}" for index in range(20)), encoding="utf-8")

    with QdrantChunkStore.in_memory() as store:
        service = DocumentRetrievalService(FakeEmbeddingClient(), store)
        result = service.index_text_document(
            path,
            default_title="Long section",
            chunk_size_words=8,
            overlap_words=2,
        )

        assert result.section_count == 1
        assert result.chunk_count == 3
        assert result.stored_count == 3


def test_search_returns_best_matching_section(tmp_path: Path) -> None:
    with QdrantChunkStore.in_memory() as store:
        service = DocumentRetrievalService(FakeEmbeddingClient(), store)
        service.index_text_document(_write_manual(tmp_path))

        response = service.search("  engine warning  ", limit=1)

        assert response.query == "engine warning"
        assert response.embedding_model == "embeddinggemma"
        assert len(response.matches) == 1
        assert response.matches[0].section_title == "Engine"
        assert response.matches[0].score == pytest.approx(1.0)


def test_search_forwards_score_threshold(tmp_path: Path) -> None:
    with QdrantChunkStore.in_memory() as store:
        service = DocumentRetrievalService(FakeEmbeddingClient(), store)
        service.index_text_document(_write_manual(tmp_path))

        response = service.search("engine", score_threshold=1.1)

        assert response.matches == ()


def test_count_is_zero_before_indexing() -> None:
    with QdrantChunkStore.in_memory() as store:
        service = DocumentRetrievalService(FakeEmbeddingClient(), store)

        assert service.count() == 0


def test_empty_search_query_is_rejected() -> None:
    with QdrantChunkStore.in_memory() as store:
        service = DocumentRetrievalService(FakeEmbeddingClient(), store)

        with pytest.raises(ValueError, match="must not be empty"):
            service.search("   ")


def test_search_before_indexing_reports_missing_collection() -> None:
    with QdrantChunkStore.in_memory() as store:
        service = DocumentRetrievalService(FakeEmbeddingClient(), store)

        with pytest.raises(RuntimeError, match="collection does not exist"):
            service.search("engine")


def test_indexing_embedding_failure_is_reported(tmp_path: Path) -> None:
    with QdrantChunkStore.in_memory() as store:
        service = DocumentRetrievalService(FakeEmbeddingClient(fail=True), store)

        with pytest.raises(RuntimeError, match="simulated embedding failure"):
            service.index_text_document(_write_manual(tmp_path))

        assert service.count() == 0


def test_search_embedding_failure_is_reported(tmp_path: Path) -> None:
    client = FakeEmbeddingClient()

    with QdrantChunkStore.in_memory() as store:
        service = DocumentRetrievalService(client, store)
        service.index_text_document(_write_manual(tmp_path))
        client.fail = True

        with pytest.raises(RuntimeError, match="simulated embedding failure"):
            service.search("engine")


def test_indexing_rejects_wrong_embedding_count(tmp_path: Path) -> None:
    client = FakeEmbeddingClient(forced_embeddings=((1.0, 0.0, 0.0),))

    with QdrantChunkStore.in_memory() as store:
        service = DocumentRetrievalService(client, store)

        with pytest.raises(RuntimeError, match="count does not match"):
            service.index_text_document(_write_manual(tmp_path))


def test_search_rejects_multiple_query_embeddings(tmp_path: Path) -> None:
    client = FakeEmbeddingClient()

    with QdrantChunkStore.in_memory() as store:
        service = DocumentRetrievalService(client, store)
        service.index_text_document(_write_manual(tmp_path))
        client.forced_embeddings = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))

        with pytest.raises(RuntimeError, match="count does not match"):
            service.search("engine")


def test_successful_embedding_response_requires_model_name(tmp_path: Path) -> None:
    client = FakeEmbeddingClient(model=None)

    with QdrantChunkStore.in_memory() as store:
        service = DocumentRetrievalService(client, store)

        with pytest.raises(RuntimeError, match="no model name"):
            service.index_text_document(_write_manual(tmp_path))


def test_inconsistent_embedding_dimensions_are_rejected(tmp_path: Path) -> None:
    client = FakeEmbeddingClient(
        forced_embeddings=((1.0, 0.0, 0.0), (0.0, 1.0)),
    )

    with QdrantChunkStore.in_memory() as store:
        service = DocumentRetrievalService(client, store)

        with pytest.raises(RuntimeError, match="inconsistent vector dimensions"):
            service.index_text_document(_write_manual(tmp_path))


def test_non_finite_embedding_values_are_rejected(tmp_path: Path) -> None:
    client = FakeEmbeddingClient(
        forced_embeddings=((float("nan"), 0.0, 0.0), (0.0, 1.0, 0.0)),
    )

    with QdrantChunkStore.in_memory() as store:
        service = DocumentRetrievalService(client, store)

        with pytest.raises(RuntimeError, match="non-finite"):
            service.index_text_document(_write_manual(tmp_path))


def test_invalid_search_limit_is_delegated_to_vector_store(tmp_path: Path) -> None:
    with QdrantChunkStore.in_memory() as store:
        service = DocumentRetrievalService(FakeEmbeddingClient(), store)
        service.index_text_document(_write_manual(tmp_path))

        with pytest.raises(ValueError, match="limit must be greater than zero"):
            service.search("engine", limit=0)


def test_search_forwards_section_title_metadata_filter(tmp_path: Path) -> None:
    with QdrantChunkStore.in_memory() as store:
        service = DocumentRetrievalService(FakeEmbeddingClient(), store)
        service.index_text_document(_write_manual(tmp_path))

        response = service.search(
            "engine",
            metadata_filter=ChunkMetadataFilter(section_titles=("Tires",)),
        )

    assert len(response.matches) == 1
    assert response.matches[0].section_title == "Tires"


def test_search_forwards_document_id_metadata_filter(tmp_path: Path) -> None:
    first_path = _write_manual(tmp_path, name="first.md")
    second_path = tmp_path / "second.md"
    second_path.write_text(
        "# Battery\nBattery voltage must be monitored.",
        encoding="utf-8",
    )

    with QdrantChunkStore.in_memory() as store:
        service = DocumentRetrievalService(FakeEmbeddingClient(), store)
        service.index_text_document(first_path)
        second_result = service.index_text_document(second_path)

        response = service.search(
            "engine",
            metadata_filter=ChunkMetadataFilter(
                document_ids=(second_result.document_id,)
            ),
        )

    assert len(response.matches) == 1
    assert response.matches[0].document_id == second_result.document_id
    assert response.matches[0].section_title == "Battery"


def test_sparse_search_returns_bm25_response_without_embedding_query(
    tmp_path: Path,
) -> None:
    client = FakeEmbeddingClient()

    with QdrantChunkStore.in_memory() as store:
        service = DocumentRetrievalService(client, store)
        service.index_text_document(_write_manual(tmp_path))
        calls_after_indexing = len(client.calls)

        response = service.search_sparse("tire pressure", limit=1)

    assert response.query == "tire pressure"
    assert response.algorithm == "bm25-local-v1"
    assert len(response.matches) == 1
    assert response.matches[0].section_title == "Tires"
    assert len(client.calls) == calls_after_indexing


def test_sparse_search_forwards_metadata_filter(tmp_path: Path) -> None:
    with QdrantChunkStore.in_memory() as store:
        service = DocumentRetrievalService(FakeEmbeddingClient(), store)
        service.index_text_document(_write_manual(tmp_path))

        response = service.search_sparse(
            "tire",
            metadata_filter=ChunkMetadataFilter(section_titles=("Engine",)),
        )

    assert response.matches == ()


def test_sparse_search_rejects_empty_query() -> None:
    with QdrantChunkStore.in_memory() as store:
        service = DocumentRetrievalService(FakeEmbeddingClient(), store)

        with pytest.raises(ValueError, match="must not be empty"):
            service.search_sparse("   ")


def test_hybrid_search_fuses_dense_and_sparse_rankings(tmp_path: Path) -> None:
    client = FakeEmbeddingClient()

    with QdrantChunkStore.in_memory() as store:
        service = DocumentRetrievalService(client, store)
        service.index_text_document(_write_manual(tmp_path))
        calls_after_indexing = len(client.calls)

        response = service.search_hybrid("tire pressure", limit=2)

    assert response.query == "tire pressure"
    assert response.algorithm == "rrf-dense-bm25-v1"
    assert response.embedding_model == "embeddinggemma"
    assert response.dense_match_count == 2
    assert response.sparse_match_count == 2
    assert response.matches[0].section_title == "Tires"
    assert response.matches[0].score > response.matches[1].score > 0
    assert len(client.calls) == calls_after_indexing + 1


def test_hybrid_search_sparse_weight_can_promote_lexical_match(tmp_path: Path) -> None:
    client = FakeEmbeddingClient()

    with QdrantChunkStore.in_memory() as store:
        service = DocumentRetrievalService(client, store)
        service.index_text_document(_write_manual(tmp_path))
        client.forced_embeddings = ((1.0, 0.0, 0.0),)

        response = service.search_hybrid(
            "tire pressure",
            limit=1,
            dense_weight=1.0,
            sparse_weight=2.0,
        )

    assert response.matches[0].section_title == "Tires"


def test_hybrid_search_returns_dense_only_matches_when_sparse_has_none(
    tmp_path: Path,
) -> None:
    with QdrantChunkStore.in_memory() as store:
        service = DocumentRetrievalService(FakeEmbeddingClient(), store)
        service.index_text_document(_write_manual(tmp_path))

        response = service.search_hybrid("automobile", limit=2)

    assert response.dense_match_count == 2
    assert response.sparse_match_count == 0
    assert len(response.matches) == 2


def test_hybrid_search_returns_sparse_only_matches_when_dense_is_filtered(
    tmp_path: Path,
) -> None:
    with QdrantChunkStore.in_memory() as store:
        service = DocumentRetrievalService(FakeEmbeddingClient(), store)
        service.index_text_document(_write_manual(tmp_path))

        response = service.search_hybrid(
            "tire pressure",
            limit=1,
            score_threshold=1.1,
        )

    assert response.dense_match_count == 0
    assert response.sparse_match_count == 2
    assert response.matches[0].section_title == "Tires"


def test_hybrid_search_forwards_metadata_filter(tmp_path: Path) -> None:
    with QdrantChunkStore.in_memory() as store:
        service = DocumentRetrievalService(FakeEmbeddingClient(), store)
        service.index_text_document(_write_manual(tmp_path))

        response = service.search_hybrid(
            "tire pressure",
            metadata_filter=ChunkMetadataFilter(section_titles=("Engine",)),
        )

    assert {match.section_title for match in response.matches} == {"Engine"}


def test_hybrid_search_rejects_empty_query() -> None:
    with QdrantChunkStore.in_memory() as store:
        service = DocumentRetrievalService(FakeEmbeddingClient(), store)

        with pytest.raises(ValueError, match="must not be empty"):
            service.search_hybrid("   ")


@pytest.mark.parametrize(
    ("limit", "candidate_limit", "message"),
    [
        (0, 20, "result limit"),
        (-1, 20, "result limit"),
        (5, 4, "candidate limit"),
    ],
)
def test_hybrid_search_rejects_invalid_limits(
    limit: int,
    candidate_limit: int,
    message: str,
) -> None:
    with QdrantChunkStore.in_memory() as store:
        service = DocumentRetrievalService(FakeEmbeddingClient(), store)

        with pytest.raises(ValueError, match=message):
            service.search_hybrid(
                "warning",
                limit=limit,
                candidate_limit=candidate_limit,
            )


@pytest.mark.parametrize("rrf_k", [0.0, float("nan")])
def test_hybrid_search_rejects_invalid_rrf_constant(rrf_k: float) -> None:
    with QdrantChunkStore.in_memory() as store:
        service = DocumentRetrievalService(FakeEmbeddingClient(), store)

        with pytest.raises(ValueError, match="reciprocal-rank constant"):
            service.search_hybrid("warning", rrf_k=rrf_k)


@pytest.mark.parametrize(
    ("weight_name", "weight", "message"),
    [
        ("dense", 0.0, "dense hybrid weight"),
        ("dense", float("nan"), "dense hybrid weight"),
        ("sparse", 0.0, "sparse hybrid weight"),
        ("sparse", float("inf"), "sparse hybrid weight"),
    ],
)
def test_hybrid_search_rejects_invalid_weights(
    weight_name: str,
    weight: float,
    message: str,
) -> None:
    with QdrantChunkStore.in_memory() as store:
        service = DocumentRetrievalService(FakeEmbeddingClient(), store)

        with pytest.raises(ValueError, match=message):
            if weight_name == "dense":
                service.search_hybrid("warning", dense_weight=weight)
            else:
                service.search_hybrid("warning", sparse_weight=weight)


def test_hybrid_search_embedding_failure_is_reported(tmp_path: Path) -> None:
    client = FakeEmbeddingClient()

    with QdrantChunkStore.in_memory() as store:
        service = DocumentRetrievalService(client, store)
        service.index_text_document(_write_manual(tmp_path))
        client.fail = True

        with pytest.raises(RuntimeError, match="simulated embedding failure"):
            service.search_hybrid("engine warning")


def _vector_result(
    *,
    chunk_id: str,
    section_title: str,
    text: str,
    score: float,
    ordinal: int,
) -> VectorSearchResult:
    return VectorSearchResult(
        chunk_id=chunk_id,
        document_id="doc-test",
        section_id=f"section-{ordinal}",
        section_title=section_title,
        ordinal=ordinal,
        text=text,
        word_count=len(text.split()),
        start_word=ordinal * 10,
        end_word=ordinal * 10 + len(text.split()),
        score=score,
    )


def test_hybrid_reranked_search_returns_explainable_response(tmp_path: Path) -> None:
    client = FakeEmbeddingClient()

    with QdrantChunkStore.in_memory() as store:
        service = DocumentRetrievalService(client, store)
        service.index_text_document(_write_manual(tmp_path))
        calls_after_indexing = len(client.calls)

        response = service.search_hybrid_reranked(
            "tire pressure",
            limit=2,
            candidate_limit=2,
        )

    assert response.query == "tire pressure"
    assert response.algorithm == "hybrid-rrf-lexical-rerank-v1"
    assert response.embedding_model == "embeddinggemma"
    assert response.dense_match_count == 2
    assert response.sparse_match_count == 2
    assert response.candidate_count == 2
    assert response.matches[0].section_title == "Tires"
    assert response.matches[0].hybrid_score > 0
    assert 0 < response.matches[0].score <= 1
    assert response.matches[0].lexical_coverage == pytest.approx(1.0)
    assert response.matches[0].exact_phrase_match
    assert len(client.calls) == calls_after_indexing + 1


def test_hybrid_reranker_can_promote_complete_term_coverage() -> None:
    first = _vector_result(
        chunk_id="first",
        section_title="General",
        text="Vehicle maintenance information.",
        score=0.032,
        ordinal=0,
    )
    second = _vector_result(
        chunk_id="second",
        section_title="Tires",
        text="A sidewall bulge requires the tire to be removed.",
        score=0.016,
        ordinal=1,
    )

    matches = DocumentRetrievalService._rerank_hybrid_matches(
        query="sidewall bulge",
        matches=(first, second),
        limit=2,
        hybrid_score_weight=0.1,
        lexical_coverage_weight=0.9,
        section_title_weight=0.0,
        exact_phrase_weight=0.0,
    )

    assert matches[0].chunk_id == "second"
    assert matches[0].original_rank == 2
    assert matches[0].lexical_coverage == pytest.approx(1.0)


def test_hybrid_reranker_uses_section_title_coverage() -> None:
    general = _vector_result(
        chunk_id="general",
        section_title="General",
        text="Charging system information.",
        score=0.02,
        ordinal=0,
    )
    battery = _vector_result(
        chunk_id="battery",
        section_title="Battery Warning",
        text="Charging system information.",
        score=0.02,
        ordinal=1,
    )

    matches = DocumentRetrievalService._rerank_hybrid_matches(
        query="battery warning",
        matches=(general, battery),
        limit=2,
        hybrid_score_weight=0.0,
        lexical_coverage_weight=0.0,
        section_title_weight=1.0,
        exact_phrase_weight=0.0,
    )

    assert matches[0].chunk_id == "battery"
    assert matches[0].section_title_coverage == pytest.approx(1.0)


def test_hybrid_reranker_rewards_exact_phrase_match() -> None:
    separated = _vector_result(
        chunk_id="separated",
        section_title="General",
        text="The battery charging system displays a warning.",
        score=0.02,
        ordinal=0,
    )
    exact = _vector_result(
        chunk_id="exact",
        section_title="General",
        text="A battery warning requires inspection.",
        score=0.02,
        ordinal=1,
    )

    matches = DocumentRetrievalService._rerank_hybrid_matches(
        query="battery warning",
        matches=(separated, exact),
        limit=2,
        hybrid_score_weight=0.0,
        lexical_coverage_weight=0.0,
        section_title_weight=0.0,
        exact_phrase_weight=1.0,
    )

    assert matches[0].chunk_id == "exact"
    assert matches[0].exact_phrase_match
    assert not matches[1].exact_phrase_match


def test_hybrid_reranked_search_forwards_metadata_filter(tmp_path: Path) -> None:
    with QdrantChunkStore.in_memory() as store:
        service = DocumentRetrievalService(FakeEmbeddingClient(), store)
        service.index_text_document(_write_manual(tmp_path))

        response = service.search_hybrid_reranked(
            "tire pressure",
            metadata_filter=ChunkMetadataFilter(section_titles=("Engine",)),
        )

    assert {match.section_title for match in response.matches} == {"Engine"}


def test_hybrid_reranked_search_handles_empty_candidate_set(tmp_path: Path) -> None:
    with QdrantChunkStore.in_memory() as store:
        service = DocumentRetrievalService(FakeEmbeddingClient(), store)
        service.index_text_document(_write_manual(tmp_path))

        response = service.search_hybrid_reranked(
            "automobile",
            score_threshold=1.1,
        )

    assert response.candidate_count == 0
    assert response.matches == ()


def test_hybrid_reranked_search_rejects_empty_query() -> None:
    with QdrantChunkStore.in_memory() as store:
        service = DocumentRetrievalService(FakeEmbeddingClient(), store)

        with pytest.raises(ValueError, match="must not be empty"):
            service.search_hybrid_reranked("   ")


@pytest.mark.parametrize(
    ("limit", "candidate_limit", "message"),
    [
        (0, 20, "result limit"),
        (-1, 20, "result limit"),
        (5, 4, "candidate limit"),
    ],
)
def test_hybrid_reranked_search_rejects_invalid_limits(
    limit: int,
    candidate_limit: int,
    message: str,
) -> None:
    with QdrantChunkStore.in_memory() as store:
        service = DocumentRetrievalService(FakeEmbeddingClient(), store)

        with pytest.raises(ValueError, match=message):
            service.search_hybrid_reranked(
                "warning",
                limit=limit,
                candidate_limit=candidate_limit,
            )


@pytest.mark.parametrize(
    ("weight_name", "weight", "message"),
    [
        ("hybrid", -0.1, "hybrid score reranking weight"),
        ("lexical", float("nan"), "lexical coverage reranking weight"),
        ("title", float("inf"), "section-title coverage reranking weight"),
        ("phrase", -1.0, "exact-phrase reranking weight"),
    ],
)
def test_hybrid_reranked_search_rejects_invalid_weights(
    weight_name: str,
    weight: float,
    message: str,
) -> None:
    with QdrantChunkStore.in_memory() as store:
        service = DocumentRetrievalService(FakeEmbeddingClient(), store)

        with pytest.raises(ValueError, match=message):
            if weight_name == "hybrid":
                service.search_hybrid_reranked(
                    "warning",
                    hybrid_score_weight=weight,
                )
            elif weight_name == "lexical":
                service.search_hybrid_reranked(
                    "warning",
                    lexical_coverage_weight=weight,
                )
            elif weight_name == "title":
                service.search_hybrid_reranked(
                    "warning",
                    section_title_weight=weight,
                )
            else:
                service.search_hybrid_reranked(
                    "warning",
                    exact_phrase_weight=weight,
                )


def test_hybrid_reranked_search_requires_one_positive_weight() -> None:
    with QdrantChunkStore.in_memory() as store:
        service = DocumentRetrievalService(FakeEmbeddingClient(), store)

        with pytest.raises(ValueError, match="At least one reranking weight"):
            service.search_hybrid_reranked(
                "warning",
                hybrid_score_weight=0.0,
                lexical_coverage_weight=0.0,
                section_title_weight=0.0,
                exact_phrase_weight=0.0,
            )


def test_hybrid_reranker_uses_original_rank_for_deterministic_ties() -> None:
    first = _vector_result(
        chunk_id="first",
        section_title="General",
        text="warning information",
        score=0.02,
        ordinal=0,
    )
    second = _vector_result(
        chunk_id="second",
        section_title="General",
        text="warning information",
        score=0.02,
        ordinal=1,
    )

    matches = DocumentRetrievalService._rerank_hybrid_matches(
        query="warning",
        matches=(first, second),
        limit=2,
        hybrid_score_weight=1.0,
        lexical_coverage_weight=1.0,
        section_title_weight=1.0,
        exact_phrase_weight=1.0,
    )

    assert [match.chunk_id for match in matches] == ["first", "second"]
    assert [match.original_rank for match in matches] == [1, 2]
