from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from fleetmind_rag.documents import ingest_text_document
from fleetmind_rag.ollama import OllamaEmbeddingResult
from fleetmind_rag.retrieval import DocumentRetrievalService
from fleetmind_rag.vector_store import ChunkMetadataFilter, QdrantChunkStore


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
