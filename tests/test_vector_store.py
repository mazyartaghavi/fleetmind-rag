from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from fleetmind_rag.documents import DocumentChunk
from fleetmind_rag.vector_store import ChunkMetadataFilter, QdrantChunkStore


def make_chunk(
    ordinal: int,
    text: str,
    *,
    document_id: str = "doc-one",
    section_id: str | None = None,
    section_title: str = "Engine warnings",
) -> DocumentChunk:
    words = text.split()
    resolved_section_id = section_id or f"{document_id}-section-001"
    return DocumentChunk(
        chunk_id=f"{resolved_section_id}-chunk-{ordinal:03d}",
        document_id=document_id,
        section_id=resolved_section_id,
        section_title=section_title,
        ordinal=ordinal,
        text=text,
        word_count=len(words),
        start_word=(ordinal - 1) * len(words),
        end_word=ordinal * len(words),
    )


def test_rejects_blank_collection_name() -> None:
    with pytest.raises(ValueError, match="collection name"):
        QdrantChunkStore(QdrantClient(":memory:"), collection_name="   ")


def test_in_memory_factory_uses_default_collection() -> None:
    with QdrantChunkStore.in_memory() as store:
        assert store.collection_name == "fleetmind_document_chunks"
        assert not store.is_closed


def test_context_manager_closes_owned_store() -> None:
    store = QdrantChunkStore.in_memory()

    with store:
        assert not store.is_closed

    with pytest.raises(RuntimeError, match="closed"):
        store.count()

    assert store.is_closed


def test_close_is_idempotent() -> None:
    store = QdrantChunkStore.in_memory()

    store.close()
    store.close()

    assert store.is_closed


def test_ensure_collection_creates_once() -> None:
    with QdrantChunkStore.in_memory() as store:
        assert store.ensure_collection(3)
        assert not store.ensure_collection(3)
        assert store.count() == 0


def test_ensure_collection_rejects_invalid_vector_size() -> None:
    with (
        QdrantChunkStore.in_memory() as store,
        pytest.raises(ValueError, match="vector size"),
    ):
        store.ensure_collection(0)


def test_ensure_collection_rejects_existing_wrong_dimension() -> None:
    with QdrantChunkStore.in_memory() as store:
        store.ensure_collection(3)

        with pytest.raises(RuntimeError, match="different vector dimension"):
            store.ensure_collection(2)


def test_ensure_collection_rejects_existing_wrong_distance() -> None:
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="fleetmind_document_chunks",
        vectors_config=VectorParams(size=2, distance=Distance.DOT),
    )

    with (
        QdrantChunkStore(client) as store,
        pytest.raises(RuntimeError, match="cosine distance"),
    ):
        store.ensure_collection(2)


def test_recreate_collection_removes_existing_points() -> None:
    with QdrantChunkStore.in_memory() as store:
        store.upsert_chunks([make_chunk(1, "stop and inspect")], [[1.0, 0.0]])

        assert store.count() == 1
        assert store.ensure_collection(2, recreate=True)
        assert store.count() == 0


def test_upsert_and_count_chunks() -> None:
    chunks = [
        make_chunk(1, "stop and inspect the engine"),
        make_chunk(2, "continue normal vehicle operation"),
    ]

    with QdrantChunkStore.in_memory() as store:
        count = store.upsert_chunks(chunks, [[1.0, 0.0], [0.0, 1.0]])

        assert count == 2
        assert store.count() == 2


def test_upsert_replaces_existing_chunk() -> None:
    chunk = make_chunk(1, "stop and inspect")

    with QdrantChunkStore.in_memory() as store:
        store.upsert_chunks([chunk], [[1.0, 0.0]])
        store.upsert_chunks([chunk], [[0.5, 0.5]])

        assert store.count() == 1


def test_upsert_rejects_empty_chunks() -> None:
    with (
        QdrantChunkStore.in_memory() as store,
        pytest.raises(ValueError, match="At least one"),
    ):
        store.upsert_chunks([], [])


def test_upsert_rejects_mismatched_counts() -> None:
    with (
        QdrantChunkStore.in_memory() as store,
        pytest.raises(ValueError, match="counts must match"),
    ):
        store.upsert_chunks([make_chunk(1, "one")], [])


def test_upsert_rejects_inconsistent_dimensions() -> None:
    chunks = [make_chunk(1, "one"), make_chunk(2, "two")]

    with (
        QdrantChunkStore.in_memory() as store,
        pytest.raises(ValueError, match="same dimension"),
    ):
        store.upsert_chunks(chunks, [[1.0, 0.0], [1.0]])


@pytest.mark.parametrize(
    "invalid_vector",
    [[], [True, 0.0], [float("nan"), 0.0], [float("inf"), 0.0]],
)
def test_upsert_rejects_invalid_vectors(invalid_vector: list[Any]) -> None:
    with QdrantChunkStore.in_memory() as store, pytest.raises(ValueError):
        store.upsert_chunks([make_chunk(1, "one")], [invalid_vector])


def test_upsert_rejects_duplicate_chunk_ids() -> None:
    chunk = make_chunk(1, "one")

    with (
        QdrantChunkStore.in_memory() as store,
        pytest.raises(ValueError, match="identifiers must be unique"),
    ):
        store.upsert_chunks([chunk, chunk], [[1.0, 0.0], [0.0, 1.0]])


def test_search_returns_ranked_chunk_metadata() -> None:
    chunks = [
        make_chunk(1, "stop and inspect the engine"),
        make_chunk(2, "continue normal vehicle operation"),
    ]

    with QdrantChunkStore.in_memory() as store:
        store.upsert_chunks(chunks, [[1.0, 0.0], [0.0, 1.0]])
        results = store.search([0.9, 0.1], limit=2)

        assert [result.chunk_id for result in results] == [
            chunks[0].chunk_id,
            chunks[1].chunk_id,
        ]
        assert results[0].document_id == "doc-one"
        assert results[0].section_title == "Engine warnings"
        assert results[0].text == "stop and inspect the engine"
        assert results[0].score > results[1].score


def test_search_applies_score_threshold() -> None:
    chunks = [make_chunk(1, "engine"), make_chunk(2, "tires")]

    with QdrantChunkStore.in_memory() as store:
        store.upsert_chunks(chunks, [[1.0, 0.0], [0.0, 1.0]])
        results = store.search([1.0, 0.0], limit=2, score_threshold=0.5)

        assert [result.chunk_id for result in results] == [chunks[0].chunk_id]


def test_search_rejects_invalid_arguments() -> None:
    with QdrantChunkStore.in_memory() as store:
        store.ensure_collection(2)

        with pytest.raises(ValueError, match="limit"):
            store.search([1.0, 0.0], limit=0)

        with pytest.raises(ValueError, match="finite"):
            store.search([1.0, 0.0], score_threshold=float("nan"))

        with pytest.raises(ValueError, match="query vector"):
            store.search([])


def test_search_requires_existing_collection() -> None:
    with (
        QdrantChunkStore.in_memory() as store,
        pytest.raises(RuntimeError, match="does not exist"),
    ):
        store.search([1.0, 0.0])


def test_search_rejects_wrong_vector_dimension() -> None:
    with QdrantChunkStore.in_memory() as store:
        store.ensure_collection(2)

        with pytest.raises(ValueError, match="dimension does not match"):
            store.search([1.0, 0.0, 0.0])


def test_local_store_rejects_file_path(tmp_path: Path) -> None:
    file_path = tmp_path / "not-a-directory"
    file_path.write_text("content", encoding="utf-8")

    with pytest.raises(ValueError, match="not a directory"):
        QdrantChunkStore.from_local_path(file_path)


def test_persistent_local_store_round_trip(tmp_path: Path) -> None:
    storage_path = tmp_path / "qdrant"
    chunk = make_chunk(1, "inspect the warning light")

    with QdrantChunkStore.from_local_path(storage_path) as store:
        store.upsert_chunks([chunk], [[1.0, 0.0]])
        assert store.count() == 1

    with QdrantChunkStore.from_local_path(storage_path) as reopened_store:
        results = reopened_store.search([1.0, 0.0])

        assert len(results) == 1
        assert results[0].chunk_id == chunk.chunk_id


def test_metadata_filter_normalizes_and_deduplicates_values() -> None:
    metadata_filter = ChunkMetadataFilter(
        document_ids=(" doc-one ", "doc-one", "doc-two"),
        section_titles=(" Tires ",),
    )

    assert metadata_filter.document_ids == ("doc-one", "doc-two")
    assert metadata_filter.section_titles == ("Tires",)


def test_metadata_filter_requires_at_least_one_criterion() -> None:
    with pytest.raises(ValueError, match="At least one"):
        ChunkMetadataFilter()


@pytest.mark.parametrize(
    ("field_name", "values"),
    [
        ("document_ids", ("",)),
        ("section_ids", ("   ",)),
        ("section_titles", ("Engine", " ")),
    ],
)
def test_metadata_filter_rejects_blank_values(
    field_name: str,
    values: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match=field_name):
        ChunkMetadataFilter(**{field_name: values})


def test_search_filters_by_document_id() -> None:
    chunks = [
        make_chunk(1, "engine warning", document_id="doc-one"),
        make_chunk(1, "engine warning", document_id="doc-two"),
    ]

    with QdrantChunkStore.in_memory() as store:
        store.upsert_chunks(chunks, [[1.0, 0.0], [1.0, 0.0]])
        results = store.search(
            [1.0, 0.0],
            metadata_filter=ChunkMetadataFilter(document_ids=("doc-two",)),
        )

    assert [result.document_id for result in results] == ["doc-two"]


def test_search_filters_by_section_title() -> None:
    chunks = [
        make_chunk(1, "engine warning", section_title="Engine"),
        make_chunk(2, "tire pressure", section_title="Tires"),
    ]

    with QdrantChunkStore.in_memory() as store:
        store.upsert_chunks(chunks, [[1.0, 0.0], [0.0, 1.0]])
        results = store.search(
            [1.0, 0.0],
            metadata_filter=ChunkMetadataFilter(section_titles=("Tires",)),
        )

    assert [result.section_title for result in results] == ["Tires"]


def test_search_uses_or_semantics_within_one_metadata_field() -> None:
    chunks = [
        make_chunk(1, "one", document_id="doc-one"),
        make_chunk(1, "two", document_id="doc-two"),
        make_chunk(1, "three", document_id="doc-three"),
    ]

    with QdrantChunkStore.in_memory() as store:
        store.upsert_chunks(
            chunks,
            [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]],
        )
        results = store.search(
            [1.0, 0.0],
            limit=3,
            metadata_filter=ChunkMetadataFilter(document_ids=("doc-one", "doc-three")),
        )

    assert {result.document_id for result in results} == {"doc-one", "doc-three"}


def test_search_uses_and_semantics_across_metadata_fields() -> None:
    chunks = [
        make_chunk(1, "one", document_id="doc-one", section_title="Engine"),
        make_chunk(1, "two", document_id="doc-two", section_title="Engine"),
        make_chunk(2, "three", document_id="doc-two", section_title="Tires"),
    ]

    with QdrantChunkStore.in_memory() as store:
        store.upsert_chunks(
            chunks,
            [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]],
        )
        results = store.search(
            [1.0, 0.0],
            limit=3,
            metadata_filter=ChunkMetadataFilter(
                document_ids=("doc-two",),
                section_titles=("Engine",),
            ),
        )

    assert len(results) == 1
    assert results[0].document_id == "doc-two"
    assert results[0].section_title == "Engine"


def test_search_returns_empty_tuple_when_metadata_filter_has_no_match() -> None:
    with QdrantChunkStore.in_memory() as store:
        store.upsert_chunks([make_chunk(1, "engine")], [[1.0, 0.0]])
        results = store.search(
            [1.0, 0.0],
            metadata_filter=ChunkMetadataFilter(document_ids=("missing",)),
        )

    assert results == ()


def test_sparse_search_ranks_exact_lexical_matches() -> None:
    chunks = [
        make_chunk(1, "battery charging system inspection"),
        make_chunk(2, "engine coolant temperature alert"),
        make_chunk(3, "battery warning requires dispatch report"),
    ]

    with QdrantChunkStore.in_memory() as store:
        store.upsert_chunks(
            chunks,
            [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
        )
        results = store.search_sparse("battery warning", limit=3)

    assert [result.chunk_id for result in results] == [
        chunks[2].chunk_id,
        chunks[0].chunk_id,
    ]
    assert results[0].score > results[1].score > 0


def test_sparse_search_is_case_and_punctuation_insensitive() -> None:
    chunk = make_chunk(1, "Pressurized coolant can cause serious burns.")

    with QdrantChunkStore.in_memory() as store:
        store.upsert_chunks([chunk], [[1.0, 0.0]])
        results = store.search_sparse("COOLANT, burns!")

    assert [result.chunk_id for result in results] == [chunk.chunk_id]


def test_sparse_search_returns_no_results_without_term_overlap() -> None:
    with QdrantChunkStore.in_memory() as store:
        store.upsert_chunks([make_chunk(1, "engine warning")], [[1.0, 0.0]])

        assert store.search_sparse("dental insurance") == ()


def test_sparse_search_applies_metadata_filter() -> None:
    chunks = [
        make_chunk(1, "warning report", section_title="Engine"),
        make_chunk(2, "warning report", section_title="Tires"),
    ]

    with QdrantChunkStore.in_memory() as store:
        store.upsert_chunks(chunks, [[1.0, 0.0], [0.0, 1.0]])
        results = store.search_sparse(
            "warning report",
            metadata_filter=ChunkMetadataFilter(section_titles=("Tires",)),
        )

    assert [result.section_title for result in results] == ["Tires"]


def test_sparse_search_uses_deterministic_tie_breaking() -> None:
    chunks = [
        make_chunk(2, "battery warning"),
        make_chunk(1, "battery warning"),
    ]

    with QdrantChunkStore.in_memory() as store:
        store.upsert_chunks(chunks, [[1.0, 0.0], [0.0, 1.0]])
        results = store.search_sparse("battery warning", limit=2)

    assert [result.ordinal for result in results] == [1, 2]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"limit": 0}, "limit"),
        ({"k1": 0.0}, "k1"),
        ({"k1": float("nan")}, "k1"),
        ({"b": -0.1}, "b parameter"),
        ({"b": 1.1}, "b parameter"),
    ],
)
def test_sparse_search_rejects_invalid_parameters(
    kwargs: dict[str, Any],
    message: str,
) -> None:
    with QdrantChunkStore.in_memory() as store:
        store.ensure_collection(2)

        with pytest.raises(ValueError, match=message):
            store.search_sparse("warning", **kwargs)


def test_sparse_search_rejects_blank_or_nonlexical_query() -> None:
    with QdrantChunkStore.in_memory() as store:
        store.ensure_collection(2)

        with pytest.raises(ValueError, match="must not be empty"):
            store.search_sparse("   ")

        with pytest.raises(ValueError, match="lexical term"):
            store.search_sparse("--- !!!")


def test_sparse_search_requires_existing_collection() -> None:
    with (
        QdrantChunkStore.in_memory() as store,
        pytest.raises(RuntimeError, match="does not exist"),
    ):
        store.search_sparse("warning")
