from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import TracebackType
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Condition,
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    PointStruct,
    VectorParams,
)

from fleetmind_rag.documents import DocumentChunk

DEFAULT_COLLECTION_NAME = "fleetmind_document_chunks"


@dataclass(frozen=True, slots=True)
class ChunkMetadataFilter:
    """Restrict chunk search by indexed Qdrant payload metadata.

    Values within one field use OR semantics. Different populated fields use AND
    semantics. For example, two document identifiers and one section title mean
    "document A or B" AND "this section title".
    """

    document_ids: tuple[str, ...] = ()
    section_ids: tuple[str, ...] = ()
    section_titles: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "document_ids",
            self._normalize_values(self.document_ids, field_name="document_ids"),
        )
        object.__setattr__(
            self,
            "section_ids",
            self._normalize_values(self.section_ids, field_name="section_ids"),
        )
        object.__setattr__(
            self,
            "section_titles",
            self._normalize_values(self.section_titles, field_name="section_titles"),
        )

        if not (self.document_ids or self.section_ids or self.section_titles):
            raise ValueError(
                "At least one chunk metadata filter criterion is required."
            )

    def to_qdrant_filter(self) -> Filter:
        """Build the Qdrant payload filter represented by this value object."""

        conditions: list[Condition] = []
        self._append_condition(conditions, "document_id", self.document_ids)
        self._append_condition(conditions, "section_id", self.section_ids)
        self._append_condition(conditions, "section_title", self.section_titles)
        return Filter(must=conditions)

    @staticmethod
    def _normalize_values(
        values: Sequence[str],
        *,
        field_name: str,
    ) -> tuple[str, ...]:
        normalized: list[str] = []
        seen: set[str] = set()

        for value in values:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    "Chunk metadata filter "
                    f"{field_name} must contain non-empty strings."
                )

            clean_value = value.strip()
            if clean_value not in seen:
                normalized.append(clean_value)
                seen.add(clean_value)

        return tuple(normalized)

    @staticmethod
    def _append_condition(
        conditions: list[Condition],
        payload_key: str,
        values: tuple[str, ...],
    ) -> None:
        if values:
            conditions.append(
                FieldCondition(
                    key=payload_key,
                    match=MatchAny(any=list(values)),
                )
            )


@dataclass(frozen=True, slots=True)
class VectorSearchResult:
    """A document chunk returned by dense or sparse retrieval."""

    chunk_id: str
    document_id: str
    section_id: str
    section_title: str
    ordinal: int
    text: str
    word_count: int
    start_word: int
    end_word: int
    score: float


class QdrantChunkStore:
    """Store and search FleetMind document chunks in Qdrant."""

    def __init__(
        self,
        client: QdrantClient,
        *,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        owns_client: bool = False,
    ) -> None:
        clean_collection_name = collection_name.strip()

        if not clean_collection_name:
            raise ValueError("The Qdrant collection name must not be empty.")

        self._client = client
        self._collection_name = clean_collection_name
        self._owns_client = owns_client
        self._closed = False

    @classmethod
    def in_memory(
        cls,
        *,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> QdrantChunkStore:
        """Create a store backed by Qdrant local in-memory mode."""

        return cls(
            QdrantClient(":memory:"),
            collection_name=collection_name,
            owns_client=True,
        )

    @classmethod
    def from_local_path(
        cls,
        path: str | Path,
        *,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> QdrantChunkStore:
        """Create a store backed by persistent Qdrant local storage."""

        storage_path = Path(path)

        if storage_path.exists() and not storage_path.is_dir():
            raise ValueError(f"Qdrant storage path is not a directory: {storage_path}")

        storage_path.mkdir(parents=True, exist_ok=True)

        return cls(
            QdrantClient(path=str(storage_path)),
            collection_name=collection_name,
            owns_client=True,
        )

    @property
    def collection_name(self) -> str:
        """Return the configured collection name."""

        return self._collection_name

    @property
    def is_closed(self) -> bool:
        """Return whether the store has been closed."""

        return self._closed

    def ensure_collection(
        self,
        vector_size: int,
        *,
        recreate: bool = False,
    ) -> bool:
        """Create the collection when needed and report whether it was created."""

        self._require_open()

        if vector_size <= 0:
            raise ValueError("The vector size must be greater than zero.")

        collection_exists = self._client.collection_exists(self._collection_name)

        if collection_exists and not recreate:
            self._validate_collection_configuration(vector_size)
            return False

        if collection_exists:
            self._client.delete_collection(self._collection_name)

        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )
        return True

    def upsert_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> int:
        """Insert or replace chunks and their embeddings."""

        self._require_open()

        if not chunks:
            raise ValueError("At least one document chunk is required.")

        if len(chunks) != len(embeddings):
            raise ValueError("Chunk and embedding counts must match.")

        parsed_embeddings = tuple(
            self._normalize_vector(vector, label="embedding") for vector in embeddings
        )
        vector_size = len(parsed_embeddings[0])

        if any(len(vector) != vector_size for vector in parsed_embeddings):
            raise ValueError("All embeddings must have the same dimension.")

        chunk_ids = [chunk.chunk_id for chunk in chunks]
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("Document chunk identifiers must be unique.")

        self.ensure_collection(vector_size)

        points = [
            PointStruct(
                id=str(uuid5(NAMESPACE_URL, chunk.chunk_id)),
                vector=list(vector),
                payload=self._chunk_payload(chunk),
            )
            for chunk, vector in zip(chunks, parsed_embeddings, strict=True)
        ]

        self._client.upsert(
            collection_name=self._collection_name,
            points=points,
            wait=True,
        )
        return len(points)

    def search(
        self,
        query_embedding: Sequence[float],
        *,
        limit: int = 5,
        score_threshold: float | None = None,
        metadata_filter: ChunkMetadataFilter | None = None,
    ) -> tuple[VectorSearchResult, ...]:
        """Return the nearest stored document chunks."""

        self._require_open()

        if limit <= 0:
            raise ValueError("The search limit must be greater than zero.")

        if score_threshold is not None and not math.isfinite(score_threshold):
            raise ValueError("The score threshold must be finite.")

        if not self._client.collection_exists(self._collection_name):
            raise RuntimeError(
                f"The Qdrant collection does not exist: {self._collection_name!r}."
            )

        query_vector = self._normalize_vector(query_embedding, label="query vector")
        expected_vector_size = self._collection_vector_size()

        if len(query_vector) != expected_vector_size:
            raise ValueError(
                "The query vector dimension does not match the Qdrant collection."
            )

        response = self._client.query_points(
            collection_name=self._collection_name,
            query=list(query_vector),
            limit=limit,
            score_threshold=score_threshold,
            query_filter=(
                metadata_filter.to_qdrant_filter()
                if metadata_filter is not None
                else None
            ),
            with_payload=True,
            with_vectors=False,
        )

        return tuple(self._parse_search_result(point) for point in response.points)

    def search_sparse(
        self,
        query: str,
        *,
        limit: int = 5,
        metadata_filter: ChunkMetadataFilter | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> tuple[VectorSearchResult, ...]:
        """Return BM25-ranked chunks without generating an embedding.

        The implementation reads payloads from Qdrant, applies any payload filter in
        Qdrant, and computes deterministic BM25 scores over the remaining chunks.
        """

        self._require_open()

        clean_query = query.strip()
        if not clean_query:
            raise ValueError("The sparse search query must not be empty.")

        if limit <= 0:
            raise ValueError("The sparse search limit must be greater than zero.")

        if not math.isfinite(k1) or k1 <= 0:
            raise ValueError(
                "The BM25 k1 parameter must be finite and greater than zero."
            )

        if not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError(
                "The BM25 b parameter must be finite and between zero and one."
            )

        if not self._client.collection_exists(self._collection_name):
            raise RuntimeError(
                f"The Qdrant collection does not exist: {self._collection_name!r}."
            )

        query_terms = tuple(dict.fromkeys(self._tokenize_lexical(clean_query)))
        if not query_terms:
            raise ValueError(
                "The sparse search query must contain at least one lexical term."
            )

        candidates = self._scroll_payload_results(metadata_filter=metadata_filter)
        tokenized_candidates = tuple(
            (candidate, self._tokenize_lexical(candidate.text))
            for candidate in candidates
        )
        searchable_candidates = tuple(
            (candidate, terms) for candidate, terms in tokenized_candidates if terms
        )

        if not searchable_candidates:
            return ()

        document_count = len(searchable_candidates)
        average_length = (
            sum(len(terms) for _, terms in searchable_candidates) / document_count
        )
        candidate_term_sets = tuple(set(terms) for _, terms in searchable_candidates)
        document_frequency = {
            term: sum(1 for term_set in candidate_term_sets if term in term_set)
            for term in query_terms
        }

        ranked: list[VectorSearchResult] = []
        for candidate, terms in searchable_candidates:
            frequencies = Counter(terms)
            document_length = len(terms)
            score = 0.0

            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if frequency == 0:
                    continue

                term_document_frequency = document_frequency[term]
                inverse_document_frequency = math.log(
                    1
                    + (document_count - term_document_frequency + 0.5)
                    / (term_document_frequency + 0.5)
                )
                normalization = frequency + k1 * (
                    1 - b + b * document_length / average_length
                )
                score += inverse_document_frequency * (
                    frequency * (k1 + 1) / normalization
                )

            if score > 0:
                ranked.append(replace(candidate, score=score))

        ranked.sort(
            key=lambda result: (
                -result.score,
                result.document_id,
                result.ordinal,
                result.chunk_id,
            )
        )
        return tuple(ranked[:limit])

    def count(self) -> int:
        """Return the exact number of points stored in the collection."""

        self._require_open()

        if not self._client.collection_exists(self._collection_name):
            return 0

        return int(
            self._client.count(
                collection_name=self._collection_name,
                exact=True,
            ).count
        )

    def close(self) -> None:
        """Close the owned Qdrant client and make the store unusable."""

        if self._closed:
            return

        if self._owns_client:
            self._client.close()

        self._closed = True

    def __enter__(self) -> QdrantChunkStore:
        self._require_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.close()

    def _scroll_payload_results(
        self,
        *,
        metadata_filter: ChunkMetadataFilter | None,
    ) -> tuple[VectorSearchResult, ...]:
        results: list[VectorSearchResult] = []
        offset: Any = None
        query_filter = (
            metadata_filter.to_qdrant_filter() if metadata_filter is not None else None
        )

        while True:
            points, next_offset = self._client.scroll(
                collection_name=self._collection_name,
                scroll_filter=query_filter,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            results.extend(
                self._parse_payload_result(point.payload, score=0.0) for point in points
            )

            if next_offset is None:
                return tuple(results)

            offset = next_offset

    @staticmethod
    def _tokenize_lexical(text: str) -> tuple[str, ...]:
        return tuple(re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", text.lower()))

    def _validate_collection_configuration(self, vector_size: int) -> None:
        configured_vector_size = self._collection_vector_size()

        if configured_vector_size != vector_size:
            raise RuntimeError(
                "The existing Qdrant collection has a different vector dimension."
            )

    def _collection_vector_size(self) -> int:
        collection = self._client.get_collection(self._collection_name)
        vectors_config = collection.config.params.vectors

        if not isinstance(vectors_config, VectorParams):
            raise RuntimeError(
                "The Qdrant collection does not use one unnamed dense vector."
            )

        if vectors_config.distance != Distance.COSINE:
            raise RuntimeError("The Qdrant collection does not use cosine distance.")

        return int(vectors_config.size)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("The Qdrant chunk store is closed.")

    @staticmethod
    def _normalize_vector(
        vector: Sequence[float],
        *,
        label: str,
    ) -> tuple[float, ...]:
        if not vector:
            raise ValueError(f"The {label} must not be empty.")

        normalized_values: list[float] = []

        for value in vector:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"The {label} must contain only numeric values.")

            normalized_value = float(value)
            if not math.isfinite(normalized_value):
                raise ValueError(f"The {label} must contain only finite values.")

            normalized_values.append(normalized_value)

        return tuple(normalized_values)

    @staticmethod
    def _chunk_payload(chunk: DocumentChunk) -> dict[str, Any]:
        return {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "section_id": chunk.section_id,
            "section_title": chunk.section_title,
            "ordinal": chunk.ordinal,
            "text": chunk.text,
            "word_count": chunk.word_count,
            "start_word": chunk.start_word,
            "end_word": chunk.end_word,
        }

    @classmethod
    def _parse_search_result(cls, point: Any) -> VectorSearchResult:
        return cls._parse_payload_result(point.payload, score=float(point.score))

    @classmethod
    def _parse_payload_result(
        cls,
        payload: Any,
        *,
        score: float,
    ) -> VectorSearchResult:
        if not isinstance(payload, dict):
            raise RuntimeError("A Qdrant search result has no valid payload.")

        return VectorSearchResult(
            chunk_id=cls._require_payload_string(payload, "chunk_id"),
            document_id=cls._require_payload_string(payload, "document_id"),
            section_id=cls._require_payload_string(payload, "section_id"),
            section_title=cls._require_payload_string(payload, "section_title"),
            ordinal=cls._require_payload_int(payload, "ordinal"),
            text=cls._require_payload_string(payload, "text"),
            word_count=cls._require_payload_int(payload, "word_count"),
            start_word=cls._require_payload_int(payload, "start_word"),
            end_word=cls._require_payload_int(payload, "end_word"),
            score=score,
        )

    @staticmethod
    def _require_payload_string(payload: dict[str, Any], field_name: str) -> str:
        value = payload.get(field_name)

        if not isinstance(value, str) or not value.strip():
            raise RuntimeError(
                f"A Qdrant search result has invalid {field_name!r} metadata."
            )

        return value.strip()

    @staticmethod
    def _require_payload_int(payload: dict[str, Any], field_name: str) -> int:
        value = payload.get(field_name)

        if isinstance(value, bool) or not isinstance(value, int):
            raise RuntimeError(
                f"A Qdrant search result has invalid {field_name!r} metadata."
            )

        return value
