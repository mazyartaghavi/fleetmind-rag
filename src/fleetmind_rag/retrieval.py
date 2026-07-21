from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from fleetmind_rag.documents import IngestedDocument, ingest_text_document
from fleetmind_rag.ollama import OllamaEmbeddingResult
from fleetmind_rag.vector_store import (
    ChunkMetadataFilter,
    QdrantChunkStore,
    VectorSearchResult,
)


class EmbeddingClient(Protocol):
    """Structural interface required from a text-embedding client."""

    def embed(
        self,
        input_value: str | list[str] | tuple[str, ...],
    ) -> OllamaEmbeddingResult:
        """Generate embeddings for one text or a batch of texts."""


@dataclass(frozen=True, slots=True)
class DocumentIndexResult:
    """Summary of indexing one ingested document."""

    document_id: str
    source_name: str
    section_count: int
    chunk_count: int
    stored_count: int
    embedding_model: str
    vector_size: int


@dataclass(frozen=True, slots=True)
class RetrievalResponse:
    """Ranked vector-search results for one normalized query."""

    query: str
    embedding_model: str
    matches: tuple[VectorSearchResult, ...]


class DocumentRetrievalService:
    """Coordinate document ingestion, embedding, indexing, and retrieval."""

    def __init__(
        self,
        embedding_client: EmbeddingClient,
        vector_store: QdrantChunkStore,
    ) -> None:
        self._embedding_client = embedding_client
        self._vector_store = vector_store

    def index_text_document(
        self,
        path: str | Path,
        *,
        default_title: str | None = None,
        chunk_size_words: int = 180,
        overlap_words: int = 30,
        encoding: str = "utf-8",
        recreate_collection: bool = False,
    ) -> DocumentIndexResult:
        """Ingest a text file and store embeddings for all generated chunks."""

        ingested_document = ingest_text_document(
            path,
            default_title=default_title,
            chunk_size_words=chunk_size_words,
            overlap_words=overlap_words,
            encoding=encoding,
        )
        return self.index_document(
            ingested_document,
            recreate_collection=recreate_collection,
        )

    def index_document(
        self,
        ingested_document: IngestedDocument,
        *,
        recreate_collection: bool = False,
    ) -> DocumentIndexResult:
        """Embed and store the chunks from an already ingested document."""

        chunks = ingested_document.chunks
        chunk_texts = [chunk.text for chunk in chunks]
        embedding_result = self._embedding_client.embed(chunk_texts)
        embeddings, embedding_model = self._validated_embedding_response(
            embedding_result,
            expected_count=len(chunks),
        )
        vector_size = len(embeddings[0])

        if recreate_collection:
            self._vector_store.ensure_collection(vector_size, recreate=True)

        stored_count = self._vector_store.upsert_chunks(chunks, embeddings)

        return DocumentIndexResult(
            document_id=ingested_document.document.document_id,
            source_name=ingested_document.document.source_name,
            section_count=len(ingested_document.sections),
            chunk_count=len(chunks),
            stored_count=stored_count,
            embedding_model=embedding_model,
            vector_size=vector_size,
        )

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        score_threshold: float | None = None,
        metadata_filter: ChunkMetadataFilter | None = None,
    ) -> RetrievalResponse:
        """Embed one query and return the nearest indexed chunks."""

        clean_query = query.strip()

        if not clean_query:
            raise ValueError("The retrieval query must not be empty.")

        embedding_result = self._embedding_client.embed(clean_query)
        embeddings, embedding_model = self._validated_embedding_response(
            embedding_result,
            expected_count=1,
        )
        matches = self._vector_store.search(
            embeddings[0],
            limit=limit,
            score_threshold=score_threshold,
            metadata_filter=metadata_filter,
        )

        return RetrievalResponse(
            query=clean_query,
            embedding_model=embedding_model,
            matches=matches,
        )

    def count(self) -> int:
        """Return the number of indexed chunks."""

        return self._vector_store.count()

    @staticmethod
    def _validated_embedding_response(
        result: OllamaEmbeddingResult,
        *,
        expected_count: int,
    ) -> tuple[tuple[tuple[float, ...], ...], str]:
        if not result.succeeded:
            raise RuntimeError(f"Embedding generation failed: {result.message}")

        if len(result.embeddings) != expected_count:
            raise RuntimeError(
                "The embedding response count does not match the requested text count."
            )

        model = result.model.strip() if result.model is not None else ""
        if not model:
            raise RuntimeError("The successful embedding response has no model name.")

        vector_size: int | None = None

        for vector in result.embeddings:
            if not vector:
                raise RuntimeError("The embedding response contains an empty vector.")

            if vector_size is None:
                vector_size = len(vector)
            elif len(vector) != vector_size:
                raise RuntimeError(
                    "The embedding response contains inconsistent vector dimensions."
                )

            for value in vector:
                if isinstance(value, bool) or not math.isfinite(float(value)):
                    raise RuntimeError(
                        "The embedding response contains a non-finite vector value."
                    )

        return result.embeddings, model
