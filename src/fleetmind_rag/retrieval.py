from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
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


@dataclass(frozen=True, slots=True)
class SparseRetrievalResponse:
    """Ranked lexical-search results for one normalized query."""

    query: str
    algorithm: str
    matches: tuple[VectorSearchResult, ...]


@dataclass(frozen=True, slots=True)
class HybridRetrievalResponse:
    """Reciprocal-rank-fused dense and sparse retrieval results."""

    query: str
    algorithm: str
    embedding_model: str
    dense_match_count: int
    sparse_match_count: int
    matches: tuple[VectorSearchResult, ...]


@dataclass(frozen=True, slots=True)
class RerankedSearchResult:
    """One hybrid candidate with transparent deterministic reranking signals."""

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
    hybrid_score: float
    original_rank: int
    lexical_coverage: float
    section_title_coverage: float
    exact_phrase_match: bool


@dataclass(frozen=True, slots=True)
class RerankedRetrievalResponse:
    """Hybrid candidates reordered by a transparent lexical relevance model."""

    query: str
    algorithm: str
    embedding_model: str
    dense_match_count: int
    sparse_match_count: int
    candidate_count: int
    matches: tuple[RerankedSearchResult, ...]


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

    def search_sparse(
        self,
        query: str,
        *,
        limit: int = 5,
        metadata_filter: ChunkMetadataFilter | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> SparseRetrievalResponse:
        """Return deterministic BM25 lexical matches without embedding the query."""

        clean_query = query.strip()
        if not clean_query:
            raise ValueError("The retrieval query must not be empty.")

        matches = self._vector_store.search_sparse(
            clean_query,
            limit=limit,
            metadata_filter=metadata_filter,
            k1=k1,
            b=b,
        )

        return SparseRetrievalResponse(
            query=clean_query,
            algorithm="bm25-local-v1",
            matches=matches,
        )

    def search_hybrid(
        self,
        query: str,
        *,
        limit: int = 5,
        candidate_limit: int = 20,
        score_threshold: float | None = None,
        metadata_filter: ChunkMetadataFilter | None = None,
        rrf_k: float = 60.0,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> HybridRetrievalResponse:
        """Fuse dense and BM25 rankings with weighted reciprocal rank fusion."""

        clean_query = query.strip()
        if not clean_query:
            raise ValueError("The retrieval query must not be empty.")

        if limit <= 0:
            raise ValueError("The hybrid result limit must be greater than zero.")

        if candidate_limit < limit:
            raise ValueError(
                "The hybrid candidate limit must be greater than or equal to "
                "the result limit."
            )

        if not math.isfinite(rrf_k) or rrf_k <= 0:
            raise ValueError(
                "The reciprocal-rank constant must be finite and greater than zero."
            )

        for label, weight in (
            ("dense", dense_weight),
            ("sparse", sparse_weight),
        ):
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError(
                    f"The {label} hybrid weight must be finite and greater than zero."
                )

        embedding_result = self._embedding_client.embed(clean_query)
        embeddings, embedding_model = self._validated_embedding_response(
            embedding_result,
            expected_count=1,
        )
        dense_matches = self._vector_store.search(
            embeddings[0],
            limit=candidate_limit,
            score_threshold=score_threshold,
            metadata_filter=metadata_filter,
        )
        sparse_matches = self._vector_store.search_sparse(
            clean_query,
            limit=candidate_limit,
            metadata_filter=metadata_filter,
            k1=k1,
            b=b,
        )
        matches = self._fuse_reciprocal_ranks(
            dense_matches=dense_matches,
            sparse_matches=sparse_matches,
            limit=limit,
            rrf_k=rrf_k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
        )

        return HybridRetrievalResponse(
            query=clean_query,
            algorithm="rrf-dense-bm25-v1",
            embedding_model=embedding_model,
            dense_match_count=len(dense_matches),
            sparse_match_count=len(sparse_matches),
            matches=matches,
        )

    def search_hybrid_reranked(
        self,
        query: str,
        *,
        limit: int = 5,
        candidate_limit: int = 20,
        score_threshold: float | None = None,
        metadata_filter: ChunkMetadataFilter | None = None,
        rrf_k: float = 60.0,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        k1: float = 1.5,
        b: float = 0.75,
        hybrid_score_weight: float = 0.45,
        lexical_coverage_weight: float = 0.35,
        section_title_weight: float = 0.15,
        exact_phrase_weight: float = 0.05,
    ) -> RerankedRetrievalResponse:
        """Rerank hybrid candidates with transparent query-coverage features."""

        clean_query = query.strip()
        if not clean_query:
            raise ValueError("The retrieval query must not be empty.")

        if limit <= 0:
            raise ValueError("The reranked result limit must be greater than zero.")

        if candidate_limit < limit:
            raise ValueError(
                "The reranking candidate limit must be greater than or equal to "
                "the result limit."
            )

        rerank_weights = (
            ("hybrid score", hybrid_score_weight),
            ("lexical coverage", lexical_coverage_weight),
            ("section-title coverage", section_title_weight),
            ("exact-phrase", exact_phrase_weight),
        )
        for label, weight in rerank_weights:
            if not math.isfinite(weight) or weight < 0:
                raise ValueError(
                    f"The {label} reranking weight must be finite and non-negative."
                )

        total_rerank_weight = sum(weight for _, weight in rerank_weights)
        if total_rerank_weight <= 0:
            raise ValueError("At least one reranking weight must be greater than zero.")

        hybrid_response = self.search_hybrid(
            clean_query,
            limit=candidate_limit,
            candidate_limit=candidate_limit,
            score_threshold=score_threshold,
            metadata_filter=metadata_filter,
            rrf_k=rrf_k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            k1=k1,
            b=b,
        )
        matches = self._rerank_hybrid_matches(
            query=clean_query,
            matches=hybrid_response.matches,
            limit=limit,
            hybrid_score_weight=hybrid_score_weight,
            lexical_coverage_weight=lexical_coverage_weight,
            section_title_weight=section_title_weight,
            exact_phrase_weight=exact_phrase_weight,
        )

        return RerankedRetrievalResponse(
            query=clean_query,
            algorithm="hybrid-rrf-lexical-rerank-v1",
            embedding_model=hybrid_response.embedding_model,
            dense_match_count=hybrid_response.dense_match_count,
            sparse_match_count=hybrid_response.sparse_match_count,
            candidate_count=len(hybrid_response.matches),
            matches=matches,
        )

    def count(self) -> int:
        """Return the number of indexed chunks."""

        return self._vector_store.count()

    @staticmethod
    def _fuse_reciprocal_ranks(
        *,
        dense_matches: tuple[VectorSearchResult, ...],
        sparse_matches: tuple[VectorSearchResult, ...],
        limit: int,
        rrf_k: float,
        dense_weight: float,
        sparse_weight: float,
    ) -> tuple[VectorSearchResult, ...]:
        representatives: dict[str, VectorSearchResult] = {}
        fused_scores: dict[str, float] = {}
        dense_ranks: dict[str, int] = {}
        sparse_ranks: dict[str, int] = {}

        for rank, match in enumerate(dense_matches, start=1):
            representatives.setdefault(match.chunk_id, match)
            dense_ranks[match.chunk_id] = rank
            fused_scores[match.chunk_id] = fused_scores.get(match.chunk_id, 0.0) + (
                dense_weight / (rrf_k + rank)
            )

        for rank, match in enumerate(sparse_matches, start=1):
            representatives.setdefault(match.chunk_id, match)
            sparse_ranks[match.chunk_id] = rank
            fused_scores[match.chunk_id] = fused_scores.get(match.chunk_id, 0.0) + (
                sparse_weight / (rrf_k + rank)
            )

        fused = [
            replace(representatives[chunk_id], score=score)
            for chunk_id, score in fused_scores.items()
        ]
        fallback_rank = len(dense_matches) + len(sparse_matches) + 1
        fused.sort(
            key=lambda match: (
                -match.score,
                dense_ranks.get(match.chunk_id, fallback_rank),
                sparse_ranks.get(match.chunk_id, fallback_rank),
                match.document_id,
                match.ordinal,
                match.chunk_id,
            )
        )
        return tuple(fused[:limit])

    @staticmethod
    def _rerank_hybrid_matches(
        *,
        query: str,
        matches: tuple[VectorSearchResult, ...],
        limit: int,
        hybrid_score_weight: float,
        lexical_coverage_weight: float,
        section_title_weight: float,
        exact_phrase_weight: float,
    ) -> tuple[RerankedSearchResult, ...]:
        if not matches:
            return ()

        query_tokens = DocumentRetrievalService._tokenize(query)
        if not query_tokens:
            raise ValueError(
                "The reranking query must contain at least one lexical term."
            )

        query_terms = tuple(dict.fromkeys(query_tokens))
        query_term_set = set(query_terms)
        query_phrase = " ".join(query_tokens)
        maximum_hybrid_score = max(match.score for match in matches)
        total_weight = (
            hybrid_score_weight
            + lexical_coverage_weight
            + section_title_weight
            + exact_phrase_weight
        )
        reranked: list[RerankedSearchResult] = []

        for original_rank, match in enumerate(matches, start=1):
            text_terms = DocumentRetrievalService._tokenize(match.text)
            title_terms = DocumentRetrievalService._tokenize(match.section_title)
            lexical_coverage = len(query_term_set.intersection(text_terms)) / len(
                query_term_set
            )
            section_title_coverage = len(
                query_term_set.intersection(title_terms)
            ) / len(query_term_set)
            exact_phrase_match = bool(
                query_phrase
                and (
                    query_phrase in " ".join(text_terms)
                    or query_phrase in " ".join(title_terms)
                )
            )
            normalized_hybrid_score = (
                match.score / maximum_hybrid_score if maximum_hybrid_score > 0 else 0.0
            )
            rerank_score = (
                hybrid_score_weight * normalized_hybrid_score
                + lexical_coverage_weight * lexical_coverage
                + section_title_weight * section_title_coverage
                + exact_phrase_weight * float(exact_phrase_match)
            ) / total_weight

            reranked.append(
                RerankedSearchResult(
                    chunk_id=match.chunk_id,
                    document_id=match.document_id,
                    section_id=match.section_id,
                    section_title=match.section_title,
                    ordinal=match.ordinal,
                    text=match.text,
                    word_count=match.word_count,
                    start_word=match.start_word,
                    end_word=match.end_word,
                    score=rerank_score,
                    hybrid_score=match.score,
                    original_rank=original_rank,
                    lexical_coverage=lexical_coverage,
                    section_title_coverage=section_title_coverage,
                    exact_phrase_match=exact_phrase_match,
                )
            )

        reranked.sort(
            key=lambda match: (
                -match.score,
                -match.lexical_coverage,
                -match.section_title_coverage,
                -int(match.exact_phrase_match),
                match.original_rank,
                match.document_id,
                match.ordinal,
                match.chunk_id,
            )
        )
        return tuple(reranked[:limit])

    @staticmethod
    def _tokenize(text: str) -> tuple[str, ...]:
        return tuple(re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", text.lower()))

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
