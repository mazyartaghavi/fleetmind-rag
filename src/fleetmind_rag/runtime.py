from __future__ import annotations

from dataclasses import dataclass, field
from types import TracebackType

from fleetmind_rag.config import FleetMindSettings
from fleetmind_rag.grounded_rag import GroundedAnswerService
from fleetmind_rag.ollama import OllamaChatClient, OllamaEmbeddingClient
from fleetmind_rag.retrieval import DocumentRetrievalService
from fleetmind_rag.vector_store import QdrantChunkStore


@dataclass(slots=True)
class FleetMindRAGRuntime:
    """Own the configured services that implement the local FleetMind RAG path."""

    settings: FleetMindSettings
    vector_store: QdrantChunkStore
    retrieval_service: DocumentRetrievalService
    grounded_answer_service: GroundedAnswerService
    _closed: bool = field(default=False, init=False, repr=False)

    @classmethod
    def from_settings(cls, settings: FleetMindSettings) -> FleetMindRAGRuntime:
        """Construct the complete local RAG runtime from validated settings."""

        base_url = str(settings.llm_base_url)
        vector_store = QdrantChunkStore.from_local_path(
            settings.qdrant_path,
            collection_name=settings.qdrant_collection,
        )

        try:
            embedding_client = OllamaEmbeddingClient(
                base_url,
                settings.embedding_model,
                timeout_seconds=settings.ollama_timeout_seconds,
            )
            retrieval_service = DocumentRetrievalService(
                embedding_client,
                vector_store,
            )
            chat_client = OllamaChatClient(
                base_url,
                settings.llm_model,
                timeout_seconds=settings.ollama_timeout_seconds,
            )
            grounded_answer_service = GroundedAnswerService(
                retrieval_service,
                chat_client,
                minimum_score=settings.minimum_grounding_score,
                max_context_chars=settings.max_context_chars,
            )
        except Exception:
            vector_store.close()
            raise

        return cls(
            settings=settings,
            vector_store=vector_store,
            retrieval_service=retrieval_service,
            grounded_answer_service=grounded_answer_service,
        )

    @property
    def is_closed(self) -> bool:
        """Return whether the runtime has released its owned resources."""

        return self._closed

    def close(self) -> None:
        """Release the owned Qdrant client exactly once."""

        if self._closed:
            return

        self.vector_store.close()
        self._closed = True

    def __enter__(self) -> FleetMindRAGRuntime:
        if self._closed:
            raise RuntimeError("The FleetMind RAG runtime is closed.")

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.close()
