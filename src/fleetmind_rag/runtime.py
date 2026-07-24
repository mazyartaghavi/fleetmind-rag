from __future__ import annotations

from dataclasses import dataclass, field
from types import TracebackType

from fleetmind_rag.adaptive_grounded_rag import AdaptiveGroundedAnswerService
from fleetmind_rag.config import FleetMindSettings
from fleetmind_rag.feedback_routing import RoutingFeedbackHistory
from fleetmind_rag.feedback_store import (
    FeedbackStoreSnapshot,
    JsonRoutingFeedbackStore,
)
from fleetmind_rag.grounded_rag import GroundedAnswerService
from fleetmind_rag.ollama import OllamaChatClient, OllamaEmbeddingClient
from fleetmind_rag.retrieval import DocumentRetrievalService
from fleetmind_rag.vector_store import QdrantChunkStore

ROUTING_FEEDBACK_FILENAME = "routing_feedback.json"


@dataclass(slots=True)
class FleetMindRAGRuntime:
    """Own the configured services that implement the local FleetMind RAG path."""

    settings: FleetMindSettings
    vector_store: QdrantChunkStore
    retrieval_service: DocumentRetrievalService
    grounded_answer_service: GroundedAnswerService
    adaptive_grounded_answer_service: AdaptiveGroundedAnswerService
    feedback_store: JsonRoutingFeedbackStore
    feedback_snapshot: FeedbackStoreSnapshot
    _closed: bool = field(default=False, init=False, repr=False)

    @classmethod
    def from_settings(cls, settings: FleetMindSettings) -> FleetMindRAGRuntime:
        """Construct the complete local RAG runtime from validated settings."""

        base_url = str(settings.llm_base_url)
        vector_store = QdrantChunkStore.from_local_path(
            settings.qdrant_path,
            collection_name=settings.qdrant_collection,
        )
        feedback_store = JsonRoutingFeedbackStore(
            settings.qdrant_path / ROUTING_FEEDBACK_FILENAME
        )

        try:
            feedback_snapshot = feedback_store.load()
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
            adaptive_grounded_answer_service = AdaptiveGroundedAnswerService(
                retrieval_service,
                chat_client,
                history=feedback_snapshot.history,
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
            adaptive_grounded_answer_service=adaptive_grounded_answer_service,
            feedback_store=feedback_store,
            feedback_snapshot=feedback_snapshot,
        )

    @property
    def is_closed(self) -> bool:
        """Return whether the runtime has released its owned resources."""

        return self._closed

    def persist_feedback(
        self,
        history: RoutingFeedbackHistory,
    ) -> FeedbackStoreSnapshot:
        """Persist updated routing history using the loaded revision."""

        if self._closed:
            raise RuntimeError("The FleetMind RAG runtime is closed.")

        snapshot = self.feedback_store.save(
            history,
            expected_revision=self.feedback_snapshot.revision,
        )
        self.feedback_snapshot = snapshot
        return snapshot

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
