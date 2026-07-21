from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic_settings import SettingsConfigDict

from fleetmind_rag.config import FleetMindSettings
from fleetmind_rag.runtime import FleetMindRAGRuntime


class _TestFleetMindSettings(FleetMindSettings):
    model_config = SettingsConfigDict(env_file=None)


@dataclass
class _FakeVectorStore:
    path: Path
    collection_name: str
    closed: bool = False

    def close(self) -> None:
        self.closed = True


@dataclass
class _FakeRetrievalService:
    embedding_client: Any
    vector_store: _FakeVectorStore


@dataclass
class _FakeGroundedAnswerService:
    retrieval_service: _FakeRetrievalService
    chat_client: Any
    minimum_score: float
    max_context_chars: int


class _FakeEmbeddingClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout_seconds: float,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.timeout_seconds = timeout_seconds


class _FakeChatClient(_FakeEmbeddingClient):
    pass


def test_runtime_builds_configured_service_graph(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = _TestFleetMindSettings(
        qdrant_path=tmp_path / "qdrant",
        qdrant_collection="fleet_test",
        minimum_grounding_score=0.61,
        max_context_chars=4096,
        ollama_timeout_seconds=45.0,
    )
    created_store = _FakeVectorStore(settings.qdrant_path, settings.qdrant_collection)

    monkeypatch.setattr(
        "fleetmind_rag.runtime.QdrantChunkStore.from_local_path",
        lambda path, *, collection_name: created_store,
    )
    monkeypatch.setattr(
        "fleetmind_rag.runtime.OllamaEmbeddingClient",
        _FakeEmbeddingClient,
    )
    monkeypatch.setattr(
        "fleetmind_rag.runtime.OllamaChatClient",
        _FakeChatClient,
    )
    monkeypatch.setattr(
        "fleetmind_rag.runtime.DocumentRetrievalService",
        _FakeRetrievalService,
    )

    def _build_grounded_service(
        retrieval_service: _FakeRetrievalService,
        chat_client: _FakeChatClient,
        *,
        minimum_score: float,
        max_context_chars: int,
    ) -> _FakeGroundedAnswerService:
        return _FakeGroundedAnswerService(
            retrieval_service,
            chat_client,
            minimum_score,
            max_context_chars,
        )

    monkeypatch.setattr(
        "fleetmind_rag.runtime.GroundedAnswerService",
        _build_grounded_service,
    )

    runtime = FleetMindRAGRuntime.from_settings(settings)

    assert runtime.settings is settings
    assert id(runtime.vector_store) == id(created_store)

    retrieval_service = cast(
        _FakeRetrievalService,
        cast(object, runtime.retrieval_service),
    )
    grounded_service = cast(
        _FakeGroundedAnswerService,
        cast(object, runtime.grounded_answer_service),
    )

    assert retrieval_service.embedding_client.base_url == "http://localhost:11434/"
    assert retrieval_service.embedding_client.model == "embeddinggemma"
    assert retrieval_service.embedding_client.timeout_seconds == 45.0
    assert grounded_service.chat_client.model == "llama3.2:3b"
    assert grounded_service.minimum_score == 0.61
    assert grounded_service.max_context_chars == 4096


def test_runtime_context_manager_closes_store() -> None:
    settings = _TestFleetMindSettings()
    store = _FakeVectorStore(Path("data/qdrant_local"), "fleetmind_documents")
    retrieval = _FakeRetrievalService(object(), store)
    grounded = _FakeGroundedAnswerService(retrieval, object(), 0.5, 6000)
    runtime = FleetMindRAGRuntime(settings, store, retrieval, grounded)  # type: ignore[arg-type]

    with runtime:
        assert runtime.settings is settings

    assert runtime.is_closed
    assert store.closed


def test_runtime_close_is_idempotent() -> None:
    settings = _TestFleetMindSettings()
    store = _FakeVectorStore(Path("data/qdrant_local"), "fleetmind_documents")
    retrieval = _FakeRetrievalService(object(), store)
    grounded = _FakeGroundedAnswerService(retrieval, object(), 0.5, 6000)
    runtime = FleetMindRAGRuntime(settings, store, retrieval, grounded)  # type: ignore[arg-type]

    runtime.close()
    runtime.close()

    assert runtime.is_closed
    assert store.closed


def test_closed_runtime_cannot_be_reentered() -> None:
    settings = _TestFleetMindSettings()
    store = _FakeVectorStore(Path("data/qdrant_local"), "fleetmind_documents")
    retrieval = _FakeRetrievalService(object(), store)
    grounded = _FakeGroundedAnswerService(retrieval, object(), 0.5, 6000)
    runtime = FleetMindRAGRuntime(settings, store, retrieval, grounded)  # type: ignore[arg-type]
    runtime.close()

    with pytest.raises(RuntimeError, match="closed"):
        runtime.__enter__()
