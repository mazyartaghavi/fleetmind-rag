from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import cast

import pytest

from fleetmind_rag.feedback_routing import (
    RoutingFeedbackHistory,
    RoutingFeedbackObservation,
)
from fleetmind_rag.feedback_store import (
    FEEDBACK_STORE_SCHEMA_VERSION,
    FeedbackStoreConflictError,
    FeedbackStoreFormatError,
    FeedbackStoreIOError,
    FeedbackStoreLockTimeoutError,
    FeedbackStoreSnapshot,
    JsonRoutingFeedbackStore,
)


def _observation(
    *,
    query: str = "What does overheating mean?",
    strategy: str = "dense",
    verdict: str = "accept",
    quality_score: float = 0.9,
    attempt_number: int = 1,
    features: tuple[str, ...] = ("conceptual",),
) -> RoutingFeedbackObservation:
    return RoutingFeedbackObservation(
        query=query,
        strategy=strategy,  # type: ignore[arg-type]
        verdict=verdict,  # type: ignore[arg-type]
        quality_score=quality_score,
        attempt_number=attempt_number,
        features=features,  # type: ignore[arg-type]
    )


def _payload(
    *,
    schema_version: object = FEEDBACK_STORE_SCHEMA_VERSION,
    revision: object = 1,
    observations: object | None = None,
) -> dict[str, object]:
    resolved_observations: object = (
        [
            {
                "query": "What does overheating mean?",
                "strategy": "dense",
                "verdict": "accept",
                "quality_score": 0.9,
                "attempt_number": 1,
                "features": ["conceptual"],
            }
        ]
        if observations is None
        else observations
    )
    return {
        "schema_version": schema_version,
        "revision": revision,
        "observations": resolved_observations,
    }


def _write_payload(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_missing_store_loads_empty_initial_snapshot(tmp_path: Path) -> None:
    store = JsonRoutingFeedbackStore(tmp_path / "feedback.json")

    snapshot = store.load()

    assert snapshot.schema_version == FEEDBACK_STORE_SCHEMA_VERSION
    assert snapshot.revision == 0
    assert snapshot.history.observations == ()
    assert not store.path.exists()


def test_store_exposes_data_and_lock_paths(tmp_path: Path) -> None:
    path = tmp_path / "feedback.json"
    store = JsonRoutingFeedbackStore(path)

    assert store.path == path
    assert store.lock_path == tmp_path / "feedback.json.lock"


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("lock_timeout_seconds", 0.0),
        ("lock_poll_interval_seconds", -1.0),
        ("stale_lock_seconds", float("inf")),
        ("stale_lock_seconds", float("nan")),
    ],
)
def test_store_rejects_invalid_lock_configuration(
    tmp_path: Path,
    keyword: str,
    value: float,
) -> None:
    arguments = {keyword: value}

    with pytest.raises(ValueError, match=keyword):
        JsonRoutingFeedbackStore(
            tmp_path / "feedback.json",
            **arguments,
        )


def test_store_rejects_blank_string_path() -> None:
    with pytest.raises(ValueError, match="path"):
        JsonRoutingFeedbackStore("  ")


@pytest.mark.parametrize("revision", [-1, True, 1.5])
def test_snapshot_rejects_invalid_revision(revision: object) -> None:
    with pytest.raises(ValueError, match="revision"):
        FeedbackStoreSnapshot(
            history=RoutingFeedbackHistory(),
            revision=revision,  # type: ignore[arg-type]
        )


def test_save_and_load_round_trip_preserves_history(tmp_path: Path) -> None:
    store = JsonRoutingFeedbackStore(tmp_path / "feedback.json")
    history = RoutingFeedbackHistory(
        (
            _observation(),
            _observation(
                query="error code P0420",
                strategy="sparse",
                verdict="rewrite",
                quality_score=0.25,
                attempt_number=2,
                features=("exact_identifier",),
            ),
        )
    )

    saved = store.save(history)
    loaded = store.load()

    assert saved == loaded
    assert loaded.revision == 1
    assert loaded.history == history
    assert not store.lock_path.exists()


def test_serialized_json_is_versioned_and_human_readable(
    tmp_path: Path,
) -> None:
    path = tmp_path / "feedback.json"
    store = JsonRoutingFeedbackStore(path)

    store.save(RoutingFeedbackHistory((_observation(),)))

    text = path.read_text(encoding="utf-8")
    payload = json.loads(text)

    assert text.endswith("\n")
    assert payload["schema_version"] == 1
    assert payload["revision"] == 1
    assert payload["observations"][0]["strategy"] == "dense"
    assert payload["observations"][0]["features"] == ["conceptual"]


def test_unicode_query_round_trips_as_utf8(tmp_path: Path) -> None:
    store = JsonRoutingFeedbackStore(tmp_path / "feedback.json")
    observation = _observation(query="بررسی هشدار باتری")

    store.save(RoutingFeedbackHistory((observation,)))

    loaded = store.load()
    assert loaded.history.observations[0].query == "بررسی هشدار باتری"
    assert "\\u" not in store.path.read_text(encoding="utf-8")


def test_save_increments_revision(tmp_path: Path) -> None:
    store = JsonRoutingFeedbackStore(tmp_path / "feedback.json")

    first = store.save(RoutingFeedbackHistory())
    second = store.save(RoutingFeedbackHistory((_observation(),)))

    assert first.revision == 1
    assert second.revision == 2
    assert store.load() == second


def test_save_accepts_matching_expected_revision(tmp_path: Path) -> None:
    store = JsonRoutingFeedbackStore(tmp_path / "feedback.json")
    first = store.save(RoutingFeedbackHistory())

    second = store.save(
        RoutingFeedbackHistory((_observation(),)),
        expected_revision=first.revision,
    )

    assert second.revision == 2


def test_save_rejects_stale_expected_revision_without_writing(
    tmp_path: Path,
) -> None:
    store = JsonRoutingFeedbackStore(tmp_path / "feedback.json")
    first = store.save(RoutingFeedbackHistory((_observation(),)))
    original_text = store.path.read_text(encoding="utf-8")

    with pytest.raises(FeedbackStoreConflictError, match="expected 0"):
        store.save(
            RoutingFeedbackHistory(),
            expected_revision=0,
        )

    assert store.load() == first
    assert store.path.read_text(encoding="utf-8") == original_text
    assert not store.lock_path.exists()


@pytest.mark.parametrize("value", [-1, True])
def test_save_rejects_invalid_expected_revision(
    tmp_path: Path,
    value: int,
) -> None:
    store = JsonRoutingFeedbackStore(tmp_path / "feedback.json")

    with pytest.raises(ValueError, match="expected_revision"):
        store.save(
            RoutingFeedbackHistory(),
            expected_revision=value,
        )


def test_append_adds_observation_without_replacing_history(
    tmp_path: Path,
) -> None:
    store = JsonRoutingFeedbackStore(tmp_path / "feedback.json")
    first_observation = _observation()
    second_observation = _observation(
        query="error code P0420",
        strategy="sparse",
        features=("exact_identifier",),
    )
    first = store.append(first_observation)
    second = store.append(
        second_observation,
        expected_revision=first.revision,
    )

    assert second.revision == 2
    assert second.history.observations == (
        first_observation,
        second_observation,
    )


def test_save_creates_missing_parent_directories(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "feedback" / "history.json"
    store = JsonRoutingFeedbackStore(path)

    store.save(RoutingFeedbackHistory())

    assert path.is_file()


def test_invalid_json_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "feedback.json"
    path.write_text("{invalid", encoding="utf-8")

    with pytest.raises(FeedbackStoreFormatError, match="invalid JSON"):
        JsonRoutingFeedbackStore(path).load()


def test_invalid_utf8_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "feedback.json"
    path.write_bytes(b"\xff\xfe")

    with pytest.raises(FeedbackStoreFormatError, match="valid UTF-8"):
        JsonRoutingFeedbackStore(path).load()


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "JSON object"),
        (
            {
                "schema_version": 1,
                "revision": 0,
            },
            "missing keys",
        ),
        (
            {
                **_payload(),
                "unexpected": True,
            },
            "unexpected keys",
        ),
        (_payload(schema_version=2), "Unsupported feedback schema"),
        (_payload(revision=-1), "non-negative"),
        (_payload(revision=True), "revision must be an integer"),
        (_payload(observations={}), "JSON array"),
    ],
)
def test_invalid_snapshot_structure_is_rejected(
    tmp_path: Path,
    payload: object,
    message: str,
) -> None:
    path = tmp_path / "feedback.json"
    _write_payload(path, payload)

    with pytest.raises(FeedbackStoreFormatError, match=message):
        JsonRoutingFeedbackStore(path).load()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("query", 7, "query must be a string"),
        ("strategy", "unknown", "strategy contains an unsupported"),
        ("verdict", "maybe", "verdict contains an unsupported"),
        ("quality_score", True, "quality_score must be a number"),
        ("quality_score", float("inf"), "quality_score must be finite"),
        ("attempt_number", 0, "routing-feedback invariants"),
        ("features", "conceptual", "features must be a JSON array"),
        ("features", ["unknown"], "is unsupported"),
    ],
)
def test_invalid_observation_is_rejected(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    path = tmp_path / "feedback.json"
    observations = cast(
        list[dict[str, object]],
        _payload()["observations"],
    )
    observation = observations[0]
    observation[field] = value
    _write_payload(path, _payload(observations=[observation]))

    with pytest.raises(FeedbackStoreFormatError, match=message):
        JsonRoutingFeedbackStore(path).load()


def test_observation_with_missing_field_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "feedback.json"
    observations = cast(
        list[dict[str, object]],
        _payload()["observations"],
    )
    observation = observations[0]
    del observation["query"]
    _write_payload(path, _payload(observations=[observation]))

    with pytest.raises(FeedbackStoreFormatError, match="missing keys"):
        JsonRoutingFeedbackStore(path).load()


def test_active_lock_times_out_without_modifying_store(
    tmp_path: Path,
) -> None:
    path = tmp_path / "feedback.json"
    store = JsonRoutingFeedbackStore(
        path,
        lock_timeout_seconds=0.01,
        lock_poll_interval_seconds=0.001,
        stale_lock_seconds=60.0,
    )
    store.lock_path.write_text("pid=123\n", encoding="utf-8")

    with pytest.raises(FeedbackStoreLockTimeoutError, match="Timed out"):
        store.save(RoutingFeedbackHistory())

    assert not path.exists()
    assert store.lock_path.exists()


def test_stale_lock_is_removed_and_write_continues(tmp_path: Path) -> None:
    path = tmp_path / "feedback.json"
    store = JsonRoutingFeedbackStore(
        path,
        stale_lock_seconds=0.01,
    )
    store.lock_path.write_text("pid=123\n", encoding="utf-8")
    old_time = time.time() - 60.0
    os.utime(store.lock_path, (old_time, old_time))

    snapshot = store.save(RoutingFeedbackHistory())

    assert snapshot.revision == 1
    assert path.exists()
    assert not store.lock_path.exists()


def test_atomic_replace_failure_preserves_previous_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "feedback.json"
    store = JsonRoutingFeedbackStore(path)
    original = store.save(RoutingFeedbackHistory((_observation(),)))
    original_text = path.read_text(encoding="utf-8")

    def _fail_replace(source: Path, destination: Path) -> None:
        del source, destination
        raise OSError("simulated replace failure")

    monkeypatch.setattr(
        "fleetmind_rag.feedback_store.os.replace",
        _fail_replace,
    )

    with pytest.raises(FeedbackStoreIOError, match="simulated"):
        store.save(RoutingFeedbackHistory())

    assert path.read_text(encoding="utf-8") == original_text
    assert store.load() == original
    assert not store.lock_path.exists()
    assert not tuple(tmp_path.glob("*.tmp"))


def test_reading_directory_as_store_reports_io_error(tmp_path: Path) -> None:
    with pytest.raises(FeedbackStoreIOError, match="Unable to read"):
        JsonRoutingFeedbackStore(tmp_path).load()
