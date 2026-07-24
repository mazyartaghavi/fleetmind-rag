from __future__ import annotations

import json
import math
import os
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, cast

from fleetmind_rag.feedback_routing import (
    FeedbackFeature,
    RoutingFeedbackHistory,
    RoutingFeedbackObservation,
)
from fleetmind_rag.routing import RetrievalStrategy

FEEDBACK_STORE_SCHEMA_VERSION: Final = 1

_STRATEGIES: Final = frozenset(
    {
        "dense",
        "sparse",
        "hybrid",
        "reranked",
    }
)
_VERDICTS: Final = frozenset({"accept", "rewrite"})
_FEATURES: Final = frozenset(
    {
        "exact_identifier",
        "quoted_phrase",
        "conceptual",
        "conditional",
        "action",
        "safety",
        "domain",
        "complex",
        "general",
    }
)
_SNAPSHOT_KEYS: Final = frozenset(
    {
        "schema_version",
        "revision",
        "observations",
    }
)
_OBSERVATION_KEYS: Final = frozenset(
    {
        "query",
        "strategy",
        "verdict",
        "quality_score",
        "attempt_number",
        "features",
    }
)


class FeedbackStoreError(RuntimeError):
    """Base error raised by persistent routing-feedback storage."""


class FeedbackStoreFormatError(FeedbackStoreError):
    """Raised when stored JSON violates the versioned feedback schema."""


class FeedbackStoreConflictError(FeedbackStoreError):
    """Raised when an optimistic revision check detects stale state."""


class FeedbackStoreLockTimeoutError(FeedbackStoreError):
    """Raised when another process retains the store lock too long."""


class FeedbackStoreIOError(FeedbackStoreError):
    """Raised when durable feedback storage cannot read or write data."""


@dataclass(frozen=True, slots=True)
class FeedbackStoreSnapshot:
    """One immutable, versioned view of persisted routing feedback."""

    history: RoutingFeedbackHistory
    revision: int
    schema_version: int = FEEDBACK_STORE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate internal snapshot invariants."""

        if self.schema_version != FEEDBACK_STORE_SCHEMA_VERSION:
            raise ValueError("schema_version must match FEEDBACK_STORE_SCHEMA_VERSION")

        if (
            isinstance(self.revision, bool)
            or not isinstance(self.revision, int)
            or self.revision < 0
        ):
            raise ValueError("revision must be a non-negative integer")


class JsonRoutingFeedbackStore:
    """Persist immutable routing feedback as atomic, versioned JSON."""

    def __init__(
        self,
        path: str | Path,
        *,
        lock_timeout_seconds: float = 5.0,
        lock_poll_interval_seconds: float = 0.05,
        stale_lock_seconds: float = 60.0,
    ) -> None:
        """Configure one cross-process routing-feedback store."""

        if isinstance(path, str) and not path.strip():
            raise ValueError("feedback store path must not be blank")

        resolved_path = Path(path).expanduser()

        if not resolved_path.name:
            raise ValueError("feedback store path must name a file")

        _validate_positive_finite(
            lock_timeout_seconds,
            field="lock_timeout_seconds",
        )
        _validate_positive_finite(
            lock_poll_interval_seconds,
            field="lock_poll_interval_seconds",
        )
        _validate_positive_finite(
            stale_lock_seconds,
            field="stale_lock_seconds",
        )

        self._path = resolved_path
        self._lock_path = resolved_path.with_name(f"{resolved_path.name}.lock")
        self._lock_timeout_seconds = lock_timeout_seconds
        self._lock_poll_interval_seconds = lock_poll_interval_seconds
        self._stale_lock_seconds = stale_lock_seconds

    @property
    def path(self) -> Path:
        """Return the configured JSON file path."""

        return self._path

    @property
    def lock_path(self) -> Path:
        """Return the sibling lock-file path."""

        return self._lock_path

    def load(self) -> FeedbackStoreSnapshot:
        """Load the latest complete snapshot or return empty initial state."""

        return self._load_unlocked()

    def save(
        self,
        history: RoutingFeedbackHistory,
        *,
        expected_revision: int | None = None,
    ) -> FeedbackStoreSnapshot:
        """Atomically replace history after an optional revision check."""

        _validate_expected_revision(expected_revision)

        with self._exclusive_lock():
            current = self._load_unlocked()
            _require_expected_revision(
                current.revision,
                expected_revision,
            )
            snapshot = FeedbackStoreSnapshot(
                history=history,
                revision=current.revision + 1,
            )
            self._write_unlocked(snapshot)
            return snapshot

    def append(
        self,
        observation: RoutingFeedbackObservation,
        *,
        expected_revision: int | None = None,
    ) -> FeedbackStoreSnapshot:
        """Atomically append one observation without losing concurrent data."""

        _validate_expected_revision(expected_revision)

        with self._exclusive_lock():
            current = self._load_unlocked()
            _require_expected_revision(
                current.revision,
                expected_revision,
            )
            snapshot = FeedbackStoreSnapshot(
                history=current.history.record(observation),
                revision=current.revision + 1,
            )
            self._write_unlocked(snapshot)
            return snapshot

    def _load_unlocked(self) -> FeedbackStoreSnapshot:
        if not self._path.exists():
            return FeedbackStoreSnapshot(
                history=RoutingFeedbackHistory(),
                revision=0,
            )

        try:
            text = self._path.read_text(encoding="utf-8")
        except OSError as error:
            raise FeedbackStoreIOError(
                f"Unable to read feedback store {self._path}: {error}"
            ) from error
        except UnicodeError as error:
            raise FeedbackStoreFormatError(
                f"Feedback store is not valid UTF-8: {self._path}"
            ) from error

        try:
            payload = cast(object, json.loads(text))
        except json.JSONDecodeError as error:
            raise FeedbackStoreFormatError(
                f"Feedback store contains invalid JSON: {self._path}"
            ) from error

        return _snapshot_from_payload(payload)

    def _write_unlocked(self, snapshot: FeedbackStoreSnapshot) -> None:
        payload = _snapshot_to_payload(snapshot)
        serialized = (
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        temporary_path: Path | None = None

        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)

            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                prefix=f".{self._path.name}.",
                suffix=".tmp",
                dir=self._path.parent,
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                temporary_file.write(serialized)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())

            os.replace(temporary_path, self._path)
            temporary_path = None
        except (OSError, UnicodeError) as error:
            raise FeedbackStoreIOError(
                f"Unable to write feedback store {self._path}: {error}"
            ) from error
        finally:
            if temporary_path is not None:
                with suppress(OSError):
                    temporary_path.unlink(missing_ok=True)

    @contextmanager
    def _exclusive_lock(self) -> Iterator[None]:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise FeedbackStoreIOError(
                f"Unable to create feedback store directory "
                f"{self._path.parent}: {error}"
            ) from error

        deadline = time.monotonic() + self._lock_timeout_seconds
        acquired = False

        while not acquired:
            descriptor: int | None = None

            try:
                descriptor = os.open(
                    self._lock_path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o600,
                )

                with os.fdopen(
                    descriptor,
                    mode="w",
                    encoding="utf-8",
                    newline="\n",
                ) as lock_file:
                    lock_file.write(f"pid={os.getpid()}\n")
                    lock_file.flush()
                    os.fsync(lock_file.fileno())

                acquired = True
            except FileExistsError:
                if self._remove_stale_lock():
                    continue

                if time.monotonic() >= deadline:
                    raise FeedbackStoreLockTimeoutError(
                        f"Timed out waiting for feedback store lock: {self._lock_path}"
                    ) from None

                time.sleep(self._lock_poll_interval_seconds)
            except OSError as error:
                if descriptor is not None:
                    with suppress(OSError):
                        os.close(descriptor)

                    with suppress(OSError):
                        self._lock_path.unlink(missing_ok=True)

                raise FeedbackStoreIOError(
                    f"Unable to acquire feedback store lock {self._lock_path}: {error}"
                ) from error

        try:
            yield
        finally:
            try:
                self._lock_path.unlink(missing_ok=True)
            except OSError as error:
                raise FeedbackStoreIOError(
                    f"Unable to release feedback store lock {self._lock_path}: {error}"
                ) from error

    def _remove_stale_lock(self) -> bool:
        try:
            age_seconds = time.time() - self._lock_path.stat().st_mtime
        except FileNotFoundError:
            return True
        except OSError as error:
            raise FeedbackStoreIOError(
                f"Unable to inspect feedback store lock {self._lock_path}: {error}"
            ) from error

        if age_seconds <= self._stale_lock_seconds:
            return False

        try:
            self._lock_path.unlink()
        except FileNotFoundError:
            pass
        except OSError as error:
            raise FeedbackStoreIOError(
                f"Unable to remove stale feedback store lock {self._lock_path}: {error}"
            ) from error

        return True


def _snapshot_to_payload(snapshot: FeedbackStoreSnapshot) -> dict[str, object]:
    observations = [
        {
            "query": observation.query,
            "strategy": observation.strategy,
            "verdict": observation.verdict,
            "quality_score": observation.quality_score,
            "attempt_number": observation.attempt_number,
            "features": list(observation.features),
        }
        for observation in snapshot.history.observations
    ]
    return {
        "schema_version": snapshot.schema_version,
        "revision": snapshot.revision,
        "observations": observations,
    }


def _snapshot_from_payload(payload: object) -> FeedbackStoreSnapshot:
    mapping = _require_mapping(payload, field="feedback snapshot")
    _require_exact_keys(
        mapping,
        expected=_SNAPSHOT_KEYS,
        field="feedback snapshot",
    )
    schema_version = _require_integer(
        mapping["schema_version"],
        field="schema_version",
    )

    if schema_version != FEEDBACK_STORE_SCHEMA_VERSION:
        raise FeedbackStoreFormatError(
            f"Unsupported feedback schema version: {schema_version}"
        )

    revision = _require_integer(
        mapping["revision"],
        field="revision",
    )

    if revision < 0:
        raise FeedbackStoreFormatError("revision must be a non-negative integer")

    raw_observations = mapping["observations"]

    if not isinstance(raw_observations, list):
        raise FeedbackStoreFormatError("observations must be a JSON array")

    observations = tuple(
        _observation_from_payload(raw_observation, index=index)
        for index, raw_observation in enumerate(raw_observations)
    )
    return FeedbackStoreSnapshot(
        history=RoutingFeedbackHistory(observations),
        revision=revision,
        schema_version=schema_version,
    )


def _observation_from_payload(
    payload: object,
    *,
    index: int,
) -> RoutingFeedbackObservation:
    field = f"observations[{index}]"
    mapping = _require_mapping(payload, field=field)
    _require_exact_keys(
        mapping,
        expected=_OBSERVATION_KEYS,
        field=field,
    )
    query = _require_string(mapping["query"], field=f"{field}.query")
    strategy_value = _require_string(
        mapping["strategy"],
        field=f"{field}.strategy",
    )

    if strategy_value not in _STRATEGIES:
        raise FeedbackStoreFormatError(
            f"{field}.strategy contains an unsupported value"
        )

    verdict_value = _require_string(
        mapping["verdict"],
        field=f"{field}.verdict",
    )

    if verdict_value not in _VERDICTS:
        raise FeedbackStoreFormatError(f"{field}.verdict contains an unsupported value")

    quality_score = _require_number(
        mapping["quality_score"],
        field=f"{field}.quality_score",
    )
    attempt_number = _require_integer(
        mapping["attempt_number"],
        field=f"{field}.attempt_number",
    )
    raw_features = mapping["features"]

    if not isinstance(raw_features, list):
        raise FeedbackStoreFormatError(f"{field}.features must be a JSON array")

    feature_values: list[str] = []

    for feature_index, raw_feature in enumerate(raw_features):
        feature = _require_string(
            raw_feature,
            field=f"{field}.features[{feature_index}]",
        )

        if feature not in _FEATURES:
            raise FeedbackStoreFormatError(
                f"{field}.features[{feature_index}] is unsupported"
            )

        feature_values.append(feature)

    strategy = cast(RetrievalStrategy, strategy_value)
    verdict = cast(Literal["accept", "rewrite"], verdict_value)
    features = cast(tuple[FeedbackFeature, ...], tuple(feature_values))

    try:
        return RoutingFeedbackObservation(
            query=query,
            strategy=strategy,
            verdict=verdict,
            quality_score=quality_score,
            attempt_number=attempt_number,
            features=features,
        )
    except (TypeError, ValueError) as error:
        raise FeedbackStoreFormatError(
            f"{field} violates routing-feedback invariants: {error}"
        ) from error


def _require_mapping(value: object, *, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise FeedbackStoreFormatError(f"{field} must be a JSON object")

    if any(not isinstance(key, str) for key in value):
        raise FeedbackStoreFormatError(f"{field} keys must be strings")

    return cast(dict[str, object], value)


def _require_exact_keys(
    mapping: dict[str, object],
    *,
    expected: frozenset[str],
    field: str,
) -> None:
    actual = frozenset(mapping)

    if actual == expected:
        return

    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    details: list[str] = []

    if missing:
        details.append(f"missing keys: {', '.join(missing)}")

    if unexpected:
        details.append(f"unexpected keys: {', '.join(unexpected)}")

    raise FeedbackStoreFormatError(f"{field} has {'; '.join(details)}")


def _require_string(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise FeedbackStoreFormatError(f"{field} must be a string")

    return value


def _require_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FeedbackStoreFormatError(f"{field} must be an integer")

    return value


def _require_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FeedbackStoreFormatError(f"{field} must be a number")

    result = float(value)

    if not math.isfinite(result):
        raise FeedbackStoreFormatError(f"{field} must be finite")

    return result


def _validate_expected_revision(expected_revision: int | None) -> None:
    if expected_revision is None:
        return

    if isinstance(expected_revision, bool) or expected_revision < 0:
        raise ValueError("expected_revision must be a non-negative integer")


def _require_expected_revision(
    actual_revision: int,
    expected_revision: int | None,
) -> None:
    if expected_revision is None or expected_revision == actual_revision:
        return

    raise FeedbackStoreConflictError(
        f"Feedback revision conflict: expected {expected_revision}, "
        f"found {actual_revision}"
    )


def _validate_positive_finite(value: float, *, field: str) -> None:
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{field} must be a positive finite number")
