from dataclasses import dataclass

import httpx


@dataclass(frozen=True, slots=True)
class OllamaHealth:
    """Result of checking whether the Ollama API is available."""

    available: bool
    version: str | None
    message: str


class OllamaHealthClient:
    """Check the availability and version of an Ollama API server."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 3.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = httpx.Timeout(timeout_seconds)
        self._transport = transport

    def check(self) -> OllamaHealth:
        """Return the current Ollama API health without raising network errors."""

        try:
            with httpx.Client(
                base_url=self._base_url,
                timeout=self._timeout,
                transport=self._transport,
            ) as client:
                response = client.get("/api/version")
                response.raise_for_status()
        except httpx.TimeoutException:
            return OllamaHealth(
                available=False,
                version=None,
                message="The Ollama health check timed out.",
            )
        except httpx.RequestError:
            return OllamaHealth(
                available=False,
                version=None,
                message="The Ollama API is unreachable.",
            )
        except httpx.HTTPStatusError as error:
            return OllamaHealth(
                available=False,
                version=None,
                message=(f"The Ollama API returned HTTP {error.response.status_code}."),
            )

        try:
            payload = response.json()
        except ValueError:
            return OllamaHealth(
                available=False,
                version=None,
                message="The Ollama API returned invalid JSON.",
            )

        version = payload.get("version") if isinstance(payload, dict) else None

        if not isinstance(version, str) or not version.strip():
            return OllamaHealth(
                available=False,
                version=None,
                message="The Ollama API response did not contain a version.",
            )

        return OllamaHealth(
            available=True,
            version=version.strip(),
            message="The Ollama API is reachable.",
        )


@dataclass(frozen=True, slots=True)
class OllamaModel:
    """A model available through the Ollama API."""

    name: str


@dataclass(frozen=True, slots=True)
class OllamaModelListResult:
    """Result of requesting the models available through Ollama."""

    succeeded: bool
    models: tuple[OllamaModel, ...]
    message: str


class OllamaModelClient:
    """Retrieve the models available through an Ollama API server."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 3.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = httpx.Timeout(timeout_seconds)
        self._transport = transport

    def list_models(self) -> OllamaModelListResult:
        """Return the models available through the configured Ollama API."""

        try:
            with httpx.Client(
                base_url=self._base_url,
                timeout=self._timeout,
                transport=self._transport,
            ) as client:
                response = client.get("/api/tags")
                response.raise_for_status()
        except httpx.TimeoutException:
            return OllamaModelListResult(
                succeeded=False,
                models=(),
                message="The Ollama model request timed out.",
            )
        except httpx.RequestError:
            return OllamaModelListResult(
                succeeded=False,
                models=(),
                message="The Ollama API is unreachable.",
            )
        except httpx.HTTPStatusError as error:
            return OllamaModelListResult(
                succeeded=False,
                models=(),
                message=(f"The Ollama API returned HTTP {error.response.status_code}."),
            )

        try:
            payload = response.json()
        except ValueError:
            return OllamaModelListResult(
                succeeded=False,
                models=(),
                message="The Ollama API returned invalid JSON.",
            )

        raw_models = payload.get("models") if isinstance(payload, dict) else None

        if not isinstance(raw_models, list):
            return OllamaModelListResult(
                succeeded=False,
                models=(),
                message=("The Ollama API response did not contain a model list."),
            )

        models: list[OllamaModel] = []

        for raw_model in raw_models:
            if not isinstance(raw_model, dict):
                return OllamaModelListResult(
                    succeeded=False,
                    models=(),
                    message="The Ollama API returned an invalid model entry.",
                )

            name = raw_model.get("name")

            if not isinstance(name, str) or not name.strip():
                return OllamaModelListResult(
                    succeeded=False,
                    models=(),
                    message="The Ollama API returned an invalid model entry.",
                )

            models.append(OllamaModel(name=name.strip()))

        return OllamaModelListResult(
            succeeded=True,
            models=tuple(models),
            message=f"Installed Ollama model count: {len(models)}.",
        )
