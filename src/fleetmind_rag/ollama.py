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
