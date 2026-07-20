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


@dataclass(frozen=True, slots=True)
class OllamaChatResult:
    """Result of requesting a chat response from Ollama."""

    succeeded: bool
    content: str | None
    model: str | None
    message: str


class OllamaChatClient:
    """Generate non-streaming chat responses through the Ollama API."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout_seconds: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("The Ollama chat model must not be empty.")

        self._base_url = base_url.rstrip("/")
        self._model = model.strip()
        self._timeout = httpx.Timeout(timeout_seconds)
        self._transport = transport

    def chat(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> OllamaChatResult:
        """Generate one complete assistant response for a user prompt."""

        clean_prompt = prompt.strip()

        if not clean_prompt:
            raise ValueError("The Ollama chat prompt must not be empty.")

        messages: list[dict[str, str]] = []

        if system_prompt is not None and system_prompt.strip():
            messages.append(
                {
                    "role": "system",
                    "content": system_prompt.strip(),
                }
            )

        messages.append(
            {
                "role": "user",
                "content": clean_prompt,
            }
        )

        try:
            with httpx.Client(
                base_url=self._base_url,
                timeout=self._timeout,
                transport=self._transport,
            ) as client:
                response = client.post(
                    "/api/chat",
                    json={
                        "model": self._model,
                        "messages": messages,
                        "stream": False,
                    },
                )
                response.raise_for_status()
        except httpx.TimeoutException:
            return OllamaChatResult(
                succeeded=False,
                content=None,
                model=None,
                message="The Ollama chat request timed out.",
            )
        except httpx.RequestError:
            return OllamaChatResult(
                succeeded=False,
                content=None,
                model=None,
                message="The Ollama API is unreachable.",
            )
        except httpx.HTTPStatusError as error:
            return OllamaChatResult(
                succeeded=False,
                content=None,
                model=None,
                message=(f"The Ollama API returned HTTP {error.response.status_code}."),
            )

        try:
            payload = response.json()
        except ValueError:
            return OllamaChatResult(
                succeeded=False,
                content=None,
                model=None,
                message="The Ollama API returned invalid JSON.",
            )

        if not isinstance(payload, dict):
            return OllamaChatResult(
                succeeded=False,
                content=None,
                model=None,
                message="The Ollama API returned an invalid chat response.",
            )

        response_message = payload.get("message")

        if not isinstance(response_message, dict):
            return OllamaChatResult(
                succeeded=False,
                content=None,
                model=None,
                message="The Ollama API returned an invalid chat response.",
            )

        content = response_message.get("content")

        if not isinstance(content, str) or not content.strip():
            return OllamaChatResult(
                succeeded=False,
                content=None,
                model=None,
                message="The Ollama API returned an empty chat response.",
            )

        response_model = payload.get("model")

        if not isinstance(response_model, str) or not response_model.strip():
            response_model = self._model

        return OllamaChatResult(
            succeeded=True,
            content=content.strip(),
            model=response_model.strip(),
            message="The Ollama chat request succeeded.",
        )


@dataclass(frozen=True, slots=True)
class OllamaEmbeddingResult:
    """Result of requesting embeddings from Ollama."""

    succeeded: bool
    embeddings: tuple[tuple[float, ...], ...]
    model: str | None
    message: str


class OllamaEmbeddingClient:
    """Generate text embeddings through the Ollama API."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout_seconds: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("The Ollama embedding model must not be empty.")

        self._base_url = base_url.rstrip("/")
        self._model = model.strip()
        self._timeout = httpx.Timeout(timeout_seconds)
        self._transport = transport

    def embed(
        self,
        input_value: str | list[str] | tuple[str, ...],
    ) -> OllamaEmbeddingResult:
        """Generate embeddings for one text or a batch of texts."""

        normalized_inputs: tuple[str, ...]
        request_input: str | list[str]

        if isinstance(input_value, str):
            cleaned_input = input_value.strip()

            if not cleaned_input:
                raise ValueError("The Ollama embedding input must not be empty.")

            normalized_inputs = (cleaned_input,)
            request_input = cleaned_input
        else:
            normalized_inputs = tuple(text.strip() for text in input_value)

            if not normalized_inputs:
                raise ValueError("The Ollama embedding input must not be empty.")

            if any(not text for text in normalized_inputs):
                raise ValueError("Ollama embedding inputs must not contain empty text.")

            request_input = list(normalized_inputs)

        try:
            with httpx.Client(
                base_url=self._base_url,
                timeout=self._timeout,
                transport=self._transport,
            ) as client:
                response = client.post(
                    "/api/embed",
                    json={
                        "model": self._model,
                        "input": request_input,
                    },
                )
                response.raise_for_status()
        except httpx.TimeoutException:
            return OllamaEmbeddingResult(
                succeeded=False,
                embeddings=(),
                model=None,
                message="The Ollama embedding request timed out.",
            )
        except httpx.RequestError:
            return OllamaEmbeddingResult(
                succeeded=False,
                embeddings=(),
                model=None,
                message="The Ollama API is unreachable.",
            )
        except httpx.HTTPStatusError as error:
            return OllamaEmbeddingResult(
                succeeded=False,
                embeddings=(),
                model=None,
                message=(f"The Ollama API returned HTTP {error.response.status_code}."),
            )

        try:
            payload = response.json()
        except ValueError:
            return OllamaEmbeddingResult(
                succeeded=False,
                embeddings=(),
                model=None,
                message="The Ollama API returned invalid JSON.",
            )

        if not isinstance(payload, dict):
            return self._invalid_response_result()

        raw_embeddings = payload.get("embeddings")

        if not isinstance(raw_embeddings, list) or not raw_embeddings:
            return self._invalid_response_result()

        parsed_embeddings: list[tuple[float, ...]] = []
        dimension: int | None = None

        for raw_vector in raw_embeddings:
            if not isinstance(raw_vector, list) or not raw_vector:
                return self._invalid_response_result()

            vector_values: list[float] = []

            for value in raw_vector:
                if isinstance(value, bool) or not isinstance(
                    value,
                    (int, float),
                ):
                    return self._invalid_response_result()

                vector_values.append(float(value))

            vector = tuple(vector_values)

            if dimension is None:
                dimension = len(vector)
            elif len(vector) != dimension:
                return self._invalid_response_result()

            parsed_embeddings.append(vector)

        if len(parsed_embeddings) != len(normalized_inputs):
            return self._invalid_response_result()

        response_model = payload.get("model")

        if not isinstance(response_model, str) or not response_model.strip():
            response_model = self._model

        return OllamaEmbeddingResult(
            succeeded=True,
            embeddings=tuple(parsed_embeddings),
            model=response_model.strip(),
            message=(
                f"Generated {len(parsed_embeddings)} Ollama embedding(s) "
                f"with dimension {dimension}."
            ),
        )

    @staticmethod
    def _invalid_response_result() -> OllamaEmbeddingResult:
        """Build a standard result for malformed embedding responses."""

        return OllamaEmbeddingResult(
            succeeded=False,
            embeddings=(),
            model=None,
            message=("The Ollama API returned an invalid embedding response."),
        )
