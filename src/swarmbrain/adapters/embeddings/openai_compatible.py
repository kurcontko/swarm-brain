"""OpenAI-compatible embedding provider for self-hosted vLLM/TEI endpoints."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from typing import Any
from urllib.parse import urlsplit

_MAX_BATCH_SIZE = 64
_REQUEST_TIMEOUT_SECONDS = 60.0
# One in-flight embedding request is retried through brief network faults so a
# multi-hour benchmark or backfill does not die on a single dropped connection.
# Protocol-shape errors (wrong dimensions, bad payload) never retry.
_TRANSIENT_ATTEMPTS = 3
_TRANSIENT_BACKOFF_SECONDS = (0.5, 2.0)


class OpenAICompatibleUnavailable(RuntimeError):
    """The endpoint is unreachable or the provider response is incompatible."""


class OpenAICompatibleEmbeddingProvider:
    """Generate embeddings from any ``/v1/embeddings``-compatible server.

    Built for a self-hosted vLLM instance serving ``Qwen/Qwen3-Embedding-0.6B``,
    but any OpenAI-compatible embeddings endpoint works.  Vectors are
    L2-normalized client-side so the stored projection does not depend on the
    server's pooling configuration.  Instruction-aware models receive the
    configured task instruction on queries only; documents embed raw, matching
    the Qwen3-Embedding usage contract.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model_id: str = "Qwen/Qwen3-Embedding-0.6B",
        dimensions: int = 1024,
        api_key: str | None = None,
        required_response_model: str | None = None,
        query_instruction: str | None = (
            "Given a coding-agent memory search query, retrieve relevant memories"
        ),
        truncate_prompt_tokens: int | None = None,
        client: Any | None = None,
    ) -> None:
        if truncate_prompt_tokens is not None and truncate_prompt_tokens < 1:
            raise ValueError("truncate_prompt_tokens must be at least 1")
        if dimensions < 2:
            raise ValueError("dimensions must be at least 2")
        if not model_id or len(model_id) > 255:
            raise ValueError("model_id must contain between 1 and 255 characters")
        if required_response_model is not None and (
            not required_response_model.strip() or len(required_response_model) > 255
        ):
            raise ValueError("required_response_model must contain between 1 and 255 characters")
        stripped = base_url.strip().rstrip("/")
        if not stripped or not stripped.startswith(("http://", "https://")):
            raise ValueError("base_url must be an http:// or https:// URL")
        parsed = urlsplit(stripped)
        if (
            not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "base_url must not contain credentials, query parameters, or a fragment"
            )
        if stripped.endswith("/v1"):
            stripped = stripped[: -len("/v1")]
        self._base_url = stripped
        self._model_id = model_id
        self._dimensions = dimensions
        self._api_key = api_key
        self._required_response_model = required_response_model
        self._query_instruction = query_instruction
        self._truncate_prompt_tokens = truncate_prompt_tokens
        self._client = client
        self._owns_client = client is None
        self._document_inputs = 0
        self._document_batch_calls = 0
        self._query_calls = 0
        self._http_attempts = 0
        self._successful_http_calls = 0

    @property
    def model_name(self) -> str:
        return self._model_id

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def required_response_model(self) -> str | None:
        return self._required_response_model

    @property
    def call_accounting(self) -> dict[str, int]:
        return {
            "document_inputs": self._document_inputs,
            "document_batch_calls": self._document_batch_calls,
            "query_calls": self._query_calls,
            "successful_http_calls": self._successful_http_calls,
            "http_attempts": self._http_attempts,
        }

    def reset_call_accounting(self) -> None:
        self._document_inputs = 0
        self._document_batch_calls = 0
        self._query_calls = 0
        self._http_attempts = 0
        self._successful_http_calls = 0

    async def close(self) -> None:
        if not self._owns_client or self._client is None:
            return
        close = getattr(self._client, "aclose", None)
        if callable(close):
            await close()
        self._client = None

    async def embed_query(self, text: str) -> tuple[float, ...]:
        self._query_calls += 1
        if self._query_instruction:
            text = f"Instruct: {self._query_instruction}\nQuery: {text}"
        vectors = await self._embed_batch([text])
        return vectors[0]

    async def embed_documents(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        self._document_batch_calls += 1
        self._document_inputs += len(texts)
        vectors: list[tuple[float, ...]] = []
        for start in range(0, len(texts), _MAX_BATCH_SIZE):
            vectors.extend(await self._embed_batch(list(texts[start : start + _MAX_BATCH_SIZE])))
        return tuple(vectors)

    async def _embed_batch(self, texts: list[str]) -> list[tuple[float, ...]]:
        if not texts:
            return []
        client = self._ensure_client()
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        payload: Any = None
        for attempt in range(_TRANSIENT_ATTEMPTS):
            try:
                self._http_attempts += 1
                body: dict[str, Any] = {"model": self._model_id, "input": texts}
                if self._truncate_prompt_tokens is not None:
                    # vLLM extension: server-side head truncation for inputs
                    # beyond the model's context window. Part of the embedding
                    # identity, so it must be reflected in any projection
                    # signature naming this provider configuration.
                    body["truncate_prompt_tokens"] = self._truncate_prompt_tokens
                response = await client.post(
                    f"{self._base_url}/v1/embeddings",
                    json=body,
                    headers=headers,
                    timeout=_REQUEST_TIMEOUT_SECONDS,
                )
                response.raise_for_status()
                payload = response.json()
                break
            except OpenAICompatibleUnavailable:
                raise
            except Exception as exc:
                if attempt + 1 == _TRANSIENT_ATTEMPTS:
                    raise OpenAICompatibleUnavailable(
                        f"embeddings request to {self._base_url} failed after "
                        f"{_TRANSIENT_ATTEMPTS} attempts: {type(exc).__name__}"
                    ) from exc
                await asyncio.sleep(_TRANSIENT_BACKOFF_SECONDS[attempt])
        if self._required_response_model is not None and payload.get("model") != (
            self._required_response_model
        ):
            raise OpenAICompatibleUnavailable(
                "embedding endpoint response model does not match the required revision"
            )
        rows = payload.get("data")
        if not isinstance(rows, list) or len(rows) != len(texts):
            raise OpenAICompatibleUnavailable(
                f"endpoint returned {0 if not isinstance(rows, list) else len(rows)} "
                f"embeddings, expected {len(texts)}"
            )
        ordered: list[tuple[float, ...] | None] = [None] * len(texts)
        for row in rows:
            index = row.get("index") if isinstance(row, dict) else None
            values = row.get("embedding") if isinstance(row, dict) else None
            if not isinstance(index, int) or not 0 <= index < len(texts):
                raise OpenAICompatibleUnavailable("endpoint returned an invalid embedding index")
            if not isinstance(values, list) or len(values) != self._dimensions:
                got = len(values) if isinstance(values, list) else "no"
                raise OpenAICompatibleUnavailable(
                    f"model returned {got} dimensions, expected {self._dimensions}"
                )
            ordered[index] = self._normalize(values)
        complete = [vector for vector in ordered if vector is not None]
        if len(complete) != len(texts):
            raise OpenAICompatibleUnavailable("endpoint returned duplicate embedding indexes")
        self._successful_http_calls += 1
        return complete

    def _normalize(self, values: list[Any]) -> tuple[float, ...]:
        floats = [float(value) for value in values]
        norm = math.sqrt(sum(value * value for value in floats))
        if not math.isfinite(norm) or norm == 0.0:
            raise OpenAICompatibleUnavailable("model returned a zero or non-finite vector")
        return tuple(value / norm for value in floats)

    def _ensure_client(self) -> Any:
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient()
        return self._client


__all__ = ["OpenAICompatibleEmbeddingProvider", "OpenAICompatibleUnavailable"]
