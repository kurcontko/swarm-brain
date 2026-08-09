"""OpenAI-compatible embedding provider for self-hosted vLLM/TEI endpoints."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from typing import Any

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
        query_instruction: str | None = (
            "Given a coding-agent memory search query, retrieve relevant memories"
        ),
        client: Any | None = None,
    ) -> None:
        if dimensions < 2:
            raise ValueError("dimensions must be at least 2")
        if not model_id or len(model_id) > 255:
            raise ValueError("model_id must contain between 1 and 255 characters")
        stripped = base_url.strip().rstrip("/")
        if not stripped or not stripped.startswith(("http://", "https://")):
            raise ValueError("base_url must be an http:// or https:// URL")
        self._base_url = stripped
        self._model_id = model_id
        self._dimensions = dimensions
        self._api_key = api_key
        self._query_instruction = query_instruction
        self._client = client

    @property
    def model_name(self) -> str:
        return self._model_id

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed_query(self, text: str) -> tuple[float, ...]:
        if self._query_instruction:
            text = f"Instruct: {self._query_instruction}\nQuery: {text}"
        vectors = await self._embed_batch([text])
        return vectors[0]

    async def embed_documents(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
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
                response = await client.post(
                    f"{self._base_url}/v1/embeddings",
                    json={"model": self._model_id, "input": texts},
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
