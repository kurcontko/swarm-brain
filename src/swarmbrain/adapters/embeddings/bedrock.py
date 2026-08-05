"""Amazon Bedrock embedding provider behind a lazy optional-SDK boundary."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from typing import Any


class BedrockUnavailable(RuntimeError):
    """The optional SDK is absent or the provider response is incompatible."""


class BedrockEmbeddingProvider:
    """Generate normalized Titan text embeddings outside database transactions."""

    def __init__(
        self,
        *,
        model_id: str = "amazon.titan-embed-text-v2:0",
        dimensions: int = 1024,
        region_name: str | None = None,
        client: Any | None = None,
    ) -> None:
        if dimensions < 2:
            raise ValueError("dimensions must be at least 2")
        if not model_id or len(model_id) > 255:
            raise ValueError("model_id must contain between 1 and 255 characters")
        self._model_id = model_id
        self._dimensions = dimensions
        self._region_name = region_name
        self._client = client

    @property
    def model_name(self) -> str:
        return self._model_id

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed_query(self, text: str) -> tuple[float, ...]:
        return await asyncio.to_thread(self._invoke, text)

    async def embed_documents(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        # Titan accepts one input per invocation.  Sequential calls avoid an
        # unbounded fan-out from one leased work item.
        vectors = []
        for text in texts:
            vectors.append(await asyncio.to_thread(self._invoke, text))
        return tuple(vectors)

    def _invoke(self, text: str) -> tuple[float, ...]:
        client = self._ensure_client()
        response = client.invoke_model(
            modelId=self._model_id,
            body=json.dumps(
                {
                    "inputText": text,
                    "dimensions": self._dimensions,
                    "normalize": True,
                }
            ),
            accept="application/json",
            contentType="application/json",
        )
        payload = json.loads(response["body"].read())
        values = tuple(float(item) for item in payload["embedding"])
        if len(values) != self._dimensions:
            raise BedrockUnavailable(
                f"model returned {len(values)} dimensions, expected {self._dimensions}"
            )
        return values

    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                import boto3
            except ImportError as exc:  # pragma: no cover - environment-specific
                raise BedrockUnavailable(
                    "boto3 is required for Bedrock embeddings; install swarmbrain[aws]"
                ) from exc
            kwargs = {"region_name": self._region_name} if self._region_name else {}
            self._client = boto3.client("bedrock-runtime", **kwargs)
        return self._client


__all__ = ["BedrockEmbeddingProvider", "BedrockUnavailable"]
