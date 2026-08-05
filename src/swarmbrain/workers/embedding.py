"""Embed-memory work handler; provider calls stay outside queue transactions."""

from __future__ import annotations

from swarmbrain.domain.memory import EmbeddingVector
from swarmbrain.domain.work import (
    EmbeddingWorkHandlerResult,
    EmbedMemoryWorkPayload,
    WorkKind,
    WorkLease,
    embedding_vector_sha256,
)
from swarmbrain.ports.embeddings import EmbeddingProvider


class EmbedMemoryHandler:
    """Turn one durable ``EMBED_MEMORY`` item into an idempotent vector upsert."""

    kind = WorkKind.EMBED_MEMORY

    def __init__(
        self,
        provider: EmbeddingProvider,
    ) -> None:
        self.provider = provider
        self.claim_model = provider.model_name

    async def handle(self, lease: WorkLease) -> EmbeddingWorkHandlerResult:
        payload = EmbedMemoryWorkPayload.model_validate(lease.item.payload)
        if payload.model != self.provider.model_name:
            raise ValueError(
                f"work requests model {payload.model!r} but the worker uses "
                f"{self.provider.model_name!r}"
            )
        (values,) = await self.provider.embed_documents([payload.content])
        if len(values) != self.provider.dimensions:
            raise ValueError(
                f"provider returned {len(values)} dimensions, expected {self.provider.dimensions}"
            )
        vector = EmbeddingVector(
            memory_id=lease.item.subject_id,
            model=payload.model,
            dimensions=len(values),
            values=values,
        )
        return EmbeddingWorkHandlerResult(
            vector=vector,
            outcome="completed",
            result={
                "model": payload.model,
                "dimensions": len(values),
                "vector_sha256": embedding_vector_sha256(vector),
            },
        )


__all__ = ["EmbedMemoryHandler"]
