"""Embedding generation and vector-index ports."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from swarmbrain.domain.agents import ActorContext
from swarmbrain.domain.common import IdempotencyKey
from swarmbrain.domain.memory import EmbeddingMatch, EmbeddingVector


@runtime_checkable
class EmbeddingProvider(Protocol):
    @property
    def model_name(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    async def embed_query(self, text: str) -> tuple[float, ...]: ...

    async def embed_documents(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]: ...


@runtime_checkable
class EmbeddingIndex(Protocol):
    """Scope-aware index; canonical memory content remains in MemoryStore."""

    async def upsert_embeddings(
        self,
        actor: ActorContext,
        vectors: Sequence[EmbeddingVector],
        *,
        idempotency_key: IdempotencyKey,
    ) -> None: ...

    async def search_embeddings(
        self,
        actor: ActorContext,
        query_vector: Sequence[float],
        *,
        model: str,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> tuple[EmbeddingMatch, ...]: ...


__all__ = ["EmbeddingIndex", "EmbeddingProvider"]
