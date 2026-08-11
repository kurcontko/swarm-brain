"""Provider boundary for bounded learned relevance reranking."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from swarmbrain.domain.reranking import (
    LearnedRerankerIdentity,
    LearnedRerankRequest,
    LearnedRerankResult,
)


@runtime_checkable
class LearnedRerankerProvider(Protocol):
    """Score exactly the candidate IDs supplied by the retrieval core.

    Implementations have no authority to generate, remove, hydrate, or expose
    candidates.  The application validates one result per input ID and every
    receipt binding before applying a score.
    """

    @property
    def identity(self) -> LearnedRerankerIdentity: ...

    async def rerank(self, request: LearnedRerankRequest) -> LearnedRerankResult: ...


__all__ = ["LearnedRerankerProvider"]
