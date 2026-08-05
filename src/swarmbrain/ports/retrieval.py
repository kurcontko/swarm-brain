"""Provider-neutral read ports for multi-lane memory retrieval."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from swarmbrain.domain.agents import ActorContext
from swarmbrain.domain.common import MemoryId
from swarmbrain.domain.memory import Memory, RecallQuery
from swarmbrain.domain.retrieval import (
    CandidateBatch,
    RetrievalPlan,
    RetrievalSignal,
    RetrievalTrace,
)


@runtime_checkable
class RetrievalGateway(Protocol):
    """Generate references for exactly one retrieval signal lane."""

    @property
    def signal(self) -> RetrievalSignal: ...

    async def retrieve(
        self,
        actor: ActorContext,
        plan: RetrievalPlan,
        query: RecallQuery,
    ) -> CandidateBatch: ...


@runtime_checkable
class CanonicalMemoryReader(Protocol):
    """Revalidate and hydrate internal candidate IDs in requested order."""

    async def hydrate_recallable(
        self,
        actor: ActorContext,
        query: RecallQuery,
        candidate_ids: Sequence[MemoryId],
    ) -> tuple[Memory, ...]: ...


@runtime_checkable
class RetrievalTraceSink(Protocol):
    async def record(self, trace: RetrievalTrace) -> None: ...


__all__ = ["CanonicalMemoryReader", "RetrievalGateway", "RetrievalTraceSink"]
