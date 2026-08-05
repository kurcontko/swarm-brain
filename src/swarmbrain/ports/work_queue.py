"""Durable leased-work boundary, distinct from the event relay outbox."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from swarmbrain.domain.agents import ActorContext
from swarmbrain.domain.work import (
    ApplyEmbeddingWorkCommand,
    ApplyExtractionWorkCommand,
    ApplyWorkResult,
    ClaimWorkCommand,
    CompleteWorkCommand,
    CompleteWorkResult,
    EnqueueWorkResult,
    FailWorkCommand,
    PreparedWorkEnqueue,
    WorkItem,
    WorkLeaseBatch,
)


@runtime_checkable
class WorkQueueStore(Protocol):
    async def enqueue_work(
        self,
        actor: ActorContext,
        prepared: PreparedWorkEnqueue,
    ) -> EnqueueWorkResult: ...

    async def claim_work(self, command: ClaimWorkCommand) -> WorkLeaseBatch: ...

    async def apply_extraction(
        self,
        command: ApplyExtractionWorkCommand,
    ) -> ApplyWorkResult: ...

    async def apply_embedding(
        self,
        command: ApplyEmbeddingWorkCommand,
    ) -> CompleteWorkResult: ...

    async def complete_work(
        self,
        command: CompleteWorkCommand,
    ) -> CompleteWorkResult: ...

    async def fail_work(self, command: FailWorkCommand) -> WorkItem: ...


__all__ = ["WorkQueueStore"]
