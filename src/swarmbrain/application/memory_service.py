from __future__ import annotations

from swarmbrain.domain.agents import ActorContext, Capability
from swarmbrain.domain.common import MemoryId
from swarmbrain.domain.evidence import RejectSourceCommand, SourceRejectionResult
from swarmbrain.domain.memory import (
    MemoryLineage,
    MemoryReviewDecision,
    MemoryState,
    RecallBundle,
    RecallQuery,
    RememberCommand,
    RememberResult,
    ReviewMemoryCommand,
    ReviewMemoryResult,
)
from swarmbrain.ports.memory_store import MemoryOperationPolicy, MemoryReviewStore, MemoryStore

from .capabilities import require_capability


class MemoryService:
    def __init__(
        self,
        store: MemoryStore,
        policy: MemoryOperationPolicy,
        *,
        review_store: MemoryReviewStore | None = None,
    ) -> None:
        self.store = store
        self.policy = policy
        self.review_store = review_store

    async def publish(self, actor: ActorContext, command: RememberCommand) -> RememberResult:
        require_capability(actor, Capability.MEMORY_PUBLISH)
        if command.desired_state is MemoryState.CONFIRMED:
            require_capability(actor, Capability.MEMORY_CONFIRM)
        if command.desired_state is MemoryState.REFUTED:
            require_capability(actor, Capability.MEMORY_REFUTE)
        return await self.store.remember(actor, command, self.policy)

    async def recall(self, actor: ActorContext, query: RecallQuery) -> RecallBundle:
        require_capability(actor, Capability.MEMORY_RECALL)
        return await self.store.recall(actor, query)

    async def lineage(self, actor: ActorContext, memory_id: MemoryId) -> MemoryLineage:
        require_capability(actor, Capability.MEMORY_RECALL)
        return await self.store.get_lineage(actor, memory_id)

    async def reject_source(
        self,
        actor: ActorContext,
        command: RejectSourceCommand,
    ) -> SourceRejectionResult:
        require_capability(actor, Capability.SOURCE_REVIEW)
        return await self.store.reject_source(actor, command)

    async def review(
        self,
        actor: ActorContext,
        command: ReviewMemoryCommand,
    ) -> ReviewMemoryResult:
        if self.review_store is None:
            raise RuntimeError("configured memory store does not support review operations")
        capability = (
            Capability.MEMORY_CONFIRM
            if command.decision is MemoryReviewDecision.CONFIRM
            else Capability.MEMORY_REFUTE
        )
        require_capability(actor, capability)
        return await self.review_store.review_memory(actor, command)
