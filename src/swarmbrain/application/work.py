"""Deterministic enqueue boundary for provider and external-sink work."""

from __future__ import annotations

from uuid import NAMESPACE_URL, uuid5

from swarmbrain.domain.agents import ActorContext, Capability
from swarmbrain.domain.work import (
    EnqueueWorkCommand,
    EnqueueWorkResult,
    PreparedWorkEnqueue,
    WorkKind,
)
from swarmbrain.ports.work_queue import WorkQueueStore

from .capabilities import require_capability


class DurableWorkService:
    """Prepare scope-owned durable work without exposing identity fields in payloads."""

    def __init__(self, store: WorkQueueStore) -> None:
        self.store = store

    async def enqueue(
        self,
        actor: ActorContext,
        command: EnqueueWorkCommand,
    ) -> EnqueueWorkResult:
        capability = (
            Capability.MEMORY_PUBLISH
            if command.kind is WorkKind.EMBED_MEMORY
            else Capability.SOURCE_INGEST
        )
        require_capability(actor, capability)
        return await self.store.enqueue_work(actor, self.prepare_enqueue(actor, command))

    @staticmethod
    def prepare_enqueue(
        actor: ActorContext,
        command: EnqueueWorkCommand,
    ) -> PreparedWorkEnqueue:
        work_id = uuid5(
            NAMESPACE_URL,
            ":".join(
                (
                    "swarmbrain-work",
                    actor.tenant_id,
                    actor.run_id,
                    command.kind.value,
                    command.dedupe_key,
                )
            ),
        )
        return PreparedWorkEnqueue(work_id=str(work_id), command=command)


__all__ = ["DurableWorkService"]
