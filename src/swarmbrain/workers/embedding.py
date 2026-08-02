"""Embed-memory work handler; provider calls stay outside queue transactions."""

from __future__ import annotations

from swarmbrain.domain.agents import ActorContext, Capability
from swarmbrain.domain.memory import EmbeddingVector
from swarmbrain.domain.work import (
    EmbedMemoryWorkPayload,
    WorkHandlerResult,
    WorkItem,
    WorkKind,
    WorkLease,
)
from swarmbrain.ports.embeddings import EmbeddingIndex, EmbeddingProvider


class EmbedMemoryHandler:
    """Turn one durable EMBED_MEMORY item into an upserted vector.

    The provider call happens between the committed claim and the fenced
    completion, per the leased-work contract. The actor used for the upsert is
    reconstructed from the durable work item's scope, never from the payload.
    """

    kind = WorkKind.EMBED_MEMORY

    def __init__(
        self,
        provider: EmbeddingProvider,
        index: EmbeddingIndex,
        *,
        harness: str = "swarmbrain-embedding-worker",
    ) -> None:
        self.provider = provider
        self.index = index
        self.harness = harness

    async def handle(self, lease: WorkLease) -> WorkHandlerResult:
        payload = EmbedMemoryWorkPayload.model_validate(lease.item.payload)
        if payload.model != self.provider.model_name:
            raise ValueError(
                f"work requests model {payload.model!r} but the worker embeds "
                f"with {self.provider.model_name!r}"
            )
        (values,) = await self.provider.embed_documents([payload.content])
        await self.index.upsert_embeddings(
            self._actor_for(lease.item),
            [
                EmbeddingVector(
                    memory_id=lease.item.subject_id,
                    model=payload.model,
                    dimensions=len(values),
                    values=values,
                )
            ],
            idempotency_key=f"embed:{lease.item.work_id}",
        )
        return WorkHandlerResult(
            outcome="completed",
            result={"model": payload.model, "dimensions": len(values)},
        )

    def _actor_for(self, item: WorkItem) -> ActorContext:
        return ActorContext(
            tenant_id=item.tenant_id,
            project_id=item.project_id,
            repository_id=item.repository_id,
            swarm_id=item.swarm_id,
            run_id=item.run_id,
            agent_id=item.requested_by_agent_id,
            harness=self.harness,
            provider="swarmbrain",
            model=self.provider.model_name,
            capabilities=frozenset({Capability.MEMORY_PUBLISH.value}),
        )


__all__ = ["EmbedMemoryHandler"]
