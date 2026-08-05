"""Cheap-to-create, strict-to-trust evidence registration boundary."""

from __future__ import annotations

from swarmbrain.domain.agents import ActorContext, Capability
from swarmbrain.domain.evidence import (
    AddEvidenceCommand,
    Evidence,
    EvidenceSource,
    RegisterEvidenceSourceCommand,
    ReviewSourceCommand,
    SourceTrust,
)
from swarmbrain.ports.memory_store import EvidenceStore

from .capabilities import require_capability


class EvidenceService:
    """Preserve evidence cheaply while keeping trust changes privileged."""

    def __init__(self, store: EvidenceStore) -> None:
        self.store = store

    async def register_source(
        self,
        actor: ActorContext,
        command: RegisterEvidenceSourceCommand,
    ) -> EvidenceSource:
        require_capability(actor, Capability.MEMORY_PUBLISH)
        if command.trust is not SourceTrust.UNKNOWN:
            require_capability(actor, Capability.SOURCE_REVIEW)
        return await self.store.register_source(actor, command)

    async def add_evidence(
        self,
        actor: ActorContext,
        command: AddEvidenceCommand,
    ) -> Evidence:
        require_capability(actor, Capability.MEMORY_PUBLISH)
        return await self.store.add_evidence(actor, command)

    async def review_source(
        self,
        actor: ActorContext,
        command: ReviewSourceCommand,
    ) -> EvidenceSource:
        require_capability(actor, Capability.SOURCE_REVIEW)
        return await self.store.review_source(actor, command)


__all__ = ["EvidenceService"]
