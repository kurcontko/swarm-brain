from __future__ import annotations

from swarmbrain.domain.agents import ActorContext, Capability
from swarmbrain.domain.conflicts import (
    ReportConflictCommand,
    ReportConflictResult,
    ResolveConflictCommand,
    ResolveConflictResult,
)
from swarmbrain.ports.memory_store import ConflictStore

from .capabilities import require_capability


class ConflictService:
    def __init__(self, store: ConflictStore) -> None:
        self.store = store

    async def report(
        self,
        actor: ActorContext,
        command: ReportConflictCommand,
    ) -> ReportConflictResult:
        require_capability(actor, Capability.CONFLICT_REPORT)
        return await self.store.report_conflict(actor, command)

    async def resolve(
        self,
        actor: ActorContext,
        command: ResolveConflictCommand,
    ) -> ResolveConflictResult:
        require_capability(actor, Capability.CONFLICT_RESOLVE)
        return await self.store.resolve_conflict(actor, command)
