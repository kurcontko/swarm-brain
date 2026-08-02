from __future__ import annotations

from swarmbrain.domain.agents import ActorContext
from swarmbrain.domain.tasks import CheckpointCommand, CheckpointResult

from .coordination import CoordinationService


class HandoffService:
    """Explicit application seam for durable checkpoint/crash handoff."""

    def __init__(self, coordination: CoordinationService) -> None:
        self.coordination = coordination

    async def checkpoint(
        self,
        actor: ActorContext,
        command: CheckpointCommand,
    ) -> CheckpointResult:
        return await self.coordination.checkpoint(actor, command)
