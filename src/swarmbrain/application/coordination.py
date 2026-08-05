from __future__ import annotations

import logging

from swarmbrain.domain.agents import ActorContext, Agent, Capability
from swarmbrain.domain.events import EventPage, RunMetrics
from swarmbrain.domain.leases import RenewLeaseCommand, RenewLeaseResult
from swarmbrain.domain.memory import RecallQuery, Visibility
from swarmbrain.domain.tasks import (
    CheckpointCommand,
    CheckpointResult,
    ClaimTaskCommand,
    ClaimTaskResult,
    CompleteTaskCommand,
    CompletionResult,
    ReleaseResult,
    ReleaseTaskCommand,
    Task,
)
from swarmbrain.ports.coordination_store import CoordinationStore

from .capabilities import require_capability
from .memory_service import MemoryService

logger = logging.getLogger(__name__)


class CoordinationService:
    def __init__(
        self,
        store: CoordinationStore,
        *,
        memory_service: MemoryService | None = None,
        initial_memory_limit: int = 12,
    ) -> None:
        self.store = store
        self.memory_service = memory_service
        self.initial_memory_limit = initial_memory_limit

    async def join(self, actor: ActorContext) -> Agent:
        require_capability(actor, Capability.RUN_JOIN)
        return await self.store.join_agent(actor)

    async def add_task(self, task: Task) -> Task:
        """Administrative bootstrap seam; callers authorize outside model-visible MCP."""

        return await self.store.add_task(task)

    async def claim(
        self,
        actor: ActorContext,
        command: ClaimTaskCommand,
    ) -> ClaimTaskResult:
        require_capability(actor, Capability.TASK_CLAIM)
        result = await self.store.claim_task(actor, command)
        if self.memory_service is None or not actor.has_capability(Capability.MEMORY_RECALL):
            return result

        query_text = " ".join(
            part for part in (result.task.title, result.task.description, *result.task.tags) if part
        )
        try:
            memory = await self.memory_service.recall(
                actor,
                RecallQuery(
                    text=query_text,
                    task_id=result.task.task_id,
                    visibilities=frozenset(Visibility),
                    limit=self.initial_memory_limit,
                ),
            )
        except Exception:
            # Claim is already committed. Initial memory is optional context,
            # so a recall outage must not hide the lease from its new owner.
            logger.warning(
                "initial memory recall failed after task claim",
                extra={"task_id": result.task.task_id, "agent_id": actor.agent_id},
                exc_info=True,
            )
            return result
        return result.model_copy(update={"memory": memory})

    async def renew(
        self,
        actor: ActorContext,
        command: RenewLeaseCommand,
    ) -> RenewLeaseResult:
        require_capability(actor, Capability.LEASE_RENEW)
        return await self.store.renew_lease(actor, command)

    async def checkpoint(
        self,
        actor: ActorContext,
        command: CheckpointCommand,
    ) -> CheckpointResult:
        require_capability(actor, Capability.TASK_CHECKPOINT)
        return await self.store.checkpoint_task(actor, command)

    async def complete(
        self,
        actor: ActorContext,
        command: CompleteTaskCommand,
    ) -> CompletionResult:
        require_capability(actor, Capability.TASK_COMPLETE)
        return await self.store.complete_task(actor, command)

    async def release(
        self,
        actor: ActorContext,
        command: ReleaseTaskCommand,
    ) -> ReleaseResult:
        require_capability(actor, Capability.TASK_RELEASE)
        return await self.store.release_task(actor, command)

    async def events(
        self,
        actor: ActorContext,
        *,
        cursor: str | None = None,
        limit: int = 100,
    ) -> EventPage:
        require_capability(actor, Capability.EVENTS_READ)
        return await self.store.list_run_events(
            actor,
            actor.run_id,
            cursor=cursor,
            limit=limit,
        )

    async def metrics(self, actor: ActorContext) -> RunMetrics:
        require_capability(actor, Capability.METRICS_READ)
        return await self.store.get_run_metrics(actor, actor.run_id)
