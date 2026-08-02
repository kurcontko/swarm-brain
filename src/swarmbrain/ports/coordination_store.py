"""Storage-neutral asynchronous coordination boundary."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from swarmbrain.domain.agents import ActorContext, Agent
from swarmbrain.domain.common import RunId
from swarmbrain.domain.events import EventPage, RunMetrics
from swarmbrain.domain.leases import RenewLeaseCommand, RenewLeaseResult
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


@runtime_checkable
class CoordinationStore(Protocol):
    """Transactional task/lease repository, separate from swarm memory.

    Implementations must make claim, completion, release, their audit event,
    and their idempotency record atomic.  The CockroachDB adapter is also
    responsible for retrying serialization failures around those boundaries.
    """

    async def join_agent(self, actor: ActorContext) -> Agent: ...

    async def add_task(self, task: Task) -> Task: ...

    async def claim_task(
        self,
        actor: ActorContext,
        command: ClaimTaskCommand,
    ) -> ClaimTaskResult: ...

    async def renew_lease(
        self,
        actor: ActorContext,
        command: RenewLeaseCommand,
    ) -> RenewLeaseResult: ...

    async def checkpoint_task(
        self,
        actor: ActorContext,
        command: CheckpointCommand,
    ) -> CheckpointResult: ...

    async def complete_task(
        self,
        actor: ActorContext,
        command: CompleteTaskCommand,
    ) -> CompletionResult: ...

    async def release_task(
        self,
        actor: ActorContext,
        command: ReleaseTaskCommand,
    ) -> ReleaseResult: ...

    async def list_run_events(
        self,
        actor: ActorContext,
        run_id: RunId,
        *,
        cursor: str | None = None,
        limit: int = 100,
    ) -> EventPage: ...

    async def get_run_metrics(
        self,
        actor: ActorContext,
        run_id: RunId,
    ) -> RunMetrics: ...


__all__ = ["CoordinationStore"]
