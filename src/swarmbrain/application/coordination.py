from __future__ import annotations

import logging
from math import isfinite

from swarmbrain.domain.activation import (
    ActivationTrigger,
    MemoryActivationCommand,
    MemoryActivationDelivery,
    MemoryActivationRequest,
)
from swarmbrain.domain.agents import ActorContext, Agent, Capability
from swarmbrain.domain.events import EventPage, RunMetrics
from swarmbrain.domain.exploration import ReadExpandMemoryRequest, ReadExpandMemoryResult
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
from swarmbrain.ports.coordination_store import CoordinationStore

from .activation import MemoryActivationService
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
        initial_memory_token_budget: int = 2048,
        initial_memory_min_score: float = 0.4,
    ) -> None:
        if type(initial_memory_limit) is not int or not 1 <= initial_memory_limit <= 100:
            raise ValueError("initial_memory_limit must be between 1 and 100")
        if (
            type(initial_memory_token_budget) is not int
            or not 1 <= initial_memory_token_budget <= 131_072
        ):
            raise ValueError("initial_memory_token_budget must be between 1 and 131072")
        if (
            isinstance(initial_memory_min_score, bool)
            or not isinstance(initial_memory_min_score, (int, float))
            or not isfinite(initial_memory_min_score)
            or not 0.0 <= initial_memory_min_score <= 1.0
        ):
            raise ValueError("initial_memory_min_score must be between 0 and 1")
        self.store = store
        self.memory_service = memory_service
        self.memory_activation = (
            MemoryActivationService(memory_service) if memory_service is not None else None
        )
        self.initial_memory_limit = initial_memory_limit
        self.initial_memory_token_budget = initial_memory_token_budget
        self.initial_memory_min_score = initial_memory_min_score

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
        if self.memory_activation is None or not actor.has_capability(Capability.MEMORY_RECALL):
            return result

        checkpoint = result.checkpoint
        activation_request = MemoryActivationRequest(
            task_id=result.task.task_id,
            lease_id=result.lease.lease_id,
            trigger=(
                ActivationTrigger.CHECKPOINT_RESUME
                if checkpoint is not None
                else (
                    ActivationTrigger.DEPENDENCY_UNBLOCKED
                    if result.unblocked_by_task_ids
                    else ActivationTrigger.TASK_CLAIM
                )
            ),
            seed_memory_ids=(checkpoint.memory_ids if checkpoint is not None else ()),
            token_budget=self.initial_memory_token_budget,
            min_score=self.initial_memory_min_score,
            limit=self.initial_memory_limit,
        )
        if result.replayed:
            try:
                recorded = await self.store.get_memory_activation(
                    actor,
                    activation_request.activation_id,
                )
            except Exception:
                # A replay lookup failure cannot hide the durable lease, and it
                # must not cause an activation that may disagree with storage.
                logger.warning(
                    "memory activation replay lookup failed",
                    extra={
                        "task_id": result.task.task_id,
                        "agent_id": actor.agent_id,
                        "activation_id": activation_request.activation_id,
                    },
                    exc_info=True,
                )
                return result
            if recorded is not None:
                # The exact context was intentionally not persisted. Return the
                # stable lease without recomputing against newer memory state.
                return result

        query_text = " ".join(
            part
            for part in (
                result.task.title,
                result.task.description,
                *result.task.tags,
                *sorted(result.task.required_capabilities),
                checkpoint.summary if checkpoint is not None else "",
                *(checkpoint.discoveries if checkpoint is not None else ()),
                *(checkpoint.remaining_work if checkpoint is not None else ()),
            )
            if part
        )
        try:
            activation = await self.memory_activation.activate(
                actor,
                activation_request,
                query_text=query_text,
            )
        except Exception:
            # Claim is already committed. No optional activation failure may
            # hide the lease from its owner; capability authorization happened
            # before the mutation and failed activation never injects context.
            logger.warning(
                "initial memory activation failed after task claim",
                extra={"task_id": result.task.task_id, "agent_id": actor.agent_id},
                exc_info=True,
            )
            return result

        try:
            await self.store.record_memory_activation(actor, activation.telemetry)
        except Exception:
            # The claim remains committed, but untracked memory must not cross
            # the agent boundary. Returning the base claim preserves its lease
            # while keeping activation/citation metrics causally honest.
            logger.warning(
                "memory activation telemetry persistence failed",
                extra={
                    "task_id": result.task.task_id,
                    "agent_id": actor.agent_id,
                    "activation_id": activation.activation_id,
                },
                exc_info=True,
            )
            return result
        return result.model_copy(
            update={
                "memory": activation.bundle,
                "activation": activation.telemetry,
                "activation_context": activation.rendered_context or None,
            }
        )

    async def activate_memory(
        self,
        actor: ActorContext,
        command: MemoryActivationCommand,
    ) -> MemoryActivationDelivery:
        """Run one deterministic in-task intervention under an owned lease."""

        require_capability(actor, Capability.MEMORY_RECALL)
        if self.memory_activation is None:
            raise RuntimeError("memory activation is not configured")
        await self.store.validate_memory_activation_scope(
            actor,
            command.task_id,
            command.lease_id,
        )
        request = command.request()
        recorded = await self.store.get_memory_activation(actor, request.activation_id)
        if recorded is not None:
            return MemoryActivationDelivery(activation=recorded, replayed=True)

        activation = await self.memory_activation.activate(
            actor,
            request,
            query_text=command.query_text,
        )
        try:
            await self.store.record_memory_activation(actor, activation.telemetry)
        except Exception:
            # A concurrent identical trigger may have committed first. Return
            # only its durable, content-free envelope; never expose context
            # that this call failed to fence at the persistence boundary.
            winner = await self.store.get_memory_activation(actor, request.activation_id)
            if winner is not None:
                return MemoryActivationDelivery(activation=winner, replayed=True)
            raise
        return MemoryActivationDelivery(
            activation=activation.telemetry,
            activation_context=activation.rendered_context or None,
        )

    async def read_expand_memory(
        self,
        actor: ActorContext,
        request: ReadExpandMemoryRequest,
    ) -> ReadExpandMemoryResult:
        """Run the bounded read/expand step for the caller's current lease."""

        require_capability(actor, Capability.MEMORY_RECALL)
        if self.memory_service is None:
            raise RuntimeError("memory retrieval is not configured")
        await self.store.validate_memory_activation_scope(
            actor,
            request.task_id,
            request.lease_id,
        )
        return await self.memory_service.read_expand(actor, request)

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
