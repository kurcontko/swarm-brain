"""Atomic in-memory implementation of the Swarm Brain v0 ports.

This adapter is intentionally feature-complete enough to exercise the same
domain invariants as the durable adapter.  It is not a persistence substitute;
process restart loses all state.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections import defaultdict, deque
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, TypeVar
from uuid import uuid4

from pydantic import BaseModel

from swarmbrain.application.errors import (
    IdempotencyConflict,
    InvalidState,
    LeaseLost,
    NoTaskAvailable,
    ResourceNotFound,
)
from swarmbrain.application.memory_policy import memory_content_text
from swarmbrain.domain.activation import MemoryActivationTelemetry
from swarmbrain.domain.agents import ActorContext, Agent
from swarmbrain.domain.common import MemoryId, RunId, SourceId, utc_now
from swarmbrain.domain.conflicts import (
    Conflict,
    ConflictResolution,
    ConflictResolutionKind,
    ConflictStatus,
    ReportConflictCommand,
    ReportConflictResult,
    ResolveConflictCommand,
    ResolveConflictResult,
)
from swarmbrain.domain.events import (
    AggregateType,
    EventPage,
    EventType,
    OutboxEvent,
    RunMetrics,
    SwarmEvent,
    enriched_payload,
)
from swarmbrain.domain.evidence import (
    AddEvidenceCommand,
    Evidence,
    EvidenceRef,
    EvidenceSource,
    RegisterEvidenceSourceCommand,
    RejectSourceCommand,
    ReviewSourceCommand,
    SourceRejectionResult,
    SourceReviewState,
    SourceTrust,
)
from swarmbrain.domain.leases import (
    LeaseStatus,
    RenewLeaseCommand,
    RenewLeaseResult,
    TaskLease,
)
from swarmbrain.domain.memory import (
    EmbeddingMatch,
    EmbeddingVector,
    Memory,
    MemoryLineage,
    MemoryLink,
    MemoryLinkKind,
    MemoryOperation,
    MemoryReviewDecision,
    MemoryState,
    RecallBundle,
    RecallHit,
    RecallQuery,
    RememberCommand,
    RememberResult,
    ReviewMemoryCommand,
    ReviewMemoryResult,
    Visibility,
)
from swarmbrain.domain.outcome_feedback import (
    MAX_OUTCOME_ASSOCIATIONS_PER_COMPLETION,
    MemoryOutcomeAssociation,
    memory_outcome_association_id,
)
from swarmbrain.domain.tasks import (
    CheckpointCommand,
    CheckpointResult,
    ClaimTaskCommand,
    ClaimTaskResult,
    CompleteTaskCommand,
    CompletionResult,
    DependencyKind,
    ReleaseResult,
    ReleaseTaskCommand,
    Task,
    TaskCheckpoint,
    TaskCompletion,
    TaskDependency,
    TaskOutcome,
    TaskStatus,
)
from swarmbrain.ports.memory_store import MemoryOperationPolicy

TModel = TypeVar("TModel", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class _IdempotencyEntry:
    fingerprint: str
    result: BaseModel


def _uuid() -> str:
    return str(uuid4())


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"[\w-]+", value.casefold()))


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


class InMemoryKernel:
    """One adapter implementing coordination, memory, evidence, and conflicts.

    A single ``asyncio.Lock`` supplies transaction-like atomicity for local
    concurrency tests. Every lookup applies the authenticated scope before it
    returns or mutates a resource.
    """

    def __init__(self, *, clock: Callable[[], datetime] = utc_now) -> None:
        self._clock = clock
        self._lock = asyncio.Lock()
        self._retrieval_snapshot_now: ContextVar[datetime | None] = ContextVar(
            f"swarmbrain_retrieval_snapshot_{id(self)}",
            default=None,
        )

        self.agents: dict[tuple[str, str, str], Agent] = {}
        self.tasks: dict[str, Task] = {}
        self.dependencies: dict[str, list[TaskDependency]] = defaultdict(list)
        self.leases: dict[str, TaskLease] = {}
        self.checkpoints: dict[str, list[TaskCheckpoint]] = defaultdict(list)
        self.completions: dict[str, TaskCompletion] = {}
        self.memory_outcome_associations: dict[str, MemoryOutcomeAssociation] = {}

        self.memories: dict[str, Memory] = {}
        self.memory_embeddings: dict[tuple[str, str], EmbeddingVector] = {}
        self.memory_links: dict[str, MemoryLink] = {}
        self.sources: dict[str, EvidenceSource] = {}
        self.evidence: dict[str, Evidence] = {}
        self.conflicts: dict[str, Conflict] = {}

        self.events: list[SwarmEvent] = []
        self.outbox: list[OutboxEvent] = []
        self._source_scope: dict[str, tuple[str, str, str, str, str]] = {}
        self._source_occurrences: dict[tuple[str, str, str], str] = {}
        self._idempotency: dict[tuple[str, str, str, str, str], _IdempotencyEntry] = {}
        self._previous_memory_state: dict[str, MemoryState] = {}
        self._crash_handoffs: dict[str, int] = defaultdict(int)
        self._memory_reuses: dict[str, int] = defaultdict(int)
        self._duplicate_claims: dict[str, int] = defaultdict(int)
        self._duplicate_replays: dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # Coordination

    async def join_agent(self, actor: ActorContext) -> Agent:
        async with self._lock:
            now = self._now()
            key = (actor.tenant_id, actor.run_id, actor.agent_id)
            current = self.agents.get(key)
            if current is None:
                agent = Agent(
                    tenant_id=actor.tenant_id,
                    project_id=actor.project_id,
                    repository_id=actor.repository_id,
                    swarm_id=actor.swarm_id,
                    run_id=actor.run_id,
                    agent_id=actor.agent_id,
                    harness=actor.harness,
                    provider=actor.provider,
                    model=actor.model,
                    capabilities=actor.capabilities,
                    joined_at=now,
                    last_seen_at=now,
                    metadata=actor.metadata,
                )
                self.agents[key] = agent
                self._emit(
                    actor,
                    EventType.AGENT_JOINED,
                    AggregateType.AGENT,
                    agent.agent_id,
                    agent.version,
                )
                return agent

            agent = current.model_copy(
                update={
                    "harness": actor.harness,
                    "provider": actor.provider,
                    "model": actor.model,
                    "capabilities": actor.capabilities,
                    "last_seen_at": now,
                    "version": current.version + 1,
                    "metadata": actor.metadata,
                }
            )
            self.agents[key] = agent
            return agent

    async def add_task(self, task: Task) -> Task:
        async with self._lock:
            if task.task_id in self.tasks:
                raise InvalidState("task already exists", task_id=task.task_id)
            self.tasks[task.task_id] = task
            return task

    async def add_dependency(self, dependency: TaskDependency) -> TaskDependency:
        """Administrative local bootstrap helper, deliberately outside MCP."""

        async with self._lock:
            if dependency.task_id not in self.tasks:
                raise ResourceNotFound("task", dependency.task_id)
            if dependency.depends_on_task_id not in self.tasks:
                raise ResourceNotFound("task", dependency.depends_on_task_id)
            task = self.tasks[dependency.task_id]
            prerequisite = self.tasks[dependency.depends_on_task_id]
            if (
                task.tenant_id,
                task.project_id,
                task.repository_id,
                task.swarm_id,
                task.run_id,
            ) != (
                prerequisite.tenant_id,
                prerequisite.project_id,
                prerequisite.repository_id,
                prerequisite.swarm_id,
                prerequisite.run_id,
            ):
                raise InvalidState("task dependencies must stay in one authenticated scope")
            self.dependencies[dependency.task_id].append(dependency)
            return dependency

    async def claim_task(
        self,
        actor: ActorContext,
        command: ClaimTaskCommand,
    ) -> ClaimTaskResult:
        async with self._lock:
            replay = self._replay(actor, "task.claim", command)
            if replay is not None:
                return replay

            now = self._now()
            self._expire_leases(actor, now)
            candidates = [
                task
                for task in self.tasks.values()
                if self._task_in_scope(actor, task)
                and (command.task_id is None or task.task_id == command.task_id)
                and self._task_is_claimable(task, now)
                and set(command.required_tags).issubset(task.tags)
                and command.required_capabilities.issubset(task.required_capabilities)
                and task.required_capabilities.issubset(actor.capabilities)
            ]
            candidates.sort(
                key=lambda task: (
                    -task.priority,
                    task.available_at or task.created_at,
                    task.created_at,
                    task.task_id,
                )
            )
            if not candidates:
                if command.task_id is not None and command.task_id not in self.tasks:
                    raise ResourceNotFound("task", command.task_id)
                self._duplicate_claims[actor.run_id] += 1
                raise NoTaskAvailable()

            task = candidates[0]
            if (
                command.expected_task_version is not None
                and task.version != command.expected_task_version
            ):
                raise InvalidState(
                    "task version changed before claim",
                    task_id=task.task_id,
                    expected_version=command.expected_task_version,
                    actual_version=task.version,
                )

            lease_id = _uuid()
            lease = TaskLease(
                lease_id=lease_id,
                task_id=task.task_id,
                run_id=task.run_id,
                owner_agent_id=actor.agent_id,
                acquired_at=now,
                expires_at=now + timedelta(seconds=command.lease_seconds),
            )
            claimed = task.model_copy(
                update={
                    "status": TaskStatus.CLAIMED,
                    "claimed_by_agent_id": actor.agent_id,
                    "active_lease_id": lease_id,
                    "updated_at": now,
                    "version": task.version + 1,
                }
            )
            self.tasks[task.task_id] = claimed
            self.leases[lease_id] = lease
            latest = self.checkpoints[task.task_id][-1] if self.checkpoints[task.task_id] else None
            unblocked_by = tuple(
                dependency.depends_on_task_id
                for dependency in self.dependencies[task.task_id]
                if dependency.kind is DependencyKind.BLOCKS
                and self.tasks[dependency.depends_on_task_id].status is TaskStatus.COMPLETED
            )
            result = ClaimTaskResult(
                task=claimed,
                lease=lease,
                checkpoint=latest,
                unblocked_by_task_ids=unblocked_by,
            )
            self._emit(
                actor,
                EventType.TASK_CLAIMED,
                AggregateType.TASK,
                claimed.task_id,
                claimed.version,
                task_id=claimed.task_id,
                task_title=claimed.title,
                payload={"lease_id": lease.lease_id},
                idempotency_key=command.idempotency_key,
            )
            return self._remember_result(actor, "task.claim", command, result)

    async def renew_lease(
        self,
        actor: ActorContext,
        command: RenewLeaseCommand,
    ) -> RenewLeaseResult:
        async with self._lock:
            replay = self._replay(actor, "lease.renew", command)
            if replay is not None:
                return replay
            lease = self._owned_lease(actor, command.lease_id, command.expected_version)
            now = self._now()
            renewed = lease.model_copy(
                update={
                    "renewed_at": now,
                    "expires_at": max(
                        lease.expires_at,
                        now + timedelta(seconds=command.extension_seconds),
                    ),
                    "version": lease.version + 1,
                }
            )
            self.leases[lease.lease_id] = renewed
            result = RenewLeaseResult(lease=renewed)
            self._emit(
                actor,
                EventType.LEASE_RENEWED,
                AggregateType.LEASE,
                renewed.lease_id,
                renewed.version,
                task_id=renewed.task_id,
                idempotency_key=command.idempotency_key,
            )
            return self._remember_result(actor, "lease.renew", command, result)

    async def checkpoint_task(
        self,
        actor: ActorContext,
        command: CheckpointCommand,
    ) -> CheckpointResult:
        async with self._lock:
            replay = self._replay(actor, "task.checkpoint", command)
            if replay is not None:
                return replay
            self._validate_citation_memory_ids(actor, command.task_id, command.memory_ids)
            task, lease = self._owned_task_and_lease(
                actor,
                command.task_id,
                command.lease_id,
                command.expected_task_version,
                command.expected_lease_version,
            )
            now = self._now()
            checkpoint = TaskCheckpoint(
                checkpoint_id=_uuid(),
                task_id=task.task_id,
                lease_id=lease.lease_id,
                run_id=task.run_id,
                agent_id=actor.agent_id,
                sequence=len(self.checkpoints[task.task_id]) + 1,
                summary=command.summary,
                discoveries=command.discoveries,
                completed_work=command.completed_work,
                remaining_work=command.remaining_work,
                memory_ids=command.memory_ids,
                artifact_ids=command.artifact_ids,
                state=command.state,
                created_at=now,
            )
            task = task.model_copy(update={"updated_at": now, "version": task.version + 1})
            lease = lease.model_copy(update={"version": lease.version + 1})
            self.tasks[task.task_id] = task
            self.leases[lease.lease_id] = lease
            self.checkpoints[task.task_id].append(checkpoint)
            result = CheckpointResult(task=task, lease=lease, checkpoint=checkpoint)
            self._emit(
                actor,
                EventType.TASK_CHECKPOINTED,
                AggregateType.TASK,
                task.task_id,
                task.version,
                task_id=task.task_id,
                task_title=task.title,
                payload={
                    "checkpoint_id": checkpoint.checkpoint_id,
                    **(
                        {"memory_ids": list(checkpoint.memory_ids)} if checkpoint.memory_ids else {}
                    ),
                },
                idempotency_key=command.idempotency_key,
            )
            return self._remember_result(actor, "task.checkpoint", command, result)

    async def complete_task(
        self,
        actor: ActorContext,
        command: CompleteTaskCommand,
    ) -> CompletionResult:
        async with self._lock:
            replay = self._replay(actor, "task.complete", command)
            if replay is not None:
                return replay
            self._validate_citation_memory_ids(actor, command.task_id, command.memory_ids)
            task, lease = self._owned_task_and_lease(
                actor,
                command.task_id,
                command.lease_id,
                command.expected_task_version,
                command.expected_lease_version,
            )
            now = self._now()
            completion = TaskCompletion(
                completion_id=_uuid(),
                task_id=task.task_id,
                lease_id=lease.lease_id,
                run_id=task.run_id,
                agent_id=actor.agent_id,
                outcome=command.outcome,
                summary=command.summary,
                memory_ids=command.memory_ids,
                artifact_ids=command.artifact_ids,
                completed_at=now,
            )
            associations = self._derive_memory_outcome_associations(
                actor,
                completion,
                now=now,
            )
            status = (
                TaskStatus.COMPLETED
                if command.outcome is TaskOutcome.SUCCEEDED
                else TaskStatus.FAILED
            )
            task = task.model_copy(
                update={
                    "status": status,
                    "claimed_by_agent_id": None,
                    "active_lease_id": None,
                    "updated_at": now,
                    "completed_at": now,
                    "version": task.version + 1,
                }
            )
            lease = lease.model_copy(
                update={
                    "status": LeaseStatus.COMPLETED,
                    "released_at": now,
                    "version": lease.version + 1,
                }
            )
            self.tasks[task.task_id] = task
            self.leases[lease.lease_id] = lease
            self.completions[task.task_id] = completion
            for association in associations:
                self.memory_outcome_associations.setdefault(
                    association.association_id,
                    association,
                )
            result = CompletionResult(task=task, lease=lease, completion=completion)
            self._emit(
                actor,
                EventType.TASK_COMPLETED,
                AggregateType.TASK,
                task.task_id,
                task.version,
                task_id=task.task_id,
                task_title=task.title,
                payload={
                    "outcome": command.outcome.value,
                    **(
                        {"memory_ids": list(completion.memory_ids)} if completion.memory_ids else {}
                    ),
                },
                idempotency_key=command.idempotency_key,
            )
            return self._remember_result(actor, "task.complete", command, result)

    async def release_task(
        self,
        actor: ActorContext,
        command: ReleaseTaskCommand,
    ) -> ReleaseResult:
        async with self._lock:
            replay = self._replay(actor, "task.release", command)
            if replay is not None:
                return replay
            task, lease = self._owned_task_and_lease(
                actor,
                command.task_id,
                command.lease_id,
                command.expected_task_version,
                command.expected_lease_version,
            )
            now = self._now()
            task = task.model_copy(
                update={
                    "status": TaskStatus.READY,
                    "claimed_by_agent_id": None,
                    "active_lease_id": None,
                    "updated_at": now,
                    "version": task.version + 1,
                }
            )
            lease = lease.model_copy(
                update={
                    "status": LeaseStatus.RELEASED,
                    "released_at": now,
                    "version": lease.version + 1,
                }
            )
            self.tasks[task.task_id] = task
            self.leases[lease.lease_id] = lease
            result = ReleaseResult(task=task, lease=lease)
            self._emit(
                actor,
                EventType.TASK_RELEASED,
                AggregateType.TASK,
                task.task_id,
                task.version,
                task_id=task.task_id,
                task_title=task.title,
                payload={"reason": command.reason},
                idempotency_key=command.idempotency_key,
            )
            return self._remember_result(actor, "task.release", command, result)

    async def list_run_events(
        self,
        actor: ActorContext,
        run_id: RunId,
        *,
        cursor: str | None = None,
        limit: int = 100,
    ) -> EventPage:
        async with self._lock:
            if run_id != actor.run_id:
                raise ResourceNotFound("run", run_id)
            start = int(cursor) if cursor else 0
            scoped = [
                event
                for event in self.events
                if event.tenant_id == actor.tenant_id and event.run_id == run_id
            ]
            page = scoped[start : start + limit]
            next_cursor = str(start + len(page)) if start + len(page) < len(scoped) else None
            return EventPage(events=tuple(page), next_cursor=next_cursor)

    async def get_run_metrics(self, actor: ActorContext, run_id: RunId) -> RunMetrics:
        async with self._lock:
            if run_id != actor.run_id:
                raise ResourceNotFound("run", run_id)
            now = self._now()
            tasks = [task for task in self.tasks.values() if self._task_in_scope(actor, task)]
            leases = [
                lease
                for lease in self.leases.values()
                if (task := self.tasks.get(lease.task_id)) is not None
                and self._task_in_scope(actor, task)
            ]
            conflicts = [
                item
                for item in self.conflicts.values()
                if item.tenant_id == actor.tenant_id and item.run_id == run_id
            ]
            citation_records = {
                (
                    checkpoint.task_id,
                    checkpoint.lease_id,
                    checkpoint.agent_id,
                    memory_id,
                )
                for task in tasks
                for checkpoint in self.checkpoints[task.task_id]
                for memory_id in checkpoint.memory_ids
            }
            citation_records.update(
                (
                    completion.task_id,
                    completion.lease_id,
                    completion.agent_id,
                    memory_id,
                )
                for task in tasks
                if (completion := self.completions.get(task.task_id)) is not None
                for memory_id in completion.memory_ids
            )
            activation_events = [
                event
                for event in self.events
                if event.tenant_id == actor.tenant_id
                and event.project_id == actor.project_id
                and event.repository_id == actor.repository_id
                and event.swarm_id == actor.swarm_id
                and event.run_id == run_id
                and event.event_type is EventType.MEMORY_ACTIVATED
            ]
            activation_keys = {
                (
                    event.task_id,
                    str(event.payload.get("lease_id") or ""),
                    event.agent_id,
                    str(memory_id),
                )
                for event in activation_events
                for memory_id in event.payload.get("memory_ids", ())
            }
            cross_agent_uses = sum(
                (memory := self.memories.get(memory_id)) is not None
                and memory.author_agent_id != consumer_agent_id
                and (task_id, lease_id, consumer_agent_id, memory_id) in activation_keys
                for task_id, lease_id, consumer_agent_id, memory_id in citation_records
            )
            activated_count = sum(
                len(tuple(dict.fromkeys(event.payload.get("memory_ids", ()))))
                for event in activation_events
            )
            return RunMetrics(
                run_id=run_id,
                measured_at=now,
                tasks_total=len(tasks),
                tasks_claimed=sum(task.status is TaskStatus.CLAIMED for task in tasks),
                tasks_completed=sum(task.status is TaskStatus.COMPLETED for task in tasks),
                tasks_failed=sum(task.status is TaskStatus.FAILED for task in tasks),
                active_leases=sum(
                    lease.status is LeaseStatus.ACTIVE and lease.expires_at > now
                    for lease in leases
                ),
                checkpoints=sum(len(self.checkpoints[task.task_id]) for task in tasks),
                crash_handoffs=self._crash_handoffs[run_id],
                memories_published=sum(
                    memory.run_id == run_id and memory.recorded_to is None
                    for memory in self.memories.values()
                ),
                memories_activated=activated_count,
                memory_activation_attempts=len(activation_events),
                memories_cited=len(citation_records),
                cross_agent_memory_uses=cross_agent_uses,
                memories_reused=self._memory_reuses[run_id],
                conflicts_open=sum(item.status is ConflictStatus.OPEN for item in conflicts),
                conflicts_resolved=sum(
                    item.status is ConflictStatus.RESOLVED for item in conflicts
                ),
                duplicate_claims_prevented=self._duplicate_claims[run_id],
                duplicate_mutations_replayed=self._duplicate_replays[run_id],
            )

    async def record_memory_activation(
        self,
        actor: ActorContext,
        telemetry: MemoryActivationTelemetry,
    ) -> None:
        """Persist one content-free activation decision per lease trigger."""

        async with self._lock:
            if telemetry.run_id != actor.run_id or telemetry.agent_id != actor.agent_id:
                raise InvalidState("memory activation actor does not match telemetry")
            existing = next(
                (event for event in self.events if event.event_id == telemetry.activation_id),
                None,
            )
            if existing is not None:
                expected = telemetry.model_dump(mode="json")
                if (
                    existing.tenant_id != actor.tenant_id
                    or existing.project_id != actor.project_id
                    or existing.repository_id != actor.repository_id
                    or existing.swarm_id != actor.swarm_id
                    or existing.run_id != actor.run_id
                    or existing.agent_id != actor.agent_id
                    or existing.task_id != telemetry.task_id
                    or existing.event_type is not EventType.MEMORY_ACTIVATED
                    or any(existing.payload.get(key) != value for key, value in expected.items())
                ):
                    raise InvalidState(
                        "memory activation identity is already used by another event",
                        activation_id=telemetry.activation_id,
                    )
                return
            task = self.tasks.get(telemetry.task_id)
            if task is None or not self._task_in_scope(actor, task):
                raise ResourceNotFound("task", telemetry.task_id)
            lease = self.leases.get(telemetry.lease_id)
            now = self._now()
            if (
                lease is None
                or lease.task_id != task.task_id
                or lease.run_id != actor.run_id
                or lease.owner_agent_id != actor.agent_id
                or lease.status is not LeaseStatus.ACTIVE
                or lease.expires_at <= now
                or task.status is not TaskStatus.CLAIMED
                or task.active_lease_id != lease.lease_id
                or task.claimed_by_agent_id != actor.agent_id
            ):
                raise ResourceNotFound("active activation lease", telemetry.lease_id)
            if telemetry.memory_ids:
                query = RecallQuery(
                    text="activation eligibility check",
                    task_id=telemetry.task_id,
                    memory_ids=frozenset(telemetry.memory_ids),
                    states=frozenset({MemoryState.CONFIRMED}),
                    visibilities=frozenset(Visibility),
                    referenced_valid_from=telemetry.referenced_valid_from,
                    referenced_valid_to=telemetry.referenced_valid_to,
                    limit=len(telemetry.memory_ids),
                )
                recallable_versions = {
                    memory.memory_id: memory.version
                    for memory in self._recallable_memories_locked(actor, query, now)
                }
                stale_ids = sorted(
                    memory_id
                    for memory_id, expected_version in telemetry.memory_versions.items()
                    if recallable_versions.get(memory_id) != expected_version
                )
                if stale_ids:
                    raise InvalidState(
                        "activated memory selection is no longer current and recallable",
                        memory_ids=stale_ids,
                    )
            self._emit(
                actor,
                EventType.MEMORY_ACTIVATED,
                AggregateType.RUN,
                actor.run_id,
                1,
                task_id=task.task_id,
                payload=telemetry.model_dump(mode="json"),
                event_id=telemetry.activation_id,
            )

    async def validate_memory_activation_scope(
        self,
        actor: ActorContext,
        task_id: str,
        lease_id: str,
    ) -> None:
        """Fail closed before an ephemeral task-scoped query is evaluated."""

        async with self._lock:
            task = self.tasks.get(task_id)
            lease = self.leases.get(lease_id)
            now = self._now()
            if task is None or not self._task_in_scope(actor, task):
                raise ResourceNotFound("task", task_id)
            if (
                lease is None
                or lease.task_id != task.task_id
                or lease.run_id != actor.run_id
                or lease.owner_agent_id != actor.agent_id
                or lease.status is not LeaseStatus.ACTIVE
                or lease.expires_at <= now
                or task.status is not TaskStatus.CLAIMED
                or task.active_lease_id != lease.lease_id
                or task.claimed_by_agent_id != actor.agent_id
            ):
                raise ResourceNotFound("active activation lease", lease_id)

    async def get_memory_activation(
        self,
        actor: ActorContext,
        activation_id: str,
    ) -> MemoryActivationTelemetry | None:
        """Return canonical telemetry for a scoped deterministic activation ID."""

        async with self._lock:
            event = next((item for item in self.events if item.event_id == activation_id), None)
            if event is None:
                return None
            if (
                event.tenant_id != actor.tenant_id
                or event.project_id != actor.project_id
                or event.repository_id != actor.repository_id
                or event.swarm_id != actor.swarm_id
                or event.run_id != actor.run_id
                or event.agent_id != actor.agent_id
                or event.event_type is not EventType.MEMORY_ACTIVATED
            ):
                raise ResourceNotFound("memory activation", activation_id)
            try:
                return MemoryActivationTelemetry.model_validate(
                    {
                        key: event.payload[key]
                        for key in MemoryActivationTelemetry.model_fields
                        if key in event.payload
                    }
                )
            except ValueError as exc:
                raise InvalidState(
                    "stored memory activation telemetry is invalid",
                    activation_id=activation_id,
                ) from exc

    async def list_memory_outcome_associations(
        self,
        actor: ActorContext,
        *,
        task_id: str | None = None,
        memory_id: str | None = None,
        limit: int = 1000,
    ) -> tuple[MemoryOutcomeAssociation, ...]:
        """Return bounded silver observations from the actor's exact run scope."""

        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("outcome association limit must be between 1 and 1000")
        async with self._lock:
            scoped = sorted(
                (
                    association
                    for association in self.memory_outcome_associations.values()
                    if association.tenant_id == actor.tenant_id
                    and association.project_id == actor.project_id
                    and association.repository_id == actor.repository_id
                    and association.swarm_id == actor.swarm_id
                    and association.run_id == actor.run_id
                    and (task_id is None or association.task_id == task_id)
                    and (memory_id is None or association.memory_id == memory_id)
                ),
                key=lambda association: (
                    association.observed_at,
                    association.association_id,
                ),
            )
            return tuple(scoped[:limit])

    # ------------------------------------------------------------------
    # Memory

    async def remember(
        self,
        actor: ActorContext,
        command: RememberCommand,
        policy: MemoryOperationPolicy,
    ) -> RememberResult:
        async with self._lock:
            replay = self._replay(actor, "memory.remember", command)
            if replay is not None:
                return replay
            if command.task_id is not None:
                self._task_target(actor, command.task_id)
            self._validate_evidence_refs(actor, command.evidence)
            self._validate_derived_evidence(actor, command)
            current = [
                memory
                for memory in self.memories.values()
                if self._memory_accessible(actor, memory, command.task_id)
                and memory.recorded_to is None
                and memory.state is not MemoryState.REFUTED
            ]
            decision = await policy.decide(command, current)
            if decision.operation is MemoryOperation.NOOP:
                result = RememberResult(operation=decision.operation, decision=decision)
                return self._remember_result(actor, "memory.remember", command, result)

            now = self._now()
            affected: list[Memory] = []
            if decision.operation is MemoryOperation.MERGE:
                target = self._memory_target(actor, decision.target_memory_ids[0])
                combined_evidence = {item.evidence_id: item for item in target.evidence}
                combined_evidence.update({item.evidence_id: item for item in command.evidence})
                merged = target.model_copy(
                    update={
                        "evidence": tuple(combined_evidence.values()),
                        "tags": tuple(dict.fromkeys((*target.tags, *command.tags))),
                        "confidence": max(target.confidence, command.confidence),
                        "occurred_at": target.occurred_at or command.occurred_at,
                        "version": target.version + 1,
                    }
                )
                self.memories[target.memory_id] = merged
                memory = merged
                affected.append(merged)
            elif decision.operation in {MemoryOperation.ADD, MemoryOperation.UPDATE}:
                supersedes_id: str | None = None
                if decision.operation is MemoryOperation.UPDATE:
                    target = self._memory_target(actor, decision.target_memory_ids[0])
                    supersedes_id = target.memory_id
                    closed_at = max(now, target.recorded_from + timedelta(microseconds=1))
                    self._previous_memory_state[target.memory_id] = target.state
                    target = target.model_copy(
                        update={
                            "state": MemoryState.SUPERSEDED,
                            "recorded_to": closed_at,
                            "superseded_by_memory_id": "00000000-0000-0000-0000-000000000000",
                            "version": target.version + 1,
                        }
                    )
                    now = closed_at
                    affected.append(target)

                memory = Memory(
                    memory_id=_uuid(),
                    tenant_id=actor.tenant_id,
                    project_id=actor.project_id,
                    repository_id=actor.repository_id,
                    swarm_id=actor.swarm_id,
                    run_id=actor.run_id,
                    task_id=command.task_id,
                    author_agent_id=actor.agent_id,
                    kind=command.kind,
                    state=command.desired_state,
                    visibility=command.visibility,
                    content=command.content,
                    title=command.title,
                    tags=command.tags,
                    confidence=command.confidence,
                    evidence=command.evidence,
                    occurred_at=command.occurred_at,
                    valid_from=command.valid_from or now,
                    valid_to=command.valid_to,
                    recorded_from=now,
                    supersedes_memory_id=supersedes_id,
                    metadata={
                        **command.metadata,
                        "policy": {
                            "operation": decision.operation.value,
                            "reason": decision.reason,
                            "confidence": decision.confidence,
                        },
                    },
                )
                self.memories[memory.memory_id] = memory
                if supersedes_id is not None:
                    predecessor = affected[0].model_copy(
                        update={"superseded_by_memory_id": memory.memory_id}
                    )
                    self.memories[predecessor.memory_id] = predecessor
                    affected[0] = predecessor
                    self._add_memory_link(
                        memory.memory_id,
                        predecessor.memory_id,
                        MemoryLinkKind.SUPERSEDES,
                        reason=decision.reason,
                    )
                affected.append(memory)
            else:
                raise InvalidState(
                    "the v0 conservative policy does not hard-delete memory",
                    operation=decision.operation.value,
                )

            for related_id in command.related_memory_ids:
                self._memory_target(actor, related_id)
                if related_id != memory.memory_id:
                    self._add_memory_link(
                        memory.memory_id,
                        related_id,
                        MemoryLinkKind.RELATED_TO,
                    )

            for source_id in command.derived_from_memory_ids:
                self._memory_target(actor, source_id)
                if source_id != memory.memory_id:
                    self._add_memory_link(
                        memory.memory_id,
                        source_id,
                        MemoryLinkKind.DERIVED_FROM,
                        reason="evidence-bound derived memory",
                    )

            event_type = (
                EventType.MEMORY_SUPERSEDED
                if decision.operation is MemoryOperation.UPDATE
                else EventType.MEMORY_ADDED
            )
            self._emit(
                actor,
                event_type,
                AggregateType.MEMORY,
                memory.memory_id,
                memory.version,
                task_id=memory.task_id,
                payload={"operation": decision.operation.value},
                idempotency_key=command.idempotency_key,
            )
            result = RememberResult(
                operation=decision.operation,
                decision=decision,
                memory=memory,
                affected_memories=tuple(affected),
            )
            return self._remember_result(actor, "memory.remember", command, result)

    async def recall(self, actor: ActorContext, query: RecallQuery) -> RecallBundle:
        async with self._lock:
            now = self._now()
            candidates = self._recallable_memories_locked(actor, query, now)

            query_tokens = _tokens(query.text)
            scored: list[RecallHit] = []
            for memory in candidates:
                haystack = " ".join(
                    (memory.title or "", memory_content_text(memory.content), *memory.tags)
                )
                memory_tokens = _tokens(haystack)
                overlap = len(query_tokens & memory_tokens) / max(1, len(query_tokens))
                substring = query.text.casefold() in haystack.casefold()
                score = min(1.0, overlap + (0.2 if substring else 0.0))
                if score <= 0.0 or score < query.min_score:
                    continue
                reasons = ("lexical_overlap",)
                scored.append(
                    RecallHit(
                        memory=memory,
                        score=score,
                        reasons=reasons,
                        evidence=memory.evidence if query.include_evidence else (),
                    )
                )
            scored.sort(
                key=lambda hit: (hit.score, hit.memory.recorded_from, hit.memory.memory_id),
                reverse=True,
            )
            hits = tuple(scored[: query.limit])
            if hits:
                self._memory_reuses[actor.run_id] += len(hits)
            return RecallBundle(
                query=query,
                hits=hits,
                generated_at=now,
                total_candidates=len(candidates),
                truncated=len(scored) > query.limit,
            )

    async def recallable_memories(
        self,
        actor: ActorContext,
        query: RecallQuery,
    ) -> tuple[Memory, ...]:
        """Adapter seam for candidate lanes; results contain canonical filters."""

        snapshot_now = self._retrieval_snapshot_now.get()
        if snapshot_now is not None:
            return tuple(self._recallable_memories_locked(actor, query, snapshot_now))
        async with self._lock:
            return tuple(self._recallable_memories_locked(actor, query, self._now()))

    async def recallable_memory_graph(
        self,
        actor: ActorContext,
        query: RecallQuery,
    ) -> tuple[tuple[Memory, ...], tuple[MemoryLink, ...]]:
        """Return one canonically filtered node/edge view for graph retrieval."""

        snapshot_now = self._retrieval_snapshot_now.get()
        if snapshot_now is not None:
            return self._recallable_memory_graph_locked(actor, query, snapshot_now)
        async with self._lock:
            return self._recallable_memory_graph_locked(actor, query, self._now())

    async def hydrate_recallable(
        self,
        actor: ActorContext,
        query: RecallQuery,
        candidate_ids: Sequence[str],
    ) -> tuple[Memory, ...]:
        """Hydrate internal IDs without exposing an existence or scope oracle."""

        ordered_ids = tuple(dict.fromkeys(candidate_ids))
        snapshot_now = self._retrieval_snapshot_now.get()
        if snapshot_now is not None:
            recallable = {
                memory.memory_id: memory
                for memory in self._recallable_memories_locked(actor, query, snapshot_now)
            }
            return tuple(
                recallable[memory_id] for memory_id in ordered_ids if memory_id in recallable
            )
        async with self._lock:
            recallable = {
                memory.memory_id: memory
                for memory in self._recallable_memories_locked(actor, query, self._now())
            }
            return tuple(
                recallable[memory_id] for memory_id in ordered_ids if memory_id in recallable
            )

    @asynccontextmanager
    async def retrieval_snapshot(self) -> AsyncIterator[None]:
        """Fence candidate generation and hydration to one local atomic snapshot."""

        if self._retrieval_snapshot_now.get() is not None:
            yield
            return
        async with self._lock:
            token = self._retrieval_snapshot_now.set(self._now())
            try:
                yield
            finally:
                self._retrieval_snapshot_now.reset(token)

    def retrieval_now(self) -> datetime:
        """Return the snapshot clock used by deterministic local retrieval."""

        return self._retrieval_snapshot_now.get() or self._now()

    async def record_retrieval_reuse(
        self,
        actor: ActorContext,
        memory_ids: Sequence[MemoryId],
    ) -> None:
        """Account final public hits once when RetrievalService owns recall."""

        count = len(tuple(dict.fromkeys(memory_ids)))
        async with self._lock:
            self._memory_reuses[actor.run_id] += count

    async def get_lineage(self, actor: ActorContext, memory_id: MemoryId) -> MemoryLineage:
        async with self._lock:
            requested = self._memory_target(actor, memory_id, require_current=False)
            known = {requested.memory_id}
            queue: deque[str] = deque(known)
            links: list[MemoryLink] = []
            while queue:
                current = queue.popleft()
                for link in self.memory_links.values():
                    if current not in {link.source_memory_id, link.target_memory_id}:
                        continue
                    other = (
                        link.target_memory_id
                        if link.source_memory_id == current
                        else link.source_memory_id
                    )
                    candidate = self.memories.get(other)
                    if candidate is None or not self._memory_accessible(
                        actor, candidate, candidate.task_id
                    ):
                        continue
                    if link not in links:
                        links.append(link)
                    if other not in known:
                        known.add(other)
                        queue.append(other)
            memories = tuple(
                sorted(
                    (self.memories[item_id] for item_id in known),
                    key=lambda item: (item.recorded_from, item.memory_id),
                )
            )
            current_ids = tuple(
                item.memory_id
                for item in memories
                if item.recorded_to is None
                and item.state not in {MemoryState.SUPERSEDED, MemoryState.REFUTED}
            )
            return MemoryLineage(
                requested_memory_id=memory_id,
                memories=memories,
                links=tuple(links),
                current_memory_ids=current_ids,
            )

    async def review_memory(
        self,
        actor: ActorContext,
        command: ReviewMemoryCommand,
    ) -> ReviewMemoryResult:
        async with self._lock:
            replay = self._replay(actor, "memory.review", command)
            if replay is not None:
                return replay
            self._validate_evidence_refs(actor, command.evidence)
            memory = self._memory_target(actor, command.memory_id)
            if memory.version != command.expected_version:
                raise InvalidState(
                    "memory version changed before review",
                    memory_id=memory.memory_id,
                    expected_version=command.expected_version,
                    actual_version=memory.version,
                )
            state = (
                MemoryState.CONFIRMED
                if command.decision is MemoryReviewDecision.CONFIRM
                else MemoryState.REFUTED
            )
            refs = {item.evidence_id: item for item in memory.evidence}
            refs.update({item.evidence_id: item for item in command.evidence})
            memory = memory.model_copy(
                update={
                    "state": state,
                    "evidence": tuple(refs.values()),
                    "version": memory.version + 1,
                }
            )
            self.memories[memory.memory_id] = memory
            result = ReviewMemoryResult(memory=memory)
            self._emit(
                actor,
                EventType.MEMORY_CONFIRMED
                if state is MemoryState.CONFIRMED
                else EventType.MEMORY_REFUTED,
                AggregateType.MEMORY,
                memory.memory_id,
                memory.version,
                task_id=memory.task_id,
                payload={"reason": command.reason},
                idempotency_key=command.idempotency_key,
            )
            return self._remember_result(actor, "memory.review", command, result)

    # ------------------------------------------------------------------
    # Embedding index

    async def upsert_embeddings(
        self,
        actor: ActorContext,
        vectors: Sequence[EmbeddingVector],
        *,
        idempotency_key: str,
    ) -> None:
        del idempotency_key  # keyed upsert is naturally idempotent in memory
        async with self._lock:
            for vector in vectors:
                memory = self.memories.get(vector.memory_id)
                if memory is None or (
                    memory.tenant_id,
                    memory.project_id,
                    memory.repository_id,
                ) != (
                    actor.tenant_id,
                    actor.project_id,
                    actor.repository_id,
                ):
                    raise ResourceNotFound("memory", vector.memory_id)
                self.memory_embeddings[(vector.memory_id, vector.model)] = vector

    async def search_embeddings(
        self,
        actor: ActorContext,
        query_vector: Sequence[float],
        *,
        model: str,
        limit: int = 10,
        min_score: float = 0.0,
        candidate_ids: Sequence[MemoryId] = (),
    ) -> tuple[EmbeddingMatch, ...]:
        if self._retrieval_snapshot_now.get() is not None:
            return self._search_embeddings_locked(
                actor,
                query_vector,
                model=model,
                limit=limit,
                min_score=min_score,
                candidate_ids=candidate_ids,
            )
        async with self._lock:
            return self._search_embeddings_locked(
                actor,
                query_vector,
                model=model,
                limit=limit,
                min_score=min_score,
                candidate_ids=candidate_ids,
            )

    def _search_embeddings_locked(
        self,
        actor: ActorContext,
        query_vector: Sequence[float],
        *,
        model: str,
        limit: int,
        min_score: float,
        candidate_ids: Sequence[MemoryId] = (),
    ) -> tuple[EmbeddingMatch, ...]:
        matches: list[EmbeddingMatch] = []
        allowed = frozenset(candidate_ids)
        for (memory_id, vector_model), vector in self.memory_embeddings.items():
            if vector_model != model or (allowed and memory_id not in allowed):
                continue
            memory = self.memories.get(memory_id)
            if memory is None or (
                memory.tenant_id,
                memory.project_id,
                memory.repository_id,
            ) != (
                actor.tenant_id,
                actor.project_id,
                actor.repository_id,
            ):
                continue
            if len(vector.values) != len(query_vector):
                continue
            score = max(0.0, min(1.0, _cosine_similarity(vector.values, query_vector)))
            if score < min_score:
                continue
            matches.append(EmbeddingMatch(memory_id=memory_id, score=score))
        matches.sort(key=lambda match: (match.score, match.memory_id), reverse=True)
        return tuple(matches[: max(1, min(100, limit))])

    # ------------------------------------------------------------------
    # Sources and evidence

    async def register_source(
        self,
        actor: ActorContext,
        command: RegisterEvidenceSourceCommand,
    ) -> EvidenceSource:
        async with self._lock:
            replay = self._replay(actor, "source.register", command)
            if replay is not None:
                return replay
            if command.task_id is not None:
                self._task_target(actor, command.task_id)
            occurrence = command.occurrence_key or command.idempotency_key
            occurrence_key = (actor.tenant_id, actor.repository_id, occurrence)
            existing_id = self._source_occurrences.get(occurrence_key)
            if existing_id is not None:
                if not self._source_accessible(actor, existing_id):
                    raise IdempotencyConflict(command.idempotency_key)
                existing = self.sources[existing_id]
                if existing.content_sha256 != command.content_sha256:
                    raise IdempotencyConflict(command.idempotency_key)
                return self._remember_result(actor, "source.register", command, existing)
            source = EvidenceSource(
                source_id=_uuid(),
                run_id=actor.run_id,
                task_id=command.task_id,
                kind=command.kind,
                uri=command.uri,
                content_sha256=command.content_sha256,
                occurrence_key=occurrence,
                trust=command.trust,
                observed_at=command.observed_at,
                recorded_at=self._now(),
                metadata=command.metadata,
            )
            self.sources[source.source_id] = source
            self._source_scope[source.source_id] = (
                actor.tenant_id,
                actor.project_id,
                actor.repository_id,
                actor.swarm_id,
                actor.run_id,
            )
            self._source_occurrences[occurrence_key] = source.source_id
            return self._remember_result(actor, "source.register", command, source)

    async def add_evidence(
        self,
        actor: ActorContext,
        command: AddEvidenceCommand,
    ) -> Evidence:
        async with self._lock:
            replay = self._replay(actor, "evidence.add", command)
            if replay is not None:
                return replay
            source = self._source_target(actor, command.source_id)
            evidence = Evidence(
                evidence_id=_uuid(),
                source_id=source.source_id,
                kind=command.kind,
                locator=command.locator,
                excerpt=command.excerpt,
                content_sha256=command.content_sha256,
                artifact=command.artifact,
                recorded_at=self._now(),
                metadata=command.metadata,
            )
            self.evidence[evidence.evidence_id] = evidence
            return self._remember_result(actor, "evidence.add", command, evidence)

    async def review_source(
        self,
        actor: ActorContext,
        command: ReviewSourceCommand,
    ) -> EvidenceSource:
        async with self._lock:
            replay = self._replay(actor, "source.review", command)
            if replay is not None:
                return replay
            source = self._source_target(actor, command.source_id)
            if source.version != command.expected_version:
                raise InvalidState(
                    "source version changed before review",
                    source_id=source.source_id,
                    expected_version=command.expected_version,
                    actual_version=source.version,
                )
            if source.review_state is SourceReviewState.REJECTED:
                raise InvalidState(
                    "source rejection is terminal",
                    source_id=source.source_id,
                )
            source = source.model_copy(
                update={
                    "review_state": command.review_state,
                    "reviewed_by_agent_id": actor.agent_id,
                    "reviewed_at": self._now(),
                    "version": source.version + 1,
                }
            )
            self.sources[source.source_id] = source
            return self._remember_result(actor, "source.review", command, source)

    async def reject_source(
        self,
        actor: ActorContext,
        command: RejectSourceCommand,
    ) -> SourceRejectionResult:
        async with self._lock:
            replay = self._replay(actor, "source.reject", command)
            if replay is not None:
                return replay
            source = self._source_target(actor, command.source_id)
            if source.version != command.expected_version:
                raise InvalidState(
                    "source version changed before rejection",
                    source_id=source.source_id,
                    expected_version=command.expected_version,
                    actual_version=source.version,
                )
            now = self._now()
            source = source.model_copy(
                update={
                    "review_state": SourceReviewState.REJECTED,
                    "reviewed_by_agent_id": actor.agent_id,
                    "reviewed_at": now,
                    "rejection_reason": command.reason,
                    "version": source.version + 1,
                }
            )
            self.sources[source.source_id] = source
            rolled_back: list[str] = []
            for memory in list(self.memories.values()):
                if memory.recorded_to is not None:
                    continue
                if not any(ref.source_id == source.source_id for ref in memory.evidence):
                    continue
                if not self._memory_accessible(actor, memory, memory.task_id):
                    continue
                remaining = tuple(
                    ref for ref in memory.evidence if ref.source_id != source.source_id
                )
                if remaining:
                    self.memories[memory.memory_id] = memory.model_copy(
                        update={"evidence": remaining, "version": memory.version + 1}
                    )
                    continue
                rejected = memory.model_copy(
                    update={"state": MemoryState.REFUTED, "version": memory.version + 1}
                )
                self.memories[memory.memory_id] = rejected
                rolled_back.append(memory.memory_id)
                predecessor_id = memory.supersedes_memory_id
                if predecessor_id is not None and predecessor_id in self.memories:
                    predecessor = self.memories[predecessor_id]
                    previous_state = self._previous_memory_state.get(
                        predecessor_id, MemoryState.TENTATIVE
                    )
                    self.memories[predecessor_id] = predecessor.model_copy(
                        update={
                            "state": previous_state,
                            "recorded_to": None,
                            "superseded_by_memory_id": None,
                            "version": predecessor.version + 1,
                        }
                    )
            result = SourceRejectionResult(
                source=source,
                rolled_back_memory_ids=tuple(rolled_back),
            )
            self._emit(
                actor,
                EventType.SOURCE_REJECTED,
                AggregateType.SOURCE,
                source.source_id,
                source.version,
                task_id=source.task_id,
                payload={"rolled_back_memory_ids": rolled_back},
                idempotency_key=command.idempotency_key,
            )
            return self._remember_result(actor, "source.reject", command, result)

    async def get_source(
        self,
        actor: ActorContext,
        source_id: SourceId,
    ) -> EvidenceSource | None:
        async with self._lock:
            source = self.sources.get(source_id)
            if source is None or not self._source_accessible(actor, source_id):
                return None
            return source

    async def get_evidence(self, actor: ActorContext, evidence_id: str) -> Evidence | None:
        async with self._lock:
            item = self.evidence.get(evidence_id)
            if item is None or not self._source_accessible(actor, item.source_id):
                return None
            return item

    # ------------------------------------------------------------------
    # Conflicts

    async def report_conflict(
        self,
        actor: ActorContext,
        command: ReportConflictCommand,
    ) -> ReportConflictResult:
        async with self._lock:
            replay = self._replay(actor, "conflict.report", command)
            if replay is not None:
                return replay
            if command.task_id is not None:
                self._task_target(actor, command.task_id)
            self._validate_evidence_refs(actor, command.evidence)
            for memory_id in command.memory_ids:
                self._memory_target(actor, memory_id, require_current=False)
            now = self._now()
            conflict = Conflict(
                conflict_id=_uuid(),
                tenant_id=actor.tenant_id,
                project_id=actor.project_id,
                repository_id=actor.repository_id,
                swarm_id=actor.swarm_id,
                run_id=actor.run_id,
                task_id=command.task_id,
                memory_ids=command.memory_ids,
                description=command.description,
                severity=command.severity,
                reported_by_agent_id=actor.agent_id,
                reported_evidence=command.evidence,
                created_at=now,
                updated_at=now,
                metadata=command.metadata,
            )
            self.conflicts[conflict.conflict_id] = conflict
            result = ReportConflictResult(conflict=conflict)
            self._emit(
                actor,
                EventType.CONFLICT_REPORTED,
                AggregateType.CONFLICT,
                conflict.conflict_id,
                conflict.version,
                task_id=conflict.task_id,
                idempotency_key=command.idempotency_key,
            )
            return self._remember_result(actor, "conflict.report", command, result)

    async def resolve_conflict(
        self,
        actor: ActorContext,
        command: ResolveConflictCommand,
    ) -> ResolveConflictResult:
        async with self._lock:
            replay = self._replay(actor, "conflict.resolve", command)
            if replay is not None:
                return replay
            conflict = self.conflicts.get(command.conflict_id)
            if conflict is None or not self._conflict_in_scope(actor, conflict):
                raise ResourceNotFound("conflict", command.conflict_id)
            if conflict.status is not ConflictStatus.OPEN:
                raise InvalidState("conflict is already settled", conflict_id=conflict.conflict_id)
            if conflict.version != command.expected_version:
                raise InvalidState(
                    "conflict version changed before resolution",
                    conflict_id=conflict.conflict_id,
                    expected_version=command.expected_version,
                    actual_version=conflict.version,
                )
            self._validate_evidence_refs(actor, command.resolution.evidence)
            referenced = set(command.resolution.supported_memory_ids) | set(
                command.resolution.refuted_memory_ids
            )
            if not referenced.issubset(conflict.memory_ids):
                raise InvalidState(
                    "resolution references memory outside the conflict",
                    conflict_id=conflict.conflict_id,
                )
            for memory_id in command.resolution.supported_memory_ids:
                memory = self._memory_target(actor, memory_id, require_current=False)
                if memory.state not in {MemoryState.SUPERSEDED, MemoryState.REFUTED}:
                    refs = {item.evidence_id: item for item in memory.evidence}
                    refs.update({item.evidence_id: item for item in command.resolution.evidence})
                    self.memories[memory_id] = memory.model_copy(
                        update={
                            "state": MemoryState.CONFIRMED,
                            "evidence": tuple(refs.values()),
                            "version": memory.version + 1,
                        }
                    )
            for memory_id in command.resolution.refuted_memory_ids:
                memory = self._memory_target(actor, memory_id, require_current=False)
                if memory.state is not MemoryState.SUPERSEDED:
                    self.memories[memory_id] = memory.model_copy(
                        update={"state": MemoryState.REFUTED, "version": memory.version + 1}
                    )
            now = self._now()
            resolution = ConflictResolution(
                **command.resolution.model_dump(),
                resolution_id=_uuid(),
                resolved_by_agent_id=actor.agent_id,
                resolved_at=now,
            )
            status = (
                ConflictStatus.DISMISSED
                if resolution.kind is ConflictResolutionKind.DISMISS
                else ConflictStatus.RESOLVED
            )
            conflict = conflict.model_copy(
                update={
                    "status": status,
                    "resolution": resolution,
                    "updated_at": now,
                    "version": conflict.version + 1,
                }
            )
            self.conflicts[conflict.conflict_id] = conflict
            result = ResolveConflictResult(conflict=conflict)
            self._emit(
                actor,
                EventType.CONFLICT_RESOLVED,
                AggregateType.CONFLICT,
                conflict.conflict_id,
                conflict.version,
                task_id=conflict.task_id,
                payload={"resolution": resolution.kind.value},
                idempotency_key=command.idempotency_key,
            )
            return self._remember_result(actor, "conflict.resolve", command, result)

    # ------------------------------------------------------------------
    # Internal transaction helpers

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None:
            raise ValueError("clock must return a timezone-aware datetime")
        return value.astimezone(UTC)

    def _fingerprint(self, command: BaseModel) -> str:
        payload = json.dumps(
            command.model_dump(mode="json", exclude_none=False),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _replay(
        self,
        actor: ActorContext,
        operation: str,
        command: BaseModel,
    ) -> Any | None:
        key = (
            actor.tenant_id,
            actor.run_id,
            actor.agent_id,
            operation,
            command.idempotency_key,
        )
        entry = self._idempotency.get(key)
        if entry is None:
            return None
        if entry.fingerprint != self._fingerprint(command):
            raise IdempotencyConflict(command.idempotency_key)
        self._duplicate_replays[actor.run_id] += 1
        if "replayed" in entry.result.__class__.model_fields:
            return entry.result.model_copy(update={"replayed": True})
        return entry.result

    def _remember_result(
        self,
        actor: ActorContext,
        operation: str,
        command: BaseModel,
        result: TModel,
    ) -> TModel:
        key = (
            actor.tenant_id,
            actor.run_id,
            actor.agent_id,
            operation,
            command.idempotency_key,
        )
        self._idempotency[key] = _IdempotencyEntry(self._fingerprint(command), result)
        return result

    def _task_in_scope(self, actor: ActorContext, task: Task) -> bool:
        return (
            task.tenant_id == actor.tenant_id
            and task.project_id == actor.project_id
            and task.repository_id == actor.repository_id
            and task.swarm_id == actor.swarm_id
            and task.run_id == actor.run_id
        )

    def _task_target(self, actor: ActorContext, task_id: str) -> Task:
        task = self.tasks.get(task_id)
        if task is None or not self._task_in_scope(actor, task):
            raise ResourceNotFound("task", task_id)
        return task

    def _task_is_claimable(self, task: Task, now: datetime) -> bool:
        if task.status not in {TaskStatus.PENDING, TaskStatus.READY}:
            return False
        if task.available_at is not None and task.available_at > now:
            return False
        for dependency in self.dependencies[task.task_id]:
            if dependency.kind is DependencyKind.BLOCKS:
                prerequisite = self.tasks.get(dependency.depends_on_task_id)
                if prerequisite is None or prerequisite.status is not TaskStatus.COMPLETED:
                    return False
        return True

    def _expire_leases(self, actor: ActorContext, now: datetime) -> None:
        for lease in list(self.leases.values()):
            if (
                lease.run_id != actor.run_id
                or lease.status is not LeaseStatus.ACTIVE
                or lease.expires_at > now
            ):
                continue
            task = self.tasks.get(lease.task_id)
            if task is None or not self._task_in_scope(actor, task):
                continue
            self.leases[lease.lease_id] = lease.model_copy(
                update={"status": LeaseStatus.EXPIRED, "version": lease.version + 1}
            )
            if task.active_lease_id == lease.lease_id:
                self.tasks[task.task_id] = task.model_copy(
                    update={
                        "status": TaskStatus.READY,
                        "claimed_by_agent_id": None,
                        "active_lease_id": None,
                        "updated_at": now,
                        "version": task.version + 1,
                    }
                )
                if self.checkpoints[task.task_id]:
                    self._crash_handoffs[task.run_id] += 1

    def _owned_lease(
        self,
        actor: ActorContext,
        lease_id: str,
        expected_version: int,
    ) -> TaskLease:
        lease = self.leases.get(lease_id)
        now = self._now()
        task = self.tasks.get(lease.task_id) if lease is not None else None
        if (
            lease is None
            or task is None
            or not self._task_in_scope(actor, task)
            or lease.run_id != actor.run_id
            or lease.owner_agent_id != actor.agent_id
            or lease.status is not LeaseStatus.ACTIVE
            or lease.expires_at <= now
        ):
            raise LeaseLost(lease_id)
        if lease.version != expected_version:
            raise LeaseLost(lease_id, "lease version changed")
        return lease

    def _owned_task_and_lease(
        self,
        actor: ActorContext,
        task_id: str,
        lease_id: str,
        expected_task_version: int,
        expected_lease_version: int,
    ) -> tuple[Task, TaskLease]:
        task = self._task_target(actor, task_id)
        lease = self._owned_lease(actor, lease_id, expected_lease_version)
        if task.active_lease_id != lease_id or lease.task_id != task_id:
            raise LeaseLost(lease_id, "lease is not current for task")
        if task.version != expected_task_version:
            raise LeaseLost(lease_id, "task version changed")
        return task, lease

    def _memory_accessible(
        self,
        actor: ActorContext,
        memory: Memory,
        requested_task_id: str | None,
    ) -> bool:
        if (
            memory.tenant_id != actor.tenant_id
            or memory.project_id != actor.project_id
            or memory.repository_id != actor.repository_id
        ):
            return False
        if memory.visibility is Visibility.REPOSITORY:
            return True
        if memory.swarm_id != actor.swarm_id or memory.run_id != actor.run_id:
            return False
        if memory.visibility is Visibility.RUN:
            return True
        return requested_task_id is not None and memory.task_id == requested_task_id

    def _recallable_memories_locked(
        self,
        actor: ActorContext,
        query: RecallQuery,
        now: datetime,
    ) -> list[Memory]:
        if query.task_id is not None:
            self._task_target(actor, query.task_id)
        system_at = query.recorded_at
        candidates: list[Memory] = []
        for memory in self.memories.values():
            if query.memory_ids and memory.memory_id not in query.memory_ids:
                continue
            if not self._memory_accessible(actor, memory, query.task_id):
                continue
            if memory.visibility not in query.visibilities:
                continue
            if query.kinds and memory.kind not in query.kinds:
                continue
            if memory.state not in query.effective_states:
                continue
            if system_at is None:
                if memory.recorded_to is not None and not query.include_superseded:
                    continue
            elif not (
                memory.recorded_from <= system_at
                and (memory.recorded_to is None or system_at < memory.recorded_to)
            ):
                continue
            if query.referenced_valid_from is not None:
                assert query.referenced_valid_to is not None
                if not (
                    memory.valid_from < query.referenced_valid_to
                    and (memory.valid_to is None or query.referenced_valid_from < memory.valid_to)
                ):
                    continue
            else:
                world_at = query.world_at or now
                if not (
                    memory.valid_from <= world_at
                    and (memory.valid_to is None or world_at < memory.valid_to)
                ):
                    continue

            if not memory.evidence:
                candidates.append(memory)
                continue
            accepted_evidence = tuple(
                reference
                for reference in memory.evidence
                if (source := self.sources.get(reference.source_id)) is not None
                and self._source_repository_accessible(actor, reference.source_id)
                and source.trust is not SourceTrust.UNTRUSTED
                and source.review_state is not SourceReviewState.REJECTED
            )
            if accepted_evidence:
                candidates.append(memory.model_copy(update={"evidence": accepted_evidence}))
            elif query.include_refuted:
                candidates.append(memory.model_copy(update={"evidence": ()}))
        return candidates

    def _recallable_memory_graph_locked(
        self,
        actor: ActorContext,
        query: RecallQuery,
        now: datetime,
    ) -> tuple[tuple[Memory, ...], tuple[MemoryLink, ...]]:
        memories = tuple(self._recallable_memories_locked(actor, query, now))
        eligible_ids = frozenset(memory.memory_id for memory in memories)
        recorded_at = query.recorded_at or now
        links = tuple(
            sorted(
                (
                    link
                    for link in self.memory_links.values()
                    if link.created_at <= recorded_at
                    and link.source_memory_id in eligible_ids
                    and link.target_memory_id in eligible_ids
                ),
                key=lambda link: (link.created_at, link.link_id),
            )
        )
        return memories, links

    def _memory_target(
        self,
        actor: ActorContext,
        memory_id: str,
        *,
        require_current: bool = True,
    ) -> Memory:
        memory = self.memories.get(memory_id)
        if memory is None or not self._memory_accessible(actor, memory, memory.task_id):
            raise ResourceNotFound("memory", memory_id)
        if require_current and memory.recorded_to is not None:
            raise InvalidState("memory is not current", memory_id=memory_id)
        return memory

    def _validate_citation_memory_ids(
        self,
        actor: ActorContext,
        task_id: str,
        memory_ids: Sequence[str],
    ) -> None:
        for memory_id in memory_ids:
            memory = self.memories.get(memory_id)
            if memory is None or not self._memory_accessible(actor, memory, task_id):
                raise ResourceNotFound("memory", memory_id)

    def _derive_memory_outcome_associations(
        self,
        actor: ActorContext,
        completion: TaskCompletion,
        *,
        now: datetime,
    ) -> tuple[MemoryOutcomeAssociation, ...]:
        """Intersect exact activation proof with completion citations.

        A citation does not carry a memory version.  To avoid guessing which
        rendering it refers to, multiple activated versions under one lease
        are treated as ambiguous and omitted.  The sole proven version must
        also remain canonically recallable at completion time.
        """

        cited_ids = tuple(dict.fromkeys(completion.memory_ids))[
            :MAX_OUTCOME_ASSOCIATIONS_PER_COMPLETION
        ]
        if not cited_ids:
            return ()
        versions_by_memory: dict[str, set[int]] = defaultdict(set)
        for event in self.events:
            if (
                event.tenant_id != actor.tenant_id
                or event.project_id != actor.project_id
                or event.repository_id != actor.repository_id
                or event.swarm_id != actor.swarm_id
                or event.run_id != actor.run_id
                or event.agent_id != actor.agent_id
                or event.task_id != completion.task_id
                or event.event_type is not EventType.MEMORY_ACTIVATED
            ):
                continue
            try:
                telemetry = MemoryActivationTelemetry.model_validate(
                    {
                        key: event.payload[key]
                        for key in MemoryActivationTelemetry.model_fields
                        if key in event.payload
                    }
                )
            except ValueError:
                continue
            if telemetry.lease_id != completion.lease_id:
                continue
            for memory_id in cited_ids:
                if (version := telemetry.memory_versions.get(memory_id)) is not None:
                    versions_by_memory[memory_id].add(version)

        unambiguous_ids = tuple(
            memory_id for memory_id in cited_ids if len(versions_by_memory[memory_id]) == 1
        )
        if not unambiguous_ids:
            return ()
        proof_query = RecallQuery(
            text="outcome association eligibility check",
            task_id=completion.task_id,
            memory_ids=frozenset(unambiguous_ids),
            states=frozenset({MemoryState.CONFIRMED}),
            visibilities=frozenset(Visibility),
            limit=len(unambiguous_ids),
        )
        current_versions = {
            memory.memory_id: memory.version
            for memory in self._recallable_memories_locked(actor, proof_query, now)
        }
        associations = []
        for memory_id in unambiguous_ids:
            memory_version = next(iter(versions_by_memory[memory_id]))
            if current_versions.get(memory_id) != memory_version:
                continue
            associations.append(
                MemoryOutcomeAssociation(
                    association_id=memory_outcome_association_id(
                        tenant_id=actor.tenant_id,
                        project_id=actor.project_id,
                        repository_id=actor.repository_id,
                        swarm_id=actor.swarm_id,
                        run_id=actor.run_id,
                        task_id=completion.task_id,
                        lease_id=completion.lease_id,
                        consumer_agent_id=actor.agent_id,
                        memory_id=memory_id,
                        memory_version=memory_version,
                    ),
                    tenant_id=actor.tenant_id,
                    project_id=actor.project_id,
                    repository_id=actor.repository_id,
                    swarm_id=actor.swarm_id,
                    run_id=actor.run_id,
                    task_id=completion.task_id,
                    lease_id=completion.lease_id,
                    consumer_agent_id=actor.agent_id,
                    memory_id=memory_id,
                    memory_version=memory_version,
                    outcome=completion.outcome,
                    observed_at=completion.completed_at,
                )
            )
        return tuple(associations)

    def _source_accessible(self, actor: ActorContext, source_id: str) -> bool:
        scope = self._source_scope.get(source_id)
        return scope == (
            actor.tenant_id,
            actor.project_id,
            actor.repository_id,
            actor.swarm_id,
            actor.run_id,
        )

    def _source_repository_accessible(self, actor: ActorContext, source_id: str) -> bool:
        """Allow citations to follow repository-visible memory across runs."""

        scope = self._source_scope.get(source_id)
        return scope is not None and scope[:3] == (
            actor.tenant_id,
            actor.project_id,
            actor.repository_id,
        )

    def _source_target(self, actor: ActorContext, source_id: str) -> EvidenceSource:
        source = self.sources.get(source_id)
        if source is None or not self._source_accessible(actor, source_id):
            raise ResourceNotFound("source", source_id)
        return source

    def _validate_evidence_refs(
        self,
        actor: ActorContext,
        references: Sequence[EvidenceRef],
    ) -> None:
        for reference in references:
            evidence = self.evidence.get(reference.evidence_id)
            if (
                evidence is None
                or evidence.source_id != reference.source_id
                or not self._source_accessible(actor, reference.source_id)
            ):
                raise ResourceNotFound("evidence", reference.evidence_id)

    def _validate_derived_evidence(
        self,
        actor: ActorContext,
        command: RememberCommand,
    ) -> None:
        if not command.derived_from_memory_ids:
            return
        proposed = {reference.evidence_id: reference for reference in command.evidence}
        for memory_id in command.derived_from_memory_ids:
            source = self._memory_target(actor, memory_id)
            if source.state is not MemoryState.CONFIRMED:
                raise InvalidState(
                    "derived lineage requires confirmed source memories",
                    memory_id=memory_id,
                )
            source_evidence = {reference.evidence_id: reference for reference in source.evidence}
            overlap = set(proposed) & set(source_evidence)
            if not overlap:
                raise InvalidState(
                    "derived lineage requires evidence from every source memory",
                    memory_id=memory_id,
                )
            if any(
                proposed[evidence_id] != source_evidence[evidence_id] for evidence_id in overlap
            ):
                raise InvalidState(
                    "derived lineage must preserve exact immutable evidence references",
                    memory_id=memory_id,
                )
            for evidence_id in overlap:
                evidence_source = self.sources.get(proposed[evidence_id].source_id)
                if evidence_source is None or (
                    evidence_source.trust is SourceTrust.UNTRUSTED
                    or evidence_source.review_state is SourceReviewState.REJECTED
                ):
                    raise InvalidState(
                        "derived lineage source evidence is no longer accepted",
                        memory_id=memory_id,
                    )
        allowed = {
            reference.evidence_id: reference
            for memory_id in command.derived_from_memory_ids
            for reference in self._memory_target(actor, memory_id).evidence
        }
        if any(
            allowed.get(evidence_id) != reference for evidence_id, reference in proposed.items()
        ):
            raise InvalidState("derived memory evidence is not supported by its lineage")

    def _conflict_in_scope(self, actor: ActorContext, conflict: Conflict) -> bool:
        return (
            conflict.tenant_id == actor.tenant_id
            and conflict.project_id == actor.project_id
            and conflict.repository_id == actor.repository_id
            and conflict.swarm_id == actor.swarm_id
            and conflict.run_id == actor.run_id
        )

    def _add_memory_link(
        self,
        source_memory_id: str,
        target_memory_id: str,
        kind: MemoryLinkKind,
        *,
        reason: str | None = None,
    ) -> MemoryLink:
        link = MemoryLink(
            link_id=_uuid(),
            source_memory_id=source_memory_id,
            target_memory_id=target_memory_id,
            kind=kind,
            reason=reason,
            created_at=self._now(),
        )
        self.memory_links[link.link_id] = link
        return link

    def _emit(
        self,
        actor: ActorContext,
        event_type: EventType | str,
        aggregate_type: AggregateType,
        aggregate_id: str,
        aggregate_version: int,
        *,
        task_id: str | None = None,
        task_title: str | None = None,
        payload: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
        event_id: str | None = None,
        outbox_id: str | None = None,
    ) -> SwarmEvent:
        now = self._now()
        event = SwarmEvent(
            event_id=event_id or _uuid(),
            event_type=event_type,
            aggregate_type=aggregate_type,
            aggregate_id=aggregate_id,
            tenant_id=actor.tenant_id,
            project_id=actor.project_id,
            repository_id=actor.repository_id,
            swarm_id=actor.swarm_id,
            run_id=actor.run_id,
            agent_id=actor.agent_id,
            task_id=task_id,
            aggregate_version=aggregate_version,
            payload=enriched_payload(
                payload,
                harness=actor.harness,
                provider=actor.provider,
                model=actor.model,
                task_title=task_title,
            ),
            occurred_at=now,
            recorded_at=now,
            idempotency_key=idempotency_key,
        )
        self.events.append(event)
        self.outbox.append(
            OutboxEvent(outbox_id=outbox_id or _uuid(), event=event, available_at=now)
        )
        return event


__all__ = ["InMemoryKernel"]
