"""Swarm event stream, audit, outbox, and metrics contracts."""

from __future__ import annotations

from enum import StrEnum

from pydantic import AwareDatetime, Field

from .common import (
    AgentId,
    ContractModel,
    EventId,
    JsonObject,
    MutationCommand,
    ProjectId,
    RepositoryId,
    RunId,
    SwarmId,
    TaskId,
    TenantId,
    UUIDString,
    utc_now,
)


class AggregateType(StrEnum):
    AGENT = "agent"
    TASK = "task"
    LEASE = "lease"
    MEMORY = "memory"
    SOURCE = "source"
    CONFLICT = "conflict"
    RUN = "run"


class EventType(StrEnum):
    AGENT_JOINED = "agent.joined"
    TASK_CLAIMED = "task.claimed"
    LEASE_RENEWED = "lease.renewed"
    TASK_CHECKPOINTED = "task.checkpointed"
    TASK_COMPLETED = "task.completed"
    TASK_RELEASED = "task.released"
    MEMORY_ADDED = "memory.added"
    MEMORY_SUPERSEDED = "memory.superseded"
    MEMORY_CONFIRMED = "memory.confirmed"
    MEMORY_REFUTED = "memory.refuted"
    SOURCE_REJECTED = "source.rejected"
    CONFLICT_REPORTED = "conflict.reported"
    CONFLICT_RESOLVED = "conflict.resolved"


class AuditOutcome(StrEnum):
    SUCCEEDED = "succeeded"
    REJECTED = "rejected"
    FAILED = "failed"
    INDETERMINATE = "indeterminate"


class OutboxStatus(StrEnum):
    PENDING = "pending"
    PUBLISHING = "publishing"
    PUBLISHED = "published"
    FAILED = "failed"


class SwarmEvent(ContractModel):
    event_id: EventId
    event_type: EventType | str
    aggregate_type: AggregateType
    aggregate_id: UUIDString
    tenant_id: TenantId
    project_id: ProjectId
    repository_id: RepositoryId
    swarm_id: SwarmId
    run_id: RunId
    agent_id: AgentId | None = None
    task_id: TaskId | None = None
    aggregate_version: int = Field(ge=1)
    payload: JsonObject = Field(default_factory=dict)
    occurred_at: AwareDatetime = Field(default_factory=utc_now)
    recorded_at: AwareDatetime = Field(default_factory=utc_now)
    correlation_id: UUIDString | None = None
    causation_id: EventId | None = None
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=255)


class AuditEvent(ContractModel):
    audit_id: UUIDString
    event_id: EventId | None = None
    tenant_id: TenantId
    run_id: RunId
    actor_agent_id: AgentId | None = None
    action: str = Field(min_length=1, max_length=255)
    resource_type: str = Field(min_length=1, max_length=100)
    resource_id: UUIDString | None = None
    outcome: AuditOutcome
    reason: str | None = Field(default=None, max_length=4096)
    occurred_at: AwareDatetime = Field(default_factory=utc_now)
    details: JsonObject = Field(default_factory=dict)


class OutboxEvent(ContractModel):
    outbox_id: UUIDString
    event: SwarmEvent
    status: OutboxStatus = OutboxStatus.PENDING
    attempts: int = Field(default=0, ge=0)
    available_at: AwareDatetime = Field(default_factory=utc_now)
    locked_until: AwareDatetime | None = None
    published_at: AwareDatetime | None = None
    last_error: str | None = Field(default=None, max_length=4096)
    version: int = Field(default=1, ge=1)


class EventPage(ContractModel):
    events: tuple[SwarmEvent, ...] = ()
    next_cursor: str | None = None


class RunMetrics(ContractModel):
    run_id: RunId
    measured_at: AwareDatetime = Field(default_factory=utc_now)
    tasks_total: int = Field(default=0, ge=0)
    tasks_claimed: int = Field(default=0, ge=0)
    tasks_completed: int = Field(default=0, ge=0)
    tasks_failed: int = Field(default=0, ge=0)
    active_leases: int = Field(default=0, ge=0)
    checkpoints: int = Field(default=0, ge=0)
    crash_handoffs: int = Field(default=0, ge=0)
    memories_published: int = Field(default=0, ge=0)
    memories_reused: int = Field(default=0, ge=0)
    conflicts_open: int = Field(default=0, ge=0)
    conflicts_resolved: int = Field(default=0, ge=0)
    duplicate_claims_prevented: int = Field(default=0, ge=0)
    duplicate_mutations_replayed: int = Field(default=0, ge=0)
    custom: dict[str, float] = Field(default_factory=dict)


class ClaimOutboxBatchCommand(MutationCommand):
    limit: int = Field(default=100, ge=1, le=1000)
    lock_seconds: int = Field(default=30, ge=1, le=600)


class MarkOutboxPublishedCommand(MutationCommand):
    outbox_id: UUIDString
    expected_version: int = Field(ge=1)


class MarkOutboxFailedCommand(MutationCommand):
    outbox_id: UUIDString
    expected_version: int = Field(ge=1)
    error: str = Field(min_length=1, max_length=4096)
    retry_at: AwareDatetime


__all__ = [
    "AggregateType",
    "AuditEvent",
    "AuditOutcome",
    "ClaimOutboxBatchCommand",
    "EventPage",
    "EventType",
    "MarkOutboxFailedCommand",
    "MarkOutboxPublishedCommand",
    "OutboxEvent",
    "OutboxStatus",
    "RunMetrics",
    "SwarmEvent",
]
