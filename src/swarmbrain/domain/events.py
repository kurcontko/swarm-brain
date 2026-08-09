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
    MEMORY_ACTIVATED = "memory.activated"
    SOURCE_REJECTED = "source.rejected"
    CONFLICT_REPORTED = "conflict.reported"
    CONFLICT_RESOLVED = "conflict.resolved"


TASK_TITLE_KEY = "task_title"
AGENT_HARNESS_KEY = "agent_harness"
AGENT_PROVIDER_KEY = "agent_provider"
AGENT_MODEL_KEY = "agent_model"
AGENT_DISPLAY_KEY = "agent_display"

EVENT_ENRICHMENT_KEYS: tuple[str, ...] = (
    TASK_TITLE_KEY,
    AGENT_HARNESS_KEY,
    AGENT_PROVIDER_KEY,
    AGENT_MODEL_KEY,
    AGENT_DISPLAY_KEY,
)


def agent_display_label(harness: str | None, provider: str | None) -> str | None:
    """Render the human label a reader sees instead of an opaque agent UUID."""

    harness = (harness or "").strip()
    provider = (provider or "").strip()
    if not harness:
        return None
    return f"{harness} ({provider})" if provider else harness


def event_enrichment(
    *,
    harness: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    task_title: str | None = None,
) -> JsonObject:
    """Additive, human-readable payload keys shared by every event emitter.

    These live in :attr:`SwarmEvent.payload` rather than the envelope so the
    wire contract stays backward compatible: readers that predate them simply
    ignore extra payload keys, and no field becomes required. Both the
    in-memory and the CockroachDB adapters call this one function so the keys
    they stamp cannot drift apart.
    """

    enrichment: JsonObject = {}
    if task_title:
        enrichment[TASK_TITLE_KEY] = task_title
    if harness:
        enrichment[AGENT_HARNESS_KEY] = harness
    if provider:
        enrichment[AGENT_PROVIDER_KEY] = provider
    if model:
        enrichment[AGENT_MODEL_KEY] = model
    display = agent_display_label(harness, provider)
    if display is not None:
        enrichment[AGENT_DISPLAY_KEY] = display
    return enrichment


def enriched_payload(
    payload: JsonObject | None,
    *,
    harness: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    task_title: str | None = None,
) -> JsonObject:
    """Merge enrichment under an emitter's own payload keys, never over them."""

    merged = event_enrichment(
        harness=harness,
        provider=provider,
        model=model,
        task_title=task_title,
    )
    merged.update(payload or {})
    return merged


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
    memories_activated: int = Field(default=0, ge=0)
    memory_activation_attempts: int = Field(default=0, ge=0)
    memories_cited: int = Field(default=0, ge=0)
    cross_agent_memory_uses: int = Field(default=0, ge=0)
    memories_reused: int = Field(
        default=0,
        ge=0,
        description=(
            "Deprecated compatibility metric. This counts memories returned by recall, "
            "not memories proven to have affected an agent outcome; use memories_activated, "
            "memories_cited, and cross_agent_memory_uses instead."
        ),
        json_schema_extra={"deprecated": True},
    )
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
    "AGENT_DISPLAY_KEY",
    "AGENT_HARNESS_KEY",
    "AGENT_MODEL_KEY",
    "AGENT_PROVIDER_KEY",
    "EVENT_ENRICHMENT_KEYS",
    "TASK_TITLE_KEY",
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
    "agent_display_label",
    "enriched_payload",
    "event_enrichment",
]
