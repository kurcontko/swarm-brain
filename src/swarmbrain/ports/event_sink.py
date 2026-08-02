"""Event publication, audit, outbox relay, and read-model ports."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from swarmbrain.domain.agents import ActorContext
from swarmbrain.domain.common import RunId
from swarmbrain.domain.events import (
    AuditEvent,
    ClaimOutboxBatchCommand,
    EventPage,
    MarkOutboxFailedCommand,
    MarkOutboxPublishedCommand,
    OutboxEvent,
    RunMetrics,
    SwarmEvent,
)


@runtime_checkable
class EventSink(Protocol):
    """At-least-once sink that deduplicates by ``SwarmEvent.event_id``."""

    async def publish(self, events: Sequence[SwarmEvent]) -> None: ...


@runtime_checkable
class AuditLog(Protocol):
    """Append-only security/operation audit boundary."""

    async def append(self, event: AuditEvent) -> None: ...


@runtime_checkable
class OutboxStore(Protocol):
    """Relay-facing outbox operations; aggregate writers enqueue atomically."""

    async def claim_batch(
        self,
        command: ClaimOutboxBatchCommand,
    ) -> tuple[OutboxEvent, ...]: ...

    async def mark_published(
        self,
        command: MarkOutboxPublishedCommand,
    ) -> OutboxEvent: ...

    async def mark_failed(
        self,
        command: MarkOutboxFailedCommand,
    ) -> OutboxEvent: ...


@runtime_checkable
class EventReader(Protocol):
    async def list_run_events(
        self,
        actor: ActorContext,
        run_id: RunId,
        *,
        cursor: str | None = None,
        limit: int = 100,
    ) -> EventPage: ...


@runtime_checkable
class MetricsReader(Protocol):
    async def get_run_metrics(
        self,
        actor: ActorContext,
        run_id: RunId,
    ) -> RunMetrics: ...


__all__ = ["AuditLog", "EventReader", "EventSink", "MetricsReader", "OutboxStore"]
