"""Capability-gated read facade for the append-only event and metric views."""

from __future__ import annotations

from swarmbrain.domain.agents import ActorContext, Capability
from swarmbrain.domain.events import EventPage, RunMetrics
from swarmbrain.ports.event_sink import EventReader, MetricsReader

from .capabilities import require_capability


class AuditService:
    def __init__(self, events: EventReader, metrics: MetricsReader) -> None:
        self._events = events
        self._metrics = metrics

    async def list_events(
        self,
        actor: ActorContext,
        *,
        cursor: str | None = None,
        limit: int = 100,
    ) -> EventPage:
        require_capability(actor, Capability.EVENTS_READ)
        return await self._events.list_run_events(
            actor,
            actor.run_id,
            cursor=cursor,
            limit=limit,
        )

    async def metrics(self, actor: ActorContext) -> RunMetrics:
        require_capability(actor, Capability.METRICS_READ)
        return await self._metrics.get_run_metrics(actor, actor.run_id)


__all__ = ["AuditService"]
