"""Lease-driven Observer/Reflector consolidation worker."""

from __future__ import annotations

from contextlib import suppress
from datetime import UTC, datetime, timedelta

from swarmbrain.application.consolidation import ConsolidationService
from swarmbrain.domain.agents import ActorContext, Capability
from swarmbrain.domain.consolidation import ConsolidationReflection, ConsolidationWorkPayload
from swarmbrain.domain.work import (
    ClaimWorkCommand,
    CompleteWorkCommand,
    CompleteWorkResult,
    FailWorkCommand,
    StageConsolidationPlanCommand,
    WorkKind,
    WorkLease,
    WorkLeaseLost,
)
from swarmbrain.ports.work_queue import WorkQueueStore


class ConsolidationWorker:
    """Stage one immutable plan, then replay governed effects until completion."""

    def __init__(
        self,
        queue: WorkQueueStore,
        consolidation: ConsolidationService,
        *,
        retry_delay_seconds: int = 30,
    ) -> None:
        if retry_delay_seconds < 1:
            raise ValueError("retry_delay_seconds must be positive")
        self.queue = queue
        self.consolidation = consolidation
        self.retry_delay_seconds = retry_delay_seconds

    async def run_once(
        self,
        worker_id: str,
        *,
        limit: int = 1,
        lease_seconds: int = 60,
    ) -> tuple[CompleteWorkResult, ...]:
        batch = await self.queue.claim_work(
            ClaimWorkCommand(
                worker_id=worker_id,
                kinds=frozenset({WorkKind.CONSOLIDATE_MEMORY}),
                limit=limit,
                lease_seconds=lease_seconds,
            )
        )
        completed: list[CompleteWorkResult] = []
        for lease in batch.leases:
            result = await self._process(lease)
            if result is not None:
                completed.append(result)
        return tuple(completed)

    async def _process(self, lease: WorkLease) -> CompleteWorkResult | None:
        current_version = lease.work_version
        try:
            payload = ConsolidationWorkPayload.model_validate(lease.item.payload)
            staged_payload = (lease.item.result or {}).get("staged_consolidation")
            if staged_payload is None:
                reflection = await self.consolidation.reflect(payload)
                staged = await self.queue.stage_consolidation_plan(
                    StageConsolidationPlanCommand(
                        work_id=lease.item.work_id,
                        worker_id=lease.worker_id,
                        lease_token=lease.lease_token,
                        lease_version=lease.lease_version,
                        expected_work_version=current_version,
                        attempt=lease.attempt,
                        reflection=reflection,
                    )
                )
                current_version = staged.item.version
            else:
                reflection = ConsolidationReflection.model_validate(staged_payload)
            actor = self._actor(lease)
            applied = await self.consolidation.apply(
                actor,
                payload,
                reflection,
                work_id=lease.item.work_id,
            )
            return await self.queue.complete_work(
                CompleteWorkCommand(
                    work_id=lease.item.work_id,
                    worker_id=lease.worker_id,
                    lease_token=lease.lease_token,
                    lease_version=lease.lease_version,
                    expected_work_version=current_version,
                    attempt=lease.attempt,
                    outcome=applied.status,
                    result={
                        "route": reflection.route.value,
                        "input_sha256": reflection.input_sha256,
                        "plan_sha256": reflection.plan_sha256,
                        "fallback_reason": reflection.fallback_reason,
                        "provider": (
                            reflection.provider.model_dump(mode="json")
                            if reflection.provider is not None
                            else None
                        ),
                        "apply": applied.model_dump(mode="json"),
                    },
                )
            )
        except WorkLeaseLost:
            return None
        except Exception as exc:
            with suppress(WorkLeaseLost):
                await self.queue.fail_work(
                    FailWorkCommand(
                        work_id=lease.item.work_id,
                        worker_id=lease.worker_id,
                        lease_token=lease.lease_token,
                        lease_version=lease.lease_version,
                        expected_work_version=current_version,
                        attempt=lease.attempt,
                        error=f"worker_{type(exc).__name__}",
                        retry_at=datetime.now(UTC) + timedelta(seconds=self.retry_delay_seconds),
                    )
                )
            return None

    @staticmethod
    def _actor(lease: WorkLease) -> ActorContext:
        item = lease.item
        return ActorContext(
            tenant_id=item.tenant_id,
            project_id=item.project_id,
            repository_id=item.repository_id,
            swarm_id=item.swarm_id,
            run_id=item.run_id,
            agent_id=item.requested_by_agent_id,
            harness="swarmbrain-worker",
            provider="consolidation",
            model="evidence-gated-v1",
            capabilities=frozenset(
                {
                    Capability.MEMORY_PUBLISH.value,
                    Capability.MEMORY_RECALL.value,
                }
            ),
        )


__all__ = ["ConsolidationWorker"]
