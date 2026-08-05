from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from conftest import make_actor, make_task, new_id
from swarmbrain.application.coordination import CoordinationService
from swarmbrain.domain.leases import TaskLease
from swarmbrain.domain.memory import RecallBundle, RecallQuery
from swarmbrain.domain.retrieval import RetrievalPurpose
from swarmbrain.domain.tasks import (
    ClaimTaskCommand,
    ClaimTaskResult,
    TaskCheckpoint,
    TaskStatus,
)


class _ClaimStore:
    def __init__(self, result: ClaimTaskResult) -> None:
        self.result = result

    async def claim_task(self, _actor: object, _command: object) -> ClaimTaskResult:
        return self.result


class _CapturingMemory:
    def __init__(self) -> None:
        self.query: RecallQuery | None = None
        self.purpose: RetrievalPurpose | None = None
        self.seeds: tuple[str, ...] = ()

    async def recall(
        self,
        _actor: object,
        query: RecallQuery,
        *,
        purpose: RetrievalPurpose,
        seed_memory_ids: tuple[str, ...],
    ) -> RecallBundle:
        self.query = query
        self.purpose = purpose
        self.seeds = seed_memory_ids
        return RecallBundle(query=query)


@pytest.mark.asyncio
async def test_task_claim_selects_server_owned_bootstrap_purpose_and_checkpoint_seeds(
    scope_ids: dict[str, str],
) -> None:
    actor = make_actor(scope_ids)
    now = datetime(2026, 8, 5, 12, tzinfo=UTC)
    task = make_task(scope_ids, title="Repair retry coordinator", created_at=now).model_copy(
        update={
            "description": "Fix SQLSTATE 40001 handling",
            "tags": ("cockroachdb", "retry"),
            "required_capabilities": frozenset({"python"}),
            "status": TaskStatus.CLAIMED,
            "claimed_by_agent_id": actor.agent_id,
            "active_lease_id": new_id(),
        }
    )
    lease = TaskLease(
        lease_id=task.active_lease_id,
        task_id=task.task_id,
        run_id=task.run_id,
        owner_agent_id=actor.agent_id,
        acquired_at=now,
        expires_at=now + timedelta(minutes=2),
    )
    memory_id = new_id()
    checkpoint = TaskCheckpoint(
        checkpoint_id=new_id(),
        task_id=task.task_id,
        lease_id=lease.lease_id,
        run_id=task.run_id,
        agent_id=actor.agent_id,
        sequence=2,
        summary="Parser repaired, transaction retry remains",
        discoveries=("RetryCoordinator owns backoff",),
        remaining_work=("Run serialization integration test",),
        memory_ids=(memory_id,),
        created_at=now,
    )
    capture = _CapturingMemory()
    service = CoordinationService(
        _ClaimStore(ClaimTaskResult(task=task, lease=lease, checkpoint=checkpoint)),  # type: ignore[arg-type]
        memory_service=capture,  # type: ignore[arg-type]
    )

    result = await service.claim(actor, ClaimTaskCommand(idempotency_key="claim-bootstrap"))

    assert result.memory is not None
    assert capture.purpose is RetrievalPurpose.TASK_BOOTSTRAP
    assert capture.seeds == (memory_id,)
    assert capture.query is not None
    assert capture.query.task_id == task.task_id
    assert "Parser repaired" in capture.query.text
    assert "RetryCoordinator owns backoff" in capture.query.text
    assert "serialization integration test" in capture.query.text
