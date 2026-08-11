from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import pytest
from psycopg.types.json import Jsonb

from conftest import make_actor
from swarmbrain.adapters.cockroach import coordination
from swarmbrain.application.errors import InvalidState
from swarmbrain.domain.events import AggregateType, EventType
from swarmbrain.domain.outcome_feedback import (
    OutcomeAssociationKind,
    memory_outcome_association_id,
)
from swarmbrain.domain.tasks import TaskOutcome, TaskStatus


def _id() -> str:
    return str(uuid4())


def _scope() -> dict[str, str]:
    return {
        "tenant_id": _id(),
        "project_id": _id(),
        "repository_id": _id(),
        "swarm_id": _id(),
        "run_id": _id(),
    }


def _task_row(now: datetime) -> dict[str, Any]:
    return {
        "id": _id(),
        **_scope(),
        "title": "Map the durable row",
        "description": "explicit columns",
        "state": "claimed",
        "priority": 7,
        "version": 3,
        "required_capabilities": ["task:complete", "task:checkpoint"],
        "tags": ["python", "storage"],
        "created_by_agent_id": _id(),
        "claimed_by_agent_id": _id(),
        "active_lease_id": _id(),
        "available_at": None,
        "metadata": '{"source":"unit"}',
        "created_at": now,
        "updated_at": now,
        "completed_at": None,
    }


def test_row_mappers_preserve_contract_types_and_json_shapes() -> None:
    now = datetime.now(UTC)
    task_row = _task_row(now)
    task = coordination._task_from_row(task_row)

    assert task.task_id == task_row["id"]
    assert task.status is TaskStatus.CLAIMED
    assert task.required_capabilities == frozenset({"task:complete", "task:checkpoint"})
    assert task.tags == ("python", "storage")
    assert task.metadata == {"source": "unit"}

    lease_row = {
        "id": task.active_lease_id,
        "task_id": task.task_id,
        "tenant_id": task.tenant_id,
        "run_id": task.run_id,
        "agent_id": task.claimed_by_agent_id,
        "status": "active",
        "version": 2,
        "claimed_at": now,
        "renewed_at": now,
        "expires_at": now + timedelta(minutes=2),
        "released_at": None,
        "completed_at": None,
        "metadata": {},
    }
    lease = coordination._lease_from_row(lease_row)
    assert lease.lease_id == task.active_lease_id
    assert lease.renewed_at is None
    assert lease.version == 2

    checkpoint_row = {
        "id": _id(),
        "task_id": task.task_id,
        "lease_id": lease.lease_id,
        "tenant_id": task.tenant_id,
        "run_id": task.run_id,
        "agent_id": lease.owner_agent_id,
        "sequence": 4,
        "summary": "checkpoint",
        "discoveries": '["fact"]',
        "completed_work": ["mapped"],
        "remaining_work": [],
        "memory_ids": [UUID(_id())],
        "artifact_ids": [UUID(_id())],
        "state": '{"phase":4}',
        "created_at": now,
    }
    checkpoint = coordination._checkpoint_from_row(checkpoint_row)
    assert checkpoint.sequence == 4
    assert checkpoint.discoveries == ("fact",)
    assert checkpoint.state == {"phase": 4}
    assert checkpoint.memory_ids == tuple(str(item) for item in checkpoint_row["memory_ids"])

    completion_row = {
        "id": _id(),
        "task_id": task.task_id,
        "lease_id": lease.lease_id,
        "tenant_id": task.tenant_id,
        "run_id": task.run_id,
        "agent_id": lease.owner_agent_id,
        "outcome": "succeeded",
        "summary": "done",
        "memory_ids": [],
        "artifact_ids": [],
        "completed_at": now,
    }
    completion = coordination._completion_from_row(completion_row)
    assert completion.outcome is TaskOutcome.SUCCEEDED

    association_memory_id = _id()
    association_row = {
        "id": memory_outcome_association_id(
            tenant_id=task.tenant_id,
            project_id=task.project_id,
            repository_id=task.repository_id,
            swarm_id=task.swarm_id,
            run_id=task.run_id,
            task_id=task.task_id,
            lease_id=lease.lease_id,
            consumer_agent_id=lease.owner_agent_id,
            memory_id=association_memory_id,
            memory_version=2,
        ),
        "kind": "observational_silver",
        "tenant_id": task.tenant_id,
        "project_id": task.project_id,
        "repository_id": task.repository_id,
        "swarm_id": task.swarm_id,
        "run_id": task.run_id,
        "task_id": task.task_id,
        "lease_id": lease.lease_id,
        "consumer_agent_id": lease.owner_agent_id,
        "memory_id": association_memory_id,
        "memory_version": 2,
        "outcome": "failed",
        "observed_at": now,
    }
    association = coordination._outcome_association_from_row(association_row)
    assert association.kind is OutcomeAssociationKind.OBSERVATIONAL_SILVER
    assert association.outcome is TaskOutcome.FAILED
    assert association.memory_id == association_memory_id

    event_row = {
        "id": _id(),
        "tenant_id": task.tenant_id,
        "project_id": task.project_id,
        "repository_id": task.repository_id,
        "swarm_id": task.swarm_id,
        "run_id": task.run_id,
        "agent_id": lease.owner_agent_id,
        "task_id": task.task_id,
        "event_type": "task.checkpointed",
        "aggregate_type": "task",
        "aggregate_id": task.task_id,
        "aggregate_version": task.version,
        "payload": '{"checkpoint_id":"' + checkpoint.checkpoint_id + '"}',
        "occurred_at": now,
        "recorded_at": now,
        "correlation_id": None,
        "causation_id": None,
        "idempotency_key": "checkpoint-4",
    }
    event = coordination._event_from_row(event_row)
    assert event.event_type == EventType.TASK_CHECKPOINTED
    assert event.aggregate_type is AggregateType.TASK
    assert event.payload == {"checkpoint_id": checkpoint.checkpoint_id}


def test_event_cursor_round_trips_keyset_and_rejects_untrusted_values() -> None:
    occurred_at = datetime(2026, 8, 2, 13, 14, 15, 123456, tzinfo=UTC)
    event_id = _id()

    cursor = coordination._encode_event_cursor(occurred_at, event_id)

    assert coordination._decode_event_cursor(cursor) == (occurred_at, event_id)
    with pytest.raises(InvalidState, match="invalid event cursor"):
        coordination._decode_event_cursor("not-a-valid-keyset-cursor")


def test_coordination_sql_is_parameterized_and_never_uses_select_star() -> None:
    statements = {
        name: value
        for name, value in vars(coordination).items()
        if name.endswith("_SQL") and isinstance(value, str)
    }

    assert statements
    for name, sql in statements.items():
        assert "SELECT *" not in sql.upper(), name
        assert "%s" in sql, name
        assert "{" not in sql and "}" not in sql, name

    assert "FOR UPDATE SKIP LOCKED" in coordination.CLAIM_CANDIDATE_SQL
    assert "t.tags @> %s::STRING[]" in coordination.CLAIM_CANDIDATE_SQL
    assert "%s::STRING[] @> t.required_capabilities" in coordination.CLAIM_CANDIDATE_SQL
    assert "dependency.kind = 'blocks'" in coordination.CLAIM_CANDIDATE_SQL
    assert "prerequisite.state = 'completed'" in (
        coordination.SELECT_COMPLETED_BLOCKING_DEPENDENCIES_SQL
    )
    assert "ORDER BY dependency.depends_on_task_id" in (
        coordination.SELECT_COMPLETED_BLOCKING_DEPENDENCIES_SQL
    )
    assert "(occurred_at, id) > (%s, %s)" in coordination.EVENT_PAGE_AFTER_SQL
    assert "event_type = 'memory.activated'" in coordination.SELECT_COMPLETION_ACTIVATIONS_SQL
    assert "payload->>'lease_id' = %s" in coordination.SELECT_COMPLETION_ACTIVATIONS_SQL
    assert "ON CONFLICT (id) DO NOTHING" in (coordination.INSERT_MEMORY_OUTCOME_ASSOCIATION_SQL)
    assert "ORDER BY observed_at, id" in coordination.LIST_MEMORY_OUTCOME_ASSOCIATIONS_SQL


class _Cursor:
    def __init__(self, row: dict[str, Any]) -> None:
        self.row = row

    async def fetchone(self) -> dict[str, Any]:
        return self.row


class _RecordingConnection:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    async def execute(self, sql: str, params: tuple[Any, ...]) -> _Cursor:
        self.calls.append((sql, params))
        return _Cursor({"id": params[0]})


@pytest.mark.asyncio
async def test_event_and_outbox_share_id_and_store_the_full_event_envelope() -> None:
    actor = make_actor(_scope())
    event_id = _id()
    connection = _RecordingConnection()
    occurred_at = datetime.now(UTC)

    event = await coordination._append_event(
        connection,
        actor,
        event_id=event_id,
        outbox_id=_id(),
        event_type=EventType.TASK_CLAIMED,
        aggregate_type=AggregateType.TASK,
        aggregate_id=_id(),
        aggregate_version=2,
        occurred_at=occurred_at,
        payload={"lease_id": _id()},
        idempotency_key="claim-once",
    )

    assert len(connection.calls) == 2
    event_params = connection.calls[0][1]
    outbox_params = connection.calls[1][1]
    assert event_params[0] == event_id
    assert outbox_params[1] == event_id
    assert outbox_params[8] == event_id
    assert isinstance(outbox_params[9], Jsonb)
    assert outbox_params[9].obj == event.model_dump(mode="json")
