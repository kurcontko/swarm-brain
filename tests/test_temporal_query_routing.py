from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from conftest import make_actor, make_task, new_id
from swarmbrain.adapters.cockroach.retrieval import build_recall_predicates
from swarmbrain.application.runtime import build_in_memory_runtime
from swarmbrain.domain.activation import ActivationTrigger, MemoryActivationCommand
from swarmbrain.domain.exploration import ReadExpandMemoryRequest
from swarmbrain.domain.memory import (
    Memory,
    MemoryKind,
    MemoryState,
    RecallQuery,
    Visibility,
)
from swarmbrain.domain.retrieval import RetrievalPurpose
from swarmbrain.domain.tasks import ClaimTaskCommand
from swarmbrain.retrieval import RetrievalPlanner

SECRET = "0123456789abcdef-temporal-routing"
VALID_FROM = datetime(2024, 1, 1, tzinfo=UTC)
VALID_TO = datetime(2024, 1, 5, tzinfo=UTC)
RUNTIME_NOW = datetime(2026, 8, 10, tzinfo=UTC)


def _memory(
    actor: object,
    *,
    content: str,
    valid_from: datetime,
    valid_to: datetime,
    task_id: str | None = None,
    recorded_from: datetime = datetime(2023, 1, 1, tzinfo=UTC),
) -> Memory:
    return Memory(
        memory_id=new_id(),
        tenant_id=actor.tenant_id,  # type: ignore[attr-defined]
        project_id=actor.project_id,  # type: ignore[attr-defined]
        repository_id=actor.repository_id,  # type: ignore[attr-defined]
        swarm_id=actor.swarm_id,  # type: ignore[attr-defined]
        run_id=actor.run_id,  # type: ignore[attr-defined]
        task_id=task_id,
        author_agent_id=new_id(),
        kind=MemoryKind.PROCEDURE,
        state=MemoryState.CONFIRMED,
        visibility=Visibility.TASK if task_id is not None else Visibility.REPOSITORY,
        content=content,
        valid_from=valid_from,
        valid_to=valid_to,
        recorded_from=recorded_from,
    )


def test_referenced_validity_contract_is_explicit_and_unambiguous() -> None:
    recorded_at = datetime(2025, 1, 1, tzinfo=UTC)
    query = RecallQuery(
        text="temporal sentinel",
        referenced_valid_from=VALID_FROM,
        referenced_valid_to=VALID_TO,
        recorded_at=recorded_at,
    )

    assert query.referenced_valid_from == VALID_FROM
    assert query.referenced_valid_to == VALID_TO
    assert query.recorded_at == recorded_at
    with pytest.raises(ValidationError, match="must be provided together"):
        RecallQuery(text="temporal sentinel", referenced_valid_from=VALID_FROM)
    with pytest.raises(ValidationError, match="must be later"):
        RecallQuery(
            text="temporal sentinel",
            referenced_valid_from=VALID_TO,
            referenced_valid_to=VALID_FROM,
        )
    with pytest.raises(ValidationError, match="world_at cannot be combined"):
        RecallQuery(
            text="temporal sentinel",
            world_at=VALID_FROM,
            referenced_valid_from=VALID_FROM,
            referenced_valid_to=VALID_TO,
        )


@pytest.mark.asyncio
async def test_in_memory_interval_overlap_replaces_default_current_point_filter(
    scope_ids: dict[str, str],
) -> None:
    runtime = build_in_memory_runtime(
        SECRET,
        clock=lambda: RUNTIME_NOW,
    )
    actor = make_actor(scope_ids)
    overlapping = _memory(
        actor,
        content="temporal sentinel overlap",
        valid_from=datetime(2024, 1, 2, tzinfo=UTC),
        valid_to=datetime(2024, 1, 4, tzinfo=UTC),
    )
    touching_before = _memory(
        actor,
        content="temporal sentinel touching before",
        valid_from=datetime(2023, 12, 20, tzinfo=UTC),
        valid_to=VALID_FROM,
    )
    touching_after = _memory(
        actor,
        content="temporal sentinel touching after",
        valid_from=VALID_TO,
        valid_to=datetime(2024, 1, 8, tzinfo=UTC),
    )
    recorded_later = _memory(
        actor,
        content="temporal sentinel recorded later",
        valid_from=datetime(2024, 1, 2, tzinfo=UTC),
        valid_to=datetime(2024, 1, 4, tzinfo=UTC),
        recorded_from=datetime(2025, 1, 1, tzinfo=UTC),
    )
    assert runtime.kernel is not None
    for memory in (overlapping, touching_before, touching_after, recorded_later):
        runtime.kernel.memories[memory.memory_id] = memory

    current = await runtime.memory.recall(actor, RecallQuery(text="temporal sentinel"))
    interval = await runtime.memory.recall(
        actor,
        RecallQuery(
            text="temporal sentinel",
            referenced_valid_from=VALID_FROM,
            referenced_valid_to=VALID_TO,
            recorded_at=datetime(2024, 6, 1, tzinfo=UTC),
        ),
    )
    point = await runtime.memory.recall(
        actor,
        RecallQuery(text="temporal sentinel", world_at=VALID_TO),
    )

    assert current.hits == ()
    assert {hit.memory.memory_id for hit in interval.hits} == {overlapping.memory_id}
    assert {hit.memory.memory_id for hit in point.hits} == {touching_after.memory_id}


def test_cockroach_predicates_route_half_open_overlap_and_keep_system_time_orthogonal(
    scope_ids: dict[str, str],
) -> None:
    actor = make_actor(scope_ids)
    now = datetime(2026, 8, 9, tzinfo=UTC)
    recorded_at = datetime(2025, 6, 1, tzinfo=UTC)
    query = RecallQuery(
        text="temporal sentinel",
        referenced_valid_from=VALID_FROM,
        referenced_valid_to=VALID_TO,
        recorded_at=recorded_at,
    )

    predicates = build_recall_predicates(actor, query, now=now)
    sql = " AND ".join(predicates.clauses)

    assert "m.recorded_from <= %s" in sql
    assert "%s < m.recorded_to" in sql
    assert "m.valid_from < %s" in sql
    assert "%s < m.valid_to" in sql
    assert "m.valid_from <= %s" not in sql
    assert predicates.parameters[-4:] == (recorded_at, recorded_at, VALID_TO, VALID_FROM)

    plan = RetrievalPlanner().plan(
        actor,
        query,
        purpose=RetrievalPurpose.HISTORICAL_AUDIT,
        available_signals=(),
    )
    assert plan.intent == "historical"
    assert plan.referenced_valid_from == VALID_FROM
    assert plan.referenced_valid_to == VALID_TO
    assert plan.recorded_at == recorded_at

    point = build_recall_predicates(actor, RecallQuery(text="sentinel"), now=now)
    point_sql = " AND ".join(point.clauses)
    assert "m.valid_from <= %s" in point_sql
    assert point.parameters[-2:] == (now, now)


@pytest.mark.asyncio
async def test_activation_and_exact_read_expand_reapply_referenced_interval(
    scope_ids: dict[str, str],
) -> None:
    runtime = build_in_memory_runtime(
        SECRET,
        clock=lambda: RUNTIME_NOW,
    )
    actor = make_actor(scope_ids)
    task = make_task(
        scope_ids,
        title="Temporal routing task",
        created_at=RUNTIME_NOW,
    )
    await runtime.kernel.add_task(task)  # type: ignore[union-attr]
    overlapping = _memory(
        actor,
        task_id=task.task_id,
        content="temporal activation sentinel overlap",
        valid_from=datetime(2024, 1, 2, tzinfo=UTC),
        valid_to=datetime(2024, 1, 4, tzinfo=UTC),
    )
    outside = _memory(
        actor,
        task_id=task.task_id,
        content="temporal activation sentinel outside",
        valid_from=VALID_TO,
        valid_to=datetime(2024, 1, 8, tzinfo=UTC),
    )
    assert runtime.kernel is not None
    runtime.kernel.memories[overlapping.memory_id] = overlapping
    runtime.kernel.memories[outside.memory_id] = outside
    claimed = await runtime.coordination.claim(
        actor,
        ClaimTaskCommand(idempotency_key="claim-temporal-routing", task_id=task.task_id),
    )

    activation = await runtime.coordination.activate_memory(
        actor,
        MemoryActivationCommand(
            task_id=task.task_id,
            lease_id=claimed.lease.lease_id,
            trigger=ActivationTrigger.EXPLICIT,
            query_text="temporal activation sentinel",
            referenced_valid_from=VALID_FROM,
            referenced_valid_to=VALID_TO,
        ),
    )
    expanded = await runtime.coordination.read_expand_memory(
        actor,
        ReadExpandMemoryRequest(
            task_id=task.task_id,
            lease_id=claimed.lease.lease_id,
            query_text="temporal activation sentinel",
            memory_ids=(overlapping.memory_id, outside.memory_id),
            max_depth=0,
            referenced_valid_from=VALID_FROM,
            referenced_valid_to=VALID_TO,
        ),
    )

    assert activation.activation.memory_ids == (overlapping.memory_id,)
    assert activation.activation_context is not None
    assert f"valid_from={overlapping.valid_from.isoformat()}" in activation.activation_context
    assert f"valid_to={overlapping.valid_to.isoformat()}" in activation.activation_context
    assert expanded.memory_ids == (overlapping.memory_id,)
    assert outside.memory_id not in expanded.context
