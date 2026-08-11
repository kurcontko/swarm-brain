"""Focused planning and adapter-parity tests for explicit temporal retrieval."""

from __future__ import annotations

import math
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from conftest import make_actor, new_id
from swarmbrain.adapters.cockroach.retrieval import (
    CockroachTemporalRetrievalGateway,
    cockroach_hybrid_retrieval_gateways,
    cockroach_retrieval_gateways,
)
from swarmbrain.adapters.memory.in_memory import InMemoryKernel
from swarmbrain.adapters.memory.retrieval import (
    InMemoryTemporalRetrievalGateway,
    in_memory_hybrid_retrieval_gateways,
    in_memory_retrieval_gateways,
)
from swarmbrain.application.retrieval_service import RetrievalService
from swarmbrain.domain.memory import Memory, MemoryState, RecallQuery, Visibility
from swarmbrain.domain.retrieval import RetrievalPurpose, RetrievalSignal
from swarmbrain.retrieval.planner import (
    OCCURRENCE_TEMPORAL_PROJECTION_ID,
    OCCURRENCE_TEMPORAL_PROJECTION_VERSION,
    TEMPORAL_PROJECTION_ID,
    TEMPORAL_PROJECTION_VERSION,
    RetrievalPlanner,
)


class _Cursor:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    async def fetchall(self) -> list[dict[str, Any]]:
        return self.rows


class _TemporalConnection:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    async def execute(self, statement: str, parameters: tuple[Any, ...] = ()) -> _Cursor:
        self.calls.append((statement, parameters))
        if (
            "EXTRACT(EPOCH FROM m.valid_from)" in statement
            or "EXTRACT(EPOCH FROM m.occurred_at)" in statement
        ):
            return _Cursor(self.rows)
        return _Cursor([])


class _Pool:
    def __init__(self, connection: _TemporalConnection) -> None:
        self.connection_value = connection

    @asynccontextmanager
    async def connection(self):  # type: ignore[no-untyped-def]
        yield self.connection_value


def _memory(
    actor: object,
    *,
    memory_id: str,
    valid_from: datetime,
    valid_to: datetime,
    occurred_at: datetime | None = None,
    recorded_from: datetime = datetime(2023, 1, 1, tzinfo=UTC),
) -> Memory:
    return Memory(
        memory_id=memory_id,
        tenant_id=actor.tenant_id,  # type: ignore[attr-defined]
        project_id=actor.project_id,  # type: ignore[attr-defined]
        repository_id=actor.repository_id,  # type: ignore[attr-defined]
        swarm_id=actor.swarm_id,  # type: ignore[attr-defined]
        run_id=actor.run_id,  # type: ignore[attr-defined]
        author_agent_id=actor.agent_id,  # type: ignore[attr-defined]
        kind="observation",
        state=MemoryState.CONFIRMED,
        visibility=Visibility.REPOSITORY,
        content=f"temporal lane {memory_id}",
        occurred_at=occurred_at,
        valid_from=valid_from,
        valid_to=valid_to,
        recorded_from=recorded_from,
    )


def _plan(actor: object, query: RecallQuery):  # type: ignore[no-untyped-def]
    return RetrievalPlanner().plan(
        actor,  # type: ignore[arg-type]
        query,
        purpose=RetrievalPurpose.HISTORICAL_AUDIT,
        available_signals=(RetrievalSignal.TEMPORAL,),
    )


def test_planner_selects_temporal_only_for_explicit_valid_time(
    scope_ids: dict[str, str],
) -> None:
    actor = make_actor(scope_ids)
    point = datetime(2024, 1, 10, tzinfo=UTC)
    recorded = datetime(2025, 1, 10, tzinfo=UTC)

    default_plan = _plan(actor, RecallQuery(text="temporal selection"))
    recorded_plan = _plan(
        actor,
        RecallQuery(text="temporal selection", recorded_at=recorded),
    )
    point_plan = _plan(actor, RecallQuery(text="temporal selection", world_at=point))
    interval_plan = _plan(
        actor,
        RecallQuery(
            text="temporal selection",
            referenced_valid_from=point - timedelta(days=2),
            referenced_valid_to=point + timedelta(days=2),
        ),
    )
    occurrence_plan = _plan(
        actor,
        RecallQuery(
            text="temporal selection",
            occurrence_time_prior_from=point - timedelta(days=2),
            occurrence_time_prior_to=point + timedelta(days=2),
        ),
    )

    assert RetrievalSignal.TEMPORAL not in default_plan.signal_lanes
    assert RetrievalSignal.TEMPORAL not in recorded_plan.signal_lanes
    for plan in (point_plan, interval_plan, occurrence_plan):
        assert plan.signal_lanes == frozenset({RetrievalSignal.TEMPORAL})
        assert 1 <= plan.lane_budgets[RetrievalSignal.TEMPORAL.value] <= 200
        assert plan.lane_weights[RetrievalSignal.TEMPORAL.value] > 0.0


@pytest.mark.asyncio
async def test_occurrence_prior_keeps_retrospective_and_unknown_time_memories_eligible(
    scope_ids: dict[str, str],
) -> None:
    actor = make_actor(scope_ids)
    now = datetime(2026, 8, 10, tzinfo=UTC)
    occurred_at = datetime(2024, 1, 10, tzinfo=UTC)
    prior_from = occurred_at - timedelta(days=2)
    prior_to = occurred_at + timedelta(days=2)
    retrospective_id, unknown_id = new_id(), new_id()
    retrospective = _memory(
        actor,
        memory_id=retrospective_id,
        occurred_at=occurred_at,
        # The event was learned much later. Canonical current validity starts
        # at observation time and must not be rewritten to the event date.
        valid_from=datetime(2026, 8, 1, tzinfo=UTC),
        valid_to=datetime(2026, 9, 1, tzinfo=UTC),
    )
    unknown = _memory(
        actor,
        memory_id=unknown_id,
        valid_from=datetime(2026, 8, 2, tzinfo=UTC),
        valid_to=datetime(2026, 9, 1, tzinfo=UTC),
    )
    query = RecallQuery(
        text="retrospective event",
        occurrence_time_prior_from=prior_from,
        occurrence_time_prior_to=prior_to,
    )
    plan = _plan(actor, query)

    kernel = InMemoryKernel(clock=lambda: now)
    kernel.memories[retrospective_id] = retrospective
    kernel.memories[unknown_id] = unknown
    eligible = await kernel.recallable_memories(actor, query)
    in_memory = await InMemoryTemporalRetrievalGateway(kernel).retrieve(actor, plan, query)

    assert {memory.memory_id for memory in eligible} == {retrospective_id, unknown_id}
    assert [candidate.canonical_id for candidate in in_memory.candidates] == [retrospective_id]
    assert in_memory.candidates[0].projection_id == OCCURRENCE_TEMPORAL_PROJECTION_ID
    assert in_memory.candidates[0].projection_version == OCCURRENCE_TEMPORAL_PROJECTION_VERSION
    assert in_memory.candidates[0].reasons == (
        "temporal_occurrence_distance",
        "temporal_target:occurrence_interval_center",
    )

    rows = [
        {
            "id": retrospective.memory_id,
            "version": retrospective.version,
            "kind": retrospective.kind,
            "metadata": retrospective.metadata,
            "occurred_at": retrospective.occurred_at,
            "recorded_from": retrospective.recorded_from,
        }
    ]
    connection = _TemporalConnection(rows)
    database = SimpleNamespace(
        pool=_Pool(connection),
        retrieval_now=lambda _connection=None: now,
    )
    cockroach = await CockroachTemporalRetrievalGateway(database).retrieve(  # type: ignore[arg-type]
        actor,
        plan,
        query,
    )

    assert [candidate.canonical_id for candidate in cockroach.candidates] == [retrospective_id]
    assert cockroach.candidates[0].raw_score == 1.0
    assert len(connection.calls) == 1
    sql, parameters = connection.calls[0]
    assert "m.occurred_at IS NOT NULL" in sql
    assert "EXTRACT(EPOCH FROM m.occurred_at)" in sql
    assert "m.valid_from <= %s" in sql
    assert now in parameters
    assert parameters[-2] == occurred_at
    assert prior_from not in parameters
    assert prior_to not in parameters


@pytest.mark.asyncio
async def test_default_retrieval_ranking_does_not_read_occurrence_time(
    scope_ids: dict[str, str],
) -> None:
    actor = make_actor(scope_ids)
    now = datetime(2026, 8, 10, tzinfo=UTC)
    first_id, second_id = new_id(), new_id()
    baseline_memories = (
        _memory(
            actor,
            memory_id=first_id,
            valid_from=now - timedelta(days=2),
            valid_to=now + timedelta(days=2),
        ).model_copy(update={"content": "cache retry alpha alpha"}),
        _memory(
            actor,
            memory_id=second_id,
            valid_from=now - timedelta(days=1),
            valid_to=now + timedelta(days=2),
        ).model_copy(update={"content": "cache retry beta"}),
    )
    query = RecallQuery(text="cache retry", limit=10)

    async def execute(with_occurrence: bool):  # type: ignore[no-untyped-def]
        kernel = InMemoryKernel(clock=lambda: now)
        for index, memory in enumerate(baseline_memories):
            occurred_at = now - timedelta(days=300 + index) if with_occurrence else None
            kernel.memories[memory.memory_id] = memory.model_copy(
                update={"occurred_at": occurred_at}
            )
        service = RetrievalService(in_memory_retrieval_gateways(kernel), kernel)
        return await service.execute(actor, query)

    baseline = await execute(False)
    enriched = await execute(True)

    def ranking(execution: object) -> tuple[tuple[object, ...], ...]:
        return tuple(
            (hit.memory.memory_id, hit.score, hit.reasons)
            for hit in execution.bundle.hits  # type: ignore[attr-defined]
        )

    assert RetrievalSignal.TEMPORAL not in baseline.trace.plan.signal_lanes
    assert RetrievalSignal.TEMPORAL not in enriched.trace.plan.signal_lanes
    assert ranking(enriched) == ranking(baseline)


@pytest.mark.asyncio
async def test_interval_lane_ranks_from_interval_center_with_deterministic_ties(
    scope_ids: dict[str, str],
) -> None:
    actor = make_actor(scope_ids)
    interval_from = datetime(2024, 1, 4, tzinfo=UTC)
    interval_to = datetime(2024, 1, 8, tzinfo=UTC)
    center_id, left_id, right_id, touching_id = (new_id() for _ in range(4))
    kernel = InMemoryKernel(clock=lambda: datetime(2026, 8, 10, tzinfo=UTC))
    memories = (
        _memory(
            actor,
            memory_id=center_id,
            valid_from=datetime(2024, 1, 6, tzinfo=UTC),
            valid_to=datetime(2024, 1, 7, tzinfo=UTC),
        ),
        _memory(
            actor,
            memory_id=left_id,
            valid_from=datetime(2024, 1, 5, tzinfo=UTC),
            valid_to=datetime(2024, 1, 9, tzinfo=UTC),
            recorded_from=datetime(2023, 1, 2, tzinfo=UTC),
        ),
        _memory(
            actor,
            memory_id=right_id,
            valid_from=datetime(2024, 1, 7, tzinfo=UTC),
            valid_to=datetime(2024, 1, 9, tzinfo=UTC),
            recorded_from=datetime(2023, 1, 3, tzinfo=UTC),
        ),
        _memory(
            actor,
            memory_id=touching_id,
            valid_from=interval_to,
            valid_to=interval_to + timedelta(days=1),
        ),
    )
    for memory in memories:
        kernel.memories[memory.memory_id] = memory
    query = RecallQuery(
        text="interval center",
        referenced_valid_from=interval_from,
        referenced_valid_to=interval_to,
    )

    batch = await InMemoryTemporalRetrievalGateway(kernel).retrieve(
        actor,
        _plan(actor, query),
        query,
    )

    assert [candidate.canonical_id for candidate in batch.candidates] == [
        center_id,
        left_id,
        right_id,
    ]
    assert [candidate.raw_score for candidate in batch.candidates] == [1.0, 0.5, 0.5]
    assert touching_id not in {candidate.canonical_id for candidate in batch.candidates}
    assert all(
        candidate.reasons == ("temporal_valid_from_distance", "temporal_target:interval_center")
        for candidate in batch.candidates
    )


@pytest.mark.asyncio
async def test_world_time_lane_has_in_memory_cockroach_parity_and_one_sql_scope(
    scope_ids: dict[str, str],
) -> None:
    actor = make_actor(scope_ids)
    target = datetime(2024, 1, 10, tzinfo=UTC)
    recorded_at = datetime(2025, 6, 1, tzinfo=UTC)
    exact_id, near_id, far_id = (new_id() for _ in range(3))
    memories = (
        _memory(
            actor,
            memory_id=exact_id,
            valid_from=target,
            valid_to=target + timedelta(days=2),
        ),
        _memory(
            actor,
            memory_id=near_id,
            valid_from=target - timedelta(days=1),
            valid_to=target + timedelta(days=2),
        ),
        _memory(
            actor,
            memory_id=far_id,
            valid_from=target - timedelta(days=9),
            valid_to=target + timedelta(days=2),
        ),
    )
    query = RecallQuery(
        text="world-time target",
        world_at=target,
        recorded_at=recorded_at,
    )
    plan = _plan(actor, query)

    kernel = InMemoryKernel(clock=lambda: datetime(2026, 8, 10, tzinfo=UTC))
    for memory in memories:
        kernel.memories[memory.memory_id] = memory
    in_memory = await InMemoryTemporalRetrievalGateway(kernel).retrieve(actor, plan, query)

    rows = [
        {
            "id": memory.memory_id,
            "version": memory.version,
            "kind": memory.kind,
            "metadata": memory.metadata,
            "valid_from": memory.valid_from,
            "recorded_from": memory.recorded_from,
        }
        for memory in reversed(memories)
    ]
    connection = _TemporalConnection(rows)
    database = SimpleNamespace(pool=_Pool(connection))
    cockroach = await CockroachTemporalRetrievalGateway(database).retrieve(  # type: ignore[arg-type]
        actor,
        plan,
        query,
    )

    def comparable(batch: object) -> tuple[tuple[object, ...], ...]:
        return tuple(
            (
                candidate.canonical_id,
                candidate.rank,
                candidate.raw_score,
                candidate.projection_id,
                candidate.projection_version,
                candidate.reasons,
            )
            for candidate in batch.candidates  # type: ignore[attr-defined]
        )

    assert comparable(cockroach) == comparable(in_memory)
    assert [candidate.raw_score for candidate in cockroach.candidates] == [1.0, 0.5, 0.1]
    assert all(
        score is not None and math.isfinite(score) and 0.0 < score <= 1.0
        for score in (candidate.raw_score for candidate in cockroach.candidates)
    )
    assert all(
        candidate.projection_id == TEMPORAL_PROJECTION_ID for candidate in cockroach.candidates
    )
    assert all(
        candidate.projection_version == TEMPORAL_PROJECTION_VERSION
        for candidate in cockroach.candidates
    )

    assert len(connection.calls) == 1
    sql, parameters = connection.calls[0]
    assert sql.count("FROM memories@primary AS m") == 1
    assert "retrieval_documents" not in sql
    assert "m.tenant_id = %s" in sql
    assert "m.project_id = %s" in sql
    assert "m.repository_id = %s" in sql
    assert "m.recorded_from <= %s" in sql
    assert "m.valid_from <= %s" in sql
    assert "s1.review_state != 'rejected'" in sql
    assert "EXTRACT(EPOCH FROM m.valid_from)" in sql
    assert sql.index("m.tenant_id = %s") < sql.index("LIMIT %s")
    assert parameters[-2] == target
    assert parameters[-1] == plan.lane_budgets[RetrievalSignal.TEMPORAL.value] + 1
    assert recorded_at in parameters


@pytest.mark.asyncio
async def test_unselected_temporal_gateway_returns_empty_without_io(
    scope_ids: dict[str, str],
) -> None:
    actor = make_actor(scope_ids)
    query = RecallQuery(
        text="system time is not event time",
        recorded_at=datetime(2025, 1, 1, tzinfo=UTC),
    )
    plan = _plan(actor, query)
    kernel = InMemoryKernel(clock=lambda: datetime(2026, 8, 10, tzinfo=UTC))
    connection = _TemporalConnection([])
    database = SimpleNamespace(pool=_Pool(connection))

    in_memory = await InMemoryTemporalRetrievalGateway(kernel).retrieve(actor, plan, query)
    cockroach = await CockroachTemporalRetrievalGateway(database).retrieve(  # type: ignore[arg-type]
        actor,
        plan,
        query,
    )

    assert RetrievalSignal.TEMPORAL not in plan.signal_lanes
    assert in_memory.candidates == cockroach.candidates == ()
    assert in_memory.examined_count == cockroach.examined_count == 0
    assert connection.calls == []


@pytest.mark.asyncio
async def test_in_memory_temporal_lane_enforces_the_bounded_plan_budget(
    scope_ids: dict[str, str],
) -> None:
    actor = make_actor(scope_ids)
    target = datetime(2024, 2, 1, tzinfo=UTC)
    query = RecallQuery(text="bounded temporal lane", world_at=target, limit=1)
    plan = _plan(actor, query)
    budget = plan.lane_budgets[RetrievalSignal.TEMPORAL.value]
    kernel = InMemoryKernel(clock=lambda: datetime(2026, 8, 10, tzinfo=UTC))
    for hours_before in range(budget + 1):
        memory = _memory(
            actor,
            memory_id=new_id(),
            valid_from=target - timedelta(hours=hours_before),
            valid_to=target + timedelta(days=1),
        )
        kernel.memories[memory.memory_id] = memory

    batch = await InMemoryTemporalRetrievalGateway(kernel).retrieve(actor, plan, query)

    assert budget == 16
    assert len(batch.candidates) == budget
    assert batch.examined_count == budget + 1
    assert batch.truncated is True
    assert tuple(candidate.rank for candidate in batch.candidates) == tuple(range(1, budget + 1))


def test_default_and_hybrid_factories_expose_exactly_one_temporal_lane() -> None:
    kernel = InMemoryKernel()
    database = SimpleNamespace(pool=_Pool(_TemporalConnection([])))
    gateway_sets = (
        in_memory_retrieval_gateways(kernel),
        in_memory_hybrid_retrieval_gateways(kernel, kernel),
        cockroach_retrieval_gateways(database),  # type: ignore[arg-type]
        cockroach_hybrid_retrieval_gateways(database),  # type: ignore[arg-type]
    )

    for gateways in gateway_sets:
        signals = tuple(gateway.signal for gateway in gateways)
        assert signals.count(RetrievalSignal.TEMPORAL) == 1
        assert len(signals) == len(set(signals))
