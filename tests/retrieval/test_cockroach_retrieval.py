from __future__ import annotations

from collections.abc import Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from conftest import make_actor, new_id
from swarmbrain.adapters.cockroach.memory import CockroachMemoryStore
from swarmbrain.adapters.cockroach.retrieval import (
    CockroachRetrievalGateway,
    cockroach_retrieval_gateways,
)
from swarmbrain.application.memory_policy import ConservativeMemoryPolicy
from swarmbrain.application.memory_service import MemoryService
from swarmbrain.application.retrieval_service import RetrievalService
from swarmbrain.domain.memory import RecallQuery
from swarmbrain.domain.retrieval import RetrievalPurpose, RetrievalSignal
from swarmbrain.retrieval import RetrievalPlanner


class _Cursor:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    async def fetchone(self) -> dict[str, Any] | None:
        return self.rows[0] if self.rows else None

    async def fetchall(self) -> list[dict[str, Any]]:
        return self.rows


class _Connection:
    def __init__(self, now: datetime) -> None:
        self.now = now
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    async def execute(self, statement: str, parameters: tuple[Any, ...] = ()) -> _Cursor:
        self.calls.append((statement, parameters))
        if "SELECT now() AS database_now" in statement:
            return _Cursor([{"database_now": self.now}])
        return _Cursor([])


class _Pool:
    def __init__(self, connection: _Connection) -> None:
        self.connection_value = connection

    @asynccontextmanager
    async def connection(self) -> Any:
        yield self.connection_value


class _SnapshotDatabase:
    def __init__(self, connection: _Connection, now: datetime) -> None:
        self.connection = connection
        self.now = now
        self.active = False
        self.snapshot_entries = 0
        self.connection_uses = 0
        self.pool = _Pool(connection)

    @asynccontextmanager
    async def retrieval_snapshot(self):  # type: ignore[no-untyped-def]
        if self.active:
            yield
            return
        self.active = True
        self.snapshot_entries += 1
        try:
            yield
        finally:
            self.active = False

    @asynccontextmanager
    async def retrieval_connection(self):  # type: ignore[no-untyped-def]
        assert self.active
        self.connection_uses += 1
        yield self.connection

    async def retrieval_now(self, _connection: object | None = None) -> datetime:
        assert self.active
        return self.now


class _ExactCandidateConnection(_Connection):
    def __init__(self, now: datetime, memory_id: str) -> None:
        super().__init__(now)
        self.memory_id = memory_id

    async def execute(self, statement: str, parameters: tuple[Any, ...] = ()) -> _Cursor:
        cursor = await super().execute(statement, parameters)
        if "FROM retrieval_exact_terms@" not in statement:
            return cursor
        return _Cursor(
            [
                {
                    "id": self.memory_id,
                    "version": 3,
                    "kind": "observation",
                    "metadata": {"retrieval": {"domain_lane": "handoff"}},
                    "recorded_from": self.now,
                    "domain_lane": None,
                    "lane_score": 1.0,
                }
            ]
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("signal", "query_text", "expected_sql"),
    (
        (RetrievalSignal.EXACT, "10000000-0000-4000-8000-000000000001", "retrieval_exact_terms"),
        (RetrievalSignal.LEXICAL, "serialization retry", "to_tsquery('simple'"),
        (RetrievalSignal.FUZZY, "RetryCoordintor", "lookup_text %% %s::STRING"),
    ),
)
async def test_cockroach_lane_applies_canonical_filters_before_limit(
    scope_ids: dict[str, str],
    signal: RetrievalSignal,
    query_text: str,
    expected_sql: str,
) -> None:
    actor = make_actor(scope_ids)
    now = datetime(2026, 8, 5, 12, tzinfo=UTC)
    connection = _Connection(now)
    database = SimpleNamespace(pool=_Pool(connection))
    query = RecallQuery(text=query_text)
    plan = RetrievalPlanner().plan(
        actor,
        query,
        purpose=RetrievalPurpose.INTERACTIVE_RECALL,
        available_signals=(signal,),
    )

    batch = await CockroachRetrievalGateway(database, signal).retrieve(  # type: ignore[arg-type]
        actor, plan, query
    )

    assert batch.candidates == ()
    lane_sql = "\n".join(
        statement for statement, _ in connection.calls if "SELECT now()" not in statement
    )
    assert expected_sql in lane_sql
    assert "m.tenant_id = %s" in lane_sql
    assert "m.project_id = %s" in lane_sql
    assert "m.repository_id = %s" in lane_sql
    assert "m.recorded_to IS NULL" in lane_sql
    assert "m.valid_from <= %s" in lane_sql
    assert "s1.review_state != 'rejected'" in lane_sql
    assert "s1.tenant_id = m.tenant_id" in lane_sql
    assert "s1.project_id = m.project_id" in lane_sql
    assert "s1.repository_id = m.repository_id" in lane_sql
    assert lane_sql.index("m.tenant_id = %s") < lane_sql.index("LIMIT %s")
    if signal is RetrievalSignal.EXACT:
        term_sql = next(
            statement
            for statement, _ in connection.calls
            if "FROM retrieval_exact_terms@" in statement
        )
        assert "SELECT DISTINCT m.id" in term_sql


@pytest.mark.asyncio
async def test_cockroach_fuzzy_query_is_normalized_and_bounded_before_sql(
    scope_ids: dict[str, str],
) -> None:
    actor = make_actor(scope_ids)
    now = datetime(2026, 8, 5, 12, tzinfo=UTC)
    connection = _Connection(now)
    database = SimpleNamespace(pool=_Pool(connection))
    raw_query = "STRAßE" + (" " * 20_000) + "x"
    query = RecallQuery(text=raw_query)
    plan = RetrievalPlanner().plan(
        actor,
        query,
        purpose=RetrievalPurpose.INTERACTIVE_RECALL,
        available_signals=(RetrievalSignal.FUZZY,),
    )

    await CockroachRetrievalGateway(  # type: ignore[arg-type]
        database,
        RetrievalSignal.FUZZY,
    ).retrieve(actor, plan, query)

    _, parameters = next(
        (statement, values)
        for statement, values in connection.calls
        if "retrieval_documents_lookup_trgm" in statement
    )
    assert "strasse" in parameters
    assert raw_query not in parameters
    assert max(len(value) for value in parameters if isinstance(value, str)) < 1_000


@pytest.mark.asyncio
async def test_cockroach_hydration_has_private_ids_and_no_recency_limit(
    scope_ids: dict[str, str],
) -> None:
    actor = make_actor(scope_ids)
    now = datetime(2026, 8, 5, 12, tzinfo=UTC)
    connection = _Connection(now)
    store = CockroachMemoryStore(SimpleNamespace(pool=_Pool(connection)))  # type: ignore[arg-type]
    candidate_ids = (new_id(), new_id())

    hydrated = await store.hydrate_recallable(
        actor,
        RecallQuery(text="retry"),
        candidate_ids,
    )

    assert hydrated == ()
    sql, parameters = next(
        (statement, params)
        for statement, params in connection.calls
        if "FROM memories AS m" in statement
    )
    assert "m.id = ANY(%s::UUID[])" in sql
    assert "LIMIT" not in sql.upper()
    assert "ORDER BY m.recorded_from" not in sql
    assert any(len(value) == 2 for value in parameters if isinstance(value, list))


@pytest.mark.asyncio
async def test_cockroach_exact_candidate_preserves_metadata_domain_lane(
    scope_ids: dict[str, str],
) -> None:
    actor = make_actor(scope_ids)
    now = datetime(2026, 8, 5, 12, tzinfo=UTC)
    memory_id = new_id()
    connection = _ExactCandidateConnection(now, memory_id)
    database = SimpleNamespace(pool=_Pool(connection))
    query = RecallQuery(text="handoff alias")
    plan = RetrievalPlanner().plan(
        actor,
        query,
        purpose=RetrievalPurpose.TASK_BOOTSTRAP,
        available_signals=(RetrievalSignal.EXACT,),
    )

    batch = await CockroachRetrievalGateway(  # type: ignore[arg-type]
        database,
        RetrievalSignal.EXACT,
    ).retrieve(actor, plan, query)

    assert len(batch.candidates) == 1
    assert batch.candidates[0].domain_lane == "handoff"


@pytest.mark.asyncio
async def test_cockroach_store_exposes_one_snapshot_to_lanes_and_hydration(
    scope_ids: dict[str, str],
) -> None:
    actor = make_actor(scope_ids)
    now = datetime(2026, 8, 5, 12, tzinfo=UTC)
    connection = _Connection(now)
    database = _SnapshotDatabase(connection, now)
    store = CockroachMemoryStore(database)  # type: ignore[arg-type]
    retrieval = RetrievalService(cockroach_retrieval_gateways(database), store)  # type: ignore[arg-type]

    class Provider:
        model_name = "snapshot-test"
        dimensions = 1024

        async def embed_query(self, _text: str) -> tuple[float, ...]:
            assert database.active is False
            return (1.0,) + (0.0,) * 1023

        async def embed_documents(self, _texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
            raise AssertionError("document embedding is not part of recall")

    service = MemoryService(
        store,
        ConservativeMemoryPolicy(),
        embeddings=Provider(),
        embedding_index=store,
        retrieval=retrieval,
        canonical_reader=store,
    )

    bundle = await service.recall(actor, RecallQuery(text="snapshot sentinel"))

    assert bundle.hits == ()
    assert bundle.generated_at == now
    assert database.snapshot_entries == 1
    assert database.connection_uses == 5
    assert database.active is False
