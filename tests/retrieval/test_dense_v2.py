from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from conftest import make_actor, new_id
from swarmbrain.adapters.cockroach.retrieval import CockroachDenseRetrievalGateway
from swarmbrain.adapters.extraction.in_memory import InMemoryWorkStore
from swarmbrain.adapters.memory import InMemoryKernel, in_memory_hybrid_retrieval_gateways
from swarmbrain.application.memory_policy import ConservativeMemoryPolicy
from swarmbrain.application.retrieval_service import RetrievalService
from swarmbrain.domain.memory import EmbeddingVector, RecallQuery, RememberCommand, Visibility
from swarmbrain.domain.retrieval import DenseQuery, RetrievalPurpose, RetrievalSignal
from swarmbrain.retrieval import RetrievalPlanner, dense_projection_signature


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
    async def connection(self):  # type: ignore[no-untyped-def]
        yield self.connection_value


@pytest.mark.asyncio
async def test_dense_candidates_join_weighted_rrf_and_trace_after_hard_prefilter(
    scope_ids: dict[str, str],
) -> None:
    actor = make_actor(scope_ids)
    kernel = InMemoryKernel()
    index = InMemoryWorkStore()
    relevant = await kernel.remember(
        actor,
        RememberCommand(
            idempotency_key="dense-v2-relevant",
            kind="observation",
            content="lease fencing invariant",
        ),
        ConservativeMemoryPolicy(),
    )
    decoy = await kernel.remember(
        actor,
        RememberCommand(
            idempotency_key="dense-v2-decoy",
            kind="observation",
            content="unrelated artifact retention",
        ),
        ConservativeMemoryPolicy(),
    )
    assert relevant.memory is not None and decoy.memory is not None
    await index.upsert_embeddings(
        actor,
        (
            EmbeddingVector(
                memory_id=relevant.memory.memory_id,
                model="mapped-v2",
                dimensions=4,
                values=(1.0, 0.0, 0.0, 0.0),
            ),
            EmbeddingVector(
                memory_id=decoy.memory.memory_id,
                model="mapped-v2",
                dimensions=4,
                values=(0.0, 1.0, 0.0, 0.0),
            ),
        ),
        idempotency_key="dense-v2-index",
    )
    service = RetrievalService(in_memory_hybrid_retrieval_gateways(kernel, index), kernel)

    execution = await service.execute(
        actor,
        RecallQuery(text="conceptual query with no lexical overlap"),
        dense_query=DenseQuery(
            model="mapped-v2",
            dimensions=4,
            values=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    assert [hit.memory.memory_id for hit in execution.bundle.hits] == [relevant.memory.memory_id]
    # 0.8 is the dense lane weight over the strongest configured lane weight,
    # asserted on the fusion stage that owns it.  The published hit score is
    # taken after relevance reranking and is a different quantity.
    assert execution.trace.fused_candidates[0].normalized_score == pytest.approx(0.8)
    assert "signal:dense" in execution.bundle.hits[0].reasons
    assert "semantic_match" in execution.bundle.hits[0].reasons
    dense_batch = next(
        batch for batch in execution.trace.batches if batch.lane is RetrievalSignal.DENSE
    )
    assert dense_batch.degraded is False
    assert dense_batch.candidates[0].resource_version == relevant.memory.version
    assert execution.trace.fused_candidates[0].contributions[0].lane is RetrievalSignal.DENSE


@pytest.mark.asyncio
async def test_dense_provider_failure_is_an_observable_degraded_lane(
    scope_ids: dict[str, str],
) -> None:
    actor = make_actor(scope_ids)
    kernel = InMemoryKernel()
    index = InMemoryWorkStore()
    service = RetrievalService(in_memory_hybrid_retrieval_gateways(kernel, index), kernel)

    execution = await service.execute(
        actor,
        RecallQuery(text="provider outage"),
        dense_query=DenseQuery(
            model="mapped-v2",
            dimensions=4,
            unavailable_reason="ProviderUnavailable",
        ),
    )

    assert execution.bundle.hits == ()
    assert execution.trace.degraded_lanes == frozenset({RetrievalSignal.DENSE})
    dense_batch = next(
        batch for batch in execution.trace.batches if batch.lane is RetrievalSignal.DENSE
    )
    assert dense_batch.degradation_reason == "ProviderUnavailable"


class _DenseConnection(_Connection):
    def __init__(self, now: datetime, memory_id: str, repository_id: str) -> None:
        super().__init__(now)
        self.memory_id = memory_id
        self.repository_id = repository_id
        self.digest = b"dense-v2-content-digest"

    async def execute(self, statement: str, parameters: tuple[Any, ...] = ()) -> _Cursor:
        cursor = await super().execute(statement, parameters)
        if "FROM retrieval_vectors_1024@" in statement:
            return _Cursor(
                [
                    {
                        "id": self.memory_id,
                        "resource_id": self.memory_id,
                        "resource_version": 7,
                        "kind": "observation",
                        "metadata": {},
                        "recorded_from": self.now,
                        "domain_lane": "knowledge",
                        "model": "dense-live-v2",
                        "dimensions": 1024,
                        "projection_id": "memory-content-v1:current:cosine",
                        "projection_signature": dense_projection_signature("dense-live-v2", 1024),
                        "indexed_at": self.now,
                        "content_sha256": self.digest,
                        "distance": 0.1,
                    }
                ]
            )
        if "FROM memories@primary AS m" in statement:
            return _Cursor(
                [
                    {
                        "id": self.memory_id,
                        "repository_id": self.repository_id,
                        "run_id": new_id(),
                        "task_id": None,
                        "visibility": "repository",
                        "version": 7,
                        "kind": "observation",
                        "metadata": {},
                        "recorded_from": self.now,
                        "normalized_sha256": self.digest,
                    }
                ]
            )
        return cursor


class _AdaptiveDenseConnection(_Connection):
    def __init__(self, now: datetime, repository_id: str) -> None:
        super().__init__(now)
        self.repository_id = repository_id
        self.memory_ids = tuple(new_id() for _ in range(33))
        self.relevant_id = self.memory_ids[-1]
        self.digest = b"adaptive-dense-content-digest"

    async def execute(self, statement: str, parameters: tuple[Any, ...] = ()) -> _Cursor:
        cursor = await super().execute(statement, parameters)
        if "retrieval_vectors_1024@retrieval_vectors_1024_ann_v2" in statement:
            limit = int(parameters[-1])
            return _Cursor(
                [
                    {
                        "id": memory_id,
                        "resource_id": memory_id,
                        "resource_version": 1,
                        "domain_lane": "knowledge",
                        "model": "dense-live-v2",
                        "dimensions": 1024,
                        "projection_id": "memory-content-v1:current:cosine",
                        "projection_signature": dense_projection_signature("dense-live-v2", 1024),
                        "indexed_at": self.now,
                        "content_sha256": self.digest,
                        "distance": 0.1 + (position / 1000),
                    }
                    for position, memory_id in enumerate(self.memory_ids[:limit])
                ]
            )
        if "FROM memories@primary AS m" in statement:
            selected = {
                str(value)
                for parameter in parameters
                if isinstance(parameter, list)
                for value in parameter
            }
            if self.relevant_id not in selected:
                return _Cursor([])
            return _Cursor(
                [
                    {
                        "id": self.relevant_id,
                        "repository_id": self.repository_id,
                        "run_id": new_id(),
                        "task_id": None,
                        "visibility": "repository",
                        "version": 1,
                        "kind": "observation",
                        "metadata": {},
                        "recorded_from": self.now,
                        "normalized_sha256": self.digest,
                    }
                ]
            )
        return cursor


@pytest.mark.asyncio
async def test_cockroach_dense_ann_binds_full_prefix_then_validates_canonical_rows(
    scope_ids: dict[str, str],
) -> None:
    actor = make_actor(scope_ids)
    now = datetime(2026, 8, 6, 12, tzinfo=UTC)
    memory_id = new_id()
    connection = _DenseConnection(now, memory_id, actor.repository_id)
    database = SimpleNamespace(pool=_Pool(connection))
    query = RecallQuery(text="semantic retry policy")
    plan = RetrievalPlanner().plan(
        actor,
        query,
        purpose=RetrievalPurpose.INTERACTIVE_RECALL,
        available_signals=(RetrievalSignal.DENSE,),
    )
    dense_query = DenseQuery(
        model="dense-live-v2",
        dimensions=1024,
        values=(1.0,) + (0.0,) * 1023,
    )
    gateway = CockroachDenseRetrievalGateway(  # type: ignore[arg-type]
        database,
        min_similarity=0.2,
        beam_size=64,
    )

    batch = await gateway.retrieve(actor, plan, query, dense_query)

    assert [candidate.canonical_id for candidate in batch.candidates] == [memory_id]
    assert batch.candidates[0].raw_score == pytest.approx(0.9)
    assert batch.candidates[0].resource_version == 7
    assert any(
        statement.strip().startswith("SET LOCAL vector_search_beam_size")
        and statement.strip().endswith("= 64")
        and parameters == ()
        for statement, parameters in connection.calls
    )
    lane_statements = [
        statement
        for statement, _ in connection.calls
        if "FROM retrieval_vectors_1024@" in statement
    ]
    assert lane_statements
    for statement in lane_statements:
        assert "retrieval_vectors_1024@retrieval_vectors_1024_ann_v2" in statement
        assert "v.tenant_id = %s" in statement
        assert "v.project_id = %s" in statement
        assert "v.repository_id = %s" in statement
        assert "v.resource_type = 'memory'" in statement
        assert "v.projection_id = %s" in statement
        assert "v.projection_signature = %s" in statement
        assert "v.scope_key = %s" in statement
        order_clause = statement.split("ORDER BY", 1)[1].split("LIMIT", 1)[0]
        assert "recorded_from" not in order_clause
        assert "m.recorded_to IS NULL" not in statement

    validation_statements = [
        statement for statement, _ in connection.calls if "FROM memories@primary AS m" in statement
    ]
    assert validation_statements
    for statement in validation_statements:
        assert "m.normalized_sha256" in statement
        assert "m.recorded_to IS NULL" in statement
        assert "s1.review_state != 'rejected'" in statement
        assert "m.id = ANY(%s::UUID[])" in statement

    exact_connection = _DenseConnection(now, memory_id, actor.repository_id)
    exact_gateway = CockroachDenseRetrievalGateway(  # type: ignore[arg-type]
        SimpleNamespace(pool=_Pool(exact_connection))
    )
    exact = await exact_gateway.retrieve_exact(actor, plan, query, dense_query)
    assert exact.candidates[0].reasons[-1] == "dense_cosine_exact"
    assert not any(
        statement.strip().startswith("SET LOCAL vector_search_beam_size")
        for statement, _ in exact_connection.calls
    )
    assert all(
        "retrieval_vectors_1024@primary" in statement
        for statement, _ in exact_connection.calls
        if "FROM retrieval_vectors_1024@" in statement
    )


@pytest.mark.asyncio
async def test_cockroach_dense_ann_adaptively_widens_after_canonical_underfill(
    scope_ids: dict[str, str],
) -> None:
    actor = make_actor(scope_ids)
    now = datetime(2026, 8, 6, 12, tzinfo=UTC)
    connection = _AdaptiveDenseConnection(now, actor.repository_id)
    query = RecallQuery(
        text="semantic retry policy",
        limit=1,
        visibilities=frozenset({Visibility.REPOSITORY}),
    )
    plan = RetrievalPlanner().plan(
        actor,
        query,
        purpose=RetrievalPurpose.INTERACTIVE_RECALL,
        available_signals=(RetrievalSignal.DENSE,),
    )
    gateway = CockroachDenseRetrievalGateway(  # type: ignore[arg-type]
        SimpleNamespace(pool=_Pool(connection)),
        overfetch_factor=1,
    )

    batch = await gateway.retrieve(
        actor,
        plan,
        query,
        DenseQuery(
            model="dense-live-v2",
            dimensions=1024,
            values=(1.0,) + (0.0,) * 1023,
        ),
    )

    assert [candidate.canonical_id for candidate in batch.candidates] == [connection.relevant_id]
    assert batch.examined_count == 65
    ann_limits = [
        int(parameters[-1])
        for statement, parameters in connection.calls
        if "retrieval_vectors_1024@retrieval_vectors_1024_ann_v2" in statement
    ]
    assert ann_limits == [32, 64]
