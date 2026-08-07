from __future__ import annotations

import os
from typing import Any
from uuid import uuid4

import pytest

from swarmbrain.adapters.cockroach import memory as memory_adapter
from swarmbrain.adapters.cockroach.coordination import CockroachCoordinationStore
from swarmbrain.adapters.cockroach.database import CockroachDatabase
from swarmbrain.adapters.cockroach.memory import CockroachMemoryStore
from swarmbrain.adapters.cockroach.retrieval import cockroach_retrieval_gateways
from swarmbrain.application.memory_policy import ConservativeMemoryPolicy
from swarmbrain.application.memory_service import MemoryService
from swarmbrain.application.retrieval_service import RetrievalService
from swarmbrain.domain.agents import ActorContext, Capability
from swarmbrain.domain.memory import MemoryKind, RecallQuery, RememberCommand, Visibility

DATABASE_URL = os.getenv("SWARMBRAIN_TEST_DATABASE_URL")
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(not DATABASE_URL, reason="SWARMBRAIN_TEST_DATABASE_URL is not set"),
]

SENTINEL = "cockroach reuse counter sentinel"


def _id() -> str:
    return str(uuid4())


def _actor(**overrides: str) -> ActorContext:
    scope = {
        "tenant_id": _id(),
        "project_id": _id(),
        "repository_id": _id(),
        "swarm_id": _id(),
        "run_id": _id(),
    }
    scope.update(overrides)
    return ActorContext(
        **scope,
        agent_id=_id(),
        harness="pytest",
        provider="local",
        model="none",
        capabilities=frozenset({Capability.MEMORY_PUBLISH.value, Capability.MEMORY_RECALL.value}),
    )


@pytest.fixture
async def database() -> CockroachDatabase:
    assert DATABASE_URL is not None
    value = CockroachDatabase(DATABASE_URL, min_size=1, max_size=4)
    await value.start()
    try:
        yield value
    finally:
        await value.close()


def _service(database: CockroachDatabase) -> tuple[MemoryService, CockroachMemoryStore]:
    store = CockroachMemoryStore(database)
    service = MemoryService(
        store,
        ConservativeMemoryPolicy(),
        retrieval=RetrievalService(cockroach_retrieval_gateways(database), store),
        canonical_reader=store,
    )
    return service, store


async def _insert_run(database: CockroachDatabase, actor: ActorContext) -> None:
    async def body(connection: Any) -> None:
        await connection.execute(
            """
            INSERT INTO runs (tenant_id, id, project_id, repository_id, swarm_id)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (
                actor.tenant_id,
                actor.run_id,
                actor.project_id,
                actor.repository_id,
                actor.swarm_id,
            ),
        )

    await database.run(body)  # type: ignore[arg-type]


async def _counter(database: CockroachDatabase, actor: ActorContext) -> dict[str, Any] | None:
    async with database.pool.connection() as connection:
        cursor = await connection.execute(
            """
            SELECT reuse_count, recall_count, project_id, repository_id, swarm_id
            FROM retrieval_reuse_counters
            WHERE tenant_id = %s AND run_id = %s
            """,
            (actor.tenant_id, actor.run_id),
        )
        return await cursor.fetchone()


async def _publish(service: MemoryService, actor: ActorContext, suffix: str) -> str:
    result = await service.publish(
        actor,
        RememberCommand(
            idempotency_key=f"reuse-{suffix}-{_id()}",
            kind=MemoryKind.INVARIANT,
            visibility=Visibility.REPOSITORY,
            content=f"{SENTINEL} {suffix}",
        ),
    )
    assert result.memory is not None
    return result.memory.memory_id


async def test_recall_increments_durable_counter_and_run_metrics(
    database: CockroachDatabase,
) -> None:
    actor = _actor()
    await _insert_run(database, actor)
    service, _ = _service(database)
    metrics_store = CockroachCoordinationStore(database)
    await _publish(service, actor, "alpha")
    await _publish(service, actor, "beta")

    first = await service.recall(actor, RecallQuery(text=SENTINEL))
    assert len(first.hits) == 2

    row = await _counter(database, actor)
    assert row is not None
    assert int(row["reuse_count"]) == 2
    assert int(row["recall_count"]) == 1
    assert row["project_id"] == actor.project_id

    metrics = await metrics_store.get_run_metrics(actor, actor.run_id)
    assert metrics.memories_reused == 2

    second = await service.recall(actor, RecallQuery(text=SENTINEL))
    assert len(second.hits) == 2

    row = await _counter(database, actor)
    assert row is not None
    assert int(row["reuse_count"]) == 4
    assert int(row["recall_count"]) == 2
    metrics = await metrics_store.get_run_metrics(actor, actor.run_id)
    assert metrics.memories_reused == 4


async def test_abstained_recall_leaves_no_counter_row(database: CockroachDatabase) -> None:
    actor = _actor()
    await _insert_run(database, actor)
    service, _ = _service(database)
    await _publish(service, actor, "gamma")

    bundle = await service.recall(actor, RecallQuery(text="entirely unrelated telemetry query"))

    assert bundle.hits == ()
    assert await _counter(database, actor) is None


async def test_failed_counter_write_never_breaks_recall(
    database: CockroachDatabase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actor = _actor()
    await _insert_run(database, actor)
    service, _ = _service(database)
    await _publish(service, actor, "delta")

    monkeypatch.setattr(
        memory_adapter,
        "RECORD_RETRIEVAL_REUSE_SQL",
        "INSERT INTO retrieval_reuse_counters_missing VALUES (%s, %s, %s, %s, %s, %s)",
    )
    bundle = await service.recall(actor, RecallQuery(text=SENTINEL))

    assert len(bundle.hits) == 1
    assert await _counter(database, actor) is None

    monkeypatch.undo()
    recovered = await service.recall(actor, RecallQuery(text=SENTINEL))
    assert len(recovered.hits) == 1
    row = await _counter(database, actor)
    assert row is not None
    assert int(row["reuse_count"]) == 1


async def test_reuse_counters_are_isolated_per_run_and_tenant(
    database: CockroachDatabase,
) -> None:
    first = _actor()
    same_tenant = _actor(
        tenant_id=first.tenant_id,
        project_id=first.project_id,
        repository_id=first.repository_id,
        swarm_id=first.swarm_id,
    )
    other_tenant = _actor()
    for actor in (first, same_tenant, other_tenant):
        await _insert_run(database, actor)
    service, _ = _service(database)
    metrics_store = CockroachCoordinationStore(database)
    await _publish(service, first, "epsilon")
    await _publish(service, other_tenant, "epsilon")

    assert len((await service.recall(first, RecallQuery(text=SENTINEL))).hits) == 1
    assert len((await service.recall(same_tenant, RecallQuery(text=SENTINEL))).hits) == 1
    assert len((await service.recall(other_tenant, RecallQuery(text=SENTINEL))).hits) == 1

    for actor in (first, same_tenant, other_tenant):
        row = await _counter(database, actor)
        assert row is not None, actor.run_id
        assert int(row["reuse_count"]) == 1
        metrics = await metrics_store.get_run_metrics(actor, actor.run_id)
        assert metrics.memories_reused == 1

    # A run identity from another tenant must never resolve this run's counter.
    foreign = first.model_copy(update={"tenant_id": other_tenant.tenant_id})
    async with database.pool.connection() as connection:
        cursor = await connection.execute(
            """
            SELECT count(*) AS total
            FROM retrieval_reuse_counters
            WHERE tenant_id = %s AND run_id = %s
            """,
            (foreign.tenant_id, foreign.run_id),
        )
        row = await cursor.fetchone()
    assert row is not None
    assert int(row["total"]) == 0
