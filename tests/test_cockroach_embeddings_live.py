"""Live CockroachDB gate for durable structured-memory vector recall."""

from __future__ import annotations

import asyncio
import os
from typing import Any
from uuid import UUID

import pytest

from conftest import make_actor, new_id
from swarmbrain.adapters.cockroach.database import CockroachDatabase
from swarmbrain.adapters.cockroach.memory import CockroachMemoryStore
from swarmbrain.adapters.cockroach.retrieval import (
    CockroachDenseRetrievalGateway,
    cockroach_hybrid_retrieval_gateways,
)
from swarmbrain.adapters.cockroach.work_store import CockroachWorkStore
from swarmbrain.adapters.embeddings.deterministic import DeterministicEmbeddingProvider
from swarmbrain.application.memory_policy import (
    ConservativeMemoryPolicy,
    memory_content_text,
    memory_text_sha256,
)
from swarmbrain.application.memory_service import SEMANTIC_MATCH_REASON, MemoryService
from swarmbrain.application.retrieval_service import RetrievalService
from swarmbrain.application.work import DurableWorkService
from swarmbrain.domain.agents import ActorContext
from swarmbrain.domain.memory import EmbeddingVector, RecallQuery, RememberCommand, Visibility
from swarmbrain.domain.retrieval import DenseQuery, RetrievalPurpose, RetrievalSignal
from swarmbrain.domain.work import (
    ApplyEmbeddingWorkCommand,
    ClaimWorkCommand,
    EnqueueWorkCommand,
    WorkKind,
    WorkLease,
    WorkLeaseLost,
    WorkStatus,
    embedding_vector_sha256,
)
from swarmbrain.retrieval import (
    DENSE_PROJECTION_ID,
    RetrievalPlanner,
    dense_projection_signature,
)
from swarmbrain.retrieval.evaluation import ann_recall_at_k
from swarmbrain.workers.durable import LeasedWorkWorker
from swarmbrain.workers.embedding import EmbedMemoryHandler

DATABASE_URL = os.getenv("SWARMBRAIN_TEST_DATABASE_URL")
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(not DATABASE_URL, reason="SWARMBRAIN_TEST_DATABASE_URL is not set"),
]

MODEL_NAME = "deterministic-live-1024-v0"
SEMANTIC_QUERY = (
    "nebula-anchor quartz willow cobalt saffron delta ember juniper "
    "kestrel lunar mosaic northstar opal"
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


def _apply_command(
    lease: WorkLease,
    values: tuple[float, ...],
) -> ApplyEmbeddingWorkCommand:
    vector = EmbeddingVector(
        memory_id=lease.item.subject_id,
        model=str(lease.item.payload["model"]),
        dimensions=len(values),
        values=values,
    )
    return ApplyEmbeddingWorkCommand(
        work_id=lease.item.work_id,
        worker_id=lease.worker_id,
        lease_token=lease.lease_token,
        lease_version=lease.lease_version,
        expected_work_version=lease.work_version,
        attempt=lease.attempt,
        vector=vector,
        result={
            "model": vector.model,
            "dimensions": vector.dimensions,
            "vector_sha256": embedding_vector_sha256(vector),
        },
    )


def _structured_content(scope_marker: str) -> dict[str, object]:
    return {
        "memory_shape": {
            "signals": ["nebula-anchor"] * 48,
            "confidence_inputs": {"observed": True, "weight": 0.73},
        },
        "extensions": [
            {
                "namespace": "org.example/live-vector",
                "payload": {
                    "free_form": [1, {"nested": "value"}],
                    "scope_marker": scope_marker,
                },
            }
        ],
    }


async def test_structured_memory_embedding_and_semantic_recall_are_fully_scoped(
    database: CockroachDatabase,
) -> None:
    tenant_id = new_id()
    project_id = new_id()
    repository_id = new_id()
    base_scope = {
        "tenant_id": tenant_id,
        "project_id": project_id,
        "repository_id": repository_id,
        "swarm_id": new_id(),
        "run_id": new_id(),
    }
    actors = (
        make_actor(base_scope),
        make_actor(
            {
                **base_scope,
                "project_id": new_id(),
                "run_id": new_id(),
            }
        ),
        make_actor(
            {
                **base_scope,
                "repository_id": new_id(),
                "run_id": new_id(),
            }
        ),
    )
    for actor in actors:
        await _insert_run(database, actor)

    memory_store = CockroachMemoryStore(database)
    work_store = CockroachWorkStore(database)
    work_service = DurableWorkService(work_store)
    provider = DeterministicEmbeddingProvider(
        dimensions=1024,
        model_name=MODEL_NAME,
    )
    service = MemoryService(
        memory_store,
        ConservativeMemoryPolicy(),
        review_store=memory_store,
        embeddings=provider,
        embedding_index=memory_store,
        work=work_service,
        retrieval=RetrievalService(
            cockroach_hybrid_retrieval_gateways(database),
            memory_store,
        ),
        canonical_reader=memory_store,
    )

    published: list[tuple[ActorContext, str, dict[str, object]]] = []
    for position, actor in enumerate(actors):
        content = _structured_content(f"scope-{position}")
        result = await service.publish(
            actor,
            RememberCommand(
                idempotency_key=f"live-vector-publish-{new_id()}",
                kind="org.example/flexible-memory",
                visibility=Visibility.REPOSITORY,
                content=content,
            ),
        )
        assert result.memory is not None
        assert result.memory.content == content
        published.append((actor, result.memory.memory_id, content))

    incompatible_model = "deterministic-live-incompatible-1024-v0"
    incompatible = await work_service.enqueue(
        actors[0],
        EnqueueWorkCommand(
            idempotency_key=f"live-vector-incompatible-{new_id()}",
            kind=WorkKind.EMBED_MEMORY,
            subject_id=published[0][1],
            payload={
                "content": memory_content_text(published[0][2]),
                "model": incompatible_model,
                "metadata": {},
            },
            dedupe_key=f"live-vector-incompatible:{published[0][1]}:{incompatible_model}",
            priority=1000,
        ),
    )

    worker = LeasedWorkWorker(
        work_store,
        [EmbedMemoryHandler(provider)],
    )
    completed = await worker.run_once(f"live-vector-worker-{new_id()}", limit=3)

    assert {item.item.subject_id for item in completed} == {
        memory_id for _, memory_id, _ in published
    }
    assert all(item.item.status is WorkStatus.COMPLETED for item in completed)

    async with database.pool.connection() as connection:
        cursor = await connection.execute(
            "SELECT status, attempts FROM outbox_work_items WHERE id = %s",
            (incompatible.item.work_id,),
        )
        incompatible_row = await cursor.fetchone()
    assert incompatible_row is not None
    assert incompatible_row["status"] == WorkStatus.PENDING.value
    assert int(incompatible_row["attempts"]) == 0

    incompatible_provider = DeterministicEmbeddingProvider(
        dimensions=1024,
        model_name=incompatible_model,
    )
    incompatible_completed = await LeasedWorkWorker(
        work_store,
        [EmbedMemoryHandler(incompatible_provider)],
    ).run_once(f"live-vector-incompatible-worker-{new_id()}", limit=1)
    assert len(incompatible_completed) == 1
    assert incompatible_completed[0].item.work_id == incompatible.item.work_id
    assert incompatible_completed[0].item.status is WorkStatus.COMPLETED

    memory_ids = [memory_id for _, memory_id, _ in published]
    async with database.pool.connection() as connection:
        cursor = await connection.execute(
            """
            SELECT memory_id, tenant_id, project_id, repository_id, model, dimensions
            FROM memory_vector_embeddings
            WHERE memory_id = ANY(%s::UUID[])
              AND model = %s
            ORDER BY memory_id
            """,
            (memory_ids, MODEL_NAME),
        )
        vector_rows = await cursor.fetchall()
        projection_cursor = await connection.execute(
            """
            SELECT canonical_id, tenant_id, project_id, repository_id,
                   projection_id, projection_signature, scope_key,
                   resource_version, content_sha256, model, dimensions
            FROM retrieval_vectors_1024
            WHERE canonical_id = ANY(%s::UUID[])
              AND projection_signature = %s
            ORDER BY canonical_id
            """,
            (memory_ids, dense_projection_signature(MODEL_NAME, 1024)),
        )
        projection_rows = await projection_cursor.fetchall()

    assert len(vector_rows) == len(published)
    rows_by_memory = {str(row["memory_id"]): row for row in vector_rows}
    for actor, memory_id, _ in published:
        row = rows_by_memory[memory_id]
        assert row["tenant_id"] == actor.tenant_id
        assert row["project_id"] == actor.project_id
        assert row["repository_id"] == actor.repository_id
        assert row["model"] == MODEL_NAME
        assert int(row["dimensions"]) == 1024
    assert len(projection_rows) == len(published)
    projections_by_memory = {str(row["canonical_id"]): row for row in projection_rows}
    for actor, memory_id, content in published:
        row = projections_by_memory[memory_id]
        assert row["tenant_id"] == actor.tenant_id
        assert row["project_id"] == actor.project_id
        assert row["repository_id"] == actor.repository_id
        assert row["projection_id"] == DENSE_PROJECTION_ID
        assert row["projection_signature"] == dense_projection_signature(MODEL_NAME, 1024)
        assert row["scope_key"] == f"repository:{actor.repository_id}"
        assert int(row["resource_version"]) == 1
        assert bytes(row["content_sha256"]) == bytes.fromhex(memory_text_sha256(content))
        assert row["model"] == MODEL_NAME
        assert int(row["dimensions"]) == 1024

    query_vector = await provider.embed_query(SEMANTIC_QUERY)
    for actor, expected_memory_id, expected_content in published:
        scoped_matches = await memory_store.search_embeddings(
            actor,
            query_vector,
            model=MODEL_NAME,
            limit=10,
            min_score=0.2,
        )
        assert {match.memory_id for match in scoped_matches} == {expected_memory_id}

        dense_query_for_actor = DenseQuery(
            model=MODEL_NAME,
            dimensions=1024,
            values=tuple(query_vector),
        )
        dense_plan_for_actor = RetrievalPlanner().plan(
            actor,
            RecallQuery(text=SEMANTIC_QUERY, min_score=0.2, limit=10),
            purpose=RetrievalPurpose.INTERACTIVE_RECALL,
            available_signals=(RetrievalSignal.DENSE,),
        )
        async with memory_store.retrieval_snapshot():
            dense_batch_for_actor = await CockroachDenseRetrievalGateway(database).retrieve(
                actor,
                dense_plan_for_actor,
                RecallQuery(text=SEMANTIC_QUERY, min_score=0.2, limit=10),
                dense_query_for_actor,
            )
        assert dense_batch_for_actor.degraded is False
        assert {candidate.canonical_id for candidate in dense_batch_for_actor.candidates} == {
            expected_memory_id
        }

        recalled = await service.recall(
            actor,
            RecallQuery(text=SEMANTIC_QUERY, min_score=0.2, limit=10),
        )
        assert [hit.memory.memory_id for hit in recalled.hits] == [expected_memory_id]
        assert recalled.hits[0].memory.content == expected_content
        assert SEMANTIC_MATCH_REASON in recalled.hits[0].reasons
        assert "signal:dense" in recalled.hits[0].reasons

    dense_query = DenseQuery(
        model=MODEL_NAME,
        dimensions=1024,
        values=tuple(query_vector),
    )
    dense_plan = RetrievalPlanner().plan(
        actors[0],
        RecallQuery(text=SEMANTIC_QUERY),
        purpose=RetrievalPurpose.INTERACTIVE_RECALL,
        available_signals=(RetrievalSignal.DENSE,),
    )
    dense_gateway = CockroachDenseRetrievalGateway(database)
    async with memory_store.retrieval_snapshot():
        ann = await dense_gateway.retrieve(
            actors[0],
            dense_plan,
            RecallQuery(text=SEMANTIC_QUERY),
            dense_query,
        )
        exact = await dense_gateway.retrieve_exact(
            actors[0],
            dense_plan,
            RecallQuery(text=SEMANTIC_QUERY),
            dense_query,
        )
    assert ann_recall_at_k(
        tuple(candidate.canonical_id for candidate in ann.candidates),
        tuple(candidate.canonical_id for candidate in exact.candidates),
        k=10,
    ) == pytest.approx(1.0)


async def test_stale_embedding_lease_cannot_write_or_overwrite_a_vector(
    database: CockroachDatabase,
) -> None:
    actor = make_actor(
        {
            "tenant_id": new_id(),
            "project_id": new_id(),
            "repository_id": new_id(),
            "swarm_id": new_id(),
            "run_id": new_id(),
        }
    )
    await _insert_run(database, actor)
    model = f"race-{new_id()}"
    memory_store = CockroachMemoryStore(database)
    work_store = CockroachWorkStore(database)
    service = MemoryService(
        memory_store,
        ConservativeMemoryPolicy(),
        review_store=memory_store,
        embeddings=DeterministicEmbeddingProvider(dimensions=1024, model_name=model),
        embedding_index=memory_store,
        work=DurableWorkService(work_store),
    )
    published = await service.publish(
        actor,
        RememberCommand(
            idempotency_key=f"stale-vector-publish-{new_id()}",
            kind="org.example/stale-vector",
            content={"race": "lease-fence"},
        ),
    )
    assert published.memory is not None

    try:
        stale = (
            await work_store.claim_work(
                ClaimWorkCommand(
                    worker_id="stale-vector-a",
                    kinds=frozenset({WorkKind.EMBED_MEMORY}),
                    embedding_model=model,
                    lease_seconds=5,
                )
            )
        ).leases[0]
        async with database.pool.connection() as connection:
            await connection.execute(
                """
                UPDATE outbox_work_items
                SET locked_until = now() - INTERVAL '1 second'
                WHERE id = %s
                """,
                (UUID(stale.item.work_id),),
            )
        current_batch = None
        for _ in range(3):
            current_batch = await work_store.claim_work(
                ClaimWorkCommand(
                    worker_id="current-vector-b",
                    kinds=frozenset({WorkKind.EMBED_MEMORY}),
                    embedding_model=model,
                    lease_seconds=30,
                )
            )
            if current_batch.leases:
                break
            # Claim uses SKIP LOCKED, so a just-committed forced-expiry update
            # may produce one transiently empty poll under live contention.
            await asyncio.sleep(0.01)
        assert current_batch is not None and len(current_batch.leases) == 1
        current = current_batch.leases[0]
        assert current.item.work_id == stale.item.work_id

        stale_values = tuple([1.0] + [0.0] * 1023)
        current_values = tuple([0.0, 1.0] + [0.0] * 1022)
        with pytest.raises(WorkLeaseLost):
            await work_store.apply_embedding(_apply_command(stale, stale_values))

        async with database.pool.connection() as connection:
            cursor = await connection.execute(
                """
                SELECT count(*) AS count
                FROM memory_vector_embeddings
                WHERE memory_id = %s AND model = %s
                """,
                (UUID(published.memory.memory_id), model),
            )
            before = await cursor.fetchone()
            projection_cursor = await connection.execute(
                """
                SELECT count(*) AS count
                FROM retrieval_vectors_1024
                WHERE canonical_id = %s AND projection_signature = %s
                """,
                (
                    UUID(published.memory.memory_id),
                    dense_projection_signature(model, 1024),
                ),
            )
            projected_before = await projection_cursor.fetchone()
        assert before is not None and int(before["count"]) == 0
        assert projected_before is not None and int(projected_before["count"]) == 0

        command = _apply_command(current, current_values)
        first = await work_store.apply_embedding(command)
        replay = await work_store.apply_embedding(command)
        assert first.replayed is False
        assert replay.replayed is True

        async with database.pool.connection() as connection:
            cursor = await connection.execute(
                """
                SELECT embedding = %s::VECTOR AS matches
                FROM memory_vector_embeddings
                WHERE memory_id = %s AND model = %s
                """,
                (
                    "[" + ",".join(repr(value) for value in current_values) + "]",
                    UUID(published.memory.memory_id),
                    model,
                ),
            )
            stored = await cursor.fetchone()
            projection_cursor = await connection.execute(
                """
                SELECT embedding = %s::VECTOR AS matches
                FROM retrieval_vectors_1024
                WHERE canonical_id = %s AND projection_signature = %s
                """,
                (
                    "[" + ",".join(repr(value) for value in current_values) + "]",
                    UUID(published.memory.memory_id),
                    dense_projection_signature(model, 1024),
                ),
            )
            projected = await projection_cursor.fetchone()
        assert stored is not None and bool(stored["matches"])
        assert projected is not None and bool(projected["matches"])
    finally:
        async with database.pool.connection() as connection:
            await connection.execute(
                "DELETE FROM outbox_events WHERE tenant_id = %s AND run_id = %s",
                (actor.tenant_id, actor.run_id),
            )
            await connection.execute(
                "DELETE FROM outbox_work_items WHERE tenant_id = %s AND run_id = %s",
                (actor.tenant_id, actor.run_id),
            )
            await connection.execute(
                "DELETE FROM memories WHERE tenant_id = %s AND run_id = %s",
                (actor.tenant_id, actor.run_id),
            )
            await connection.execute(
                "DELETE FROM idempotency_records WHERE tenant_id = %s AND run_id = %s",
                (actor.tenant_id, actor.run_id),
            )
            await connection.execute(
                "DELETE FROM runs WHERE tenant_id = %s AND id = %s",
                (actor.tenant_id, actor.run_id),
            )
