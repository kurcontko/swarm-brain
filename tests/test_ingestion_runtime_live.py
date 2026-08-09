from __future__ import annotations

import json
import os
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest
from psycopg.types.json import Jsonb

from conftest import make_actor
from swarmbrain.adapters.cockroach.database import CockroachDatabase
from swarmbrain.adapters.extraction import STRUCTURED_MEMORY_SOURCE_KIND
from swarmbrain.application.runtime import build_runtime
from swarmbrain.config import ApiSettings, BackendKind, EmbeddingsKind
from swarmbrain.domain.agents import ActorContext
from swarmbrain.domain.common import ContractModel
from swarmbrain.domain.evidence import RejectSourceCommand
from swarmbrain.domain.memory import RecallQuery
from swarmbrain.transports.http.app import create_app

DATABASE_URL = os.getenv("SWARMBRAIN_TEST_DATABASE_URL")
TOKEN_SECRET = "0123456789abcdef-ingestion-runtime-live"
EMBEDDING_MODEL = "deterministic-ingestion-live-1024-v0"
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(not DATABASE_URL, reason="SWARMBRAIN_TEST_DATABASE_URL is not set"),
]


class _Ack(ContractModel):
    ok: bool = True


def _settings(
    *,
    embedding_model: str = EMBEDDING_MODEL,
    embeddings: EmbeddingsKind = EmbeddingsKind.DETERMINISTIC,
) -> ApiSettings:
    assert DATABASE_URL is not None
    return ApiSettings(
        backend=BackendKind.COCKROACH,
        token_secret=TOKEN_SECRET,
        database_url=DATABASE_URL,
        database_pool_min_size=1,
        database_pool_max_size=4,
        embeddings=embeddings,
        embeddings_model=(embedding_model if embeddings is not EmbeddingsKind.NONE else None),
        embeddings_dimensions=1024,
        ingest_use_provider=False,
        ingest_priority=1000,
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
    async def body(connection: Any) -> _Ack:
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
        return _Ack()

    await database.run(body)


async def _cleanup(database: CockroachDatabase, actor: ActorContext) -> None:
    async def body(connection: Any) -> _Ack:
        scope = (actor.tenant_id, actor.run_id)
        await connection.execute(
            "DELETE FROM outbox_events WHERE tenant_id = %s AND run_id = %s",
            scope,
        )
        await connection.execute(
            "DELETE FROM outbox_work_items WHERE tenant_id = %s AND run_id = %s",
            scope,
        )
        await connection.execute(
            "DELETE FROM memory_conflicts WHERE tenant_id = %s AND run_id = %s",
            scope,
        )
        await connection.execute(
            """
            DELETE FROM memories
            WHERE tenant_id = %s AND project_id = %s AND repository_id = %s
              AND swarm_id = %s AND run_id = %s
            """,
            (
                actor.tenant_id,
                actor.project_id,
                actor.repository_id,
                actor.swarm_id,
                actor.run_id,
            ),
        )
        await connection.execute(
            "DELETE FROM evidence WHERE tenant_id = %s",
            (actor.tenant_id,),
        )
        await connection.execute(
            """
            DELETE FROM sources
            WHERE tenant_id = %s AND project_id = %s AND repository_id = %s
              AND swarm_id = %s AND run_id = %s
            """,
            (
                actor.tenant_id,
                actor.project_id,
                actor.repository_id,
                actor.swarm_id,
                actor.run_id,
            ),
        )
        await connection.execute(
            "DELETE FROM action_attempts WHERE tenant_id = %s AND run_id = %s",
            scope,
        )
        await connection.execute(
            "DELETE FROM idempotency_records WHERE tenant_id = %s AND run_id = %s",
            scope,
        )
        await connection.execute(
            "DELETE FROM runs WHERE tenant_id = %s AND id = %s",
            scope,
        )
        return _Ack()

    await database.run(body)


async def _effect_snapshot(
    database: CockroachDatabase,
    actor: ActorContext,
    source_id: str,
) -> tuple[int, int, int, int]:
    async with database.pool.connection() as connection:
        cursor = await connection.execute(
            """
            SELECT
                (
                    SELECT count(*)
                    FROM memories AS memory
                    WHERE memory.tenant_id = %s
                      AND memory.project_id = %s
                      AND memory.repository_id = %s
                      AND memory.swarm_id = %s
                      AND memory.run_id = %s
                      AND memory.source_id = %s
                ) AS memory_count,
                (
                    SELECT count(*)
                    FROM evidence AS evidence
                    WHERE evidence.tenant_id = %s
                      AND evidence.source_id = %s
                ) AS evidence_count,
                (
                    SELECT count(*)
                    FROM outbox_work_effects AS effect
                    JOIN outbox_work_items AS work ON work.id = effect.work_id
                    WHERE work.tenant_id = %s
                      AND work.project_id = %s
                      AND work.repository_id = %s
                      AND work.swarm_id = %s
                      AND work.run_id = %s
                      AND work.subject_id = %s
                ) AS effect_count,
                (
                    SELECT count(*)
                    FROM memory_evidence AS link
                    JOIN memories AS memory ON memory.id = link.memory_id
                    JOIN evidence AS evidence ON evidence.id = link.evidence_id
                    WHERE memory.tenant_id = %s
                      AND memory.project_id = %s
                      AND memory.repository_id = %s
                      AND memory.swarm_id = %s
                      AND memory.run_id = %s
                      AND memory.source_id = %s
                      AND evidence.tenant_id = %s
                      AND evidence.source_id = %s
                ) AS association_count
            """,
            (
                actor.tenant_id,
                actor.project_id,
                actor.repository_id,
                actor.swarm_id,
                actor.run_id,
                UUID(source_id),
                actor.tenant_id,
                UUID(source_id),
                actor.tenant_id,
                actor.project_id,
                actor.repository_id,
                actor.swarm_id,
                actor.run_id,
                UUID(source_id),
                actor.tenant_id,
                actor.project_id,
                actor.repository_id,
                actor.swarm_id,
                actor.run_id,
                UUID(source_id),
                actor.tenant_id,
                UUID(source_id),
            ),
        )
        row = await cursor.fetchone()
    assert row is not None
    return (
        int(row["memory_count"]),
        int(row["evidence_count"]),
        int(row["effect_count"]),
        int(row["association_count"]),
    )


async def _insert_pending_extraction(
    database: CockroachDatabase,
    actor: ActorContext,
    source_id: str,
    *,
    extractor_name: str,
    extractor_revision: str,
) -> str:
    work_id = uuid4()

    async def body(connection: Any) -> _Ack:
        await connection.execute(
            """
            INSERT INTO outbox_work_items (
                id, tenant_id, project_id, repository_id, swarm_id, run_id,
                requested_by_agent_id, kind, subject_id, payload, dedupe_key,
                priority
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, 'extract_source', %s, %s, %s, %s
            )
            """,
            (
                work_id,
                actor.tenant_id,
                actor.project_id,
                actor.repository_id,
                actor.swarm_id,
                actor.run_id,
                actor.agent_id,
                UUID(source_id),
                Jsonb(
                    {
                        "extractor_name": extractor_name,
                        "extractor_revision": extractor_revision,
                        "use_provider": False,
                    }
                ),
                f"reject-fence:{source_id}:{work_id}",
                1000,
            ),
        )
        return _Ack()

    await database.run(body)
    return str(work_id)


async def _work_state(
    database: CockroachDatabase,
    actor: ActorContext,
    work_id: str,
) -> tuple[str, str | None]:
    async with database.pool.connection() as connection:
        cursor = await connection.execute(
            """
            SELECT status, outcome
            FROM outbox_work_items
            WHERE id = %s
              AND tenant_id = %s
              AND project_id = %s
              AND repository_id = %s
              AND swarm_id = %s
              AND run_id = %s
            """,
            (
                UUID(work_id),
                actor.tenant_id,
                actor.project_id,
                actor.repository_id,
                actor.swarm_id,
                actor.run_id,
            ),
        )
        row = await cursor.fetchone()
    assert row is not None
    return str(row["status"]), row["outcome"]


async def _work_diagnostic(
    database: CockroachDatabase,
    actor: ActorContext,
    work_id: str,
) -> tuple[str, str | None, str | None, int]:
    async with database.pool.connection() as connection:
        cursor = await connection.execute(
            """
            SELECT status, outcome, last_error, attempts
            FROM outbox_work_items
            WHERE id = %s
              AND tenant_id = %s
              AND project_id = %s
              AND repository_id = %s
              AND swarm_id = %s
              AND run_id = %s
            """,
            (
                UUID(work_id),
                actor.tenant_id,
                actor.project_id,
                actor.repository_id,
                actor.swarm_id,
                actor.run_id,
            ),
        )
        row = await cursor.fetchone()
    assert row is not None
    return (
        str(row["status"]),
        row["outcome"],
        row["last_error"],
        int(row["attempts"]),
    )


async def _claimability_snapshot(
    database: CockroachDatabase,
    actor: ActorContext,
    work_id: str,
) -> tuple[str, str | None, str | None, bool, bool]:
    """Expose only the durable predicates a compatible worker must satisfy."""

    async with database.pool.connection() as connection:
        cursor = await connection.execute(
            """
            SELECT
                work.status,
                work.payload->>'extractor_name' AS extractor_name,
                work.payload->>'extractor_revision' AS extractor_revision,
                work.available_at <= now() AS is_available,
                EXISTS (
                    SELECT 1
                    FROM sources AS source
                    WHERE source.id = work.subject_id
                      AND source.review_state != 'rejected'
                      AND source.trust_label != 'untrusted'
                ) AS source_is_eligible
            FROM outbox_work_items AS work
            WHERE work.id = %s
              AND work.tenant_id = %s
              AND work.project_id = %s
              AND work.repository_id = %s
              AND work.swarm_id = %s
              AND work.run_id = %s
            """,
            (
                UUID(work_id),
                actor.tenant_id,
                actor.project_id,
                actor.repository_id,
                actor.swarm_id,
                actor.run_id,
            ),
        )
        row = await cursor.fetchone()
    assert row is not None
    return (
        str(row["status"]),
        row["extractor_name"],
        row["extractor_revision"],
        bool(row["is_available"]),
        bool(row["source_is_eligible"]),
    )


async def _memory_state(
    database: CockroachDatabase,
    actor: ActorContext,
    memory_id: str,
) -> str:
    async with database.pool.connection() as connection:
        cursor = await connection.execute(
            """
            SELECT state
            FROM memories
            WHERE id = %s
              AND tenant_id = %s
              AND project_id = %s
              AND repository_id = %s
              AND swarm_id = %s
              AND run_id = %s
            """,
            (
                UUID(memory_id),
                actor.tenant_id,
                actor.project_id,
                actor.repository_id,
                actor.swarm_id,
                actor.run_id,
            ),
        )
        row = await cursor.fetchone()
    assert row is not None
    return str(row["state"])


async def _embedding_snapshot(
    database: CockroachDatabase,
    actor: ActorContext,
    memory_id: str,
) -> tuple[str, int, str, int, int]:
    async with database.pool.connection() as connection:
        cursor = await connection.execute(
            """
            SELECT work.status, work.attempts, work.payload->>'model' AS model,
                   count(*) OVER () AS child_count,
                   (
                       SELECT count(*)
                       FROM memory_vector_embeddings AS vector
                       WHERE vector.memory_id = work.subject_id
                         AND vector.tenant_id = work.tenant_id
                         AND vector.project_id = work.project_id
                         AND vector.repository_id = work.repository_id
                         AND vector.model = work.payload->>'model'
                   ) AS vector_count
            FROM outbox_work_items AS work
            WHERE work.kind = 'embed_memory'
              AND work.subject_id = %s
              AND work.tenant_id = %s
              AND work.project_id = %s
              AND work.repository_id = %s
              AND work.swarm_id = %s
              AND work.run_id = %s
            """,
            (
                UUID(memory_id),
                actor.tenant_id,
                actor.project_id,
                actor.repository_id,
                actor.swarm_id,
                actor.run_id,
            ),
        )
        row = await cursor.fetchone()
    assert row is not None
    return (
        str(row["status"]),
        int(row["attempts"]),
        str(row["model"]),
        int(row["child_count"]),
        int(row["vector_count"]),
    )


async def test_structured_ingestion_survives_restarts_and_applies_effects_once(
    database: CockroachDatabase,
    scope_ids: dict[str, str],
) -> None:
    actor = make_actor(scope_ids)
    await _insert_run(database, actor)

    content = {
        "rule": "retry serializable transactions only on SQLSTATE 40001",
        "contexts": ["cockroachdb", "ingestion-live-gate"],
    }
    envelope = {
        "memories": [
            {
                "kind": "org.sen/transaction-rule",
                "content": content,
                "title": "CockroachDB retry rule",
                "tags": ["cockroachdb", "retry"],
                "confidence": 0.9,
            }
        ]
    }

    try:
        runtime_a = build_runtime(_settings())
        app_a = create_app(runtime_a)
        token = runtime_a.tokens.issue(actor)
        headers = {
            "Authorization": f"Bearer {token}",
            "Idempotency-Key": f"live-ingest-{uuid4()}",
        }
        async with (
            app_a.router.lifespan_context(app_a),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app_a),
                base_url="http://runtime-a",
            ) as client,
        ):
            response = await client.post(
                "/v1/sources:ingest",
                headers=headers,
                json={
                    "kind": STRUCTURED_MEMORY_SOURCE_KIND,
                    "content": json.dumps(envelope),
                    "metadata": {"gate": "restart-exactly-once"},
                },
            )
        assert response.status_code == 202, response.text
        receipt = response.json()
        source_id = receipt["source"]["source_id"]
        source_version = int(receipt["source"]["version"])
        assert receipt["chunk_count"] == 1
        assert receipt["replayed"] is False
        assert await _claimability_snapshot(database, actor, receipt["work_id"]) == (
            "pending",
            "default-rules",
            "1",
            True,
            True,
        )

        runtime_b = build_runtime(_settings())
        await runtime_b.start()
        try:
            extracted = await runtime_b.extraction_worker(retry_delay_seconds=1).run_once(
                f"ingestion-live-b-{uuid4()}"
            )
        finally:
            await runtime_b.close()
        assert len(extracted) == 1, await _work_diagnostic(database, actor, receipt["work_id"])
        memory_id = extracted[0].effects[0].resource_id

        # Extraction must synchronously maintain the canonical lexical
        # projection. It cannot rely on the optional embedding worker to make a
        # newly materialized memory discoverable.
        lexical_runtime = build_runtime(_settings(embeddings=EmbeddingsKind.NONE))
        await lexical_runtime.start()
        try:
            lexical = await lexical_runtime.memory.recall(
                actor,
                RecallQuery(text="serializable transactions SQLSTATE 40001", min_score=0.01),
            )
        finally:
            await lexical_runtime.close()
        assert [hit.memory.memory_id for hit in lexical.hits] == [memory_id]
        assert all("semantic_match" not in hit.reasons for hit in lexical.hits)

        assert await _embedding_snapshot(database, actor, memory_id) == (
            "pending",
            0,
            EMBEDDING_MODEL,
            1,
            0,
        )

        wrong_model = "deterministic-ingestion-live-wrong-1024-v0"
        runtime_wrong = build_runtime(_settings(embedding_model=wrong_model))
        await runtime_wrong.start()
        try:
            wrong_completed = await runtime_wrong.embedding_worker().run_once(
                f"ingestion-live-wrong-{uuid4()}"
            )
        finally:
            await runtime_wrong.close()
        assert wrong_completed == ()
        assert await _embedding_snapshot(database, actor, memory_id) == (
            "pending",
            0,
            EMBEDDING_MODEL,
            1,
            0,
        )

        runtime_c = build_runtime(_settings())
        await runtime_c.start()
        try:
            embedded = await runtime_c.embedding_worker().run_once(f"ingestion-live-c-{uuid4()}")
        finally:
            await runtime_c.close()
        assert len(embedded) == 1
        assert await _embedding_snapshot(database, actor, memory_id) == (
            "completed",
            1,
            EMBEDDING_MODEL,
            1,
            1,
        )

        runtime_d = build_runtime(_settings())
        assert runtime_d.extraction is not None
        app_c = create_app(runtime_d)
        auth = {"Authorization": f"Bearer {runtime_d.tokens.issue(actor)}"}
        async with (
            app_c.router.lifespan_context(app_c),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app_c),
                base_url="http://runtime-c",
            ) as client,
        ):
            status = await client.get(
                f"/v1/sources/{source_id}/extraction",
                headers=auth,
            )
            assert status.status_code == 200, status.text
            completed = status.json()
            assert completed["status"] == "completed"
            assert completed["candidate_count"] == 1
            assert completed["attempts"] == 1
            assert len(completed["memory_ids"]) == 1
            assert completed["memory_ids"][0] == memory_id

            recalled = await client.post(
                "/v1/memories:recall",
                headers=auth,
                json={
                    "text": "40001 " + " ".join(f"opaque{position}" for position in range(128)),
                    "min_score": 0.01,
                    "limit": 10,
                },
            )
            assert recalled.status_code == 200, recalled.text
            hits = recalled.json()["hits"]
            assert [hit["memory"]["memory_id"] for hit in hits] == [memory_id]
            assert hits[0]["memory"]["content"] == content
            assert hits[0]["memory"]["state"] == "tentative"
            assert hits[0]["memory"]["visibility"] == "run"
            assert "semantic_match" in hits[0]["reasons"]
            assert len(hits[0]["evidence"]) == 1
            assert hits[0]["evidence"][0]["locator"] == "source"
            assert hits[0]["evidence"][0]["excerpt"] is None
            assert hits[0]["evidence"][0]["metadata"]["attribution"] == "whole_source"

            first_snapshot = await _effect_snapshot(database, actor, source_id)
            assert first_snapshot == (1, 1, 1, 1)
            second_cycle = await runtime_d.extraction_worker().run_once(
                f"ingestion-live-c-{uuid4()}"
            )
            assert second_cycle == ()
            assert (
                await runtime_d.embedding_worker().run_once(
                    f"ingestion-live-embedding-replay-{uuid4()}"
                )
                == ()
            )
            assert await _effect_snapshot(database, actor, source_id) == first_snapshot

            pending_work_id = await _insert_pending_extraction(
                database,
                actor,
                source_id,
                extractor_name=runtime_d.extraction.deterministic.name,
                extractor_revision=runtime_d.extraction.deterministic.revision,
            )
            assert await _work_state(database, actor, pending_work_id) == (
                "pending",
                None,
            )
            rejected = await runtime_d.memory.reject_source(
                actor,
                RejectSourceCommand(
                    idempotency_key=f"live-reject-{uuid4()}",
                    source_id=source_id,
                    expected_version=source_version,
                    reason="live gate rejects the source and fences queued reprocessing",
                ),
            )
            assert rejected.rolled_back_memory_ids == (memory_id,)
            assert rejected.source.review_state.value == "rejected"
            assert await _memory_state(database, actor, memory_id) == "refuted"
            assert await _work_state(database, actor, receipt["work_id"]) == (
                "completed",
                "completed",
            )
            assert await _work_state(database, actor, pending_work_id) == (
                "cancelled",
                "source_rejected",
            )

            cancelled = await client.get(
                f"/v1/sources/{source_id}/extraction",
                headers=auth,
            )
            assert cancelled.status_code == 200, cancelled.text
            assert cancelled.json()["status"] == "cancelled"
            assert cancelled.json()["outcome"] == "source_rejected"

            after_rejection = await client.post(
                "/v1/memories:recall",
                headers=auth,
                json={"text": "serializable SQLSTATE 40001 retry", "limit": 10},
            )
            assert after_rejection.status_code == 200, after_rejection.text
            assert after_rejection.json()["hits"] == []
    finally:
        await _cleanup(database, actor)
