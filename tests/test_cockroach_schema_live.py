from __future__ import annotations

import os
import re
from uuid import uuid4

import pytest
from psycopg import sql

from swarmbrain.adapters.cockroach.database import (
    REQUIRED_COLUMNS,
    REQUIRED_INDEXES,
    CockroachDatabase,
)

DATABASE_URL = os.getenv("SWARMBRAIN_TEST_DATABASE_URL")
VECTOR_1024 = "[1" + ",0" * 1023 + "]"
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(not DATABASE_URL, reason="SWARMBRAIN_TEST_DATABASE_URL is not set"),
]


@pytest.fixture
async def database() -> CockroachDatabase:
    assert DATABASE_URL is not None
    value = CockroachDatabase(DATABASE_URL, min_size=1, max_size=2)
    await value.start()
    try:
        yield value
    finally:
        await value.close()


async def test_show_create_contains_every_verified_table_and_column(
    database: CockroachDatabase,
) -> None:
    async with database.pool.connection() as connection:
        for table, columns in sorted(REQUIRED_COLUMNS.items()):
            cursor = await connection.execute(
                sql.SQL("SHOW CREATE TABLE {}").format(sql.Identifier(table))
            )
            row = await cursor.fetchone()
            assert row is not None, table
            statement = str(row["create_statement"])
            assert "PRIMARY KEY" in statement, table
            for column in columns:
                assert re.search(rf"\b{re.escape(column)}\b", statement), (table, column)


async def test_explain_accepts_every_critical_covering_index(
    database: CockroachDatabase,
) -> None:
    identifier = str(uuid4())
    queries: dict[str, tuple[str, tuple[object, ...]]] = {
        "tasks_claim_queue": (
            """
            SELECT id, required_capabilities, tags, title, description, version
            FROM tasks@tasks_claim_queue
            WHERE tenant_id = %s AND run_id = %s AND state = 'ready'
            ORDER BY priority DESC, created_at, id
            LIMIT 1
            """,
            (identifier, identifier),
        ),
        "task_checkpoints_latest": (
            """
            SELECT lease_id, agent_id, summary, discoveries, completed_work,
                   remaining_work, memory_ids, artifact_ids, state, created_at
            FROM task_checkpoints@task_checkpoints_latest
            WHERE task_id = %s
            ORDER BY sequence DESC
            LIMIT 1
            """,
            (identifier,),
        ),
        "memories_current_scope": (
            """
            SELECT id, task_id, agent_id, kind, confidence, valid_from,
                   valid_to, content
            FROM memories@memories_current_scope
            WHERE tenant_id = %s AND repository_id = %s AND run_id = %s
              AND visibility = 'run' AND state = 'tentative'
              AND recorded_to IS NULL
            ORDER BY recorded_from DESC, id DESC
            LIMIT 20
            """,
            (identifier, identifier, identifier),
        ),
        "memory_vector_embeddings_ann": (
            """
            SELECT memory_id
            FROM memory_vector_embeddings@memory_vector_embeddings_ann
            WHERE tenant_id = %s AND project_id = %s AND repository_id = %s
              AND model = %s
            ORDER BY embedding <=> %s::VECTOR
            LIMIT 20
            """,
            (identifier, identifier, identifier, "deterministic-v0", VECTOR_1024),
        ),
        "retrieval_vectors_1024_ann_v2": (
            """
            SELECT canonical_id
            FROM retrieval_vectors_1024@retrieval_vectors_1024_ann_v2
            WHERE tenant_id = %s AND project_id = %s AND repository_id = %s
              AND resource_type = 'memory' AND projection_id = %s
              AND projection_signature = %s AND scope_key = %s
            ORDER BY embedding <=> %s::VECTOR
            LIMIT 20
            """,
            (
                identifier,
                identifier,
                identifier,
                "memory-content-v1:current:cosine",
                "dense-v2|memory-content-v1:current:cosine|normalization=provider|"
                "truncation=provider|dimensions=1024|model=test",
                f"repository:{identifier}",
                VECTOR_1024,
            ),
        ),
        "evidence_source_lookup": (
            """
            SELECT id, tenant_id, kind, locator, content_sha256, excerpt,
                   artifact, metadata, recorded_at
            FROM evidence@evidence_source_lookup
            WHERE source_id = %s
            ORDER BY id
            """,
            (identifier,),
        ),
        "memory_evidence_by_evidence": (
            """
            SELECT memory_id, relation, created_at
            FROM memory_evidence@memory_evidence_by_evidence
            WHERE evidence_id = %s
            ORDER BY memory_id
            """,
            (identifier,),
        ),
        "memory_links_from_type": (
            """
            SELECT target_memory_id, link_type, created_at, id
            FROM memory_links@memory_links_from_type
            WHERE source_memory_id = %s AND link_type = %s
            ORDER BY created_at DESC, id
            LIMIT 9
            """,
            (identifier, "supports"),
        ),
        "memory_links_to_type": (
            """
            SELECT source_memory_id, link_type, created_at, id
            FROM memory_links@memory_links_to_type
            WHERE target_memory_id = %s AND link_type = %s
            ORDER BY created_at DESC, id
            LIMIT 9
            """,
            (identifier, "supports"),
        ),
        "outbox_events_unpublished": (
            """
            SELECT id, event_id, tenant_id, run_id, aggregate_type,
                   aggregate_id, aggregate_version, event_type, payload, status, version
            FROM outbox_events@outbox_events_unpublished
            WHERE status IN ('pending', 'publishing', 'failed')
              AND available_at <= now()
            ORDER BY available_at, locked_until, id
            LIMIT 20
            """,
            (),
        ),
        "outbox_work_items_expired_leases": (
            """
            SELECT id, attempts, max_attempts
            FROM outbox_work_items@outbox_work_items_expired_leases
            WHERE status = 'leased' AND locked_until <= now()
            ORDER BY locked_until, id
            LIMIT 20
            """,
            (),
        ),
        "idempotency_records_lookup": (
            """
            SELECT request_sha256, status, response_status, response_body,
                   resource_id, expires_at
            FROM idempotency_records@idempotency_records_lookup
            WHERE tenant_id = %s AND run_id = %s AND actor_id = %s
              AND operation = %s AND idempotency_key = %s
            """,
            (identifier, identifier, identifier, "test", "key"),
        ),
        "swarm_events_run_timeline": (
            """
            SELECT id, event_type, aggregate_type, aggregate_id,
                   aggregate_version, payload, recorded_at
            FROM swarm_events@swarm_events_run_timeline
            WHERE tenant_id = %s AND run_id = %s
              AND (occurred_at, id) > (%s::TIMESTAMPTZ, %s::UUID)
            ORDER BY occurred_at, id
            LIMIT 20
            """,
            (identifier, identifier, "1970-01-01T00:00:00Z", identifier),
        ),
    }
    assert queries.keys() <= REQUIRED_INDEXES

    async with database.pool.connection() as connection:
        for index_name, (query, parameters) in queries.items():
            cursor = await connection.execute("EXPLAIN " + query, parameters)
            plan = "\n".join(
                str(value) for row in await cursor.fetchall() for value in row.values()
            )
            assert index_name in plan, plan
            if index_name == "retrieval_vectors_1024_ann_v2":
                assert "vector search" in plan.lower(), plan
