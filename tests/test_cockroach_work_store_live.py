from __future__ import annotations

import asyncio
import hashlib
import json
import os
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import pytest
from psycopg.errors import UniqueViolation

from conftest import make_actor
from swarmbrain.adapters.cockroach.database import CockroachDatabase
from swarmbrain.adapters.cockroach.work_store import CockroachWorkStore
from swarmbrain.adapters.extraction import CodingRuleExtractor
from swarmbrain.application.extraction import ExtractionService
from swarmbrain.application.work import DurableWorkService
from swarmbrain.domain.agents import ActorContext
from swarmbrain.domain.common import ContractModel
from swarmbrain.domain.evidence import EvidenceKind
from swarmbrain.domain.extraction import (
    ExtractionCandidate,
    ExtractionOutcome,
    ExtractionProvenance,
    ExtractionStage,
    IngestRawSourceCommand,
    ProviderDescriptor,
)
from swarmbrain.domain.work import (
    ApplyExtractionWorkCommand,
    ClaimWorkCommand,
    EnqueueWorkCommand,
    FailWorkCommand,
    WorkEffect,
    WorkHandlerResult,
    WorkKind,
    WorkLeaseLost,
    WorkStatus,
)
from swarmbrain.workers import LeasedWorkWorker

DATABASE_URL = os.getenv("SWARMBRAIN_TEST_DATABASE_URL")
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(not DATABASE_URL, reason="SWARMBRAIN_TEST_DATABASE_URL is not set"),
]


class _Ack(ContractModel):
    ok: bool = True


class _CountingProvider:
    descriptor = ProviderDescriptor(provider="live-test", model="validated-candidate")

    def __init__(self) -> None:
        self.calls = 0

    async def extract(self, request: Any) -> list[dict[str, object]]:
        self.calls += 1
        excerpt = request.raw_content[:3]
        return [
            {
                "kind": "observation",
                "content": "Provider observed a Python source file",
                "tags": ["provider", "source-code"],
                "confidence": 0.75,
                "spans": [
                    {
                        "chunk_index": 0,
                        "char_start": 0,
                        "char_end": len(excerpt),
                        "excerpt": excerpt,
                    }
                ],
            }
        ]


class _CountingWorkHandler:
    def __init__(self, kind: WorkKind) -> None:
        self.kind = kind
        self.calls = 0

    async def handle(self, lease: Any) -> WorkHandlerResult:
        self.calls += 1
        return WorkHandlerResult(
            result={
                "kind": self.kind.value,
                "subject_id": lease.item.subject_id,
            }
        )


def _id() -> str:
    return str(uuid4())


def _candidate_hash(candidate: ExtractionCandidate) -> str:
    payload = json.dumps(
        candidate.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _provenance(attempt: int, input_sha256: str) -> ExtractionProvenance:
    timestamp = datetime.now(UTC)
    return ExtractionProvenance(
        stage=ExtractionStage.DETERMINISTIC,
        attempt=attempt,
        outcome=ExtractionOutcome.SUCCEEDED,
        input_sha256=input_sha256,
        output_sha256=hashlib.sha256(b"[]").hexdigest(),
        started_at=timestamp,
        finished_at=timestamp,
    )


@pytest.fixture
async def database() -> CockroachDatabase:
    assert DATABASE_URL is not None
    value = CockroachDatabase(DATABASE_URL, min_size=1, max_size=8)
    await value.start()
    try:
        yield value
    finally:
        await value.close()


async def _insert_run(database: CockroachDatabase, actor: ActorContext) -> None:
    async def transaction(connection: Any) -> _Ack:
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

    await database.run(transaction)


async def _cleanup(database: CockroachDatabase, actor: ActorContext) -> None:
    async def transaction(connection: Any) -> _Ack:
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
            "DELETE FROM memories WHERE tenant_id = %s AND run_id = %s",
            scope,
        )
        await connection.execute(
            "DELETE FROM evidence WHERE tenant_id = %s",
            (actor.tenant_id,),
        )
        await connection.execute(
            "DELETE FROM sources WHERE tenant_id = %s AND run_id = %s",
            scope,
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

    await database.run(transaction)


async def _install_work_id_blocker(
    database: CockroachDatabase,
    actor: ActorContext,
    work_id: str,
) -> None:
    async def transaction(connection: Any) -> _Ack:
        await connection.execute(
            """
            INSERT INTO outbox_work_items (
                id, tenant_id, project_id, repository_id, swarm_id, run_id,
                requested_by_agent_id, kind, subject_id, dedupe_key
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, 'persist_artifact', %s, %s)
            """,
            (
                work_id,
                actor.tenant_id,
                actor.project_id,
                actor.repository_id,
                actor.swarm_id,
                actor.run_id,
                actor.agent_id,
                _id(),
                f"atomicity-blocker:{_id()}",
            ),
        )
        return _Ack()

    await database.run(transaction)


async def _remove_work_id_blocker(
    database: CockroachDatabase,
    actor: ActorContext,
    work_id: str,
) -> None:
    async def transaction(connection: Any) -> _Ack:
        await connection.execute(
            """
            DELETE FROM outbox_work_items
            WHERE id = %s AND tenant_id = %s AND run_id = %s
            """,
            (work_id, actor.tenant_id, actor.run_id),
        )
        return _Ack()

    await database.run(transaction)


async def _expire_lease(database: CockroachDatabase, work_id: str) -> None:
    async def transaction(connection: Any) -> _Ack:
        await connection.execute(
            """
            UPDATE outbox_work_items
            SET locked_until = now() - INTERVAL '1 second'
            WHERE id = %s AND status = 'leased'
            """,
            (work_id,),
        )
        return _Ack()

    await database.run(transaction)


async def _assert_atomic_rollback(
    database: CockroachDatabase,
    source_id: str,
    work_dedupe_key: str,
) -> None:
    async with database.pool.connection() as connection:
        cursor = await connection.execute(
            """
            SELECT
                (SELECT count(*) FROM sources WHERE id = %s) AS source_count,
                (SELECT count(*) FROM source_chunks WHERE source_id = %s) AS chunk_count,
                (
                    SELECT count(*)
                    FROM outbox_work_items
                    WHERE dedupe_key = %s
                ) AS queued_count
            """,
            (source_id, source_id, work_dedupe_key),
        )
        row = await cursor.fetchone()
    assert row is not None
    assert int(row["source_count"]) == 0
    assert int(row["chunk_count"]) == 0
    assert int(row["queued_count"]) == 0


async def _assert_persisted_result(
    database: CockroachDatabase,
    actor: ActorContext,
    source_id: str,
    work_id: str,
    raw_content: str,
    chunk_count: int,
    effect_count: int,
) -> None:
    async with database.pool.connection() as connection:
        cursor = await connection.execute(
            """
            SELECT content
            FROM sources
            WHERE id = %s AND tenant_id = %s AND run_id = %s
            """,
            (source_id, actor.tenant_id, actor.run_id),
        )
        source = await cursor.fetchone()
        assert source is not None
        assert str(source["content"]) == raw_content

        cursor = await connection.execute(
            """
            SELECT content
            FROM source_chunks
            WHERE source_id = %s
            ORDER BY chunk_index
            """,
            (source_id,),
        )
        chunks = await cursor.fetchall()
        assert len(chunks) == chunk_count
        assert "".join(str(row["content"]) for row in chunks) == raw_content

        cursor = await connection.execute(
            """
            SELECT
                work.status,
                work.attempts,
                (SELECT count(*) FROM outbox_work_effects WHERE work_id = work.id)
                    AS effect_count,
                (SELECT count(*) FROM outbox_work_attempts WHERE work_id = work.id)
                    AS attempt_stage_count
            FROM outbox_work_items AS work
            WHERE work.id = %s
            """,
            (work_id,),
        )
        work = await cursor.fetchone()
        assert work is not None
        assert str(work["status"]) == WorkStatus.COMPLETED.value
        assert int(work["attempts"]) == 2
        assert int(work["effect_count"]) == effect_count
        assert int(work["attempt_stage_count"]) == 3

        cursor = await connection.execute(
            """
            SELECT memory.state, memory.visibility, memory.supersedes_id,
                   count(link.evidence_id) AS evidence_count
            FROM outbox_work_effects AS effect
            JOIN memories AS memory ON memory.id = effect.resource_id
            LEFT JOIN memory_evidence AS link ON link.memory_id = memory.id
            WHERE effect.work_id = %s
            GROUP BY memory.id, memory.state, memory.visibility, memory.supersedes_id
            ORDER BY memory.id
            """,
            (work_id,),
        )
        memories = await cursor.fetchall()
        assert len(memories) == effect_count
        assert all(str(row["state"]) == "tentative" for row in memories)
        assert all(str(row["visibility"]) == "run" for row in memories)
        assert all(row["supersedes_id"] is None for row in memories)
        assert all(int(row["evidence_count"]) > 0 for row in memories)

        cursor = await connection.execute(
            """
            SELECT
                (SELECT count(*) FROM swarm_events
                 WHERE tenant_id = %s AND run_id = %s AND event_type = 'memory.added')
                    AS event_count,
                (SELECT count(*) FROM outbox_events
                 WHERE tenant_id = %s AND run_id = %s AND event_type = 'memory.added')
                    AS outbox_count
            """,
            (actor.tenant_id, actor.run_id, actor.tenant_id, actor.run_id),
        )
        events = await cursor.fetchone()
        assert events is not None
        assert int(events["event_count"]) == effect_count
        assert int(events["outbox_count"]) == effect_count


async def _assert_index_plans(
    database: CockroachDatabase,
    source_id: str,
    resource_id: str,
) -> None:
    queries = (
        (
            "outbox_work_items_claim",
            """
            SELECT id
            FROM outbox_work_items@outbox_work_items_claim
            WHERE status IN ('pending', 'retry', 'leased')
              AND available_at <= now()
            ORDER BY priority DESC, available_at, created_at, id
            LIMIT 10
            """,
            (),
        ),
        (
            "outbox_work_items_subject",
            """
            SELECT id, tenant_id, run_id, status, outcome
            FROM outbox_work_items@outbox_work_items_subject
            WHERE subject_id = %s AND kind = 'extract_source'
            ORDER BY created_at DESC, id
            LIMIT 10
            """,
            (source_id,),
        ),
        (
            "outbox_work_effects_resource",
            """
            SELECT work_id, effect_key, payload_sha256, created_at
            FROM outbox_work_effects@outbox_work_effects_resource
            WHERE resource_id = %s AND kind = 'memory'
            """,
            (resource_id,),
        ),
    )
    async with database.pool.connection() as connection:
        for index_name, query, parameters in queries:
            cursor = await connection.execute("EXPLAIN " + query, parameters)
            rows = await cursor.fetchall()
            plan = "\n".join(str(value) for row in rows for value in row.values())
            assert index_name in plan


async def test_durable_extraction_work_queue_and_plans(
    database: CockroachDatabase,
    scope_ids: dict[str, str],
) -> None:
    actor = make_actor(scope_ids)
    await _insert_run(database, actor)
    store = CockroachWorkStore(database)
    provider = _CountingProvider()
    service = ExtractionService(store, CodingRuleExtractor(), provider=provider)
    raw_content = "def load_config():\n    pass\n# TODO: validate the config\n"
    command = IngestRawSourceCommand(
        idempotency_key=f"raw-live-{_id()}",
        kind=EvidenceKind.SOURCE_CODE,
        content=raw_content,
        observed_at=datetime.now(UTC),
        occurrence_key=f"src/config.py@{_id()}",
        use_provider=True,
        chunk_chars=1024,
    )
    prepared = service.prepare_ingest(actor, command)

    try:
        await _install_work_id_blocker(database, actor, prepared.work_id)
        with pytest.raises(UniqueViolation) as collision:
            await service.ingest(actor, command)
        assert getattr(collision.value, "sqlstate", None) == "23505"
        await _assert_atomic_rollback(
            database,
            prepared.source_id,
            prepared.work_dedupe_key,
        )
        await _remove_work_id_blocker(database, actor, prepared.work_id)

        ingested = await service.ingest(actor, command)
        replayed_ingest = await service.ingest(actor, command)
        assert replayed_ingest.replayed is True
        assert replayed_ingest == ingested.model_copy(update={"replayed": True})

        claims = await asyncio.gather(
            store.claim_work(
                ClaimWorkCommand(
                    worker_id="live-worker-a",
                    kinds=frozenset({WorkKind.EXTRACT_SOURCE}),
                    lease_seconds=5,
                )
            ),
            store.claim_work(
                ClaimWorkCommand(
                    worker_id="live-worker-b",
                    kinds=frozenset({WorkKind.EXTRACT_SOURCE}),
                    lease_seconds=5,
                )
            ),
        )
        leases = tuple(lease for batch in claims for lease in batch.leases)
        assert len(leases) == 1
        stale = leases[0]

        await _expire_lease(database, stale.item.work_id)
        handoff = await store.claim_work(
            ClaimWorkCommand(
                worker_id="live-worker-handoff",
                kinds=frozenset({WorkKind.EXTRACT_SOURCE}),
                lease_seconds=30,
            )
        )
        assert len(handoff.leases) == 1
        current = handoff.leases[0]
        assert current.attempt == stale.attempt + 1
        assert current.lease_version == stale.lease_version + 1

        with pytest.raises(WorkLeaseLost):
            await store.apply_extraction(
                ApplyExtractionWorkCommand(
                    work_id=stale.item.work_id,
                    worker_id=stale.worker_id,
                    lease_token=stale.lease_token,
                    lease_version=stale.lease_version,
                    expected_work_version=stale.work_version,
                    attempt=stale.attempt,
                    provenance=(_provenance(stale.attempt, ingested.source.content_sha256),),
                )
            )

        request = await store.load_extraction_input(current)
        extraction = await service.extract(request, use_provider=True)
        effects = tuple(
            WorkEffect(
                effect_key=f"memory:{digest}",
                payload_sha256=digest,
                candidate=candidate,
            )
            for candidate in extraction.candidates
            for digest in (_candidate_hash(candidate),)
        )
        apply_command = ApplyExtractionWorkCommand(
            work_id=current.item.work_id,
            worker_id=current.worker_id,
            lease_token=current.lease_token,
            lease_version=current.lease_version,
            expected_work_version=current.work_version,
            attempt=current.attempt,
            effects=effects,
            provenance=extraction.provenance,
            outcome=extraction.status.value,
            result={
                "route": extraction.route.value,
                "status": extraction.status.value,
                "candidate_count": len(extraction.candidates),
            },
        )

        first = await store.apply_extraction(apply_command)
        repeated = await store.apply_extraction(apply_command)

        assert first.replayed is False
        assert repeated.replayed is True
        assert repeated.effects == first.effects
        assert provider.calls == 1
        assert effects
        await _assert_persisted_result(
            database,
            actor,
            ingested.source.source_id,
            ingested.work_id,
            raw_content,
            len(ingested.chunks),
            len(effects),
        )
        await _assert_index_plans(
            database,
            ingested.source.source_id,
            first.effects[0].resource_id,
        )
    finally:
        await _cleanup(database, actor)


async def test_embedding_and_artifact_handlers_use_durable_generic_work(
    database: CockroachDatabase,
    scope_ids: dict[str, str],
) -> None:
    actor = make_actor(scope_ids)
    await _insert_run(database, actor)
    store = CockroachWorkStore(database)
    service = DurableWorkService(store)
    timestamp = datetime.now(UTC)
    embedding = EnqueueWorkCommand(
        idempotency_key=f"embed-live-{_id()}",
        kind=WorkKind.EMBED_MEMORY,
        subject_id=_id(),
        dedupe_key=f"memory:model-live:{_id()}",
        payload={"content": "A live durable memory", "model": "model-live"},
        available_at=timestamp,
    )
    artifact_content = b"live durable artifact"
    artifact = EnqueueWorkCommand(
        idempotency_key=f"artifact-live-{_id()}",
        kind=WorkKind.PERSIST_ARTIFACT,
        subject_id=_id(),
        dedupe_key=f"artifact:live:{_id()}",
        payload={
            "name": "live.txt",
            "media_type": "text/plain",
            "sha256": hashlib.sha256(artifact_content).hexdigest(),
            "size_bytes": len(artifact_content),
            "staging_uri": "file:///staging/live.txt",
        },
        available_at=timestamp,
    )

    try:
        queued_embedding = await service.enqueue(actor, embedding)
        replayed_embedding = await service.enqueue(actor, embedding)
        queued_artifact = await service.enqueue(actor, artifact)
        assert replayed_embedding.replayed is True
        assert replayed_embedding.item.work_id == queued_embedding.item.work_id

        embedding_handler = _CountingWorkHandler(WorkKind.EMBED_MEMORY)
        artifact_handler = _CountingWorkHandler(WorkKind.PERSIST_ARTIFACT)
        worker = LeasedWorkWorker(store, (embedding_handler, artifact_handler))
        completed = await worker.run_once("live-handler-worker", limit=2)
        empty = await worker.run_once("live-handler-worker", limit=2)

        assert len(completed) == 2
        assert empty == ()
        assert embedding_handler.calls == 1
        assert artifact_handler.calls == 1
        assert {item.item.work_id for item in completed} == {
            queued_embedding.item.work_id,
            queued_artifact.item.work_id,
        }

        async with database.pool.connection() as connection:
            cursor = await connection.execute(
                """
                SELECT kind, status, attempts, outcome, result
                FROM outbox_work_items
                WHERE tenant_id = %s AND run_id = %s
                  AND kind IN ('embed_memory', 'persist_artifact')
                ORDER BY kind
                """,
                (actor.tenant_id, actor.run_id),
            )
            rows = await cursor.fetchall()
        assert len(rows) == 2
        assert all(str(row["status"]) == WorkStatus.COMPLETED.value for row in rows)
        assert all(int(row["attempts"]) == 1 for row in rows)
        assert all(str(row["outcome"]) == "completed" for row in rows)
        assert {str(row["result"]["kind"]) for row in rows} == {
            WorkKind.EMBED_MEMORY.value,
            WorkKind.PERSIST_ARTIFACT.value,
        }

        retry_command = artifact.model_copy(
            update={
                "idempotency_key": f"artifact-retry-live-{_id()}",
                "subject_id": _id(),
                "dedupe_key": f"artifact:retry-live:{_id()}",
            }
        )
        retried = await service.enqueue(actor, retry_command)
        lease = (
            await store.claim_work(
                ClaimWorkCommand(
                    worker_id="live-failure-worker",
                    kinds=frozenset({WorkKind.PERSIST_ARTIFACT}),
                )
            )
        ).leases[0]
        failure = FailWorkCommand(
            work_id=retried.item.work_id,
            worker_id=lease.worker_id,
            lease_token=lease.lease_token,
            lease_version=lease.lease_version,
            expected_work_version=lease.work_version,
            attempt=lease.attempt,
            error="handler_LiveFailure",
            retry_at=datetime.now(UTC) + timedelta(seconds=60),
        )
        first_failure = await store.fail_work(failure)
        replayed_failure = await store.fail_work(failure)
        assert replayed_failure == first_failure
        assert replayed_failure.status is WorkStatus.RETRY
    finally:
        await _cleanup(database, actor)
