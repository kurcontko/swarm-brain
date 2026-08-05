"""CockroachDB raw-ingest and fenced durable-work repository."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from datetime import timedelta
from typing import Any
from uuid import UUID, uuid5

from psycopg.types.json import Jsonb

from swarmbrain.application.errors import IdempotencyConflict, ResourceNotFound
from swarmbrain.application.memory_policy import (
    memory_content_text,
    memory_text_sha256,
)
from swarmbrain.domain.agents import ActorContext
from swarmbrain.domain.events import AggregateType, EventType, SwarmEvent
from swarmbrain.domain.evidence import EvidenceSource, SourceReviewState, SourceTrust
from swarmbrain.domain.extraction import (
    ExtractionInput,
    ExtractionProvenance,
    PreparedSourceIngest,
    SourceChunk,
    SourceIngestResult,
)
from swarmbrain.domain.work import (
    AppliedWorkEffect,
    ApplyEmbeddingWorkCommand,
    ApplyExtractionWorkCommand,
    ApplyWorkResult,
    ClaimWorkCommand,
    CompleteWorkCommand,
    CompleteWorkResult,
    EnqueueWorkResult,
    FailWorkCommand,
    PreparedWorkEnqueue,
    SourceExtractionStatus,
    WorkCompletionConflict,
    WorkEffect,
    WorkEffectConflict,
    WorkEffectKind,
    WorkFailureConflict,
    WorkItem,
    WorkKind,
    WorkLease,
    WorkLeaseBatch,
    WorkLeaseLost,
    WorkStatus,
    durable_work_id,
    source_extraction_state,
)

from .database import CockroachDatabase
from .vector import vector_literal

WORK_COLUMNS = """
    id, tenant_id, project_id, repository_id, swarm_id, run_id, task_id,
    requested_by_agent_id, kind, subject_id, payload, dedupe_key, priority,
    status, attempts, max_attempts, available_at, locked_by, lease_token,
    lease_version, locked_until, outcome, result, last_error, created_at,
    updated_at, completed_at, version
"""
WORK_ALIAS_COLUMNS = """
    work.id, work.tenant_id, work.project_id, work.repository_id,
    work.swarm_id, work.run_id, work.task_id, work.requested_by_agent_id,
    work.kind, work.subject_id, work.payload, work.dedupe_key, work.priority,
    work.status, work.attempts, work.max_attempts, work.available_at,
    work.locked_by, work.lease_token, work.lease_version, work.locked_until,
    work.outcome, work.result, work.last_error, work.created_at,
    work.updated_at, work.completed_at, work.version
"""


def _uuid(value: str | UUID) -> UUID:
    return value if isinstance(value, UUID) else UUID(value)


def _json_object(value: object | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("Cockroach JSONB value did not decode to an object")
    return dict(value)


def _work_from_row(row: Mapping[str, Any]) -> WorkItem:
    return WorkItem(
        work_id=str(row["id"]),
        tenant_id=str(row["tenant_id"]),
        project_id=str(row["project_id"]),
        repository_id=str(row["repository_id"]),
        swarm_id=str(row["swarm_id"]),
        run_id=str(row["run_id"]),
        task_id=str(row["task_id"]) if row.get("task_id") is not None else None,
        requested_by_agent_id=str(row["requested_by_agent_id"]),
        kind=WorkKind(str(row["kind"])),
        subject_id=str(row["subject_id"]),
        payload=_json_object(row.get("payload")) or {},
        dedupe_key=str(row["dedupe_key"]),
        priority=int(row["priority"]),
        status=WorkStatus(str(row["status"])),
        attempts=int(row["attempts"]),
        max_attempts=int(row["max_attempts"]),
        available_at=row["available_at"],
        locked_by=row.get("locked_by"),
        lease_token=(str(row["lease_token"]) if row.get("lease_token") is not None else None),
        lease_version=int(row["lease_version"]),
        locked_until=row.get("locked_until"),
        outcome=row.get("outcome"),
        result=_json_object(row.get("result")),
        last_error=row.get("last_error"),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        completed_at=row.get("completed_at"),
        version=int(row["version"]),
    )


def _source_from_row(row: Mapping[str, Any]) -> EvidenceSource:
    return EvidenceSource(
        source_id=str(row["id"]),
        run_id=str(row["run_id"]),
        task_id=str(row["task_id"]) if row.get("task_id") is not None else None,
        kind=str(row["source_type"]),
        uri=row.get("uri"),
        content_sha256=bytes(row["content_sha256"]).hex(),
        occurrence_key=str(row["occurrence_key"]),
        trust=SourceTrust(str(row["trust_label"])),
        review_state=SourceReviewState(str(row["review_state"])),
        observed_at=row.get("valid_at") or row["recorded_at"],
        recorded_at=row["recorded_at"],
        reviewed_by_agent_id=(
            str(row["reviewed_by_agent_id"])
            if row.get("reviewed_by_agent_id") is not None
            else None
        ),
        reviewed_at=row.get("reviewed_at"),
        rejection_reason=row.get("rejection_reason"),
        version=int(row["version"]),
        metadata=_json_object(row.get("metadata")) or {},
    )


class CockroachWorkStore:
    """Share one database while keeping remote/model work outside `database.run`."""

    def __init__(
        self,
        database: CockroachDatabase,
    ) -> None:
        self.database = database

    async def enqueue_work(
        self,
        actor: ActorContext,
        prepared: PreparedWorkEnqueue,
    ) -> EnqueueWorkResult:
        command = prepared.command

        async def body(connection: Any) -> EnqueueWorkResult:
            if command.task_id is not None:
                await self._require_task(connection, actor, command.task_id)
            cursor = await connection.execute(
                f"""
                INSERT INTO outbox_work_items (
                    id, tenant_id, project_id, repository_id, swarm_id, run_id,
                    task_id, requested_by_agent_id, kind, subject_id, payload,
                    dedupe_key, priority, max_attempts, available_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s
                )
                ON CONFLICT (tenant_id, run_id, kind, dedupe_key) DO NOTHING
                RETURNING {WORK_COLUMNS}
                """,
                (
                    _uuid(prepared.work_id),
                    actor.tenant_id,
                    actor.project_id,
                    actor.repository_id,
                    actor.swarm_id,
                    actor.run_id,
                    _uuid(command.task_id) if command.task_id else None,
                    actor.agent_id,
                    str(command.kind),
                    _uuid(command.subject_id),
                    Jsonb(command.payload),
                    command.dedupe_key,
                    command.priority,
                    command.max_attempts,
                    command.available_at,
                ),
            )
            row = await cursor.fetchone()
            replayed = row is None
            if row is None:
                cursor = await connection.execute(
                    f"""
                    SELECT {WORK_COLUMNS}
                    FROM outbox_work_items
                    WHERE tenant_id = %s
                      AND run_id = %s
                      AND kind = %s
                      AND dedupe_key = %s
                    FOR UPDATE
                    """,
                    (
                        actor.tenant_id,
                        actor.run_id,
                        command.kind.value,
                        command.dedupe_key,
                    ),
                )
                row = await cursor.fetchone()
            if row is None or (
                str(row["id"]) != prepared.work_id
                or str(row["project_id"]) != actor.project_id
                or str(row["repository_id"]) != actor.repository_id
                or str(row["swarm_id"]) != actor.swarm_id
                or str(row["subject_id"]) != command.subject_id
                or (str(row["task_id"]) if row.get("task_id") is not None else None)
                != command.task_id
                or (_json_object(row.get("payload")) or {}) != command.payload
            ):
                raise IdempotencyConflict(command.idempotency_key)
            return EnqueueWorkResult(item=_work_from_row(row), replayed=replayed)

        return await self.database.run_idempotent(
            actor,
            "work.enqueue",
            command,
            EnqueueWorkResult,
            body,
        )

    async def ingest_source(
        self,
        actor: ActorContext,
        prepared: PreparedSourceIngest,
    ) -> SourceIngestResult:
        command = prepared.command
        occurrence = command.occurrence_key or command.idempotency_key

        async def body(connection: Any) -> SourceIngestResult:
            if command.task_id is not None:
                await self._require_task(connection, actor, command.task_id)

            source_cursor = await connection.execute(
                """
                INSERT INTO sources (
                    id, tenant_id, project_id, repository_id, swarm_id, run_id,
                    task_id, agent_id, source_type, occurrence_key, uri,
                    content_sha256, content, trust_label, review_state, valid_at,
                    metadata
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, 'pending', %s, %s
                )
                ON CONFLICT (tenant_id, repository_id, run_id, occurrence_key)
                DO NOTHING
                RETURNING
                    id, run_id, task_id, source_type, uri, content_sha256,
                    occurrence_key, trust_label, review_state, valid_at,
                    recorded_at, reviewed_by_agent_id, reviewed_at,
                    rejection_reason, version, metadata
                """,
                (
                    _uuid(prepared.source_id),
                    actor.tenant_id,
                    actor.project_id,
                    actor.repository_id,
                    actor.swarm_id,
                    actor.run_id,
                    _uuid(command.task_id) if command.task_id else None,
                    actor.agent_id,
                    str(command.kind),
                    occurrence,
                    command.uri,
                    bytes.fromhex(command.content_sha256),
                    command.content,
                    command.trust.value,
                    command.observed_at,
                    Jsonb(command.metadata),
                ),
            )
            source_row = await source_cursor.fetchone()
            inserted_source = source_row is not None
            if source_row is None:
                existing_cursor = await connection.execute(
                    """
                    SELECT
                        id, tenant_id, project_id, repository_id, swarm_id, run_id,
                        task_id, source_type, uri, content_sha256, occurrence_key,
                        trust_label, review_state, valid_at, recorded_at,
                        reviewed_by_agent_id, reviewed_at, rejection_reason,
                        version, metadata
                    FROM sources
                    WHERE tenant_id = %s
                      AND repository_id = %s
                      AND run_id = %s
                      AND occurrence_key = %s
                    FOR UPDATE
                    """,
                    (actor.tenant_id, actor.repository_id, actor.run_id, occurrence),
                )
                source_row = await existing_cursor.fetchone()
                if source_row is None or (
                    str(source_row["id"]) != prepared.source_id
                    or str(source_row["project_id"]) != actor.project_id
                    or str(source_row["swarm_id"]) != actor.swarm_id
                    or bytes(source_row["content_sha256"]).hex() != command.content_sha256
                ):
                    raise IdempotencyConflict(command.idempotency_key)

            if inserted_source:
                await connection.execute(
                    """
                    INSERT INTO source_chunks (
                        id, source_id, chunk_index, content, token_count,
                        char_start, char_end
                    )
                    SELECT id, %s, chunk_index, content, token_count, char_start, char_end
                    FROM UNNEST(
                        %s::UUID[], %s::INT8[], %s::STRING[], %s::INT8[],
                        %s::INT8[], %s::INT8[]
                    ) AS chunks(id, chunk_index, content, token_count, char_start, char_end)
                    """,
                    (
                        _uuid(prepared.source_id),
                        [_uuid(chunk.chunk_id) for chunk in prepared.chunks],
                        [chunk.chunk_index for chunk in prepared.chunks],
                        [chunk.content for chunk in prepared.chunks],
                        [chunk.token_count for chunk in prepared.chunks],
                        [chunk.char_start for chunk in prepared.chunks],
                        [chunk.char_end for chunk in prepared.chunks],
                    ),
                )

            chunks_cursor = await connection.execute(
                """
                SELECT id, source_id, chunk_index, content, token_count,
                       char_start, char_end, metadata
                FROM source_chunks
                WHERE source_id = %s
                ORDER BY chunk_index
                """,
                (_uuid(prepared.source_id),),
            )
            chunks = tuple(self._chunk_from_row(row) for row in await chunks_cursor.fetchall())
            self._require_exact_chunks(prepared, chunks)

            cancellation_reason = self._cancellation_reason(prepared)
            work_cursor = await connection.execute(
                f"""
                INSERT INTO outbox_work_items (
                    id, tenant_id, project_id, repository_id, swarm_id, run_id,
                    task_id, requested_by_agent_id, kind, subject_id, payload,
                    dedupe_key, priority, status, outcome
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, 'extract_source', %s,
                    %s, %s, %s, %s, %s
                )
                ON CONFLICT (tenant_id, run_id, kind, dedupe_key) DO NOTHING
                RETURNING {WORK_COLUMNS}
                """,
                (
                    _uuid(prepared.work_id),
                    actor.tenant_id,
                    actor.project_id,
                    actor.repository_id,
                    actor.swarm_id,
                    actor.run_id,
                    _uuid(command.task_id) if command.task_id else None,
                    actor.agent_id,
                    _uuid(prepared.source_id),
                    Jsonb(
                        {
                            "extractor_name": prepared.extractor_name,
                            "extractor_revision": prepared.extractor_revision,
                            "use_provider": command.use_provider,
                            "embedding_model": prepared.embedding_model,
                        }
                    ),
                    prepared.work_dedupe_key,
                    command.priority,
                    (
                        WorkStatus.CANCELLED.value
                        if cancellation_reason is not None
                        else WorkStatus.PENDING.value
                    ),
                    cancellation_reason,
                ),
            )
            work_row = await work_cursor.fetchone()
            if work_row is None:
                work_cursor = await connection.execute(
                    f"""
                    SELECT {WORK_COLUMNS}
                    FROM outbox_work_items
                    WHERE tenant_id = %s
                      AND run_id = %s
                      AND kind = 'extract_source'
                      AND dedupe_key = %s
                    FOR UPDATE
                    """,
                    (actor.tenant_id, actor.run_id, prepared.work_dedupe_key),
                )
                work_row = await work_cursor.fetchone()
                if work_row is None or (
                    str(work_row["id"]) != prepared.work_id
                    or str(work_row["subject_id"]) != prepared.source_id
                ):
                    raise IdempotencyConflict(command.idempotency_key)

            return SourceIngestResult(
                source=_source_from_row(source_row),
                work_id=str(work_row["id"]),
                chunk_count=len(chunks),
                replayed=not inserted_source,
            )

        return await self.database.run_idempotent(
            actor,
            "source.ingest_raw",
            command,
            SourceIngestResult,
            body,
        )

    async def get_extraction_status(
        self,
        actor: ActorContext,
        source_id: str,
    ) -> SourceExtractionStatus:
        async def body(connection: Any) -> SourceExtractionStatus:
            cursor = await connection.execute(
                f"""
                SELECT {WORK_ALIAS_COLUMNS}
                FROM outbox_work_items AS work
                JOIN sources AS source ON source.id = work.subject_id
                WHERE source.id = %s
                  AND source.tenant_id = %s
                  AND source.project_id = %s
                  AND source.repository_id = %s
                  AND source.swarm_id = %s
                  AND source.run_id = %s
                  AND work.kind = 'extract_source'
                ORDER BY work.created_at DESC, work.id DESC
                LIMIT 1
                """,
                (
                    _uuid(source_id),
                    actor.tenant_id,
                    actor.project_id,
                    actor.repository_id,
                    actor.swarm_id,
                    actor.run_id,
                ),
            )
            row = await cursor.fetchone()
            if row is None:
                raise ResourceNotFound("source", source_id)
            item = _work_from_row(row)
            effects_cursor = await connection.execute(
                """
                SELECT resource_id
                FROM outbox_work_effects
                WHERE work_id = %s AND kind = 'memory'
                ORDER BY resource_id
                """,
                (_uuid(item.work_id),),
            )
            memory_ids = tuple(
                str(effect["resource_id"]) for effect in await effects_cursor.fetchall()
            )
            return SourceExtractionStatus(
                source_id=source_id,
                work_id=item.work_id,
                status=source_extraction_state(item.status),
                attempts=item.attempts,
                max_attempts=item.max_attempts,
                outcome=item.outcome,
                route=(item.result or {}).get("route"),
                candidate_count=(item.result or {}).get("candidate_count"),
                memory_ids=memory_ids,
                created_at=item.created_at,
                updated_at=item.updated_at,
                completed_at=item.completed_at,
            )

        return await self.database.run(body)

    async def claim_work(self, command: ClaimWorkCommand) -> WorkLeaseBatch:
        async def body(connection: Any) -> WorkLeaseBatch:
            await connection.execute(
                """
                UPDATE outbox_work_items
                SET status = 'failed',
                    outcome = 'failed',
                    last_error = 'lease_expired',
                    locked_by = NULL,
                    lease_token = NULL,
                    locked_until = NULL,
                    completed_at = now(),
                    updated_at = now(),
                    version = version + 1
                WHERE status = 'leased'
                  AND locked_until <= now()
                  AND attempts >= max_attempts
                """
            )
            cursor = await connection.execute(
                f"""
                WITH candidates AS (
                    SELECT work.id
                    FROM outbox_work_items AS work
                    WHERE work.kind = ANY(%s::STRING[])
                      AND work.attempts < work.max_attempts
                      AND work.available_at <= now()
                      AND (
                          %s::STRING IS NULL
                          OR work.kind != 'extract_source'
                          OR (
                              work.payload->>'extractor_name' = %s
                              AND work.payload->>'extractor_revision' = %s
                          )
                      )
                      AND (
                          %s::STRING IS NULL
                          OR work.kind != 'embed_memory'
                          OR work.payload->>'model' = %s
                      )
                      AND (
                          work.status IN ('pending', 'retry')
                          OR (work.status = 'leased' AND work.locked_until <= now())
                      )
                      AND (
                          work.kind != 'extract_source'
                          OR EXISTS (
                              SELECT 1
                              FROM sources AS source
                              WHERE source.id = work.subject_id
                                AND source.review_state != 'rejected'
                                AND source.trust_label != 'untrusted'
                          )
                      )
                    ORDER BY
                        work.priority DESC, work.available_at, work.created_at, work.id
                    LIMIT %s
                    FOR UPDATE SKIP LOCKED
                ), claimed AS (
                    UPDATE outbox_work_items AS work
                    SET status = 'leased',
                        attempts = work.attempts + 1,
                        locked_by = %s,
                        lease_token = gen_random_uuid(),
                        lease_version = work.lease_version + 1,
                        locked_until = now() + %s,
                        updated_at = now(),
                        version = work.version + 1
                    FROM candidates
                    WHERE work.id = candidates.id
                    RETURNING {WORK_ALIAS_COLUMNS}
                )
                SELECT {WORK_COLUMNS}
                FROM claimed
                ORDER BY priority DESC, available_at, created_at, id
                """,
                (
                    sorted(kind.value for kind in command.kinds),
                    command.extractor_name,
                    command.extractor_name,
                    command.extractor_revision,
                    command.embedding_model,
                    command.embedding_model,
                    command.limit,
                    command.worker_id,
                    timedelta(seconds=command.lease_seconds),
                ),
            )
            rows = await cursor.fetchall()
            leases = []
            for row in rows:
                item = _work_from_row(row)
                assert (
                    item.locked_by is not None
                    and item.lease_token is not None
                    and item.locked_until is not None
                )
                leases.append(
                    WorkLease(
                        item=item,
                        worker_id=item.locked_by,
                        lease_token=item.lease_token,
                        lease_version=item.lease_version,
                        work_version=item.version,
                        attempt=item.attempts,
                        locked_until=item.locked_until,
                    )
                )
            return WorkLeaseBatch(leases=tuple(leases))

        return await self.database.run(body)

    async def load_extraction_input(self, lease: WorkLease) -> ExtractionInput:
        async def body(connection: Any) -> ExtractionInput:
            cursor = await connection.execute(
                """
                SELECT
                    work.id AS work_id, work.attempts, work.status, work.locked_by,
                    work.lease_token, work.lease_version, work.locked_until,
                    work.version AS work_version, now() AS database_now,
                    source.id, source.run_id, source.task_id, source.source_type,
                    source.uri, source.content_sha256, source.occurrence_key,
                    source.trust_label, source.review_state, source.valid_at,
                    source.recorded_at, source.reviewed_by_agent_id,
                    source.reviewed_at, source.rejection_reason, source.version,
                    source.metadata, source.content
                FROM outbox_work_items AS work
                JOIN sources AS source ON source.id = work.subject_id
                WHERE work.id = %s
                FOR UPDATE OF work
                """,
                (_uuid(lease.item.work_id),),
            )
            row = await cursor.fetchone()
            if row is None:
                raise WorkLeaseLost(lease.item.work_id)
            self._require_lease_row(
                row,
                worker_id=lease.worker_id,
                lease_token=lease.lease_token,
                lease_version=lease.lease_version,
                work_version=lease.work_version,
                attempt=lease.attempt,
            )
            chunks_cursor = await connection.execute(
                """
                SELECT id, source_id, chunk_index, content, token_count,
                       char_start, char_end, metadata
                FROM source_chunks
                WHERE source_id = %s
                ORDER BY chunk_index
                """,
                (_uuid(str(row["id"])),),
            )
            chunk_rows = await chunks_cursor.fetchall()
            chunks = tuple(self._chunk_from_row(chunk) for chunk in chunk_rows)
            return ExtractionInput(
                work_id=str(row["work_id"]),
                attempt=int(row["attempts"]),
                source=_source_from_row(row),
                raw_content=str(row["content"]),
                chunks=chunks,
            )

        return await self.database.run(body)

    async def apply_extraction(
        self,
        command: ApplyExtractionWorkCommand,
    ) -> ApplyWorkResult:
        async def body(connection: Any) -> ApplyWorkResult:
            row = await self._locked_work_row(connection, command.work_id)
            item = _work_from_row(row)
            if item.status is WorkStatus.COMPLETED:
                self._require_completion_fence(item, command)
                if item.outcome != command.outcome or (item.result or {}) != command.result:
                    raise WorkEffectConflict("completed_result")
                effects = await self._load_effects(connection, command)
                return ApplyWorkResult(item=item, effects=effects, replayed=True)

            self._require_lease_row(
                row,
                worker_id=command.worker_id,
                lease_token=command.lease_token,
                lease_version=command.lease_version,
                work_version=command.expected_work_version,
                attempt=command.attempt,
            )
            source_cursor = await connection.execute(
                """
                SELECT id, source_type, content, content_sha256, valid_at, recorded_at
                FROM sources
                WHERE id = %s
                  AND tenant_id = %s
                  AND project_id = %s
                  AND repository_id = %s
                  AND swarm_id = %s
                  AND run_id = %s
                  AND review_state != 'rejected'
                  AND trust_label != 'untrusted'
                """,
                (
                    _uuid(item.subject_id),
                    item.tenant_id,
                    item.project_id,
                    item.repository_id,
                    item.swarm_id,
                    item.run_id,
                ),
            )
            source_row = await source_cursor.fetchone()
            if source_row is None:
                raise WorkLeaseLost(command.work_id)

            chunks_cursor = await connection.execute(
                """
                SELECT chunk_index, content, char_start, char_end
                FROM source_chunks
                WHERE source_id = %s
                ORDER BY chunk_index
                """,
                (_uuid(item.subject_id),),
            )
            self._validate_effect_spans(
                command.effects,
                source_row.get("content"),
                await chunks_cursor.fetchall(),
            )

            await self._insert_attempts(connection, command.work_id, command.provenance)
            applied: list[AppliedWorkEffect] = []
            for effect in command.effects:
                applied_effect = await self._apply_memory_effect(
                    connection,
                    item,
                    source_row,
                    effect,
                )
                applied.append(applied_effect)
                if isinstance(item.payload.get("embedding_model"), str):
                    await self._enqueue_effect_embedding(
                        connection,
                        item,
                        effect,
                        applied_effect,
                    )

            update_cursor = await connection.execute(
                f"""
                UPDATE outbox_work_items
                SET status = 'completed',
                    outcome = %s,
                    result = %s,
                    last_error = NULL,
                    locked_until = NULL,
                    completed_at = now(),
                    updated_at = now(),
                    version = version + 1
                WHERE id = %s
                  AND status = 'leased'
                  AND locked_by = %s
                  AND lease_token = %s
                  AND lease_version = %s
                  AND version = %s
                  AND locked_until > now()
                RETURNING {WORK_COLUMNS}
                """,
                (
                    command.outcome,
                    Jsonb(command.result),
                    _uuid(command.work_id),
                    command.worker_id,
                    _uuid(command.lease_token),
                    command.lease_version,
                    command.expected_work_version,
                ),
            )
            updated = await update_cursor.fetchone()
            if updated is None:
                raise WorkLeaseLost(command.work_id)
            return ApplyWorkResult(item=_work_from_row(updated), effects=tuple(applied))

        return await self.database.run(body)

    async def apply_embedding(
        self,
        command: ApplyEmbeddingWorkCommand,
    ) -> CompleteWorkResult:
        """Atomically fence the lease, upsert its vector, and complete the work."""

        literal = vector_literal(command.vector.values)

        async def body(connection: Any) -> CompleteWorkResult:
            row = await self._locked_work_row(connection, command.work_id)
            item = _work_from_row(row)
            self._validate_embedding_apply(item, command)
            if item.status is WorkStatus.COMPLETED:
                self._require_completion_fence(item, command)
                if item.outcome != command.outcome or (item.result or {}) != command.result:
                    raise WorkCompletionConflict(command.work_id)
                replay_cursor = await connection.execute(
                    """
                    SELECT dimensions, embedding = %s::VECTOR AS vector_matches
                    FROM memory_vector_embeddings
                    WHERE memory_id = %s
                      AND tenant_id = %s
                      AND project_id = %s
                      AND repository_id = %s
                      AND model = %s
                    """,
                    (
                        literal,
                        _uuid(command.vector.memory_id),
                        item.tenant_id,
                        item.project_id,
                        item.repository_id,
                        command.vector.model,
                    ),
                )
                replay_row = await replay_cursor.fetchone()
                if replay_row is None or (
                    int(replay_row["dimensions"]) != command.vector.dimensions
                    or not bool(replay_row["vector_matches"])
                ):
                    raise WorkCompletionConflict("completed_embedding_vector")
                return CompleteWorkResult(item=item, replayed=True)

            self._require_lease_row(
                row,
                worker_id=command.worker_id,
                lease_token=command.lease_token,
                lease_version=command.lease_version,
                work_version=command.expected_work_version,
                attempt=command.attempt,
            )
            vector_cursor = await connection.execute(
                """
                INSERT INTO memory_vector_embeddings (
                    memory_id, tenant_id, project_id, repository_id,
                    model, dimensions, embedding
                )
                SELECT memory.id, memory.tenant_id, memory.project_id,
                       memory.repository_id, %s, %s, %s::VECTOR
                FROM memories AS memory
                WHERE memory.id = %s
                  AND memory.tenant_id = %s
                  AND memory.project_id = %s
                  AND memory.repository_id = %s
                  AND memory.swarm_id = %s
                  AND memory.run_id = %s
                ON CONFLICT (memory_id, model) DO UPDATE SET
                    dimensions = excluded.dimensions,
                    embedding = excluded.embedding,
                    created_at = now()
                RETURNING memory_id
                """,
                (
                    command.vector.model,
                    command.vector.dimensions,
                    literal,
                    _uuid(command.vector.memory_id),
                    item.tenant_id,
                    item.project_id,
                    item.repository_id,
                    item.swarm_id,
                    item.run_id,
                ),
            )
            if await vector_cursor.fetchone() is None:
                raise ResourceNotFound("memory", command.vector.memory_id)

            update_cursor = await connection.execute(
                f"""
                UPDATE outbox_work_items
                SET status = 'completed',
                    outcome = %s,
                    result = %s,
                    last_error = NULL,
                    locked_until = NULL,
                    completed_at = now(),
                    updated_at = now(),
                    version = version + 1
                WHERE id = %s
                  AND status = 'leased'
                  AND locked_by = %s
                  AND lease_token = %s
                  AND lease_version = %s
                  AND version = %s
                  AND locked_until > now()
                RETURNING {WORK_COLUMNS}
                """,
                (
                    command.outcome,
                    Jsonb(command.result),
                    _uuid(command.work_id),
                    command.worker_id,
                    _uuid(command.lease_token),
                    command.lease_version,
                    command.expected_work_version,
                ),
            )
            updated = await update_cursor.fetchone()
            if updated is None:
                # Raising rolls back the vector upsert in this same transaction.
                raise WorkLeaseLost(command.work_id)
            return CompleteWorkResult(item=_work_from_row(updated))

        return await self.database.run(body)

    async def complete_work(self, command: CompleteWorkCommand) -> CompleteWorkResult:
        async def body(connection: Any) -> CompleteWorkResult:
            row = await self._locked_work_row(connection, command.work_id)
            item = _work_from_row(row)
            if item.status is WorkStatus.COMPLETED:
                self._require_completion_fence(item, command)
                if item.outcome != command.outcome or (item.result or {}) != command.result:
                    raise WorkCompletionConflict(command.work_id)
                return CompleteWorkResult(item=item, replayed=True)

            self._require_lease_row(
                row,
                worker_id=command.worker_id,
                lease_token=command.lease_token,
                lease_version=command.lease_version,
                work_version=command.expected_work_version,
                attempt=command.attempt,
            )
            if item.kind in {WorkKind.EXTRACT_SOURCE, WorkKind.EMBED_MEMORY}:
                raise WorkCompletionConflict(f"{item.kind.value}_requires_fenced_apply")
            cursor = await connection.execute(
                f"""
                UPDATE outbox_work_items
                SET status = 'completed',
                    outcome = %s,
                    result = %s,
                    last_error = NULL,
                    locked_until = NULL,
                    completed_at = now(),
                    updated_at = now(),
                    version = version + 1
                WHERE id = %s
                  AND status = 'leased'
                  AND locked_by = %s
                  AND lease_token = %s
                  AND lease_version = %s
                  AND version = %s
                  AND locked_until > now()
                RETURNING {WORK_COLUMNS}
                """,
                (
                    command.outcome,
                    Jsonb(command.result),
                    _uuid(command.work_id),
                    command.worker_id,
                    _uuid(command.lease_token),
                    command.lease_version,
                    command.expected_work_version,
                ),
            )
            updated = await cursor.fetchone()
            if updated is None:
                raise WorkLeaseLost(command.work_id)
            return CompleteWorkResult(item=_work_from_row(updated))

        return await self.database.run(body)

    async def fail_work(self, command: FailWorkCommand) -> WorkItem:
        async def body(connection: Any) -> WorkItem:
            row = await self._locked_work_row(connection, command.work_id)
            item = _work_from_row(row)
            if item.status in {WorkStatus.RETRY, WorkStatus.FAILED}:
                self._require_failure_replay(item, command)
                return item
            self._require_lease_row(
                row,
                worker_id=command.worker_id,
                lease_token=command.lease_token,
                lease_version=command.lease_version,
                work_version=command.expected_work_version,
                attempt=command.attempt,
            )
            await self._insert_attempts(connection, command.work_id, command.provenance)
            cursor = await connection.execute(
                f"""
                UPDATE outbox_work_items
                SET status = CASE WHEN attempts < max_attempts THEN 'retry' ELSE 'failed' END,
                    available_at = %s,
                    locked_until = NULL,
                    outcome = CASE WHEN attempts < max_attempts THEN NULL ELSE 'failed' END,
                    last_error = %s,
                    completed_at = CASE WHEN attempts < max_attempts THEN NULL ELSE now() END,
                    updated_at = now(),
                    version = version + 1
                WHERE id = %s
                  AND status = 'leased'
                  AND locked_by = %s
                  AND lease_token = %s
                  AND lease_version = %s
                  AND version = %s
                  AND locked_until > now()
                RETURNING {WORK_COLUMNS}
                """,
                (
                    command.retry_at,
                    command.error,
                    _uuid(command.work_id),
                    command.worker_id,
                    _uuid(command.lease_token),
                    command.lease_version,
                    command.expected_work_version,
                ),
            )
            updated = await cursor.fetchone()
            if updated is None:
                raise WorkLeaseLost(command.work_id)
            return _work_from_row(updated)

        return await self.database.run(body)

    @staticmethod
    def _require_failure_replay(item: WorkItem, command: FailWorkCommand) -> None:
        expected_status = (
            WorkStatus.RETRY if item.attempts < item.max_attempts else WorkStatus.FAILED
        )
        if (
            item.status is not expected_status
            or item.locked_by != command.worker_id
            or item.lease_token != command.lease_token
            or item.lease_version != command.lease_version
            or item.attempts != command.attempt
            or item.version != command.expected_work_version + 1
        ):
            raise WorkLeaseLost(command.work_id)
        if item.last_error != command.error or item.available_at != command.retry_at:
            raise WorkFailureConflict(command.work_id)

    async def _locked_work_row(self, connection: Any, work_id: str) -> Mapping[str, Any]:
        cursor = await connection.execute(
            f"""
            SELECT {WORK_COLUMNS}, now() AS database_now
            FROM outbox_work_items
            WHERE id = %s
            FOR UPDATE
            """,
            (_uuid(work_id),),
        )
        row = await cursor.fetchone()
        if row is None:
            raise WorkLeaseLost(work_id)
        return row

    @staticmethod
    async def _require_task(
        connection: Any,
        actor: ActorContext,
        task_id: str,
    ) -> None:
        cursor = await connection.execute(
            """
            SELECT id
            FROM tasks
            WHERE id = %s
              AND tenant_id = %s
              AND project_id = %s
              AND repository_id = %s
              AND swarm_id = %s
              AND run_id = %s
            """,
            (
                _uuid(task_id),
                actor.tenant_id,
                actor.project_id,
                actor.repository_id,
                actor.swarm_id,
                actor.run_id,
            ),
        )
        if await cursor.fetchone() is None:
            raise ResourceNotFound("task", task_id)

    @staticmethod
    def _require_lease_row(
        row: Mapping[str, Any],
        *,
        worker_id: str,
        lease_token: str,
        lease_version: int,
        work_version: int,
        attempt: int,
    ) -> None:
        if (
            str(row["status"]) != WorkStatus.LEASED.value
            or row.get("locked_by") != worker_id
            or str(row.get("lease_token")) != lease_token
            or int(row["lease_version"]) != lease_version
            or int(row.get("work_version", row["version"])) != work_version
            or int(row["attempts"]) != attempt
            or row.get("locked_until") is None
            or row["locked_until"] <= row["database_now"]
        ):
            raise WorkLeaseLost(str(row.get("work_id", row.get("id", "unknown"))))

    @staticmethod
    def _require_completion_fence(
        item: WorkItem,
        command: (ApplyExtractionWorkCommand | ApplyEmbeddingWorkCommand | CompleteWorkCommand),
    ) -> None:
        if (
            item.locked_by != command.worker_id
            or item.lease_token != command.lease_token
            or item.lease_version != command.lease_version
            or item.attempts != command.attempt
            or item.version != command.expected_work_version + 1
        ):
            raise WorkLeaseLost(command.work_id)

    @staticmethod
    def _validate_embedding_apply(
        item: WorkItem,
        command: ApplyEmbeddingWorkCommand,
    ) -> None:
        if item.kind is not WorkKind.EMBED_MEMORY:
            raise WorkCompletionConflict("apply_embedding_requires_embed_memory")
        if (
            item.subject_id != command.vector.memory_id
            or item.payload.get("model") != command.vector.model
        ):
            raise WorkCompletionConflict("embedding_vector_does_not_match_work")

    @staticmethod
    async def _insert_attempts(
        connection: Any,
        work_id: str,
        provenance: Sequence[ExtractionProvenance],
    ) -> None:
        for item in provenance:
            await connection.execute(
                """
                INSERT INTO outbox_work_attempts (
                    work_id, attempt, stage, outcome, provider, model, revision,
                    prompt_id, prompt_sha256, input_sha256, output_sha256,
                    candidate_count, error, started_at, finished_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (work_id, attempt, stage) DO NOTHING
                """,
                (
                    _uuid(work_id),
                    item.attempt,
                    item.stage.value,
                    item.outcome.value,
                    item.provider,
                    item.model,
                    item.revision,
                    item.prompt_id,
                    bytes.fromhex(item.prompt_sha256) if item.prompt_sha256 else None,
                    bytes.fromhex(item.input_sha256),
                    bytes.fromhex(item.output_sha256) if item.output_sha256 else None,
                    item.candidate_count,
                    item.error,
                    item.started_at,
                    item.finished_at,
                ),
            )

    async def _apply_memory_effect(
        self,
        connection: Any,
        item: WorkItem,
        source_row: Mapping[str, Any],
        effect: WorkEffect,
    ) -> AppliedWorkEffect:
        existing_cursor = await connection.execute(
            """
            SELECT kind, payload_sha256, resource_id
            FROM outbox_work_effects
            WHERE work_id = %s AND effect_key = %s
            """,
            (_uuid(item.work_id), effect.effect_key),
        )
        existing = await existing_cursor.fetchone()
        if existing is not None:
            if (
                str(existing["kind"]) != effect.kind.value
                or bytes(existing["payload_sha256"]).hex() != effect.payload_sha256
            ):
                raise WorkEffectConflict(effect.effect_key)
            return AppliedWorkEffect(
                work_id=item.work_id,
                effect_key=effect.effect_key,
                kind=effect.kind,
                payload_sha256=effect.payload_sha256,
                resource_id=str(existing["resource_id"]),
            )

        memory_id = uuid5(_uuid(item.work_id), f"effect:{effect.effect_key}:memory")
        evidence_ids: list[UUID] = []
        if not effect.candidate.spans:
            evidence_id = uuid5(
                _uuid(item.work_id),
                f"effect:{effect.effect_key}:evidence:source",
            )
            evidence_ids.append(evidence_id)
            await connection.execute(
                """
                INSERT INTO evidence (
                    id, tenant_id, kind, locator, content_sha256, excerpt,
                    source_id, metadata
                ) VALUES (%s, %s, %s, 'source', %s, NULL, %s, %s)
                ON CONFLICT (id) DO NOTHING
                """,
                (
                    evidence_id,
                    item.tenant_id,
                    str(source_row["source_type"]),
                    source_row["content_sha256"],
                    _uuid(item.subject_id),
                    Jsonb(
                        {
                            "work_id": item.work_id,
                            "effect_key": effect.effect_key,
                            "attribution": "whole_source",
                        }
                    ),
                ),
            )
        for position, span in enumerate(effect.candidate.spans):
            evidence_id = uuid5(
                _uuid(item.work_id),
                f"effect:{effect.effect_key}:evidence:{position}",
            )
            evidence_ids.append(evidence_id)
            await connection.execute(
                """
                INSERT INTO evidence (
                    id, tenant_id, kind, locator, content_sha256, excerpt,
                    source_id, metadata
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
                """,
                (
                    evidence_id,
                    item.tenant_id,
                    str(source_row["source_type"]),
                    f"chunk:{span.chunk_index}:{span.char_start}-{span.char_end}",
                    hashlib.sha256(span.excerpt.encode("utf-8")).digest(),
                    span.excerpt,
                    _uuid(item.subject_id),
                    Jsonb({"work_id": item.work_id, "effect_key": effect.effect_key}),
                ),
            )

        candidate = effect.candidate
        memory_cursor = await connection.execute(
            """
            INSERT INTO memories (
                id, tenant_id, project_id, repository_id, swarm_id, run_id,
                task_id, agent_id, kind, state, visibility, content,
                content_json, title, tags, normalized_sha256, dedup_scope,
                confidence, valid_from,
                valid_to, source_id, policy_reason, policy_confidence, metadata
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, 'tentative', 'run',
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 1.0, %s
            )
            ON CONFLICT DO NOTHING
            RETURNING id, recorded_from, version
            """,
            (
                memory_id,
                item.tenant_id,
                item.project_id,
                item.repository_id,
                item.swarm_id,
                item.run_id,
                _uuid(item.task_id) if item.task_id else None,
                item.requested_by_agent_id,
                str(candidate.kind),
                memory_content_text(candidate.content),
                Jsonb(candidate.content) if not isinstance(candidate.content, str) else None,
                candidate.title,
                list(candidate.tags),
                bytes.fromhex(memory_text_sha256(candidate.content)),
                f"run:{item.run_id}",
                candidate.confidence,
                candidate.valid_from or source_row.get("valid_at") or source_row["recorded_at"],
                candidate.valid_to,
                _uuid(item.subject_id),
                "validated extraction candidate; confirmation requires review",
                Jsonb({"extraction": {"work_id": item.work_id, "effect_key": effect.effect_key}}),
            ),
        )
        inserted = await memory_cursor.fetchone()
        inserted_new_memory = inserted is not None
        memory_recorded_at = inserted.get("recorded_from") if inserted is not None else None
        memory_version = int(inserted.get("version", 1)) if inserted is not None else 1
        if inserted is None:
            existing_memory_cursor = await connection.execute(
                """
                SELECT id, recorded_from, version
                FROM memories
                WHERE id = %s
                FOR UPDATE
                """,
                (memory_id,),
            )
            existing_memory = await existing_memory_cursor.fetchone()
            if existing_memory is None:
                raise RuntimeError("memory effect conflict did not resolve by effect identity")
            memory_id = _uuid(str(existing_memory["id"]))
            memory_recorded_at = existing_memory["recorded_from"]
            memory_version = int(existing_memory["version"])

        for evidence_id in evidence_ids:
            await connection.execute(
                """
                INSERT INTO memory_evidence (memory_id, evidence_id, relation)
                VALUES (%s, %s, 'derived_from')
                ON CONFLICT (memory_id, evidence_id, relation) DO NOTHING
                """,
                (memory_id, evidence_id),
            )

        if inserted_new_memory:
            await self._emit_memory_effect(
                connection,
                item,
                effect,
                memory_id,
                memory_version,
                memory_recorded_at,
            )

        await connection.execute(
            """
            INSERT INTO outbox_work_effects (
                work_id, effect_key, kind, payload_sha256, resource_id
            ) VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (work_id, effect_key) DO NOTHING
            """,
            (
                _uuid(item.work_id),
                effect.effect_key,
                effect.kind.value,
                bytes.fromhex(effect.payload_sha256),
                memory_id,
            ),
        )
        return AppliedWorkEffect(
            work_id=item.work_id,
            effect_key=effect.effect_key,
            kind=WorkEffectKind.MEMORY,
            payload_sha256=effect.payload_sha256,
            resource_id=str(memory_id),
        )

    @staticmethod
    async def _enqueue_effect_embedding(
        connection: Any,
        item: WorkItem,
        effect: WorkEffect,
        applied: AppliedWorkEffect,
    ) -> None:
        embedding_model = item.payload.get("embedding_model")
        if not isinstance(embedding_model, str):
            return
        dedupe_key = f"embed:{applied.resource_id}:{embedding_model}"
        work_id = durable_work_id(
            item.tenant_id,
            item.run_id,
            WorkKind.EMBED_MEMORY,
            dedupe_key,
        )
        payload = {
            "content": memory_content_text(effect.candidate.content),
            "model": embedding_model,
            "metadata": {},
        }
        cursor = await connection.execute(
            """
            INSERT INTO outbox_work_items (
                id, tenant_id, project_id, repository_id, swarm_id, run_id,
                task_id, requested_by_agent_id, kind, subject_id, payload,
                dedupe_key, priority
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, 'embed_memory', %s,
                %s, %s, %s
            )
            ON CONFLICT (tenant_id, run_id, kind, dedupe_key) DO NOTHING
            RETURNING id, subject_id, payload
            """,
            (
                _uuid(work_id),
                item.tenant_id,
                item.project_id,
                item.repository_id,
                item.swarm_id,
                item.run_id,
                _uuid(item.task_id) if item.task_id else None,
                item.requested_by_agent_id,
                _uuid(applied.resource_id),
                Jsonb(payload),
                dedupe_key,
                item.priority,
            ),
        )
        row = await cursor.fetchone()
        if row is None:
            cursor = await connection.execute(
                """
                SELECT id, subject_id, payload
                FROM outbox_work_items
                WHERE tenant_id = %s
                  AND run_id = %s
                  AND kind = 'embed_memory'
                  AND dedupe_key = %s
                FOR UPDATE
                """,
                (item.tenant_id, item.run_id, dedupe_key),
            )
            row = await cursor.fetchone()
        if row is None or (
            str(row["id"]) != work_id
            or str(row["subject_id"]) != applied.resource_id
            or (_json_object(row.get("payload")) or {}) != payload
        ):
            raise WorkEffectConflict(f"embedding_work:{applied.resource_id}")

    @staticmethod
    async def _emit_memory_effect(
        connection: Any,
        item: WorkItem,
        effect: WorkEffect,
        memory_id: UUID,
        memory_version: int,
        occurred_at: Any,
    ) -> None:
        event_id = uuid5(_uuid(item.work_id), f"effect:{effect.effect_key}:event")
        outbox_id = uuid5(_uuid(item.work_id), f"effect:{effect.effect_key}:outbox")
        event = SwarmEvent(
            event_id=str(event_id),
            event_type=EventType.MEMORY_ADDED,
            aggregate_type=AggregateType.MEMORY,
            aggregate_id=str(memory_id),
            tenant_id=item.tenant_id,
            project_id=item.project_id,
            repository_id=item.repository_id,
            swarm_id=item.swarm_id,
            run_id=item.run_id,
            agent_id=item.requested_by_agent_id,
            task_id=item.task_id,
            aggregate_version=memory_version,
            payload={"work_id": item.work_id, "effect_key": effect.effect_key},
            occurred_at=occurred_at,
            recorded_at=occurred_at,
            idempotency_key=f"work:{item.work_id}:{effect.effect_key}",
        )
        await connection.execute(
            """
            INSERT INTO swarm_events (
                id, tenant_id, project_id, repository_id, swarm_id, run_id,
                agent_id, task_id, event_type, aggregate_type, aggregate_id,
                aggregate_version, payload, occurred_at, recorded_at,
                idempotency_key
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s
            )
            ON CONFLICT (id) DO NOTHING
            """,
            (
                event_id,
                item.tenant_id,
                item.project_id,
                item.repository_id,
                item.swarm_id,
                item.run_id,
                item.requested_by_agent_id,
                _uuid(item.task_id) if item.task_id else None,
                EventType.MEMORY_ADDED.value,
                AggregateType.MEMORY.value,
                str(memory_id),
                memory_version,
                Jsonb(event.payload),
                occurred_at,
                occurred_at,
                event.idempotency_key,
            ),
        )
        await connection.execute(
            """
            INSERT INTO outbox_events (
                id, event_id, tenant_id, run_id, aggregate_type, aggregate_id,
                aggregate_version, event_type, dedupe_key, payload, status,
                available_at, occurred_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                'pending', %s, %s
            )
            ON CONFLICT (dedupe_key) DO NOTHING
            """,
            (
                outbox_id,
                event_id,
                item.tenant_id,
                item.run_id,
                AggregateType.MEMORY.value,
                str(memory_id),
                memory_version,
                EventType.MEMORY_ADDED.value,
                f"event:{event_id}",
                Jsonb(event.model_dump(mode="json")),
                occurred_at,
                occurred_at,
            ),
        )

    @staticmethod
    async def _load_effects(
        connection: Any,
        command: ApplyExtractionWorkCommand,
    ) -> tuple[AppliedWorkEffect, ...]:
        keys_cursor = await connection.execute(
            """
            SELECT effect_key
            FROM outbox_work_effects
            WHERE work_id = %s
            ORDER BY effect_key
            """,
            (_uuid(command.work_id),),
        )
        durable_keys = {str(row["effect_key"]) for row in await keys_cursor.fetchall()}
        requested_keys = {effect.effect_key for effect in command.effects}
        if durable_keys != requested_keys:
            raise WorkEffectConflict("completed_effect_set")
        loaded: list[AppliedWorkEffect] = []
        for effect in command.effects:
            cursor = await connection.execute(
                """
                SELECT kind, payload_sha256, resource_id
                FROM outbox_work_effects
                WHERE work_id = %s AND effect_key = %s
                """,
                (_uuid(command.work_id), effect.effect_key),
            )
            row = await cursor.fetchone()
            if row is None or (
                str(row["kind"]) != effect.kind.value
                or bytes(row["payload_sha256"]).hex() != effect.payload_sha256
            ):
                raise WorkEffectConflict(effect.effect_key)
            loaded.append(
                AppliedWorkEffect(
                    work_id=command.work_id,
                    effect_key=effect.effect_key,
                    kind=effect.kind,
                    payload_sha256=effect.payload_sha256,
                    resource_id=str(row["resource_id"]),
                )
            )
        return tuple(loaded)

    @staticmethod
    def _chunk_from_row(row: Mapping[str, Any]) -> SourceChunk:
        content = str(row["content"])
        return SourceChunk(
            chunk_id=str(row["id"]),
            source_id=str(row["source_id"]),
            chunk_index=int(row["chunk_index"]),
            content=content,
            content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            token_count=int(row["token_count"]),
            char_start=int(row["char_start"]),
            char_end=int(row["char_end"]),
            metadata=_json_object(row.get("metadata")) or {},
        )

    @staticmethod
    def _require_exact_chunks(
        prepared: PreparedSourceIngest,
        chunks: Sequence[SourceChunk],
    ) -> None:
        if (
            not chunks
            or [chunk.chunk_index for chunk in chunks] != list(range(len(chunks)))
            or "".join(chunk.content for chunk in chunks) != prepared.command.content
        ):
            raise IdempotencyConflict(prepared.command.idempotency_key)
        expected_start = 0
        for chunk in chunks:
            if chunk.source_id != prepared.source_id or chunk.char_start != expected_start:
                raise IdempotencyConflict(prepared.command.idempotency_key)
            expected_start = chunk.char_end

    @staticmethod
    def _cancellation_reason(prepared: PreparedSourceIngest) -> str | None:
        if prepared.command.trust is SourceTrust.UNTRUSTED:
            return "source_untrusted"
        return None

    @staticmethod
    def _validate_effect_spans(
        effects: Sequence[WorkEffect],
        raw_content: object | None,
        chunk_rows: Sequence[Mapping[str, Any]],
    ) -> None:
        if not isinstance(raw_content, str):
            raise WorkEffectConflict("evidence_source_missing")
        chunks = {int(row["chunk_index"]): row for row in chunk_rows}
        for effect in effects:
            for span in effect.candidate.spans:
                chunk = chunks.get(span.chunk_index)
                if (
                    chunk is None
                    or span.char_start < int(chunk["char_start"])
                    or span.char_end > int(chunk["char_end"])
                    or raw_content[span.char_start : span.char_end] != span.excerpt
                ):
                    raise WorkEffectConflict(f"evidence_span:{effect.effect_key}")


__all__ = ["CockroachWorkStore", "WORK_COLUMNS"]
