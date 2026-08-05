"""Async CockroachDB connection and idempotent transaction kernel."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable
from datetime import timedelta
from typing import Any, TypeVar
from uuid import NAMESPACE_URL, UUID, uuid5

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel

from swarmbrain.application.errors import AmbiguousCommit, IdempotencyConflict
from swarmbrain.domain.agents import ActorContext

from .retry import AmbiguousTransactionResult, RetryPolicy, run_serializable

ModelT = TypeVar("ModelT", bound=BaseModel)
TransactionBody = Callable[[Any], Awaitable[ModelT]]

SCHEMA_VERSION = 6
REQUIRED_TABLES = frozenset(
    {
        "runs",
        "agents",
        "agent_tokens",
        "tasks",
        "task_dependencies",
        "task_leases",
        "task_checkpoints",
        "task_completions",
        "sources",
        "source_chunks",
        "source_extractions",
        "memories",
        "memory_embeddings",
        "memory_vector_embeddings",
        "evidence",
        "memory_evidence",
        "memory_links",
        "memory_conflicts",
        "memory_conflict_members",
        "conflict_evidence",
        "idempotency_records",
        "action_attempts",
        "outbox_events",
        "outbox_work_items",
        "outbox_work_attempts",
        "outbox_work_effects",
        "swarm_events",
        "swarmbrain_schema_versions",
    }
)
REQUIRED_COLUMNS = {
    "swarmbrain_schema_versions": {"version", "schema_sha256"},
    "runs": {"tenant_id", "id", "repository_id"},
    "agents": {"tenant_id", "run_id", "id", "version"},
    "agent_tokens": {"id", "tenant_id", "run_id", "agent_id", "token_hash"},
    "tasks": {"id", "tenant_id", "run_id", "state", "version", "active_lease_id"},
    "task_dependencies": {"task_id", "depends_on_task_id", "kind"},
    "task_leases": {"id", "task_id", "status", "version", "expires_at"},
    "task_checkpoints": {"id", "task_id", "lease_id", "sequence"},
    "task_completions": {"id", "task_id", "lease_id", "outcome"},
    "sources": {
        "id",
        "tenant_id",
        "repository_id",
        "swarm_id",
        "content",
        "review_state",
        "version",
    },
    "source_chunks": {"id", "source_id", "chunk_index", "content", "char_start", "char_end"},
    "memories": {
        "id",
        "tenant_id",
        "repository_id",
        "state",
        "dedup_scope",
        "normalized_sha256",
        "content_json",
        "previous_state",
        "recorded_to",
        "version",
    },
    "evidence": {"id", "source_id", "tenant_id"},
    "memory_embeddings": {"memory_id", "model", "dimensions", "embedding"},
    "memory_vector_embeddings": {
        "memory_id",
        "tenant_id",
        "project_id",
        "repository_id",
        "model",
        "dimensions",
        "embedding",
    },
    "memory_evidence": {"memory_id", "evidence_id", "relation"},
    "memory_links": {
        "id",
        "source_memory_id",
        "target_memory_id",
        "link_type",
        "reason",
    },
    "memory_conflicts": {"id", "tenant_id", "run_id", "state", "version"},
    "memory_conflict_members": {"conflict_id", "memory_id", "position"},
    "conflict_evidence": {"conflict_id", "evidence_id", "relation"},
    "idempotency_records": {
        "id",
        "tenant_id",
        "run_id",
        "actor_id",
        "operation",
        "idempotency_key",
        "request_sha256",
        "response_body",
    },
    "action_attempts": {"id", "operation", "attempt", "status", "sqlstate"},
    "outbox_events": {
        "id",
        "event_id",
        "dedupe_key",
        "payload",
        "status",
        "available_at",
        "locked_until",
        "version",
    },
    "source_extractions": {
        "id",
        "source_id",
        "extractor_name",
        "extractor_version",
        "status",
    },
    "outbox_work_items": {
        "id",
        "tenant_id",
        "run_id",
        "kind",
        "subject_id",
        "payload",
        "dedupe_key",
        "status",
        "attempts",
        "max_attempts",
        "available_at",
        "locked_by",
        "lease_token",
        "lease_version",
        "locked_until",
        "outcome",
        "result",
        "version",
    },
    "outbox_work_attempts": {
        "work_id",
        "attempt",
        "stage",
        "outcome",
        "input_sha256",
        "candidate_count",
        "started_at",
        "finished_at",
    },
    "outbox_work_effects": {
        "work_id",
        "effect_key",
        "kind",
        "payload_sha256",
        "resource_id",
    },
    "swarm_events": {"id", "tenant_id", "run_id", "event_type", "payload"},
}
REQUIRED_INDEXES = frozenset(
    {
        "tasks_claim_queue",
        "task_dependencies_reverse",
        "task_leases_one_active_per_task",
        "task_leases_agent_active",
        "task_checkpoints_latest",
        "sources_scope_lookup",
        "memories_current_scope",
        "memories_task_current",
        "memories_source",
        "memories_one_successor",
        "memories_current_fingerprint",
        "memory_vector_embeddings_ann",
        "evidence_source_lookup",
        "memory_evidence_by_evidence",
        "memory_links_from",
        "memory_links_to",
        "idempotency_records_lookup",
        "outbox_events_unpublished",
        "source_extractions_scope",
        "outbox_work_items_claim",
        "outbox_work_items_expired_leases",
        "outbox_work_items_subject",
        "outbox_work_attempts_timeline",
        "outbox_work_effects_resource",
        "swarm_events_run_timeline",
    }
)


class SchemaNotInstalled(RuntimeError):
    """Raised when a Cockroach backend is started before explicit installation."""


def command_fingerprint(command: BaseModel) -> bytes:
    payload = json.dumps(
        command.model_dump(mode="json", exclude_none=False),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).digest()


def _replayed(result: ModelT) -> ModelT:
    if "replayed" in result.__class__.model_fields:
        return result.model_copy(update={"replayed": True})
    return result


class CockroachDatabase:
    """Own an async pool and the one safe SERIALIZABLE mutation primitive.

    Startup verifies a pre-installed schema but never runs DDL. Mutation
    closures may perform database work only; external calls belong in outbox
    workers after the transaction commits.
    """

    def __init__(
        self,
        database_url: str,
        *,
        min_size: int = 1,
        max_size: int = 12,
        retry_policy: RetryPolicy | None = None,
        pool: Any | None = None,
    ) -> None:
        if not database_url:
            raise ValueError("database_url is required")
        if min_size < 0 or max_size < 1 or min_size > max_size:
            raise ValueError("invalid CockroachDB pool bounds")
        self.database_url = database_url
        self.retry_policy = retry_policy or RetryPolicy()
        self.pool = pool or AsyncConnectionPool(
            conninfo=database_url,
            min_size=min_size,
            max_size=max_size,
            open=False,
            kwargs={"row_factory": dict_row},
            name="swarmbrain",
        )
        self._manage_pool = pool is None
        self._opened = False

    async def start(self, *, verify_schema: bool = True) -> None:
        opened_here = self._manage_pool and not self._opened
        try:
            if opened_here:
                await self.pool.open(wait=True)
            self._opened = True
            if verify_schema:
                await self.verify_schema()
        except Exception:
            if opened_here:
                await self.pool.close()
            self._opened = False
            raise

    async def close(self) -> None:
        if self._manage_pool and self._opened:
            await self.pool.close()
        self._opened = False

    async def health(self) -> None:
        async with self.pool.connection() as connection:
            cursor = await connection.execute("SELECT 1 AS healthy")
            row = await cursor.fetchone()
        if row is None or int(row["healthy"]) != 1:
            raise RuntimeError("CockroachDB health query returned an invalid result")

    async def verify_schema(self) -> None:
        from .schema import read_schema

        expected_digest = hashlib.sha256(read_schema().encode("utf-8")).digest()
        try:
            async with self.pool.connection() as connection:
                version_cursor = await connection.execute(
                    """
                    SELECT version, schema_sha256
                    FROM swarmbrain_schema_versions
                    WHERE version = %s
                    """,
                    (SCHEMA_VERSION,),
                )
                version_row = await version_cursor.fetchone()
                columns_cursor = await connection.execute(
                    """
                    SELECT table_name, column_name
                    FROM information_schema.columns
                    WHERE table_schema = current_schema()
                      AND table_name = ANY(%s::STRING[])
                    """,
                    (sorted(REQUIRED_COLUMNS),),
                )
                column_rows = await columns_cursor.fetchall()
                indexes_cursor = await connection.execute(
                    """
                    SELECT indexname
                    FROM pg_catalog.pg_indexes
                    WHERE schemaname = current_schema()
                      AND indexname = ANY(%s::STRING[])
                    """,
                    (sorted(REQUIRED_INDEXES),),
                )
                index_rows = await indexes_cursor.fetchall()
        except Exception as exc:
            if getattr(exc, "sqlstate", None) == "42P01":
                raise SchemaNotInstalled(
                    "Swarm Brain schema is not installed; run swarmbrain-schema install"
                ) from exc
            raise

        present: dict[str, set[str]] = {}
        for row in column_rows:
            present.setdefault(str(row["table_name"]), set()).add(str(row["column_name"]))
        missing = {
            table: sorted(columns - present.get(table, set()))
            for table, columns in REQUIRED_COLUMNS.items()
            if not columns.issubset(present.get(table, set()))
        }
        present_indexes = {str(row["indexname"]) for row in index_rows}
        missing_indexes = sorted(REQUIRED_INDEXES - present_indexes)
        checksum_matches = (
            version_row is not None and bytes(version_row["schema_sha256"]) == expected_digest
        )
        if version_row is None or missing or missing_indexes or not checksum_matches:
            suffix = f"; incompatible columns: {missing}" if missing else ""
            if missing_indexes:
                suffix += f"; missing indexes: {', '.join(missing_indexes)}"
            if version_row is not None and not checksum_matches:
                suffix += "; schema checksum differs"
            raise SchemaNotInstalled(
                f"Swarm Brain schema version {SCHEMA_VERSION} is not installed{suffix}"
            )

    async def run(self, body: TransactionBody[ModelT]) -> ModelT:
        return await run_serializable(
            self.pool,
            body,
            policy=self.retry_policy,
        )

    async def run_idempotent(
        self,
        actor: ActorContext,
        operation: str,
        command: BaseModel,
        result_model: type[ModelT],
        body: TransactionBody[ModelT],
        *,
        retention: timedelta = timedelta(days=30),
    ) -> ModelT:
        """Run and persist business result/event/outbox/idempotency atomically."""

        if not operation or len(operation) > 255:
            raise ValueError("operation must contain at most 255 characters")
        fingerprint = command_fingerprint(command)
        record_id = uuid5(
            NAMESPACE_URL,
            ":".join(
                (
                    "swarmbrain-idempotency",
                    actor.tenant_id,
                    actor.run_id,
                    actor.agent_id,
                    operation,
                    command.idempotency_key,
                )
            ),
        )

        async def transaction(connection: Any) -> ModelT:
            replay = await self._read_idempotency(
                connection,
                actor,
                operation,
                command.idempotency_key,
                fingerprint,
                result_model,
                for_update=True,
            )
            if replay is not None:
                return replay

            await connection.execute(
                """
                INSERT INTO idempotency_records (
                    id, tenant_id, run_id, actor_id, operation, idempotency_key,
                    request_sha256, status, expires_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, 'started', now() + %s
                )
                """,
                (
                    record_id,
                    actor.tenant_id,
                    actor.run_id,
                    actor.agent_id,
                    operation,
                    command.idempotency_key,
                    fingerprint,
                    retention,
                ),
            )
            result = await body(connection)
            await connection.execute(
                """
                UPDATE idempotency_records
                SET status = 'completed',
                    response_status = 200,
                    response_body = %s,
                    completed_at = now()
                WHERE id = %s
                  AND status = 'started'
                """,
                (Jsonb(result.model_dump(mode="json")), record_id),
            )
            return result

        async def observe(attempt: int, sqlstate: str | None) -> None:
            if sqlstate is None:
                return
            await self._record_attempt(
                actor,
                operation,
                attempt,
                sqlstate,
                record_id if sqlstate == "00000" else None,
            )

        try:
            return await run_serializable(
                self.pool,
                transaction,
                policy=self.retry_policy,
                on_attempt=observe,
            )
        except AmbiguousTransactionResult:
            try:
                return await self.resolve_idempotency(
                    actor,
                    operation,
                    command.idempotency_key,
                    fingerprint,
                    result_model,
                )
            except AmbiguousTransactionResult as exc:
                raise AmbiguousCommit() from exc
        except Exception as exc:
            # A concurrent insert for the same key may surface as a uniqueness
            # error rather than 40001. Resolve the committed row; never replay
            # the business closure from this branch.
            if getattr(exc, "sqlstate", None) == "23505":
                try:
                    return await self.resolve_idempotency(
                        actor,
                        operation,
                        command.idempotency_key,
                        fingerprint,
                        result_model,
                        attempts=3,
                    )
                except AmbiguousTransactionResult as resolution_error:
                    raise exc from resolution_error
            raise

    async def resolve_idempotency(
        self,
        actor: ActorContext,
        operation: str,
        idempotency_key: str,
        fingerprint: bytes,
        result_model: type[ModelT],
        *,
        attempts: int = 5,
    ) -> ModelT:
        """Resolve an uncertain commit by reading only; never rerun its body."""

        for attempt in range(attempts):
            async with self.pool.connection() as connection:
                result = await self._read_idempotency(
                    connection,
                    actor,
                    operation,
                    idempotency_key,
                    fingerprint,
                    result_model,
                    for_update=False,
                )
            if result is not None:
                return result
            if attempt + 1 < attempts:
                await asyncio.sleep(0.05 * (attempt + 1))
        raise AmbiguousTransactionResult(
            "transaction outcome is still ambiguous after idempotency lookup"
        )

    async def _read_idempotency(
        self,
        connection: Any,
        actor: ActorContext,
        operation: str,
        idempotency_key: str,
        fingerprint: bytes,
        result_model: type[ModelT],
        *,
        for_update: bool,
    ) -> ModelT | None:
        lock = " FOR UPDATE" if for_update else ""
        cursor = await connection.execute(
            """
            SELECT request_sha256, status, response_body
            FROM idempotency_records
            WHERE tenant_id = %s
              AND run_id = %s
              AND actor_id = %s
              AND operation = %s
              AND idempotency_key = %s
            """
            + lock,
            (actor.tenant_id, actor.run_id, actor.agent_id, operation, idempotency_key),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        if bytes(row["request_sha256"]) != fingerprint:
            raise IdempotencyConflict(idempotency_key)
        if row["status"] != "completed" or row["response_body"] is None:
            return None
        return _replayed(result_model.model_validate(row["response_body"]))

    async def _record_attempt(
        self,
        actor: ActorContext,
        operation: str,
        attempt: int,
        sqlstate: str,
        record_id: UUID | None,
    ) -> None:
        status = "succeeded" if sqlstate == "00000" else "failed"
        error_code = None
        if sqlstate == "40001":
            error_code = "serialization_retry"
        elif sqlstate == "40003" or sqlstate.startswith("08"):
            error_code = "ambiguous_result"
        try:
            async with self.pool.connection() as connection:
                await connection.execute(
                    """
                    INSERT INTO action_attempts (
                        idempotency_record_id, tenant_id, run_id, agent_id,
                        operation, attempt, status, sqlstate, error_code
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        record_id,
                        actor.tenant_id,
                        actor.run_id,
                        actor.agent_id,
                        operation,
                        attempt,
                        status,
                        None if sqlstate == "00000" else sqlstate,
                        error_code,
                    ),
                )
        except Exception:
            # Observability must not turn a committed idempotent mutation into
            # a client-visible failure. The business result remains canonical.
            return


__all__ = [
    "CockroachDatabase",
    "REQUIRED_COLUMNS",
    "REQUIRED_INDEXES",
    "REQUIRED_TABLES",
    "SCHEMA_VERSION",
    "SchemaNotInstalled",
    "command_fingerprint",
]
