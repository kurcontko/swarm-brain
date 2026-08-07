"""Async CockroachDB connection and idempotent transaction kernel."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from contextvars import ContextVar
from datetime import datetime, timedelta
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

SCHEMA_VERSION = 8
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
        "retrieval_documents",
        "retrieval_exact_terms",
        "memory_embeddings",
        "memory_vector_embeddings",
        "retrieval_vectors_1024",
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
    "retrieval_documents": {
        "tenant_id",
        "project_id",
        "repository_id",
        "projection_id",
        "scope_key",
        "resource_type",
        "resource_id",
        "resource_version",
        "canonical_id",
        "domain_lane",
        "search_text",
        "lookup_text",
        "search_tsv",
        "content_sha256",
        "indexed_at",
    },
    "retrieval_exact_terms": {
        "tenant_id",
        "project_id",
        "repository_id",
        "projection_id",
        "scope_key",
        "normalized_term",
        "term_kind",
        "resource_type",
        "resource_id",
        "resource_version",
        "indexed_at",
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
    "retrieval_vectors_1024": {
        "tenant_id",
        "project_id",
        "repository_id",
        "projection_id",
        "projection_signature",
        "scope_key",
        "resource_type",
        "resource_id",
        "resource_version",
        "canonical_id",
        "domain_lane",
        "content_sha256",
        "model",
        "dimensions",
        "embedding",
        "indexed_at",
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
        "retrieval_documents_fts",
        "retrieval_documents_lookup_trgm",
        "retrieval_exact_terms_by_resource",
        "memory_vector_embeddings_ann",
        "retrieval_vectors_1024_ann_v2",
        "retrieval_vectors_1024_by_canonical",
        "evidence_source_lookup",
        "memory_evidence_by_evidence",
        "memory_links_from",
        "memory_links_from_type",
        "memory_links_to",
        "memory_links_to_type",
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

# These retrieval indexes are correctness-critical: a same-named B-tree or a
# different column order can pass a name-only check while turning every recall
# into a broad scan. pg_indexes renders Cockroach inverted indexes as USING gin.
REQUIRED_RETRIEVAL_INDEX_SHAPES: dict[str, tuple[str, tuple[str, ...], tuple[str, ...]]] = {
    "retrieval_documents_fts": (
        "using gin",
        (
            "tenant_id",
            "project_id",
            "repository_id",
            "projection_id",
            "scope_key",
            "search_tsv",
        ),
        (),
    ),
    "retrieval_documents_lookup_trgm": (
        "using gin",
        (
            "tenant_id",
            "project_id",
            "repository_id",
            "projection_id",
            "scope_key",
            "lookup_text",
        ),
        ("gin_trgm_ops",),
    ),
    "retrieval_exact_terms_pkey": (
        "using btree",
        (
            "tenant_id",
            "project_id",
            "repository_id",
            "projection_id",
            "scope_key",
            "normalized_term",
            "term_kind",
            "resource_type",
            "resource_id",
        ),
        ("create unique index",),
    ),
    "retrieval_exact_terms_by_resource": (
        "using btree",
        (
            "tenant_id",
            "project_id",
            "repository_id",
            "projection_id",
            "scope_key",
            "resource_type",
            "resource_id",
        ),
        (),
    ),
    "memory_links_from_type": (
        "using btree",
        ("source_memory_id", "link_type", "created_at", "id"),
        ("created_at desc",),
    ),
    "memory_links_to_type": (
        "using btree",
        ("target_memory_id", "link_type", "created_at", "id"),
        ("created_at desc",),
    ),
}


def incompatible_retrieval_schema_objects(
    index_rows: Sequence[dict[str, Any]],
    retrieval_documents_ddl: str,
    dense_vectors_ddl: str | None = None,
) -> tuple[str, ...]:
    """Return critical retrieval objects whose definitions are not compatible."""

    definitions = {
        str(row["indexname"]): " ".join(str(row.get("indexdef") or "").lower().split())
        for row in index_rows
    }
    incompatible: list[str] = []
    for name, (method, columns, fragments) in REQUIRED_RETRIEVAL_INDEX_SHAPES.items():
        definition = definitions.get(name, "")
        position = definition.find(method)
        if position < 0 or any(fragment not in definition for fragment in fragments):
            incompatible.append(name)
            continue
        for column in columns:
            position = definition.find(column, position + 1)
            if position < 0:
                incompatible.append(name)
                break

    ddl = " ".join(retrieval_documents_ddl.lower().split())
    required_ddl_fragments = (
        "search_text string not null",
        "lookup_text string not null",
        "search_tsv tsvector",
        "to_tsvector('simple'",
        "search_text)) stored",
    )
    if any(fragment not in ddl for fragment in required_ddl_fragments):
        incompatible.append("retrieval_documents")
    else:
        primary_position = ddl.find("primary key (")
        if primary_position < 0:
            incompatible.append("retrieval_documents")
        primary_columns = (
            ()
            if primary_position < 0
            else (
                "tenant_id",
                "project_id",
                "repository_id",
                "projection_id",
                "scope_key",
                "resource_type",
                "resource_id",
            )
        )
        for column in primary_columns:
            primary_position = ddl.find(column, primary_position + 1)
            if primary_position < 0:
                incompatible.append("retrieval_documents")
                break

    if dense_vectors_ddl is not None:
        dense_ddl = " ".join(dense_vectors_ddl.lower().split())
        dense_fragments = (
            "projection_id string not null",
            "projection_signature string not null",
            "scope_key string not null",
            "resource_version int8 not null",
            "content_sha256 bytes not null",
            "embedding vector(1024) not null",
        )
        dense_primary = (
            "tenant_id",
            "project_id",
            "repository_id",
            "projection_signature",
            "scope_key",
            "resource_type",
            "resource_id",
        )
        dense_position = dense_ddl.find("primary key (")
        if any(fragment not in dense_ddl for fragment in dense_fragments) or dense_position < 0:
            incompatible.append("retrieval_vectors_1024")
        else:
            for column in dense_primary:
                dense_position = dense_ddl.find(column, dense_position + 1)
                if dense_position < 0:
                    incompatible.append("retrieval_vectors_1024")
                    break

        vector_definition = definitions.get("retrieval_vectors_1024_ann_v2", "")
        vector_position = 0
        vector_columns = (
            "tenant_id",
            "project_id",
            "repository_id",
            "resource_type",
            "projection_id",
            "projection_signature",
            "scope_key",
            "embedding",
        )
        if "vector_cosine_ops" not in vector_definition:
            incompatible.append("retrieval_vectors_1024_ann_v2")
        else:
            for column in vector_columns:
                vector_position = vector_definition.find(column, vector_position + 1)
                if vector_position < 0:
                    incompatible.append("retrieval_vectors_1024_ann_v2")
                    break
    return tuple(dict.fromkeys(incompatible))


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
        self._retrieval_snapshot: ContextVar[tuple[Any, datetime, asyncio.Lock] | None] = (
            ContextVar(
                f"swarmbrain_cockroach_retrieval_snapshot_{id(self)}",
                default=None,
            )
        )

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

    @asynccontextmanager
    async def retrieval_snapshot(self) -> AsyncIterator[None]:
        """Share one SERIALIZABLE read snapshot across all retrieval lanes."""

        if self._retrieval_snapshot.get() is not None:
            yield
            return
        async with self.pool.connection() as connection, connection.transaction():
            cursor = await connection.execute("SELECT now() AS database_now")
            row = await cursor.fetchone()
            if row is None:
                raise RuntimeError("database clock query returned no row")
            token = self._retrieval_snapshot.set((connection, row["database_now"], asyncio.Lock()))
            try:
                yield
            finally:
                self._retrieval_snapshot.reset(token)

    @asynccontextmanager
    async def retrieval_connection(self) -> AsyncIterator[Any]:
        """Reuse the active retrieval transaction, or lease a standalone connection."""

        snapshot = self._retrieval_snapshot.get()
        if snapshot is not None:
            # One connection supplies the common MVCC snapshot. Serialize lane
            # work and isolate each unit behind a savepoint so a degraded SQL
            # lane cannot leave the outer read transaction in 25P02.
            async with snapshot[2], snapshot[0].transaction():
                yield snapshot[0]
            return
        async with self.pool.connection() as connection:
            yield connection

    async def retrieval_now(self, connection: Any | None = None) -> datetime:
        """Return the transaction-stable database clock for retrieval."""

        snapshot = self._retrieval_snapshot.get()
        if snapshot is not None:
            return snapshot[1]
        if connection is None:
            async with self.pool.connection() as leased:
                return await self.retrieval_now(leased)
        cursor = await connection.execute("SELECT now() AS database_now")
        row = await cursor.fetchone()
        if row is None:
            raise RuntimeError("database clock query returned no row")
        return row["database_now"]

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
                    SELECT indexname, indexdef
                    FROM pg_catalog.pg_indexes
                    WHERE schemaname = current_schema()
                      AND indexname = ANY(%s::STRING[])
                    """,
                    (sorted(REQUIRED_INDEXES | REQUIRED_RETRIEVAL_INDEX_SHAPES.keys()),),
                )
                index_rows = await indexes_cursor.fetchall()
                create_cursor = await connection.execute("SHOW CREATE TABLE retrieval_documents")
                create_row = await create_cursor.fetchone()
                dense_create_cursor = await connection.execute(
                    "SHOW CREATE TABLE retrieval_vectors_1024"
                )
                dense_create_row = await dense_create_cursor.fetchone()
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
        retrieval_ddl = "" if create_row is None else str(create_row["create_statement"])
        dense_ddl = "" if dense_create_row is None else str(dense_create_row["create_statement"])
        incompatible_objects = incompatible_retrieval_schema_objects(
            index_rows,
            retrieval_ddl,
            dense_ddl,
        )
        checksum_matches = (
            version_row is not None and bytes(version_row["schema_sha256"]) == expected_digest
        )
        if (
            version_row is None
            or missing
            or missing_indexes
            or incompatible_objects
            or not checksum_matches
        ):
            suffix = f"; incompatible columns: {missing}" if missing else ""
            if missing_indexes:
                suffix += f"; missing indexes: {', '.join(missing_indexes)}"
            if incompatible_objects:
                suffix += "; incompatible retrieval objects: " + ", ".join(incompatible_objects)
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
