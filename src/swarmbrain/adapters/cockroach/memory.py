"""CockroachDB temporal memory, evidence, and conflict repository.

All mutation closures passed to :class:`CockroachDatabase` contain database
work only. Entity, event, and outbox UUIDs are allocated before the closure so
whole-transaction ``40001`` retries cannot change externally visible identity.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from datetime import timedelta
from typing import Any
from uuid import UUID, uuid4

from psycopg.types.json import Jsonb

from swarmbrain.application.errors import IdempotencyConflict, InvalidState, ResourceNotFound
from swarmbrain.application.memory_policy import memory_content_text, memory_text_sha256
from swarmbrain.domain.agents import ActorContext
from swarmbrain.domain.conflicts import (
    Conflict,
    ConflictResolution,
    ConflictResolutionKind,
    ConflictStatus,
    ReportConflictCommand,
    ReportConflictResult,
    ResolveConflictCommand,
    ResolveConflictResult,
)
from swarmbrain.domain.events import AggregateType, EventType, SwarmEvent
from swarmbrain.domain.evidence import (
    AddEvidenceCommand,
    Evidence,
    EvidenceRef,
    EvidenceSource,
    RegisterEvidenceSourceCommand,
    RejectSourceCommand,
    ReviewSourceCommand,
    SourceRejectionResult,
)
from swarmbrain.domain.memory import (
    EmbeddingMatch,
    EmbeddingVector,
    Memory,
    MemoryLineage,
    MemoryOperation,
    MemoryReviewDecision,
    MemoryState,
    RecallBundle,
    RecallHit,
    RecallQuery,
    RememberCommand,
    RememberResult,
    ReviewMemoryCommand,
    ReviewMemoryResult,
    Visibility,
)
from swarmbrain.ports.memory_store import MemoryOperationPolicy

from .database import CockroachDatabase
from .memory_mappers import (
    conflict_from_parts,
    dedup_scope,
    evidence_from_row,
    lexical_score,
    link_from_row,
    memory_from_row,
    source_from_row,
)
from .vector import COCKROACH_VECTOR_DIMENSIONS, vector_literal

SOURCE_COLUMNS = """
    id, tenant_id, project_id, repository_id, swarm_id, run_id, task_id,
    agent_id, source_type, occurrence_key, uri, content_sha256, trust_label,
    review_state, valid_at, recorded_at, reviewed_by_agent_id, reviewed_at,
    rejection_reason, version, metadata
"""

EVIDENCE_COLUMNS = """
    id, tenant_id, kind, locator, content_sha256, excerpt, source_id,
    artifact, metadata, recorded_at
"""
EVIDENCE_ALIAS_COLUMNS = ", ".join(f"e.{column.strip()}" for column in EVIDENCE_COLUMNS.split(","))

MEMORY_COLUMNS = """
    m.id, m.tenant_id, m.project_id, m.repository_id, m.swarm_id, m.run_id,
    m.task_id, m.agent_id, m.kind, m.state, m.visibility, m.content,
    m.content_json, m.title, m.tags, m.normalized_sha256, m.dedup_scope,
    m.confidence, m.valid_from,
    m.valid_to, m.recorded_from, m.recorded_to, m.supersedes_id,
    m.superseded_by_id, m.source_id, m.previous_state, m.policy_reason,
    m.policy_confidence, m.version, m.metadata
"""

CONFLICT_COLUMNS = """
    id, tenant_id, project_id, repository_id, swarm_id, run_id, task_id,
    reported_by_agent_id, description, severity, state, resolution,
    resolved_by_agent_id, metadata, created_at, updated_at, resolved_at, version
"""


def _uuid(value: str | UUID) -> UUID:
    return value if isinstance(value, UUID) else UUID(value)


def _uuid_array(values: Sequence[str]) -> list[UUID]:
    return [_uuid(value) for value in values]


def _digest(value: str | None) -> bytes | None:
    return None if value is None else bytes.fromhex(value)


class CockroachMemoryStore:
    """Implement memory/evidence/conflict ports over one Cockroach database."""

    def __init__(self, database: CockroachDatabase) -> None:
        self.database = database

    # ------------------------------------------------------------------
    # Sources and evidence

    async def register_source(
        self,
        actor: ActorContext,
        command: RegisterEvidenceSourceCommand,
    ) -> EvidenceSource:
        source_id = uuid4()
        event_id = uuid4()
        outbox_id = uuid4()
        occurrence = command.occurrence_key or command.idempotency_key

        async def body(connection: Any) -> EvidenceSource:
            if command.task_id is not None:
                await self._require_task(connection, actor, command.task_id)

            cursor = await connection.execute(
                f"""
                SELECT {SOURCE_COLUMNS}
                FROM sources
                WHERE tenant_id = %s
                  AND repository_id = %s
                  AND run_id = %s
                  AND occurrence_key = %s
                FOR UPDATE
                """,
                (actor.tenant_id, actor.repository_id, actor.run_id, occurrence),
            )
            existing = await cursor.fetchone()
            if existing is not None:
                if (
                    str(existing["project_id"]) != actor.project_id
                    or str(existing["swarm_id"]) != actor.swarm_id
                    or bytes(existing["content_sha256"]) != bytes.fromhex(command.content_sha256)
                ):
                    raise IdempotencyConflict(command.idempotency_key)
                return source_from_row(existing)

            cursor = await connection.execute(
                f"""
                INSERT INTO sources (
                    id, tenant_id, project_id, repository_id, swarm_id, run_id,
                    task_id, agent_id, source_type, occurrence_key, uri,
                    content_sha256, content, trust_label, review_state, valid_at,
                    metadata
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, NULL, %s, 'pending', %s, %s
                )
                RETURNING {SOURCE_COLUMNS}
                """,
                (
                    source_id,
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
                    command.trust.value,
                    command.observed_at,
                    Jsonb(command.metadata),
                ),
            )
            row = await cursor.fetchone()
            if row is None:
                raise RuntimeError("source insert returned no row")
            source = source_from_row(row)
            await self._emit(
                connection,
                actor,
                event_id=event_id,
                outbox_id=outbox_id,
                event_type="source.registered",
                aggregate_type=AggregateType.SOURCE,
                aggregate_id=source.source_id,
                aggregate_version=source.version,
                task_id=source.task_id,
                payload={"kind": str(source.kind)},
                idempotency_key=command.idempotency_key,
                occurred_at=source.recorded_at,
            )
            return source

        return await self.database.run_idempotent(
            actor,
            "source.register",
            command,
            EvidenceSource,
            body,
        )

    async def add_evidence(
        self,
        actor: ActorContext,
        command: AddEvidenceCommand,
    ) -> Evidence:
        evidence_id = uuid4()
        event_id = uuid4()
        outbox_id = uuid4()

        async def body(connection: Any) -> Evidence:
            source = await self._require_source(
                connection,
                actor,
                command.source_id,
                for_update=False,
            )
            cursor = await connection.execute(
                f"""
                INSERT INTO evidence (
                    id, tenant_id, kind, locator, content_sha256, excerpt,
                    source_id, artifact, metadata
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING {EVIDENCE_COLUMNS}
                """,
                (
                    evidence_id,
                    actor.tenant_id,
                    str(command.kind),
                    command.locator,
                    _digest(command.content_sha256),
                    command.excerpt,
                    _uuid(command.source_id),
                    Jsonb(command.artifact.model_dump(mode="json"))
                    if command.artifact is not None
                    else None,
                    Jsonb(command.metadata),
                ),
            )
            row = await cursor.fetchone()
            if row is None:
                raise RuntimeError("evidence insert returned no row")
            evidence = evidence_from_row(row)
            await self._emit(
                connection,
                actor,
                event_id=event_id,
                outbox_id=outbox_id,
                event_type="evidence.added",
                aggregate_type=AggregateType.SOURCE,
                aggregate_id=source.source_id,
                aggregate_version=source.version,
                task_id=source.task_id,
                payload={"evidence_id": evidence.evidence_id},
                idempotency_key=command.idempotency_key,
                occurred_at=evidence.recorded_at,
            )
            return evidence

        return await self.database.run_idempotent(
            actor,
            "evidence.add",
            command,
            Evidence,
            body,
        )

    async def review_source(
        self,
        actor: ActorContext,
        command: ReviewSourceCommand,
    ) -> EvidenceSource:
        event_id = uuid4()
        outbox_id = uuid4()

        async def body(connection: Any) -> EvidenceSource:
            source = await self._require_source(
                connection,
                actor,
                command.source_id,
                for_update=True,
            )
            if source.version != command.expected_version:
                raise InvalidState(
                    "source version changed before review",
                    source_id=source.source_id,
                    expected_version=command.expected_version,
                    actual_version=source.version,
                )
            cursor = await connection.execute(
                f"""
                UPDATE sources
                SET review_state = %s,
                    reviewed_by_agent_id = %s,
                    reviewed_at = now(),
                    rejection_reason = NULL,
                    version = version + 1
                WHERE id = %s
                  AND tenant_id = %s
                  AND project_id = %s
                  AND repository_id = %s
                  AND swarm_id = %s
                  AND run_id = %s
                  AND version = %s
                RETURNING {SOURCE_COLUMNS}
                """,
                (
                    command.review_state.value,
                    actor.agent_id,
                    _uuid(command.source_id),
                    actor.tenant_id,
                    actor.project_id,
                    actor.repository_id,
                    actor.swarm_id,
                    actor.run_id,
                    command.expected_version,
                ),
            )
            row = await cursor.fetchone()
            if row is None:
                raise InvalidState("source changed during review", source_id=source.source_id)
            reviewed = source_from_row(row)
            await self._emit(
                connection,
                actor,
                event_id=event_id,
                outbox_id=outbox_id,
                event_type="source.reviewed",
                aggregate_type=AggregateType.SOURCE,
                aggregate_id=reviewed.source_id,
                aggregate_version=reviewed.version,
                task_id=reviewed.task_id,
                payload={
                    "review_state": reviewed.review_state.value,
                    "reason": command.reason,
                },
                idempotency_key=command.idempotency_key,
                occurred_at=reviewed.reviewed_at or reviewed.recorded_at,
            )
            return reviewed

        return await self.database.run_idempotent(
            actor,
            "source.review",
            command,
            EvidenceSource,
            body,
        )

    async def get_source(
        self,
        actor: ActorContext,
        source_id: str,
    ) -> EvidenceSource | None:
        async with self.database.pool.connection() as connection:
            return await self._find_source(connection, actor, source_id, for_update=False)

    async def get_evidence(self, actor: ActorContext, evidence_id: str) -> Evidence | None:
        async with self.database.pool.connection() as connection:
            cursor = await connection.execute(
                f"""
                SELECT {EVIDENCE_ALIAS_COLUMNS}
                FROM evidence AS e
                JOIN sources AS s ON s.id = e.source_id
                WHERE e.id = %s
                  AND s.tenant_id = %s
                  AND s.project_id = %s
                  AND s.repository_id = %s
                  AND s.swarm_id = %s
                  AND s.run_id = %s
                """,
                (
                    _uuid(evidence_id),
                    actor.tenant_id,
                    actor.project_id,
                    actor.repository_id,
                    actor.swarm_id,
                    actor.run_id,
                ),
            )
            row = await cursor.fetchone()
            return None if row is None else evidence_from_row(row)

    # ------------------------------------------------------------------
    # Embedding index

    async def upsert_embeddings(
        self,
        actor: ActorContext,
        vectors: Sequence[EmbeddingVector],
        *,
        idempotency_key: str,
    ) -> None:
        """Upsert fixed-width vectors after deriving scope from the memory row."""

        del idempotency_key  # (memory_id, model) is the natural idempotency key
        prepared: list[tuple[EmbeddingVector, str]] = []
        for vector in vectors:
            if vector.dimensions != COCKROACH_VECTOR_DIMENSIONS:
                raise ValueError(
                    f"CockroachDB vector index requires {COCKROACH_VECTOR_DIMENSIONS} dimensions"
                )
            prepared.append((vector, vector_literal(vector.values)))
        if not prepared:
            return

        async with self.database.pool.connection() as connection:
            for vector, literal in prepared:
                cursor = await connection.execute(
                    """
                    INSERT INTO memory_vector_embeddings (
                        memory_id, tenant_id, project_id, repository_id,
                        model, dimensions, embedding
                    )
                    SELECT m.id, m.tenant_id, m.project_id, m.repository_id,
                           %s, %s, %s::VECTOR
                    FROM memories AS m
                    WHERE m.id = %s
                      AND m.tenant_id = %s
                      AND m.project_id = %s
                      AND m.repository_id = %s
                    ON CONFLICT (memory_id, model) DO UPDATE SET
                        dimensions = excluded.dimensions,
                        embedding = excluded.embedding,
                        created_at = now()
                    """,
                    (
                        vector.model,
                        vector.dimensions,
                        literal,
                        _uuid(vector.memory_id),
                        actor.tenant_id,
                        actor.project_id,
                        actor.repository_id,
                    ),
                )
                if cursor.rowcount != 1:
                    raise ResourceNotFound("memory", vector.memory_id)

    async def search_embeddings(
        self,
        actor: ActorContext,
        query_vector: Sequence[float],
        *,
        model: str,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> tuple[EmbeddingMatch, ...]:
        """Return scoped ANN ids; callers must re-fetch through ``recall``."""

        if not model or len(model) > 255:
            raise ValueError("model must contain between 1 and 255 characters")
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        if not 0.0 <= min_score <= 1.0:
            raise ValueError("min_score must be between 0 and 1")
        literal = vector_literal(query_vector)
        async with self.database.pool.connection() as connection:
            cursor = await connection.execute(
                """
                SELECT memory_id, embedding <=> %s::VECTOR AS distance
                FROM memory_vector_embeddings
                WHERE tenant_id = %s
                  AND project_id = %s
                  AND repository_id = %s
                  AND model = %s
                ORDER BY embedding <=> %s::VECTOR
                LIMIT %s
                """,
                (
                    literal,
                    actor.tenant_id,
                    actor.project_id,
                    actor.repository_id,
                    model,
                    literal,
                    limit,
                ),
            )
            rows = await cursor.fetchall()
        matches = []
        for row in rows:
            score = max(0.0, min(1.0, 1.0 - float(row["distance"])))
            if score >= min_score:
                matches.append(EmbeddingMatch(memory_id=str(row["memory_id"]), score=score))
        return tuple(matches)

    # ------------------------------------------------------------------
    # Memory

    async def remember(
        self,
        actor: ActorContext,
        command: RememberCommand,
        policy: MemoryOperationPolicy,
    ) -> RememberResult:
        current = await self._current_memories_for_policy(actor, command)
        initial_decision = await policy.decide(command, current)

        memory_id = uuid4()
        supersession_link_id = uuid4()
        related_link_ids = {memory: uuid4() for memory in command.related_memory_ids}
        event_id = uuid4()
        outbox_id = uuid4()
        normalized_sha256 = bytes.fromhex(memory_text_sha256(command.content))
        identity_scope = dedup_scope(command.visibility, actor.run_id, command.task_id)

        async def body(connection: Any) -> RememberResult:
            if command.task_id is not None:
                await self._require_task(connection, actor, command.task_id)
            validated_evidence = await self._validate_evidence_refs(
                connection,
                actor,
                command.evidence,
            )
            decision = initial_decision
            if decision.operation is MemoryOperation.NOOP:
                return RememberResult(operation=decision.operation, decision=decision)
            if decision.operation is MemoryOperation.DELETE:
                raise InvalidState(
                    "the v0 conservative policy does not hard-delete memory",
                    operation=decision.operation.value,
                )

            if decision.operation is MemoryOperation.MERGE:
                if not decision.target_memory_ids:
                    raise InvalidState("merge decision has no target memory")
                target = await self._require_memory(
                    connection,
                    actor,
                    decision.target_memory_ids[0],
                    require_current=True,
                    for_update=True,
                )
                combined_tags = tuple(dict.fromkeys((*target.tags, *command.tags)))
                cursor = await connection.execute(
                    """
                    UPDATE memories
                    SET tags = %s,
                        confidence = GREATEST(confidence, %s),
                        version = version + 1
                    WHERE id = %s
                      AND version = %s
                      AND recorded_to IS NULL
                    """,
                    (
                        list(combined_tags),
                        command.confidence,
                        _uuid(target.memory_id),
                        target.version,
                    ),
                )
                if cursor.rowcount != 1:
                    raise InvalidState("memory changed during merge", memory_id=target.memory_id)
                await self._insert_memory_evidence(
                    connection,
                    target.memory_id,
                    tuple(validated_evidence.values()),
                    relation="supports",
                )
                memory = (await self._load_memories_by_ids(connection, actor, (target.memory_id,)))[
                    target.memory_id
                ]
                affected = (memory,)
            elif decision.operation in {MemoryOperation.ADD, MemoryOperation.UPDATE}:
                predecessor: Memory | None = None
                recorded_from = await self._database_now(connection)
                supersedes_id: str | None = None
                if decision.operation is MemoryOperation.UPDATE:
                    if not decision.target_memory_ids:
                        raise InvalidState("update decision has no target memory")
                    predecessor = await self._require_memory(
                        connection,
                        actor,
                        decision.target_memory_ids[0],
                        require_current=True,
                        for_update=True,
                    )
                    recorded_from = max(
                        recorded_from,
                        predecessor.recorded_from + timedelta(microseconds=1),
                    )
                    supersedes_id = predecessor.memory_id

                metadata = {
                    **command.metadata,
                    "policy": {
                        "operation": decision.operation.value,
                        "reason": decision.reason,
                        "confidence": decision.confidence,
                    },
                }
                await connection.execute(
                    """
                    INSERT INTO memories (
                        id, tenant_id, project_id, repository_id, swarm_id, run_id,
                        task_id, agent_id, kind, state, visibility, content,
                        content_json, title, tags, normalized_sha256, dedup_scope,
                        confidence, valid_from,
                        valid_to, recorded_from, supersedes_id, source_id,
                        policy_reason, policy_confidence, metadata
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    """,
                    (
                        memory_id,
                        actor.tenant_id,
                        actor.project_id,
                        actor.repository_id,
                        actor.swarm_id,
                        actor.run_id,
                        _uuid(command.task_id) if command.task_id else None,
                        actor.agent_id,
                        str(command.kind),
                        command.desired_state.value,
                        command.visibility.value,
                        memory_content_text(command.content),
                        Jsonb(command.content) if not isinstance(command.content, str) else None,
                        command.title,
                        list(command.tags),
                        normalized_sha256,
                        identity_scope,
                        command.confidence,
                        command.valid_from or recorded_from,
                        command.valid_to,
                        recorded_from,
                        _uuid(supersedes_id) if supersedes_id else None,
                        _uuid(next(iter(validated_evidence.values())).source_id)
                        if validated_evidence
                        else None,
                        decision.reason,
                        decision.confidence,
                        Jsonb(metadata),
                    ),
                )
                if predecessor is not None:
                    cursor = await connection.execute(
                        """
                        UPDATE memories
                        SET state = 'superseded',
                            previous_state = state,
                            recorded_to = %s,
                            superseded_by_id = %s,
                            version = version + 1
                        WHERE id = %s
                          AND version = %s
                          AND recorded_to IS NULL
                          AND state != 'superseded'
                        """,
                        (
                            recorded_from,
                            memory_id,
                            _uuid(predecessor.memory_id),
                            predecessor.version,
                        ),
                    )
                    if cursor.rowcount != 1:
                        raise InvalidState(
                            "memory changed during supersession",
                            memory_id=predecessor.memory_id,
                        )
                await self._insert_memory_evidence(
                    connection,
                    str(memory_id),
                    tuple(validated_evidence.values()),
                    relation="supports",
                )
                if predecessor is not None:
                    await connection.execute(
                        """
                        INSERT INTO memory_links (
                            id, source_memory_id, target_memory_id, link_type, reason
                        ) VALUES (%s, %s, %s, 'supersedes', %s)
                        ON CONFLICT (source_memory_id, target_memory_id, link_type)
                        DO NOTHING
                        """,
                        (
                            supersession_link_id,
                            memory_id,
                            _uuid(predecessor.memory_id),
                            decision.reason,
                        ),
                    )
                loaded = await self._load_memories_by_ids(
                    connection,
                    actor,
                    tuple(
                        item
                        for item in (
                            predecessor.memory_id if predecessor else None,
                            str(memory_id),
                        )
                        if item is not None
                    ),
                )
                memory = loaded[str(memory_id)]
                affected = (
                    (loaded[predecessor.memory_id], memory)
                    if predecessor is not None
                    else (memory,)
                )
            else:
                raise InvalidState(
                    "unsupported memory operation", operation=decision.operation.value
                )

            for related_id in command.related_memory_ids:
                await self._require_memory(
                    connection,
                    actor,
                    related_id,
                    require_current=True,
                    for_update=False,
                )
                if related_id == memory.memory_id:
                    continue
                await connection.execute(
                    """
                    INSERT INTO memory_links (
                        id, source_memory_id, target_memory_id, link_type
                    ) VALUES (%s, %s, %s, 'related_to')
                    ON CONFLICT (source_memory_id, target_memory_id, link_type)
                    DO NOTHING
                    """,
                    (
                        related_link_ids[related_id],
                        _uuid(memory.memory_id),
                        _uuid(related_id),
                    ),
                )

            event_type = (
                EventType.MEMORY_SUPERSEDED
                if decision.operation is MemoryOperation.UPDATE
                else EventType.MEMORY_ADDED
            )
            now = await self._database_now(connection)
            await self._emit(
                connection,
                actor,
                event_id=event_id,
                outbox_id=outbox_id,
                event_type=event_type,
                aggregate_type=AggregateType.MEMORY,
                aggregate_id=memory.memory_id,
                aggregate_version=memory.version,
                task_id=memory.task_id,
                payload={"operation": decision.operation.value},
                idempotency_key=command.idempotency_key,
                occurred_at=now,
            )
            return RememberResult(
                operation=decision.operation,
                decision=decision,
                memory=memory,
                affected_memories=affected,
            )

        return await self.database.run_idempotent(
            actor,
            "memory.remember",
            command,
            RememberResult,
            body,
        )

    async def recall(self, actor: ActorContext, query: RecallQuery) -> RecallBundle:
        async with self.database.pool.connection() as connection:
            if query.task_id is not None:
                await self._require_task(connection, actor, query.task_id, for_update=False)
            now = await self._database_now(connection)

            clauses = [
                "m.tenant_id = %s",
                "m.project_id = %s",
                "m.repository_id = %s",
            ]
            parameters: list[Any] = [
                actor.tenant_id,
                actor.project_id,
                actor.repository_id,
            ]
            visibility_clauses: list[str] = []
            if Visibility.REPOSITORY in query.visibilities:
                visibility_clauses.append("m.visibility = 'repository'")
            if Visibility.RUN in query.visibilities:
                visibility_clauses.append(
                    "(m.visibility = 'run' AND m.swarm_id = %s AND m.run_id = %s)"
                )
                parameters.extend((actor.swarm_id, actor.run_id))
            if Visibility.TASK in query.visibilities and query.task_id is not None:
                visibility_clauses.append(
                    "(m.visibility = 'task' AND m.swarm_id = %s AND m.run_id = %s "
                    "AND m.task_id = %s)"
                )
                parameters.extend((actor.swarm_id, actor.run_id, _uuid(query.task_id)))
            if not visibility_clauses:
                return RecallBundle(query=query, generated_at=now)
            clauses.append("(" + " OR ".join(visibility_clauses) + ")")

            clauses.append("m.state = ANY(%s::STRING[])")
            parameters.append(sorted(state.value for state in query.effective_states))
            if query.memory_ids:
                clauses.append("m.id = ANY(%s::UUID[])")
                parameters.append(sorted(_uuid(item) for item in query.memory_ids))
            if query.kinds:
                clauses.append("m.kind = ANY(%s::STRING[])")
                parameters.append(sorted(str(kind) for kind in query.kinds))

            if query.recorded_at is not None:
                clauses.extend(
                    (
                        "m.recorded_from <= %s",
                        "(m.recorded_to IS NULL OR %s < m.recorded_to)",
                    )
                )
                parameters.extend((query.recorded_at, query.recorded_at))
            elif not query.include_superseded:
                clauses.append("m.recorded_to IS NULL")

            world_at = query.world_at or now
            clauses.extend(
                (
                    "m.valid_from <= %s",
                    "(m.valid_to IS NULL OR %s < m.valid_to)",
                )
            )
            parameters.extend((world_at, world_at))

            if not query.include_refuted:
                # Evidence-less memory remains recallable. Evidence-backed memory
                # must retain at least one non-rejected/non-untrusted source. The
                # association rows themselves remain append-only for audit.
                clauses.append(
                    """
                    (
                        NOT EXISTS (
                            SELECT 1
                            FROM memory_evidence AS me0
                            WHERE me0.memory_id = m.id
                        )
                        OR EXISTS (
                            SELECT 1
                            FROM memory_evidence AS me1
                            JOIN evidence AS e1 ON e1.id = me1.evidence_id
                            JOIN sources AS s1 ON s1.id = e1.source_id
                            WHERE me1.memory_id = m.id
                              AND s1.review_state != 'rejected'
                              AND s1.trust_label != 'untrusted'
                        )
                    )
                    """
                )

            candidate_limit = min(2000, max(100, query.limit * 20))
            sql = f"""
                SELECT {MEMORY_COLUMNS}, count(*) OVER () AS total_count
                FROM memories AS m
                WHERE {" AND ".join(clauses)}
                ORDER BY m.recorded_from DESC, m.id DESC
                LIMIT %s
            """
            parameters.append(candidate_limit)
            cursor = await connection.execute(sql, tuple(parameters))
            rows = await cursor.fetchall()
            if not rows:
                return RecallBundle(query=query, generated_at=now)

            ids = tuple(str(row["id"]) for row in rows)
            evidence = await self._load_memory_evidence(
                connection,
                ids,
                include_rejected=query.include_refuted,
            )
            memories = [memory_from_row(row, evidence.get(str(row["id"]), ())) for row in rows]
            scored: list[RecallHit] = []
            for memory in memories:
                score, reasons = lexical_score(query.text, memory)
                if score < query.min_score:
                    continue
                scored.append(
                    RecallHit(
                        memory=memory,
                        score=score,
                        reasons=reasons,
                        evidence=memory.evidence if query.include_evidence else (),
                    )
                )
            scored.sort(
                key=lambda hit: (hit.score, hit.memory.recorded_from, hit.memory.memory_id),
                reverse=True,
            )
            total_candidates = int(rows[0]["total_count"])
            return RecallBundle(
                query=query,
                hits=tuple(scored[: query.limit]),
                generated_at=now,
                total_candidates=total_candidates,
                truncated=len(scored) > query.limit or total_candidates > candidate_limit,
            )

    async def get_lineage(self, actor: ActorContext, memory_id: str) -> MemoryLineage:
        async with self.database.pool.connection() as connection:
            await self._require_memory(
                connection,
                actor,
                memory_id,
                require_current=False,
                for_update=False,
            )
            cursor = await connection.execute(
                f"""
                WITH RECURSIVE lineage(memory_id) AS (
                    VALUES (%s::UUID)
                    UNION
                    SELECT candidate.id
                    FROM lineage
                    JOIN memory_links AS ml
                      ON ml.source_memory_id = lineage.memory_id
                      OR ml.target_memory_id = lineage.memory_id
                    JOIN memories AS candidate
                      ON candidate.id = CASE
                          WHEN ml.source_memory_id = lineage.memory_id
                              THEN ml.target_memory_id
                          ELSE ml.source_memory_id
                      END
                    WHERE candidate.tenant_id = %s
                      AND candidate.project_id = %s
                      AND candidate.repository_id = %s
                      AND (
                          candidate.visibility = 'repository'
                          OR (
                              candidate.swarm_id = %s
                              AND candidate.run_id = %s
                              AND candidate.visibility IN ('run', 'task')
                          )
                      )
                )
                SELECT {MEMORY_COLUMNS}
                FROM memories AS m
                JOIN lineage ON lineage.memory_id = m.id
                WHERE m.tenant_id = %s
                  AND m.project_id = %s
                  AND m.repository_id = %s
                  AND (
                      m.visibility = 'repository'
                      OR (
                          m.swarm_id = %s
                          AND m.run_id = %s
                          AND m.visibility IN ('run', 'task')
                      )
                  )
                ORDER BY m.recorded_from, m.id
                """,
                (
                    _uuid(memory_id),
                    actor.tenant_id,
                    actor.project_id,
                    actor.repository_id,
                    actor.swarm_id,
                    actor.run_id,
                    actor.tenant_id,
                    actor.project_id,
                    actor.repository_id,
                    actor.swarm_id,
                    actor.run_id,
                ),
            )
            rows = await cursor.fetchall()
            ids = tuple(str(row["id"]) for row in rows)
            evidence = await self._load_memory_evidence(connection, ids, include_rejected=True)
            memories = tuple(memory_from_row(row, evidence.get(str(row["id"]), ())) for row in rows)
            cursor = await connection.execute(
                """
                SELECT id, source_memory_id, target_memory_id, link_type, reason,
                       metadata, created_at
                FROM memory_links
                WHERE source_memory_id = ANY(%s::UUID[])
                  AND target_memory_id = ANY(%s::UUID[])
                ORDER BY created_at, id
                """,
                (_uuid_array(ids), _uuid_array(ids)),
            )
            links = tuple(link_from_row(row) for row in await cursor.fetchall())
            current_ids = tuple(
                memory.memory_id
                for memory in memories
                if memory.recorded_to is None
                and memory.state not in {MemoryState.SUPERSEDED, MemoryState.REFUTED}
            )
            return MemoryLineage(
                requested_memory_id=memory_id,
                memories=memories,
                links=links,
                current_memory_ids=current_ids,
            )

    async def review_memory(
        self,
        actor: ActorContext,
        command: ReviewMemoryCommand,
    ) -> ReviewMemoryResult:
        event_id = uuid4()
        outbox_id = uuid4()

        async def body(connection: Any) -> ReviewMemoryResult:
            evidence = await self._validate_evidence_refs(connection, actor, command.evidence)
            memory = await self._require_memory(
                connection,
                actor,
                command.memory_id,
                require_current=True,
                for_update=True,
            )
            if memory.version != command.expected_version:
                raise InvalidState(
                    "memory version changed before review",
                    memory_id=memory.memory_id,
                    expected_version=command.expected_version,
                    actual_version=memory.version,
                )
            state = (
                MemoryState.CONFIRMED
                if command.decision is MemoryReviewDecision.CONFIRM
                else MemoryState.REFUTED
            )
            cursor = await connection.execute(
                """
                UPDATE memories
                SET state = %s,
                    version = version + 1
                WHERE id = %s
                  AND version = %s
                  AND recorded_to IS NULL
                """,
                (state.value, _uuid(memory.memory_id), memory.version),
            )
            if cursor.rowcount != 1:
                raise InvalidState("memory changed during review", memory_id=memory.memory_id)
            await self._insert_memory_evidence(
                connection,
                memory.memory_id,
                tuple(evidence.values()),
                relation="supports" if state is MemoryState.CONFIRMED else "refutes",
            )
            reviewed = (await self._load_memories_by_ids(connection, actor, (memory.memory_id,)))[
                memory.memory_id
            ]
            now = await self._database_now(connection)
            await self._emit(
                connection,
                actor,
                event_id=event_id,
                outbox_id=outbox_id,
                event_type=(
                    EventType.MEMORY_CONFIRMED
                    if state is MemoryState.CONFIRMED
                    else EventType.MEMORY_REFUTED
                ),
                aggregate_type=AggregateType.MEMORY,
                aggregate_id=reviewed.memory_id,
                aggregate_version=reviewed.version,
                task_id=reviewed.task_id,
                payload={"reason": command.reason},
                idempotency_key=command.idempotency_key,
                occurred_at=now,
            )
            return ReviewMemoryResult(memory=reviewed)

        return await self.database.run_idempotent(
            actor,
            "memory.review",
            command,
            ReviewMemoryResult,
            body,
        )

    async def reject_source(
        self,
        actor: ActorContext,
        command: RejectSourceCommand,
    ) -> SourceRejectionResult:
        event_id = uuid4()
        outbox_id = uuid4()

        async def body(connection: Any) -> SourceRejectionResult:
            source = await self._require_source(
                connection,
                actor,
                command.source_id,
                for_update=True,
            )
            if source.version != command.expected_version:
                raise InvalidState(
                    "source version changed before rejection",
                    source_id=source.source_id,
                    expected_version=command.expected_version,
                    actual_version=source.version,
                )
            cursor = await connection.execute(
                f"""
                UPDATE sources
                SET review_state = 'rejected',
                    reviewed_by_agent_id = %s,
                    reviewed_at = now(),
                    rejection_reason = %s,
                    version = version + 1
                WHERE id = %s
                  AND tenant_id = %s
                  AND project_id = %s
                  AND repository_id = %s
                  AND swarm_id = %s
                  AND run_id = %s
                  AND version = %s
                RETURNING {SOURCE_COLUMNS}
                """,
                (
                    actor.agent_id,
                    command.reason,
                    _uuid(source.source_id),
                    actor.tenant_id,
                    actor.project_id,
                    actor.repository_id,
                    actor.swarm_id,
                    actor.run_id,
                    source.version,
                ),
            )
            source_row = await cursor.fetchone()
            if source_row is None:
                raise InvalidState("source changed during rejection", source_id=source.source_id)
            rejected_source = source_from_row(source_row)

            # Make queue state agree with source policy and fence any worker
            # that leased the source before this rejection committed.
            await connection.execute(
                """
                UPDATE outbox_work_items
                SET status = 'cancelled',
                    outcome = 'source_rejected',
                    locked_by = NULL,
                    lease_token = NULL,
                    lease_version = lease_version + 1,
                    locked_until = NULL,
                    completed_at = now(),
                    updated_at = now(),
                    version = version + 1
                WHERE subject_id = %s
                  AND kind = 'extract_source'
                  AND tenant_id = %s
                  AND project_id = %s
                  AND repository_id = %s
                  AND swarm_id = %s
                  AND run_id = %s
                  AND status IN ('pending', 'retry', 'leased')
                """,
                (
                    _uuid(source.source_id),
                    actor.tenant_id,
                    actor.project_id,
                    actor.repository_id,
                    actor.swarm_id,
                    actor.run_id,
                ),
            )

            cursor = await connection.execute(
                f"""
                SELECT {MEMORY_COLUMNS}
                FROM memories AS m
                WHERE m.id IN (
                    SELECT me.memory_id
                    FROM memory_evidence AS me
                    JOIN evidence AS e ON e.id = me.evidence_id
                    WHERE e.source_id = %s
                )
                  AND m.recorded_to IS NULL
                  AND m.tenant_id = %s
                  AND m.project_id = %s
                  AND m.repository_id = %s
                  AND (
                      m.visibility = 'repository'
                      OR (m.swarm_id = %s AND m.run_id = %s)
                  )
                FOR UPDATE
                """,
                (
                    _uuid(source.source_id),
                    actor.tenant_id,
                    actor.project_id,
                    actor.repository_id,
                    actor.swarm_id,
                    actor.run_id,
                ),
            )
            rows = await cursor.fetchall()
            memory_ids = tuple(str(row["id"]) for row in rows)
            acceptable: set[str] = set()
            if memory_ids:
                cursor = await connection.execute(
                    """
                    SELECT DISTINCT me.memory_id
                    FROM memory_evidence AS me
                    JOIN evidence AS e ON e.id = me.evidence_id
                    JOIN sources AS s ON s.id = e.source_id
                    WHERE me.memory_id = ANY(%s::UUID[])
                      AND e.source_id != %s
                      AND s.review_state != 'rejected'
                      AND s.trust_label != 'untrusted'
                    """,
                    (_uuid_array(memory_ids), _uuid(source.source_id)),
                )
                acceptable = {str(row["memory_id"]) for row in await cursor.fetchall()}

            rolled_back: list[str] = []
            for row in rows:
                memory_id = str(row["id"])
                if memory_id in acceptable:
                    await connection.execute(
                        "UPDATE memories SET version = version + 1 WHERE id = %s",
                        (_uuid(memory_id),),
                    )
                    continue
                rolled_back.append(memory_id)
                predecessor_id = row.get("supersedes_id")
                await connection.execute(
                    """
                    UPDATE memories
                    SET state = 'refuted',
                        supersedes_id = NULL,
                        version = version + 1
                    WHERE id = %s
                    """,
                    (_uuid(memory_id),),
                )
                if predecessor_id is not None:
                    await self._require_memory(
                        connection,
                        actor,
                        str(predecessor_id),
                        require_current=False,
                        for_update=True,
                    )
                    await connection.execute(
                        """
                        UPDATE memories
                        SET state = COALESCE(previous_state, 'tentative'),
                            recorded_to = NULL,
                            superseded_by_id = NULL,
                            version = version + 1
                        WHERE id = %s
                          AND superseded_by_id = %s
                        """,
                        (_uuid(str(predecessor_id)), _uuid(memory_id)),
                    )

            result = SourceRejectionResult(
                source=rejected_source,
                rolled_back_memory_ids=tuple(sorted(rolled_back)),
            )
            await self._emit(
                connection,
                actor,
                event_id=event_id,
                outbox_id=outbox_id,
                event_type=EventType.SOURCE_REJECTED,
                aggregate_type=AggregateType.SOURCE,
                aggregate_id=rejected_source.source_id,
                aggregate_version=rejected_source.version,
                task_id=rejected_source.task_id,
                payload={"rolled_back_memory_ids": sorted(rolled_back)},
                idempotency_key=command.idempotency_key,
                occurred_at=rejected_source.reviewed_at or rejected_source.recorded_at,
            )
            return result

        return await self.database.run_idempotent(
            actor,
            "source.reject",
            command,
            SourceRejectionResult,
            body,
        )

    # ------------------------------------------------------------------
    # Conflicts

    async def report_conflict(
        self,
        actor: ActorContext,
        command: ReportConflictCommand,
    ) -> ReportConflictResult:
        conflict_id = uuid4()
        event_id = uuid4()
        outbox_id = uuid4()

        async def body(connection: Any) -> ReportConflictResult:
            if command.task_id is not None:
                await self._require_task(connection, actor, command.task_id)
            evidence = await self._validate_evidence_refs(connection, actor, command.evidence)
            for memory_id in command.memory_ids:
                await self._require_memory(
                    connection,
                    actor,
                    memory_id,
                    require_current=False,
                    for_update=False,
                )
            now = await self._database_now(connection)
            await connection.execute(
                """
                INSERT INTO memory_conflicts (
                    id, tenant_id, project_id, repository_id, swarm_id, run_id,
                    task_id, reported_by_agent_id, description, severity, state,
                    metadata, created_at, updated_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'open', %s, %s, %s
                )
                """,
                (
                    conflict_id,
                    actor.tenant_id,
                    actor.project_id,
                    actor.repository_id,
                    actor.swarm_id,
                    actor.run_id,
                    _uuid(command.task_id) if command.task_id else None,
                    actor.agent_id,
                    command.description,
                    command.severity.value,
                    Jsonb(command.metadata),
                    now,
                    now,
                ),
            )
            for position, memory_id in enumerate(command.memory_ids):
                await connection.execute(
                    """
                    INSERT INTO memory_conflict_members (conflict_id, memory_id, position)
                    VALUES (%s, %s, %s)
                    """,
                    (conflict_id, _uuid(memory_id), position),
                )
            for reference in evidence.values():
                await connection.execute(
                    """
                    INSERT INTO conflict_evidence (conflict_id, evidence_id, relation)
                    VALUES (%s, %s, 'reported')
                    ON CONFLICT (conflict_id, evidence_id, relation) DO NOTHING
                    """,
                    (conflict_id, _uuid(reference.evidence_id)),
                )
            conflict = await self._require_conflict(
                connection,
                actor,
                str(conflict_id),
                for_update=False,
            )
            await self._emit(
                connection,
                actor,
                event_id=event_id,
                outbox_id=outbox_id,
                event_type=EventType.CONFLICT_REPORTED,
                aggregate_type=AggregateType.CONFLICT,
                aggregate_id=conflict.conflict_id,
                aggregate_version=conflict.version,
                task_id=conflict.task_id,
                payload={"memory_ids": list(conflict.memory_ids)},
                idempotency_key=command.idempotency_key,
                occurred_at=now,
            )
            return ReportConflictResult(conflict=conflict)

        return await self.database.run_idempotent(
            actor,
            "conflict.report",
            command,
            ReportConflictResult,
            body,
        )

    async def resolve_conflict(
        self,
        actor: ActorContext,
        command: ResolveConflictCommand,
    ) -> ResolveConflictResult:
        resolution_id = uuid4()
        event_id = uuid4()
        outbox_id = uuid4()

        async def body(connection: Any) -> ResolveConflictResult:
            conflict = await self._require_conflict(
                connection,
                actor,
                command.conflict_id,
                for_update=True,
            )
            if conflict.status is not ConflictStatus.OPEN:
                raise InvalidState(
                    "conflict is already settled",
                    conflict_id=conflict.conflict_id,
                )
            if conflict.version != command.expected_version:
                raise InvalidState(
                    "conflict version changed before resolution",
                    conflict_id=conflict.conflict_id,
                    expected_version=command.expected_version,
                    actual_version=conflict.version,
                )
            evidence = await self._validate_evidence_refs(
                connection,
                actor,
                command.resolution.evidence,
            )
            referenced = set(command.resolution.supported_memory_ids) | set(
                command.resolution.refuted_memory_ids
            )
            if not referenced.issubset(conflict.memory_ids):
                raise InvalidState(
                    "resolution references memory outside the conflict",
                    conflict_id=conflict.conflict_id,
                )

            for memory_id in command.resolution.supported_memory_ids:
                memory = await self._require_memory(
                    connection,
                    actor,
                    memory_id,
                    require_current=False,
                    for_update=True,
                )
                if memory.state not in {MemoryState.SUPERSEDED, MemoryState.REFUTED}:
                    cursor = await connection.execute(
                        """
                        UPDATE memories
                        SET state = 'confirmed', version = version + 1
                        WHERE id = %s AND version = %s
                        """,
                        (_uuid(memory_id), memory.version),
                    )
                    if cursor.rowcount != 1:
                        raise InvalidState(
                            "memory changed during conflict resolution",
                            memory_id=memory_id,
                        )
                    await self._insert_memory_evidence(
                        connection,
                        memory_id,
                        tuple(evidence.values()),
                        relation="supports",
                    )
            for memory_id in command.resolution.refuted_memory_ids:
                memory = await self._require_memory(
                    connection,
                    actor,
                    memory_id,
                    require_current=False,
                    for_update=True,
                )
                if memory.state is not MemoryState.SUPERSEDED:
                    cursor = await connection.execute(
                        """
                        UPDATE memories
                        SET state = 'refuted', version = version + 1
                        WHERE id = %s AND version = %s
                        """,
                        (_uuid(memory_id), memory.version),
                    )
                    if cursor.rowcount != 1:
                        raise InvalidState(
                            "memory changed during conflict resolution",
                            memory_id=memory_id,
                        )

            now = await self._database_now(connection)
            resolution = ConflictResolution(
                **command.resolution.model_dump(),
                resolution_id=str(resolution_id),
                resolved_by_agent_id=actor.agent_id,
                resolved_at=now,
            )
            status = (
                ConflictStatus.DISMISSED
                if resolution.kind is ConflictResolutionKind.DISMISS
                else ConflictStatus.RESOLVED
            )
            cursor = await connection.execute(
                """
                UPDATE memory_conflicts
                SET state = %s,
                    resolution = %s,
                    resolved_by_agent_id = %s,
                    resolved_at = %s,
                    updated_at = %s,
                    version = version + 1
                WHERE id = %s
                  AND tenant_id = %s
                  AND project_id = %s
                  AND repository_id = %s
                  AND swarm_id = %s
                  AND run_id = %s
                  AND state = 'open'
                  AND version = %s
                """,
                (
                    status.value,
                    Jsonb(resolution.model_dump(mode="json")),
                    actor.agent_id,
                    now,
                    now,
                    _uuid(conflict.conflict_id),
                    actor.tenant_id,
                    actor.project_id,
                    actor.repository_id,
                    actor.swarm_id,
                    actor.run_id,
                    conflict.version,
                ),
            )
            if cursor.rowcount != 1:
                raise InvalidState(
                    "conflict changed during resolution",
                    conflict_id=conflict.conflict_id,
                )
            for reference in evidence.values():
                await connection.execute(
                    """
                    INSERT INTO conflict_evidence (conflict_id, evidence_id, relation)
                    VALUES (%s, %s, 'supports_resolution')
                    ON CONFLICT (conflict_id, evidence_id, relation) DO NOTHING
                    """,
                    (_uuid(conflict.conflict_id), _uuid(reference.evidence_id)),
                )
            resolved = await self._require_conflict(
                connection,
                actor,
                conflict.conflict_id,
                for_update=False,
            )
            await self._emit(
                connection,
                actor,
                event_id=event_id,
                outbox_id=outbox_id,
                event_type=EventType.CONFLICT_RESOLVED,
                aggregate_type=AggregateType.CONFLICT,
                aggregate_id=resolved.conflict_id,
                aggregate_version=resolved.version,
                task_id=resolved.task_id,
                payload={"resolution": resolution.kind.value},
                idempotency_key=command.idempotency_key,
                occurred_at=now,
            )
            return ResolveConflictResult(conflict=resolved)

        return await self.database.run_idempotent(
            actor,
            "conflict.resolve",
            command,
            ResolveConflictResult,
            body,
        )

    # ------------------------------------------------------------------
    # Scoped query and transaction helpers

    @staticmethod
    async def _database_now(connection: Any) -> Any:
        cursor = await connection.execute("SELECT now() AS database_now")
        row = await cursor.fetchone()
        if row is None:
            raise RuntimeError("database clock query returned no row")
        return row["database_now"]

    async def _require_task(
        self,
        connection: Any,
        actor: ActorContext,
        task_id: str,
        *,
        for_update: bool = False,
    ) -> None:
        lock = " FOR UPDATE" if for_update else ""
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
            """
            + lock,
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

    async def _find_source(
        self,
        connection: Any,
        actor: ActorContext,
        source_id: str,
        *,
        for_update: bool,
    ) -> EvidenceSource | None:
        lock = " FOR UPDATE" if for_update else ""
        cursor = await connection.execute(
            f"""
            SELECT {SOURCE_COLUMNS}
            FROM sources
            WHERE id = %s
              AND tenant_id = %s
              AND project_id = %s
              AND repository_id = %s
              AND swarm_id = %s
              AND run_id = %s
            """
            + lock,
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
        return None if row is None else source_from_row(row)

    async def _require_source(
        self,
        connection: Any,
        actor: ActorContext,
        source_id: str,
        *,
        for_update: bool,
    ) -> EvidenceSource:
        source = await self._find_source(
            connection,
            actor,
            source_id,
            for_update=for_update,
        )
        if source is None:
            raise ResourceNotFound("source", source_id)
        return source

    async def _validate_evidence_refs(
        self,
        connection: Any,
        actor: ActorContext,
        references: Sequence[EvidenceRef],
    ) -> dict[str, EvidenceRef]:
        if not references:
            return {}
        ids = tuple(dict.fromkeys(reference.evidence_id for reference in references))
        cursor = await connection.execute(
            """
            SELECT e.id, e.tenant_id, e.kind, e.locator, e.content_sha256,
                   e.excerpt, e.source_id, e.artifact, e.metadata, e.recorded_at
            FROM evidence AS e
            JOIN sources AS s ON s.id = e.source_id
            WHERE e.id = ANY(%s::UUID[])
              AND e.tenant_id = %s
              AND s.tenant_id = %s
              AND s.project_id = %s
              AND s.repository_id = %s
              AND s.swarm_id = %s
              AND s.run_id = %s
              AND s.review_state != 'rejected'
              AND s.trust_label != 'untrusted'
            """,
            (
                _uuid_array(ids),
                actor.tenant_id,
                actor.tenant_id,
                actor.project_id,
                actor.repository_id,
                actor.swarm_id,
                actor.run_id,
            ),
        )
        found = {str(row["id"]): evidence_from_row(row) for row in await cursor.fetchall()}
        validated: dict[str, EvidenceRef] = {}
        for reference in references:
            evidence = found.get(reference.evidence_id)
            if evidence is None or evidence.source_id != reference.source_id:
                raise ResourceNotFound("evidence", reference.evidence_id)
            validated[reference.evidence_id] = evidence.as_ref()
        return validated

    async def _load_memory_evidence(
        self,
        connection: Any,
        memory_ids: Sequence[str],
        *,
        include_rejected: bool,
    ) -> dict[str, tuple[EvidenceRef, ...]]:
        if not memory_ids:
            return {}
        trust_clause = ""
        if not include_rejected:
            trust_clause = " AND s.review_state != 'rejected' AND s.trust_label != 'untrusted'"
        cursor = await connection.execute(
            """
            SELECT me.memory_id, e.id, e.tenant_id, e.kind, e.locator,
                   e.content_sha256, e.excerpt, e.source_id, e.artifact,
                   e.metadata, e.recorded_at
            FROM memory_evidence AS me
            JOIN evidence AS e ON e.id = me.evidence_id
            JOIN sources AS s ON s.id = e.source_id
            WHERE me.memory_id = ANY(%s::UUID[])
            """
            + trust_clause
            + " ORDER BY me.memory_id, e.recorded_at, e.id",
            (_uuid_array(memory_ids),),
        )
        grouped: dict[str, dict[str, EvidenceRef]] = defaultdict(dict)
        for row in await cursor.fetchall():
            memory_id = str(row["memory_id"])
            evidence = evidence_from_row(row).as_ref()
            grouped[memory_id][evidence.evidence_id] = evidence
        return {memory_id: tuple(items.values()) for memory_id, items in grouped.items()}

    async def _load_memories_by_ids(
        self,
        connection: Any,
        actor: ActorContext,
        memory_ids: Sequence[str],
        *,
        include_rejected_evidence: bool = True,
    ) -> dict[str, Memory]:
        if not memory_ids:
            return {}
        cursor = await connection.execute(
            f"""
            SELECT {MEMORY_COLUMNS}
            FROM memories AS m
            WHERE m.id = ANY(%s::UUID[])
              AND m.tenant_id = %s
              AND m.project_id = %s
              AND m.repository_id = %s
              AND (
                  m.visibility = 'repository'
                  OR (
                      m.swarm_id = %s
                      AND m.run_id = %s
                      AND m.visibility IN ('run', 'task')
                  )
              )
            """,
            (
                _uuid_array(memory_ids),
                actor.tenant_id,
                actor.project_id,
                actor.repository_id,
                actor.swarm_id,
                actor.run_id,
            ),
        )
        rows = await cursor.fetchall()
        evidence = await self._load_memory_evidence(
            connection,
            tuple(str(row["id"]) for row in rows),
            include_rejected=include_rejected_evidence,
        )
        return {
            str(row["id"]): memory_from_row(row, evidence.get(str(row["id"]), ())) for row in rows
        }

    async def _require_memory(
        self,
        connection: Any,
        actor: ActorContext,
        memory_id: str,
        *,
        require_current: bool,
        for_update: bool,
    ) -> Memory:
        lock = " FOR UPDATE" if for_update else ""
        cursor = await connection.execute(
            f"""
            SELECT {MEMORY_COLUMNS}
            FROM memories AS m
            WHERE m.id = %s
              AND m.tenant_id = %s
              AND m.project_id = %s
              AND m.repository_id = %s
              AND (
                  m.visibility = 'repository'
                  OR (
                      m.swarm_id = %s
                      AND m.run_id = %s
                      AND m.visibility IN ('run', 'task')
                  )
              )
            """
            + lock,
            (
                _uuid(memory_id),
                actor.tenant_id,
                actor.project_id,
                actor.repository_id,
                actor.swarm_id,
                actor.run_id,
            ),
        )
        row = await cursor.fetchone()
        if row is None:
            raise ResourceNotFound("memory", memory_id)
        if require_current and row.get("recorded_to") is not None:
            raise InvalidState("memory is not current", memory_id=memory_id)
        evidence = await self._load_memory_evidence(
            connection,
            (memory_id,),
            include_rejected=True,
        )
        return memory_from_row(row, evidence.get(memory_id, ()))

    async def _current_memories_for_policy(
        self,
        actor: ActorContext,
        command: RememberCommand,
    ) -> tuple[Memory, ...]:
        async with self.database.pool.connection() as connection:
            visibility_clause = """
                m.visibility = 'repository'
                OR (m.visibility = 'run' AND m.swarm_id = %s AND m.run_id = %s)
            """
            parameters: list[Any] = [
                actor.tenant_id,
                actor.project_id,
                actor.repository_id,
                actor.swarm_id,
                actor.run_id,
            ]
            if command.task_id is not None:
                visibility_clause += """
                    OR (m.visibility = 'task' AND m.swarm_id = %s
                        AND m.run_id = %s AND m.task_id = %s)
                """
                parameters.extend((actor.swarm_id, actor.run_id, _uuid(command.task_id)))
            cursor = await connection.execute(
                f"""
                SELECT {MEMORY_COLUMNS}
                FROM memories AS m
                WHERE m.tenant_id = %s
                  AND m.project_id = %s
                  AND m.repository_id = %s
                  AND ({visibility_clause})
                  AND m.recorded_to IS NULL
                  AND m.state != 'refuted'
                  AND (
                      NOT EXISTS (
                          SELECT 1 FROM memory_evidence AS me0
                          WHERE me0.memory_id = m.id
                      )
                      OR EXISTS (
                          SELECT 1
                          FROM memory_evidence AS me1
                          JOIN evidence AS e1 ON e1.id = me1.evidence_id
                          JOIN sources AS s1 ON s1.id = e1.source_id
                          WHERE me1.memory_id = m.id
                            AND s1.review_state != 'rejected'
                            AND s1.trust_label != 'untrusted'
                      )
                  )
                ORDER BY m.recorded_from, m.id
                """,
                tuple(parameters),
            )
            rows = await cursor.fetchall()
            ids = tuple(str(row["id"]) for row in rows)
            evidence = await self._load_memory_evidence(
                connection,
                ids,
                include_rejected=False,
            )
            return tuple(memory_from_row(row, evidence.get(str(row["id"]), ())) for row in rows)

    @staticmethod
    async def _insert_memory_evidence(
        connection: Any,
        memory_id: str,
        evidence: Sequence[EvidenceRef],
        *,
        relation: str,
    ) -> None:
        for reference in evidence:
            await connection.execute(
                """
                INSERT INTO memory_evidence (memory_id, evidence_id, relation)
                VALUES (%s, %s, %s)
                ON CONFLICT (memory_id, evidence_id, relation) DO NOTHING
                """,
                (_uuid(memory_id), _uuid(reference.evidence_id), relation),
            )

    async def _require_conflict(
        self,
        connection: Any,
        actor: ActorContext,
        conflict_id: str,
        *,
        for_update: bool,
    ) -> Conflict:
        lock = " FOR UPDATE" if for_update else ""
        cursor = await connection.execute(
            f"""
            SELECT {CONFLICT_COLUMNS}
            FROM memory_conflicts
            WHERE id = %s
              AND tenant_id = %s
              AND project_id = %s
              AND repository_id = %s
              AND swarm_id = %s
              AND run_id = %s
            """
            + lock,
            (
                _uuid(conflict_id),
                actor.tenant_id,
                actor.project_id,
                actor.repository_id,
                actor.swarm_id,
                actor.run_id,
            ),
        )
        row = await cursor.fetchone()
        if row is None:
            raise ResourceNotFound("conflict", conflict_id)
        cursor = await connection.execute(
            """
            SELECT memory_id
            FROM memory_conflict_members
            WHERE conflict_id = %s
            ORDER BY position
            """,
            (_uuid(conflict_id),),
        )
        memory_ids = tuple(str(item["memory_id"]) for item in await cursor.fetchall())
        cursor = await connection.execute(
            """
            SELECT e.id, e.tenant_id, e.kind, e.locator, e.content_sha256,
                   e.excerpt, e.source_id, e.artifact, e.metadata, e.recorded_at
            FROM conflict_evidence AS ce
            JOIN evidence AS e ON e.id = ce.evidence_id
            WHERE ce.conflict_id = %s
              AND ce.relation = 'reported'
            ORDER BY e.recorded_at, e.id
            """,
            (_uuid(conflict_id),),
        )
        reported_evidence = tuple(
            evidence_from_row(item).as_ref() for item in await cursor.fetchall()
        )
        return conflict_from_parts(row, memory_ids, reported_evidence)

    @staticmethod
    async def _emit(
        connection: Any,
        actor: ActorContext,
        *,
        event_id: UUID,
        outbox_id: UUID,
        event_type: EventType | str,
        aggregate_type: AggregateType,
        aggregate_id: str,
        aggregate_version: int,
        task_id: str | None,
        payload: dict[str, Any],
        idempotency_key: str,
        occurred_at: Any,
    ) -> SwarmEvent:
        event = SwarmEvent(
            event_id=str(event_id),
            event_type=event_type,
            aggregate_type=aggregate_type,
            aggregate_id=aggregate_id,
            tenant_id=actor.tenant_id,
            project_id=actor.project_id,
            repository_id=actor.repository_id,
            swarm_id=actor.swarm_id,
            run_id=actor.run_id,
            agent_id=actor.agent_id,
            task_id=task_id,
            aggregate_version=aggregate_version,
            payload=payload,
            occurred_at=occurred_at,
            recorded_at=occurred_at,
            idempotency_key=idempotency_key,
        )
        event_type_value = (
            event.event_type.value if isinstance(event.event_type, EventType) else event.event_type
        )
        await connection.execute(
            """
            INSERT INTO swarm_events (
                id, tenant_id, project_id, repository_id, swarm_id, run_id,
                agent_id, task_id, event_type, aggregate_type, aggregate_id,
                aggregate_version, payload, occurred_at, recorded_at,
                correlation_id, causation_id, idempotency_key
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, NULL, NULL, %s
            )
            """,
            (
                event_id,
                actor.tenant_id,
                actor.project_id,
                actor.repository_id,
                actor.swarm_id,
                actor.run_id,
                actor.agent_id,
                _uuid(task_id) if task_id else None,
                event_type_value,
                event.aggregate_type.value,
                event.aggregate_id,
                event.aggregate_version,
                Jsonb(event.payload),
                event.occurred_at,
                event.recorded_at,
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
            """,
            (
                outbox_id,
                event_id,
                actor.tenant_id,
                actor.run_id,
                event.aggregate_type.value,
                event.aggregate_id,
                event.aggregate_version,
                event_type_value,
                f"event:{event.event_id}",
                Jsonb(event.model_dump(mode="json")),
                event.occurred_at,
                event.occurred_at,
            ),
        )
        return event


CockroachMemoryRepository = CockroachMemoryStore

__all__ = ["CockroachMemoryRepository", "CockroachMemoryStore"]
