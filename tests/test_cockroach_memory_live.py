from __future__ import annotations

import asyncio
import hashlib
import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest

from swarmbrain.adapters.cockroach.database import CockroachDatabase
from swarmbrain.adapters.cockroach.memory import CockroachMemoryStore
from swarmbrain.application.errors import InvalidState
from swarmbrain.application.memory_policy import ConservativeMemoryPolicy
from swarmbrain.domain.agents import ActorContext
from swarmbrain.domain.conflicts import (
    ConflictResolutionKind,
    ConflictResolutionProposal,
    ConflictStatus,
    ReportConflictCommand,
    ResolveConflictCommand,
)
from swarmbrain.domain.events import EventType
from swarmbrain.domain.evidence import (
    AddEvidenceCommand,
    EvidenceKind,
    RegisterEvidenceSourceCommand,
    RejectSourceCommand,
    SourceReviewState,
)
from swarmbrain.domain.memory import (
    MemoryKind,
    MemoryOperation,
    MemoryState,
    RecallQuery,
    RememberCommand,
    Visibility,
)

DATABASE_URL = os.getenv("SWARMBRAIN_TEST_DATABASE_URL")
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(not DATABASE_URL, reason="SWARMBRAIN_TEST_DATABASE_URL is not set"),
]


def _id() -> str:
    return str(uuid4())


def _actor() -> ActorContext:
    return ActorContext(
        tenant_id=_id(),
        project_id=_id(),
        repository_id=_id(),
        swarm_id=_id(),
        run_id=_id(),
        agent_id=_id(),
        harness="pytest",
        provider="local",
        model="none",
        capabilities=frozenset(),
    )


class _InjectedSerializationFailure(RuntimeError):
    sqlstate = "40001"


class _InjectOnce:
    def __init__(self, marker: str) -> None:
        self.marker = marker
        self.triggered = False


class _InjectingConnection:
    def __init__(self, delegate: Any, injection: _InjectOnce) -> None:
        self.delegate = delegate
        self.injection = injection

    def transaction(self) -> Any:
        return self.delegate.transaction()

    async def execute(
        self,
        query: Any,
        params: Any = None,
        **kwargs: Any,
    ) -> Any:
        cursor = await self.delegate.execute(query, params, **kwargs)
        if not self.injection.triggered and self.injection.marker in str(query):
            self.injection.triggered = True
            raise _InjectedSerializationFailure("injected transaction restart")
        return cursor


class _InjectingPool:
    def __init__(self, delegate: Any, injection: _InjectOnce) -> None:
        self.delegate = delegate
        self.injection = injection

    @asynccontextmanager
    async def connection(self) -> Any:
        async with self.delegate.connection() as connection:
            yield _InjectingConnection(connection, self.injection)


def _injecting_store(
    database: CockroachDatabase,
    marker: str,
) -> tuple[CockroachMemoryStore, _InjectOnce]:
    injection = _InjectOnce(marker)
    retrying_database = CockroachDatabase(
        database.database_url,
        pool=_InjectingPool(database.pool, injection),
    )
    return CockroachMemoryStore(retrying_database), injection


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


async def test_temporal_memory_evidence_conflicts_and_outbox_are_atomic(
    database: CockroachDatabase,
) -> None:
    actor = _actor()
    await _insert_run(database, actor)
    store = CockroachMemoryStore(database)
    policy = ConservativeMemoryPolicy()
    observed_at = datetime.now(UTC)
    source_content = b"the authoritative SQL port is 26257"

    source = await store.register_source(
        actor,
        RegisterEvidenceSourceCommand(
            idempotency_key=f"source-{_id()}",
            occurrence_key=f"command-output-{_id()}",
            kind=EvidenceKind.COMMAND_OUTPUT,
            content_sha256=hashlib.sha256(source_content).hexdigest(),
            observed_at=observed_at,
            uri="command://cockroach-sql-show-port",
        ),
    )
    evidence = await store.add_evidence(
        actor,
        AddEvidenceCommand(
            idempotency_key=f"evidence-{_id()}",
            source_id=source.source_id,
            kind=EvidenceKind.COMMAND_OUTPUT,
            locator="stdout:1",
            excerpt="sql port: 26257",
            content_sha256=hashlib.sha256(b"sql port: 26257").hexdigest(),
        ),
    )

    first_command = RememberCommand(
        idempotency_key=f"remember-first-{_id()}",
        kind=MemoryKind.HYPOTHESIS,
        visibility=Visibility.REPOSITORY,
        content="The database port is 5432",
    )
    first = await store.remember(actor, first_command, policy)
    replay = await store.remember(actor, first_command, policy)
    assert first.operation is MemoryOperation.ADD
    assert first.memory is not None
    assert replay.replayed is True
    assert replay.memory == first.memory

    correction = await store.remember(
        actor,
        RememberCommand(
            idempotency_key=f"remember-correction-{_id()}",
            kind=MemoryKind.INVARIANT,
            desired_state=MemoryState.CONFIRMED,
            visibility=Visibility.REPOSITORY,
            content="The CockroachDB SQL port is 26257",
            evidence=(evidence.as_ref(),),
            supersedes_memory_id=first.memory.memory_id,
        ),
        policy,
    )
    assert correction.operation is MemoryOperation.UPDATE
    assert correction.memory is not None

    current = await store.recall(actor, RecallQuery(text="database SQL port"))
    assert [hit.memory.memory_id for hit in current.hits] == [correction.memory.memory_id]
    lineage = await store.get_lineage(actor, correction.memory.memory_id)
    assert {item.memory_id for item in lineage.memories} == {
        first.memory.memory_id,
        correction.memory.memory_id,
    }
    assert lineage.current_memory_ids == (correction.memory.memory_id,)

    reported = await store.report_conflict(
        actor,
        ReportConflictCommand(
            idempotency_key=f"conflict-report-{_id()}",
            memory_ids=(first.memory.memory_id, correction.memory.memory_id),
            description="Two incompatible SQL ports",
            evidence=(evidence.as_ref(),),
        ),
    )
    resolving_store, resolution_retry = _injecting_store(database, "UPDATE memory_conflicts")
    resolved = await resolving_store.resolve_conflict(
        actor,
        ResolveConflictCommand(
            idempotency_key=f"conflict-resolve-{_id()}",
            conflict_id=reported.conflict.conflict_id,
            expected_version=reported.conflict.version,
            resolution=ConflictResolutionProposal(
                kind=ConflictResolutionKind.PREFER_NEWER,
                rationale="The captured CockroachDB output is authoritative",
                evidence=(evidence.as_ref(),),
                supported_memory_ids=(correction.memory.memory_id,),
                refuted_memory_ids=(first.memory.memory_id,),
            ),
        ),
    )
    assert resolution_retry.triggered is True
    assert resolved.conflict.status is ConflictStatus.RESOLVED

    rejection_key = f"source-reject-{_id()}"
    rejecting_store, rejection_retry = _injecting_store(
        database,
        "SET state = 'refuted',\n                        supersedes_id = NULL",
    )
    rejected = await rejecting_store.reject_source(
        actor,
        RejectSourceCommand(
            idempotency_key=rejection_key,
            source_id=source.source_id,
            expected_version=source.version,
            reason="The captured command output was from the wrong cluster",
        ),
    )
    assert rejection_retry.triggered is True
    assert rejected.source.review_state is SourceReviewState.REJECTED
    assert rejected.rolled_back_memory_ids == (correction.memory.memory_id,)

    after_rejection = await store.recall(actor, RecallQuery(text="database port"))
    assert [hit.memory.memory_id for hit in after_rejection.hits] == [first.memory.memory_id]
    historical_lineage = await store.get_lineage(actor, correction.memory.memory_id)
    assert len(historical_lineage.memories) == 2
    assert historical_lineage.current_memory_ids == (first.memory.memory_id,)

    async with database.pool.connection() as connection:
        cursor = await connection.execute(
            """
            SELECT count(*) AS associations
            FROM memory_evidence
            WHERE memory_id = %s AND evidence_id = %s
            """,
            (correction.memory.memory_id, evidence.evidence_id),
        )
        association_row = await cursor.fetchone()
        assert association_row is not None
        assert int(association_row["associations"]) == 1

        cursor = await connection.execute(
            """
            SELECT relation
            FROM conflict_evidence
            WHERE conflict_id = %s AND evidence_id = %s
            ORDER BY relation
            """,
            (reported.conflict.conflict_id, evidence.evidence_id),
        )
        assert {str(row["relation"]) for row in await cursor.fetchall()} == {
            "reported",
            "supports_resolution",
        }

        cursor = await connection.execute(
            """
            SELECT o.event_id, o.dedupe_key, o.payload, e.event_type
            FROM outbox_events AS o
            JOIN swarm_events AS e ON e.id = o.event_id
            WHERE e.tenant_id = %s
              AND e.run_id = %s
              AND e.idempotency_key = %s
            """,
            (actor.tenant_id, actor.run_id, rejection_key),
        )
        outbox = await cursor.fetchone()
        assert outbox is not None
        assert outbox["dedupe_key"] == f"event:{outbox['event_id']}"
        assert outbox["payload"]["event_id"] == str(outbox["event_id"])
        assert outbox["payload"]["event_type"] == EventType.SOURCE_REJECTED.value
        assert outbox["event_type"] == EventType.SOURCE_REJECTED.value


async def test_concurrent_writers_cannot_create_divergent_successors(
    database: CockroachDatabase,
) -> None:
    first_actor = _actor()
    second_actor = first_actor.model_copy(update={"agent_id": _id(), "harness": "other"})
    await _insert_run(database, first_actor)
    store = CockroachMemoryStore(database)
    policy = ConservativeMemoryPolicy()
    base = await store.remember(
        first_actor,
        RememberCommand(
            idempotency_key=f"branch-base-{_id()}",
            kind=MemoryKind.HYPOTHESIS,
            visibility=Visibility.REPOSITORY,
            content="The serializer uses revision zero",
        ),
        policy,
    )
    assert base.memory is not None

    outcomes = await asyncio.gather(
        store.remember(
            first_actor,
            RememberCommand(
                idempotency_key=f"branch-first-{_id()}",
                kind=MemoryKind.OBSERVATION,
                visibility=Visibility.REPOSITORY,
                content="The serializer uses revision one",
                supersedes_memory_id=base.memory.memory_id,
            ),
            policy,
        ),
        store.remember(
            second_actor,
            RememberCommand(
                idempotency_key=f"branch-second-{_id()}",
                kind=MemoryKind.OBSERVATION,
                visibility=Visibility.REPOSITORY,
                content="The serializer uses revision two",
                supersedes_memory_id=base.memory.memory_id,
            ),
            policy,
        ),
        return_exceptions=True,
    )
    successes = [item for item in outcomes if not isinstance(item, BaseException)]
    failures = [item for item in outcomes if isinstance(item, BaseException)]

    assert len(successes) == 1
    assert len(failures) == 1
    assert isinstance(failures[0], InvalidState)
    assert successes[0].memory is not None

    async with database.pool.connection() as connection:
        cursor = await connection.execute(
            """
            SELECT id, state, recorded_to, supersedes_id, superseded_by_id
            FROM memories
            WHERE tenant_id = %s
              AND repository_id = %s
              AND (id = %s OR supersedes_id = %s)
            ORDER BY recorded_from, id
            """,
            (
                first_actor.tenant_id,
                first_actor.repository_id,
                base.memory.memory_id,
                base.memory.memory_id,
            ),
        )
        rows = await cursor.fetchall()
        assert len(rows) == 2
        predecessor = next(row for row in rows if str(row["id"]) == base.memory.memory_id)
        successor = next(row for row in rows if row["supersedes_id"] is not None)
        assert predecessor["state"] == MemoryState.SUPERSEDED.value
        assert predecessor["recorded_to"] is not None
        assert str(predecessor["superseded_by_id"]) == str(successor["id"])
        assert successor["recorded_to"] is None
