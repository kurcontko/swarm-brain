from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest

from swarmbrain.adapters.cockroach.memory import CockroachMemoryStore
from swarmbrain.adapters.cockroach.memory_mappers import (
    conflict_from_parts,
    dedup_scope,
    evidence_from_row,
    lexical_score,
    memory_from_row,
    source_from_row,
)
from swarmbrain.domain.agents import ActorContext
from swarmbrain.domain.conflicts import ConflictStatus
from swarmbrain.domain.events import AggregateType, EventType
from swarmbrain.domain.evidence import EvidenceKind, SourceReviewState, SourceTrust
from swarmbrain.domain.memory import MemoryKind, RecallQuery, Visibility
from swarmbrain.ports.memory_store import (
    ConflictStore,
    EvidenceStore,
    MemoryReviewStore,
    MemoryStore,
)


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


def _source_row(*, now: datetime, source_id: str, run_id: str) -> dict[str, Any]:
    return {
        "id": UUID(source_id),
        "tenant_id": _id(),
        "project_id": _id(),
        "repository_id": _id(),
        "swarm_id": _id(),
        "run_id": run_id,
        "task_id": None,
        "agent_id": _id(),
        "source_type": "document",
        "occurrence_key": "docs/readme#1",
        "uri": "file:///repo/README.md",
        "content_sha256": bytes.fromhex("ab" * 32),
        "trust_label": "trusted",
        "review_state": "approved",
        "valid_at": now,
        "recorded_at": now,
        "reviewed_by_agent_id": UUID(_id()),
        "reviewed_at": now,
        "rejection_reason": None,
        "version": 2,
        "metadata": {"branch": "main"},
    }


def _memory_row(*, now: datetime, memory_id: str, actor: ActorContext) -> dict[str, Any]:
    return {
        "id": UUID(memory_id),
        "tenant_id": actor.tenant_id,
        "project_id": actor.project_id,
        "repository_id": actor.repository_id,
        "swarm_id": actor.swarm_id,
        "run_id": actor.run_id,
        "task_id": None,
        "agent_id": actor.agent_id,
        "kind": "invariant",
        "state": "confirmed",
        "visibility": "repository",
        "content": "CockroachDB retries serialization failures",
        "title": "Retry invariant",
        "tags": ["cockroachdb", "retry"],
        "normalized_sha256": bytes.fromhex("cd" * 32),
        "dedup_scope": "repository",
        "confidence": Decimal("0.9750"),
        "occurred_at": now - timedelta(days=7),
        "valid_from": now,
        "valid_to": None,
        "recorded_from": now,
        "recorded_to": None,
        "supersedes_id": None,
        "superseded_by_id": None,
        "source_id": None,
        "previous_state": None,
        "policy_reason": "new",
        "policy_confidence": Decimal("0.9000"),
        "version": 1,
        "metadata": {"origin": "test"},
    }


def test_store_satisfies_all_memory_contract_protocols() -> None:
    store = CockroachMemoryStore(SimpleNamespace())  # type: ignore[arg-type]

    assert isinstance(store, MemoryStore)
    assert isinstance(store, MemoryReviewStore)
    assert isinstance(store, EvidenceStore)
    assert isinstance(store, ConflictStore)


def test_cockroach_rows_map_to_strict_domain_contracts() -> None:
    now = datetime(2026, 8, 2, 8, 30, tzinfo=UTC)
    actor = _actor()
    source_id = _id()
    evidence_id = _id()
    memory_id = _id()

    source = source_from_row(_source_row(now=now, source_id=source_id, run_id=actor.run_id))
    assert source.source_id == source_id
    assert source.kind is EvidenceKind.DOCUMENT
    assert source.trust is SourceTrust.TRUSTED
    assert source.review_state is SourceReviewState.APPROVED
    assert source.content_sha256 == "ab" * 32

    evidence = evidence_from_row(
        {
            "id": UUID(evidence_id),
            "tenant_id": actor.tenant_id,
            "kind": "document",
            "locator": "README.md:12",
            "content_sha256": bytes.fromhex("ef" * 32),
            "excerpt": "Use retry loops.",
            "source_id": UUID(source_id),
            "artifact": None,
            "metadata": {"line": 12},
            "recorded_at": now,
        }
    )
    memory = memory_from_row(
        _memory_row(now=now, memory_id=memory_id, actor=actor),
        (evidence.as_ref(),),
    )
    assert memory.memory_id == memory_id
    assert memory.kind is MemoryKind.INVARIANT
    assert memory.confidence == pytest.approx(0.975)
    assert memory.evidence == (evidence.as_ref(),)
    assert memory.occurred_at == now - timedelta(days=7)

    conflict = conflict_from_parts(
        {
            "id": UUID(_id()),
            "tenant_id": actor.tenant_id,
            "project_id": actor.project_id,
            "repository_id": actor.repository_id,
            "swarm_id": actor.swarm_id,
            "run_id": actor.run_id,
            "task_id": None,
            "reported_by_agent_id": actor.agent_id,
            "description": "Two incompatible ports",
            "severity": "high",
            "state": "open",
            "resolution": None,
            "resolved_by_agent_id": None,
            "metadata": {},
            "created_at": now,
            "updated_at": now,
            "resolved_at": None,
            "version": 1,
        },
        (memory_id, _id()),
        (evidence.as_ref(),),
    )
    assert conflict.status is ConflictStatus.OPEN
    assert conflict.reported_evidence == (evidence.as_ref(),)


def test_cockroach_mapper_preserves_structured_content_and_custom_labels() -> None:
    now = datetime(2026, 8, 2, 8, 30, tzinfo=UTC)
    actor = _actor()
    row = _memory_row(now=now, memory_id=_id(), actor=actor)
    row.update(
        {
            "kind": "org.acme/preference",
            "content": '{"preference":"vim"}',
            "content_json": {"preference": "vim", "contexts": ["terminal"]},
        }
    )

    memory = memory_from_row(row)

    assert memory.kind == "org.acme/preference"
    assert memory.content == {"preference": "vim", "contexts": ["terminal"]}
    score, _ = lexical_score("vim", memory)
    assert score > 0


def test_dedup_scope_and_lexical_ranking_are_deterministic() -> None:
    now = datetime(2026, 8, 2, 8, 30, tzinfo=UTC)
    actor = _actor()
    task_id = _id()
    memory = memory_from_row(_memory_row(now=now, memory_id=_id(), actor=actor))

    assert dedup_scope(Visibility.REPOSITORY, actor.run_id, None) == "repository"
    assert dedup_scope(Visibility.RUN, actor.run_id, None) == f"run:{actor.run_id}"
    assert dedup_scope(Visibility.TASK, actor.run_id, task_id) == f"task:{task_id}"
    with pytest.raises(ValueError, match="task_id"):
        dedup_scope(Visibility.TASK, actor.run_id, None)

    score, reasons = lexical_score("CockroachDB serialization retry", memory)
    assert score == pytest.approx(1.0)
    assert reasons == ("lexical_overlap",)


class _Cursor:
    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self.rows = rows or []

    async def fetchone(self) -> dict[str, Any] | None:
        return self.rows[0] if self.rows else None

    async def fetchall(self) -> list[dict[str, Any]]:
        return self.rows


class _RecordingConnection:
    def __init__(self, now: datetime) -> None:
        self.now = now
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    async def execute(
        self,
        sql: str,
        parameters: tuple[Any, ...] = (),
    ) -> _Cursor:
        self.calls.append((sql, parameters))
        if "SELECT now() AS database_now" in sql:
            return _Cursor([{"database_now": self.now}])
        return _Cursor()


class _Pool:
    def __init__(self, connection: _RecordingConnection) -> None:
        self.value = connection

    @asynccontextmanager
    async def connection(self) -> Any:
        yield self.value


@pytest.mark.asyncio
async def test_recall_applies_scope_trust_and_time_in_sql_before_ranking() -> None:
    now = datetime(2026, 8, 2, 8, 30, tzinfo=UTC)
    actor = _actor()
    connection = _RecordingConnection(now)
    database = SimpleNamespace(pool=_Pool(connection))
    store = CockroachMemoryStore(database)  # type: ignore[arg-type]

    result = await store.recall(actor, RecallQuery(text="serialization retry"))

    assert result.hits == ()
    query_sql = next(
        sql
        for sql, _ in connection.calls
        if "FROM retrieval_documents@retrieval_documents_fts" in sql
    )
    assert "INNER LOOKUP JOIN memories@primary AS m" in query_sql
    assert "m.tenant_id = %s" in query_sql
    assert "m.project_id = %s" in query_sql
    assert "m.repository_id = %s" in query_sql
    assert "m.recorded_to IS NULL" in query_sql
    assert "m.valid_from <= %s" in query_sql
    assert "s1.review_state != 'rejected'" in query_sql
    assert "s1.trust_label != 'untrusted'" in query_sql


class _ReuseDatabase:
    """Stand in for the transaction kernel used by the durable reuse counter."""

    def __init__(self, now: datetime, failure: Exception | None = None) -> None:
        self.now = now
        self.failure = failure
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    async def run(self, body: Any) -> Any:
        if self.failure is not None:
            raise self.failure
        connection = _RecordingConnection(self.now)
        result = await body(connection)
        self.calls.extend(connection.calls)
        return result


@pytest.mark.asyncio
async def test_retrieval_reuse_write_is_scoped_deduplicated_and_content_free() -> None:
    now = datetime(2026, 8, 2, 8, 30, tzinfo=UTC)
    actor = _actor()
    first, second = _id(), _id()
    database = _ReuseDatabase(now)
    store = CockroachMemoryStore(database)  # type: ignore[arg-type]

    await store.record_retrieval_reuse(actor, (first, second, first))

    assert len(database.calls) == 1
    sql, parameters = database.calls[0]
    assert "INSERT INTO retrieval_reuse_counters" in sql
    assert "ON CONFLICT (tenant_id, run_id) DO UPDATE" in sql
    assert parameters == (
        actor.tenant_id,
        actor.run_id,
        actor.project_id,
        actor.repository_id,
        actor.swarm_id,
        2,
    )
    assert first not in parameters
    assert second not in parameters


@pytest.mark.asyncio
async def test_empty_retrieval_records_an_activation_attempt_without_activated_hits() -> None:
    database = _ReuseDatabase(datetime(2026, 8, 2, 8, 30, tzinfo=UTC))
    store = CockroachMemoryStore(database)  # type: ignore[arg-type]

    actor = _actor()
    await store.record_retrieval_reuse(actor, ())

    assert len(database.calls) == 1
    _sql, parameters = database.calls[0]
    assert parameters == (
        actor.tenant_id,
        actor.run_id,
        actor.project_id,
        actor.repository_id,
        actor.swarm_id,
        0,
    )


@pytest.mark.asyncio
async def test_failed_retrieval_reuse_write_is_logged_and_never_raised(
    caplog: pytest.LogCaptureFixture,
) -> None:
    actor = _actor()
    database = _ReuseDatabase(
        datetime(2026, 8, 2, 8, 30, tzinfo=UTC),
        failure=RuntimeError("counter write failed"),
    )
    store = CockroachMemoryStore(database)  # type: ignore[arg-type]

    with caplog.at_level("WARNING"):
        await store.record_retrieval_reuse(actor, (_id(),))

    assert database.calls == []
    assert "durable retrieval reuse counter write failed" in caplog.text


@pytest.mark.asyncio
async def test_event_and_full_outbox_envelope_share_one_stable_identity() -> None:
    now = datetime(2026, 8, 2, 8, 30, tzinfo=UTC)
    actor = _actor()
    connection = _RecordingConnection(now)
    event_id = uuid4()
    outbox_id = uuid4()
    aggregate_id = _id()

    event = await CockroachMemoryStore._emit(
        connection,
        actor,
        event_id=event_id,
        outbox_id=outbox_id,
        event_type=EventType.MEMORY_ADDED,
        aggregate_type=AggregateType.MEMORY,
        aggregate_id=aggregate_id,
        aggregate_version=3,
        task_id=None,
        payload={"operation": "add"},
        idempotency_key="remember-1",
        occurred_at=now,
    )

    assert len(connection.calls) == 2
    event_parameters = connection.calls[0][1]
    outbox_parameters = connection.calls[1][1]
    envelope = outbox_parameters[9].obj
    assert event.event_id == str(event_id)
    assert event_parameters[0] == event_id
    assert event_parameters[10] == aggregate_id
    assert outbox_parameters[0] == outbox_id
    assert outbox_parameters[1] == event_id
    assert outbox_parameters[5] == aggregate_id
    assert outbox_parameters[8] == f"event:{event_id}"
    assert envelope["event_id"] == str(event_id)
    # The emitter's own keys survive; actor enrichment is purely additive.
    assert envelope["payload"] == {
        "operation": "add",
        "agent_harness": "pytest",
        "agent_provider": "local",
        "agent_model": "none",
        "agent_display": "pytest (local)",
    }
    assert envelope["idempotency_key"] == "remember-1"
