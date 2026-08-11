from __future__ import annotations

import hashlib
import os
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest

from swarmbrain.adapters.cockroach.database import CockroachDatabase
from swarmbrain.adapters.cockroach.memory import CockroachMemoryStore
from swarmbrain.adapters.cockroach.retrieval import (
    CockroachGraphRetrievalGateway,
    CockroachRetrievalGateway,
    cockroach_retrieval_gateways,
)
from swarmbrain.adapters.cockroach.schema import install_schema
from swarmbrain.application.memory_policy import ConservativeMemoryPolicy
from swarmbrain.application.retrieval_service import RetrievalService
from swarmbrain.domain.agents import ActorContext, Capability
from swarmbrain.domain.evidence import (
    AddEvidenceCommand,
    EvidenceKind,
    RegisterEvidenceSourceCommand,
    SourceTrust,
)
from swarmbrain.domain.memory import MemoryState, RecallQuery, RememberCommand, Visibility
from swarmbrain.domain.retrieval import (
    CandidateBatch,
    FusedCandidate,
    RetrievalPurpose,
    RetrievalSignal,
)
from swarmbrain.retrieval import RetrievalPlanner
from swarmbrain.retrieval.projection import normalize_term, trigram_similarity

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
        capabilities=frozenset({Capability.MEMORY_PUBLISH.value, Capability.MEMORY_RECALL.value}),
    )


class _RecordingConnection:
    def __init__(
        self,
        connection: Any,
        records: list[tuple[str, tuple[Any, ...] | None]],
    ) -> None:
        self.connection = connection
        self.records = records

    async def execute(
        self,
        statement: str,
        parameters: tuple[Any, ...] | None = None,
    ) -> Any:
        self.records.append((statement, parameters))
        return await self.connection.execute(statement, parameters)


class _RecordingLease:
    def __init__(self, lease: Any, records: list[tuple[str, tuple[Any, ...] | None]]) -> None:
        self.lease = lease
        self.records = records

    async def __aenter__(self) -> _RecordingConnection:
        connection = await self.lease.__aenter__()
        return _RecordingConnection(connection, self.records)

    async def __aexit__(self, *args: object) -> Any:
        return await self.lease.__aexit__(*args)


class _RecordingPool:
    def __init__(self, pool: Any, records: list[tuple[str, tuple[Any, ...] | None]]) -> None:
        self.pool = pool
        self.records = records

    def connection(self, *args: object, **kwargs: object) -> _RecordingLease:
        return _RecordingLease(self.pool.connection(*args, **kwargs), self.records)


def _runtime_statement(
    records: list[tuple[str, tuple[Any, ...] | None]],
    marker: str,
) -> tuple[str, tuple[Any, ...]]:
    matches = [record for record in records if marker in record[0]]
    assert matches, [statement for statement, _ in records]
    statement, parameters = matches[0]
    assert parameters is not None
    return statement, parameters


async def _explain_runtime_statement(
    database: CockroachDatabase,
    statement: tuple[str, tuple[Any, ...]],
) -> str:
    sql, parameters = statement
    async with database.pool.connection() as connection:
        cursor = await connection.execute("EXPLAIN " + sql, parameters)
        return "\n".join(str(next(iter(row.values()))) for row in await cursor.fetchall())


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


async def test_live_exact_fts_trigram_rrf_abstention_and_explain(
    database: CockroachDatabase,
) -> None:
    actor = _actor()
    await _insert_run(database, actor)
    store = CockroachMemoryStore(database)
    published = await store.remember(
        actor,
        RememberCommand(
            idempotency_key=f"retrieval-v7-{_id()}",
            kind="procedure",
            desired_state=MemoryState.CONFIRMED,
            visibility=Visibility.REPOSITORY,
            title="RetryCoordinator",
            content="CockroachDB serialization sentinel uses bounded exponential backoff",
            tags=("retry",),
        ),
        ConservativeMemoryPolicy(),
    )
    assert published.memory is not None
    memory_id = published.memory.memory_id
    records: list[tuple[str, tuple[Any, ...] | None]] = []
    recording_database = CockroachDatabase(
        database.database_url,
        pool=_RecordingPool(database.pool, records),
    )
    retrieval = RetrievalService(cockroach_retrieval_gateways(recording_database), store)

    exact = await retrieval.execute(actor, RecallQuery(text="RetryCoordinator", min_score=0.6))
    exact_statement = _runtime_statement(records, "FROM retrieval_exact_terms@")
    records.clear()
    lexical = await retrieval.execute(
        actor, RecallQuery(text="serialization sentinel", min_score=0.01)
    )
    fts_statement = _runtime_statement(
        records,
        "FROM retrieval_documents@retrieval_documents_fts",
    )
    records.clear()
    fuzzy = await retrieval.execute(actor, RecallQuery(text="RetryCoordintor"))
    trgm_statement = _runtime_statement(
        records,
        "FROM retrieval_documents@retrieval_documents_lookup_trgm",
    )
    records.clear()
    absent = await retrieval.execute(actor, RecallQuery(text="quasar neutrino observatory"))

    assert exact.bundle.hits[0].memory.memory_id == memory_id
    assert lexical.bundle.hits[0].memory.memory_id == memory_id
    assert fuzzy.bundle.hits[0].memory.memory_id == memory_id
    assert absent.bundle.hits == ()
    assert {batch.lane.value for batch in lexical.trace.batches} == {
        "exact",
        "lexical",
        "fuzzy",
        "graph",
    }
    assert lexical.trace.fusion_version == "weighted-rrf-v1"

    left = "RetryCoordinator\nretry"
    right = "RetryCoordintor"
    async with database.pool.connection() as connection:
        cursor = await connection.execute(
            "SELECT similarity(%s::STRING, %s::STRING) AS score",
            (normalize_term(left), normalize_term(right)),
        )
        similarity_row = await cursor.fetchone()
    assert similarity_row is not None
    assert float(similarity_row["score"]) == pytest.approx(trigram_similarity(left, right))

    exact_plan = await _explain_runtime_statement(database, exact_statement)
    fts_plan = await _explain_runtime_statement(database, fts_statement)
    trgm_plan = await _explain_runtime_statement(database, trgm_statement)

    assert "retrieval_documents_fts" in fts_plan
    assert "retrieval_documents_lookup_trgm" in trgm_plan
    assert "retrieval_exact_terms_pkey" in exact_plan
    for plan in (exact_plan, fts_plan, trgm_plan):
        assert "lookup join" in plan.lower()
        assert "FULL SCAN" not in plan.upper()


async def test_live_failed_sql_lane_rolls_back_to_savepoint_without_poisoning_snapshot(
    database: CockroachDatabase,
) -> None:
    actor = _actor()
    await _insert_run(database, actor)
    store = CockroachMemoryStore(database)
    published = await store.remember(
        actor,
        RememberCommand(
            idempotency_key=f"retrieval-savepoint-{_id()}",
            kind="procedure",
            desired_state=MemoryState.CONFIRMED,
            visibility=Visibility.REPOSITORY,
            title="Retrieval Savepoint Sentinel",
            content="valid exact candidate survives one broken SQL lane",
        ),
        ConservativeMemoryPolicy(),
    )
    assert published.memory is not None

    class BrokenSqlGateway:
        signal = RetrievalSignal.FUZZY

        async def retrieve(self, *_args: object) -> CandidateBatch:
            async with database.retrieval_connection() as connection:
                await connection.execute(
                    "SELECT retrieval_lane_missing_column FROM memories LIMIT 1"
                )
            raise AssertionError("invalid SQL unexpectedly succeeded")

    retrieval = RetrievalService(
        (
            BrokenSqlGateway(),
            CockroachRetrievalGateway(database, RetrievalSignal.EXACT),
        ),
        store,
    )

    execution = await retrieval.execute(
        actor,
        RecallQuery(text="Retrieval Savepoint Sentinel", min_score=0.6),
    )

    assert execution.bundle.hits[0].memory.memory_id == published.memory.memory_id
    fuzzy = next(batch for batch in execution.trace.batches if batch.lane is RetrievalSignal.FUZZY)
    assert fuzzy.degraded is True
    assert fuzzy.degradation_reason == "UndefinedColumn"


async def test_live_trust_before_limit_audit_sanitization_and_cross_run_parity(
    database: CockroachDatabase,
) -> None:
    actor = _actor()
    await _insert_run(database, actor)
    store = CockroachMemoryStore(database)
    now = datetime.now(UTC)

    async def evidence(trust: SourceTrust, suffix: str) -> tuple[object, str]:
        payload = f"live-source-{suffix}".encode()
        source = await store.register_source(
            actor,
            RegisterEvidenceSourceCommand(
                idempotency_key=f"live-source-{suffix}-{_id()}",
                occurrence_key=f"live-source-{suffix}-{_id()}",
                kind=EvidenceKind.DOCUMENT,
                content_sha256=hashlib.sha256(payload).hexdigest(),
                observed_at=now,
                trust=trust,
            ),
        )
        item = await store.add_evidence(
            actor,
            AddEvidenceCommand(
                idempotency_key=f"live-evidence-{suffix}-{_id()}",
                source_id=source.source_id,
                kind=EvidenceKind.DOCUMENT,
                excerpt=f"live citation {suffix}",
            ),
        )
        return item.as_ref(), source.source_id

    trusted, _ = await evidence(SourceTrust.TRUSTED, "trusted")
    poisoned, poisoned_source_id = await evidence(SourceTrust.TRUSTED, "poisoned")
    published = await store.remember(
        actor,
        RememberCommand(
            idempotency_key=f"live-mixed-trust-{_id()}",
            kind="invariant",
            desired_state=MemoryState.CONFIRMED,
            visibility=Visibility.REPOSITORY,
            content="live mixed trust canonical sentinel",
            evidence=(trusted, poisoned),  # type: ignore[arg-type]
        ),
        ConservativeMemoryPolicy(),
    )
    assert published.memory is not None
    poisoned_only = await store.remember(
        actor,
        RememberCommand(
            idempotency_key=f"live-poisoned-only-{_id()}",
            kind="observation",
            desired_state=MemoryState.CONFIRMED,
            visibility=Visibility.REPOSITORY,
            content="live poisoned only trust sentinel",
            evidence=(poisoned,),  # type: ignore[arg-type]
        ),
        ConservativeMemoryPolicy(),
    )
    assert poisoned_only.memory is not None
    async with database.pool.connection() as connection:
        await connection.execute(
            "UPDATE sources SET trust_label = 'untrusted' WHERE id = %s",
            (poisoned_source_id,),
        )
    retrieval = RetrievalService(cockroach_retrieval_gateways(database), store)

    recalled = await retrieval.execute(
        actor,
        RecallQuery(
            text="live mixed trust canonical sentinel",
            include_evidence=True,
        ),
    )

    hit = next(
        item for item in recalled.bundle.hits if item.memory.memory_id == published.memory.memory_id
    )
    assert hit.memory.evidence == (trusted,)
    assert hit.evidence == (trusted,)

    rejected = await retrieval.execute(
        actor,
        RecallQuery(text="live poisoned only trust sentinel", include_evidence=True),
    )
    assert poisoned_only.memory.memory_id not in {
        item.memory.memory_id for item in rejected.bundle.hits
    }

    audited = await retrieval.execute(
        actor,
        RecallQuery(
            text="live poisoned only trust sentinel",
            include_refuted=True,
            include_evidence=True,
        ),
    )
    audited_hit = next(
        item
        for item in audited.bundle.hits
        if item.memory.memory_id == poisoned_only.memory.memory_id
    )
    assert audited_hit.memory.evidence == ()
    assert audited_hit.evidence == ()

    next_run = actor.model_copy(update={"swarm_id": _id(), "run_id": _id(), "agent_id": _id()})
    await _insert_run(database, next_run)
    cross_run = await retrieval.execute(
        next_run,
        RecallQuery(text="live mixed trust canonical sentinel", include_evidence=True),
    )
    cross_run_hit = next(
        item
        for item in cross_run.bundle.hits
        if item.memory.memory_id == published.memory.memory_id
    )
    assert cross_run_hit.memory.evidence == (trusted,)


async def test_live_graph_lane_expands_related_memories_for_two_bounded_hops(
    database: CockroachDatabase,
) -> None:
    actor = _actor()
    await _insert_run(database, actor)
    store = CockroachMemoryStore(database)
    leaf = await store.remember(
        actor,
        RememberCommand(
            idempotency_key=f"graph-leaf-{_id()}",
            kind="outcome",
            desired_state=MemoryState.CONFIRMED,
            visibility=Visibility.REPOSITORY,
            content="owner escalation destination",
        ),
        ConservativeMemoryPolicy(),
    )
    assert leaf.memory is not None
    middle = await store.remember(
        actor,
        RememberCommand(
            idempotency_key=f"graph-middle-{_id()}",
            kind="procedure",
            desired_state=MemoryState.CONFIRMED,
            visibility=Visibility.REPOSITORY,
            content="rollback checklist dependency",
            related_memory_ids=(leaf.memory.memory_id,),
        ),
        ConservativeMemoryPolicy(),
    )
    assert middle.memory is not None
    root = await store.remember(
        actor,
        RememberCommand(
            idempotency_key=f"graph-root-{_id()}",
            kind="warning",
            desired_state=MemoryState.CONFIRMED,
            visibility=Visibility.REPOSITORY,
            content="live graph root qzv sentinel",
            related_memory_ids=(middle.memory.memory_id,),
        ),
        ConservativeMemoryPolicy(),
    )
    assert root.memory is not None

    query = RecallQuery(text="live graph root qzv sentinel", limit=10)
    plan = RetrievalPlanner().plan(
        actor,
        query,
        purpose=RetrievalPurpose.PLANNING,
        available_signals=(RetrievalSignal.GRAPH,),
    )
    records: list[tuple[str, tuple[Any, ...] | None]] = []
    recording_database = CockroachDatabase(
        database.database_url,
        pool=_RecordingPool(database.pool, records),
    )
    direct_graph = await CockroachGraphRetrievalGateway(recording_database).expand(
        actor,
        plan,
        query,
        (
            FusedCandidate(
                canonical_id=root.memory.memory_id,
                raw_rrf=0.1,
                normalized_score=1.0,
                contributions=(),
            ),
        ),
    )
    assert direct_graph.degraded is False
    edge_statement = _runtime_statement(records, "WITH frontier (node_id)")
    edge_plan = await _explain_runtime_statement(database, edge_statement)
    assert "memory_links_from_type" in edge_plan
    assert "memory_links_to_type" in edge_plan
    assert "FULL SCAN" not in edge_plan.upper()

    execution = await RetrievalService(cockroach_retrieval_gateways(database), store).execute(
        actor,
        query,
        purpose=RetrievalPurpose.PLANNING,
    )

    graph_batch = next(
        batch for batch in execution.trace.batches if batch.lane is RetrievalSignal.GRAPH
    )
    assert graph_batch.degraded is False, graph_batch.degradation_reason
    assert [candidate.canonical_id for candidate in graph_batch.candidates] == [
        middle.memory.memory_id,
        leaf.memory.memory_id,
    ]
    assert graph_batch.candidates[1].path[0] == f"memory:{root.memory.memory_id}"
    assert graph_batch.candidates[1].path[-1] == f"memory:{leaf.memory.memory_id}"
    assert len(graph_batch.candidates[1].path) == 5
    assert {middle.memory.memory_id, leaf.memory.memory_id}.issubset(
        {hit.memory.memory_id for hit in execution.bundle.hits}
    )


async def test_live_referenced_validity_routes_by_half_open_interval(
    database: CockroachDatabase,
) -> None:
    actor = _actor()
    await _insert_run(database, actor)
    store = CockroachMemoryStore(database)
    sentinel = f"live temporal route {_id()}"
    referenced_from = datetime(2024, 1, 1, tzinfo=UTC)
    referenced_to = datetime(2024, 1, 5, tzinfo=UTC)
    overlapping = await store.remember(
        actor,
        RememberCommand(
            idempotency_key=f"temporal-overlap-{_id()}",
            kind="procedure",
            desired_state=MemoryState.CONFIRMED,
            visibility=Visibility.REPOSITORY,
            content=f"{sentinel} overlapping",
            valid_from=datetime(2024, 1, 2, tzinfo=UTC),
            valid_to=datetime(2024, 1, 4, tzinfo=UTC),
        ),
        ConservativeMemoryPolicy(),
    )
    touching_after = await store.remember(
        actor,
        RememberCommand(
            idempotency_key=f"temporal-touching-{_id()}",
            kind="procedure",
            desired_state=MemoryState.CONFIRMED,
            visibility=Visibility.REPOSITORY,
            content=f"{sentinel} touching",
            valid_from=referenced_to,
            valid_to=datetime(2024, 1, 8, tzinfo=UTC),
        ),
        ConservativeMemoryPolicy(),
    )
    assert overlapping.memory is not None
    assert touching_after.memory is not None
    retrieval = RetrievalService(cockroach_retrieval_gateways(database), store)

    current = await retrieval.execute(actor, RecallQuery(text=sentinel))
    interval = await retrieval.execute(
        actor,
        RecallQuery(
            text=sentinel,
            referenced_valid_from=referenced_from,
            referenced_valid_to=referenced_to,
        ),
    )
    point = await retrieval.execute(
        actor,
        RecallQuery(text=sentinel, world_at=referenced_to),
    )

    assert overlapping.memory.memory_id not in {hit.memory.memory_id for hit in current.bundle.hits}
    assert {hit.memory.memory_id for hit in interval.bundle.hits} == {overlapping.memory.memory_id}
    assert {hit.memory.memory_id for hit in point.bundle.hits} == {touching_after.memory.memory_id}


async def test_live_v7_install_rebuilds_legacy_projection_with_application_contract(
    database: CockroachDatabase,
) -> None:
    assert DATABASE_URL is not None
    actor = _actor()
    await _insert_run(database, actor)
    store = CockroachMemoryStore(database)
    published = await store.remember(
        actor,
        RememberCommand(
            idempotency_key=f"legacy-rebuild-{_id()}",
            kind="procedure",
            desired_state=MemoryState.CONFIRMED,
            visibility=Visibility.REPOSITORY,
            title="Retry   Coordinator",
            content="legacy projection payload",
            tags=("critical",),
            metadata={
                "aliases": ["legacy-retry"],
                "paths": ["src/legacy.py"],
                "commands": ["pytest tests/test_legacy.py -q"],
            },
        ),
        ConservativeMemoryPolicy(),
    )
    assert published.memory is not None
    memory_id = published.memory.memory_id

    async with database.pool.connection() as connection:
        await connection.execute(
            "DELETE FROM retrieval_exact_terms WHERE resource_id = %s",
            (memory_id,),
        )
        await connection.execute(
            "DELETE FROM retrieval_documents WHERE resource_id = %s",
            (memory_id,),
        )
        await connection.execute(
            """
            INSERT INTO retrieval_documents (
                tenant_id, project_id, repository_id, projection_id, scope_key,
                resource_type, resource_id, resource_version, canonical_id,
                domain_lane, search_text, lookup_text, content_sha256, indexed_at
            )
            SELECT tenant_id, project_id, repository_id, 'memory-simple-v1',
                   'repository:stale-scope', 'memory', id, version, id,
                   'knowledge', 'stale', 'stale', normalized_sha256, recorded_from
            FROM memories
            WHERE id = %s
            """,
            (memory_id,),
        )
        await connection.execute(
            """
            INSERT INTO retrieval_exact_terms (
                tenant_id, project_id, repository_id, projection_id, scope_key,
                normalized_term, term_kind, resource_type, resource_id,
                resource_version, indexed_at
            )
            SELECT tenant_id, project_id, repository_id, 'memory-simple-v1',
                   'repository:stale-scope', 'stale', 'alias', 'memory', id,
                   version, recorded_from
            FROM memories
            WHERE id = %s
            """,
            (memory_id,),
        )

    await install_schema(DATABASE_URL)
    async with database.pool.connection() as connection:
        stale_cursor = await connection.execute(
            """
            SELECT
                (SELECT count(*)
                 FROM retrieval_documents
                 WHERE resource_id = %s
                   AND scope_key = 'repository:stale-scope') AS stale_documents,
                (SELECT count(*)
                 FROM retrieval_exact_terms
                 WHERE resource_id = %s
                   AND scope_key = 'repository:stale-scope') AS stale_terms
            """,
            (memory_id, memory_id),
        )
        stale_row = await stale_cursor.fetchone()
    assert stale_row is not None
    assert int(stale_row["stale_documents"]) == 0
    assert int(stale_row["stale_terms"]) == 0
    retrieval = RetrievalService(cockroach_retrieval_gateways(database), store)
    for term in (
        "retry coordinator",
        "critical",
        "src/legacy.py",
        "legacy-retry",
        "pytest tests/test_legacy.py -q",
    ):
        execution = await retrieval.execute(actor, RecallQuery(text=term, min_score=0.6))
        assert any(hit.memory.memory_id == memory_id for hit in execution.bundle.hits), term
        exact_batch = next(
            batch for batch in execution.trace.batches if batch.lane.value == "exact"
        )
        assert any(candidate.canonical_id == memory_id for candidate in exact_batch.candidates)
