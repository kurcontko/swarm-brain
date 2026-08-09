from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest

from conftest import make_task
from swarmbrain.adapters.cockroach import memory as memory_adapter
from swarmbrain.adapters.cockroach.coordination import CockroachCoordinationStore
from swarmbrain.adapters.cockroach.database import CockroachDatabase
from swarmbrain.adapters.cockroach.memory import CockroachMemoryStore
from swarmbrain.adapters.cockroach.retrieval import cockroach_retrieval_gateways
from swarmbrain.application.errors import InvalidState
from swarmbrain.application.memory_policy import ConservativeMemoryPolicy
from swarmbrain.application.memory_service import MemoryService
from swarmbrain.application.retrieval_service import RetrievalService
from swarmbrain.domain.activation import (
    ActivationDecision,
    ActivationReason,
    ActivationTrigger,
    MemoryActivationTelemetry,
    memory_activation_id,
)
from swarmbrain.domain.agents import ActorContext, Capability
from swarmbrain.domain.evidence import (
    AddEvidenceCommand,
    EvidenceKind,
    RegisterEvidenceSourceCommand,
    RejectSourceCommand,
)
from swarmbrain.domain.memory import (
    MemoryKind,
    MemoryState,
    RecallQuery,
    RememberCommand,
    Visibility,
)
from swarmbrain.domain.retrieval import RetrievalPurpose
from swarmbrain.domain.tasks import CheckpointCommand, ClaimTaskCommand, CompleteTaskCommand

DATABASE_URL = os.getenv("SWARMBRAIN_TEST_DATABASE_URL")
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(not DATABASE_URL, reason="SWARMBRAIN_TEST_DATABASE_URL is not set"),
]

SENTINEL = "cockroach reuse counter sentinel"


def _id() -> str:
    return str(uuid4())


def _actor(**overrides: str) -> ActorContext:
    scope = {
        "tenant_id": _id(),
        "project_id": _id(),
        "repository_id": _id(),
        "swarm_id": _id(),
        "run_id": _id(),
    }
    scope.update(overrides)
    return ActorContext(
        **scope,
        agent_id=_id(),
        harness="pytest",
        provider="local",
        model="none",
        capabilities=frozenset(
            {
                Capability.MEMORY_PUBLISH.value,
                Capability.MEMORY_RECALL.value,
                Capability.MEMORY_CONFIRM.value,
            }
        ),
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


def _service(database: CockroachDatabase) -> tuple[MemoryService, CockroachMemoryStore]:
    store = CockroachMemoryStore(database)
    service = MemoryService(
        store,
        ConservativeMemoryPolicy(),
        retrieval=RetrievalService(cockroach_retrieval_gateways(database), store),
        canonical_reader=store,
    )
    return service, store


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


async def _counter(database: CockroachDatabase, actor: ActorContext) -> dict[str, Any] | None:
    async with database.pool.connection() as connection:
        cursor = await connection.execute(
            """
            SELECT reuse_count, recall_count, project_id, repository_id, swarm_id
            FROM retrieval_reuse_counters
            WHERE tenant_id = %s AND run_id = %s
            """,
            (actor.tenant_id, actor.run_id),
        )
        return await cursor.fetchone()


async def _publish(service: MemoryService, actor: ActorContext, suffix: str) -> str:
    result = await service.publish(
        actor,
        RememberCommand(
            idempotency_key=f"reuse-{suffix}-{_id()}",
            kind=MemoryKind.INVARIANT,
            visibility=Visibility.REPOSITORY,
            content=f"{SENTINEL} {suffix}",
            desired_state=MemoryState.CONFIRMED,
        ),
    )
    assert result.memory is not None
    return result.memory.memory_id


async def test_recall_increments_durable_counter_and_run_metrics(
    database: CockroachDatabase,
) -> None:
    actor = _actor()
    await _insert_run(database, actor)
    service, _ = _service(database)
    metrics_store = CockroachCoordinationStore(database)
    await _publish(service, actor, "alpha")
    await _publish(service, actor, "beta")

    first = await service.recall(actor, RecallQuery(text=SENTINEL))
    assert len(first.hits) == 2

    row = await _counter(database, actor)
    assert row is not None
    assert int(row["reuse_count"]) == 2
    assert int(row["recall_count"]) == 1
    assert row["project_id"] == actor.project_id

    metrics = await metrics_store.get_run_metrics(actor, actor.run_id)
    assert metrics.memories_reused == 2

    second = await service.recall(actor, RecallQuery(text=SENTINEL))
    assert len(second.hits) == 2

    row = await _counter(database, actor)
    assert row is not None
    assert int(row["reuse_count"]) == 4
    assert int(row["recall_count"]) == 2
    metrics = await metrics_store.get_run_metrics(actor, actor.run_id)
    assert metrics.memories_reused == 4


async def test_abstained_recall_records_attempt_without_activation(
    database: CockroachDatabase,
) -> None:
    actor = _actor()
    await _insert_run(database, actor)
    service, _ = _service(database)
    await _publish(service, actor, "gamma")

    bundle = await service.recall(actor, RecallQuery(text="entirely unrelated telemetry query"))

    assert bundle.hits == ()
    row = await _counter(database, actor)
    assert row is not None
    assert int(row["reuse_count"]) == 0
    assert int(row["recall_count"]) == 1
    metrics = await CockroachCoordinationStore(database).get_run_metrics(actor, actor.run_id)
    assert metrics.memories_activated == 0
    assert metrics.memory_activation_attempts == 0


async def test_failed_counter_write_never_breaks_recall(
    database: CockroachDatabase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actor = _actor()
    await _insert_run(database, actor)
    service, _ = _service(database)
    await _publish(service, actor, "delta")

    monkeypatch.setattr(
        memory_adapter,
        "RECORD_RETRIEVAL_REUSE_SQL",
        "INSERT INTO retrieval_reuse_counters_missing VALUES (%s, %s, %s, %s, %s, %s)",
    )
    bundle = await service.recall(actor, RecallQuery(text=SENTINEL))

    assert len(bundle.hits) == 1
    assert await _counter(database, actor) is None

    monkeypatch.undo()
    recovered = await service.recall(actor, RecallQuery(text=SENTINEL))
    assert len(recovered.hits) == 1
    row = await _counter(database, actor)
    assert row is not None
    assert int(row["reuse_count"]) == 1


async def test_reuse_counters_are_isolated_per_run_and_tenant(
    database: CockroachDatabase,
) -> None:
    first = _actor()
    same_tenant = _actor(
        tenant_id=first.tenant_id,
        project_id=first.project_id,
        repository_id=first.repository_id,
        swarm_id=first.swarm_id,
    )
    other_tenant = _actor()
    for actor in (first, same_tenant, other_tenant):
        await _insert_run(database, actor)
    service, _ = _service(database)
    metrics_store = CockroachCoordinationStore(database)
    await _publish(service, first, "epsilon")
    await _publish(service, other_tenant, "epsilon")

    assert len((await service.recall(first, RecallQuery(text=SENTINEL))).hits) == 1
    assert len((await service.recall(same_tenant, RecallQuery(text=SENTINEL))).hits) == 1
    assert len((await service.recall(other_tenant, RecallQuery(text=SENTINEL))).hits) == 1

    for actor in (first, same_tenant, other_tenant):
        row = await _counter(database, actor)
        assert row is not None, actor.run_id
        assert int(row["reuse_count"]) == 1
        metrics = await metrics_store.get_run_metrics(actor, actor.run_id)
        assert metrics.memories_reused == 1

    # A run identity from another tenant must never resolve this run's counter.
    foreign = first.model_copy(update={"tenant_id": other_tenant.tenant_id})
    async with database.pool.connection() as connection:
        cursor = await connection.execute(
            """
            SELECT count(*) AS total
            FROM retrieval_reuse_counters
            WHERE tenant_id = %s AND run_id = %s
            """,
            (foreign.tenant_id, foreign.run_id),
        )
        row = await cursor.fetchone()
    assert row is not None
    assert int(row["total"]) == 0


async def test_metrics_distinguish_cross_agent_citations_from_activation(
    database: CockroachDatabase,
) -> None:
    publisher = _actor()
    consumer = publisher.model_copy(
        update={"agent_id": _id(), "provider": "other-provider", "model": "other-model"}
    )
    coordination = CockroachCoordinationStore(database)
    await coordination.join_agent(publisher)
    await coordination.join_agent(consumer)
    task = await coordination.add_task(
        make_task(
            {
                "tenant_id": publisher.tenant_id,
                "project_id": publisher.project_id,
                "repository_id": publisher.repository_id,
                "swarm_id": publisher.swarm_id,
                "run_id": publisher.run_id,
            },
            title="Cite cross-agent evidence",
        )
    )
    memory_service, _ = _service(database)
    memory_id = await _publish(memory_service, publisher, "cross-agent-citation")
    claim = await coordination.claim_task(
        consumer,
        ClaimTaskCommand(idempotency_key="cross-agent-citation-claim", task_id=task.task_id),
    )
    await coordination.record_memory_activation(
        consumer,
        MemoryActivationTelemetry(
            activation_id=memory_activation_id(
                task.task_id,
                claim.lease.lease_id,
                ActivationTrigger.TASK_CLAIM,
            ),
            run_id=consumer.run_id,
            agent_id=consumer.agent_id,
            task_id=task.task_id,
            lease_id=claim.lease.lease_id,
            trigger=ActivationTrigger.TASK_CLAIM,
            decision=ActivationDecision.RECALL,
            purpose=RetrievalPurpose.TASK_BOOTSTRAP,
            reason=ActivationReason.CONTEXT_ACTIVATED,
            memory_ids=(memory_id,),
            memory_versions={memory_id: 1},
            token_budget=2_048,
            estimated_tokens=32,
            min_score=0.4,
            candidate_count=1,
        ),
    )
    checkpoint = await coordination.checkpoint_task(
        consumer,
        CheckpointCommand(
            idempotency_key="cross-agent-citation-checkpoint",
            task_id=task.task_id,
            lease_id=claim.lease.lease_id,
            expected_task_version=claim.task.version,
            expected_lease_version=claim.lease.version,
            summary="Used the published invariant",
            memory_ids=(memory_id,),
        ),
    )
    await coordination.complete_task(
        consumer,
        CompleteTaskCommand(
            idempotency_key="cross-agent-citation-complete",
            task_id=task.task_id,
            lease_id=checkpoint.lease.lease_id,
            expected_task_version=checkpoint.task.version,
            expected_lease_version=checkpoint.lease.version,
            summary="Verified the invariant",
            memory_ids=(memory_id,),
        ),
    )

    metrics = await coordination.get_run_metrics(consumer, consumer.run_id)
    assert metrics.memories_activated == 1
    assert metrics.memory_activation_attempts == 1
    assert metrics.memories_cited == 1
    assert metrics.cross_agent_memory_uses == 1


async def test_activation_event_revalidates_selected_memory(
    database: CockroachDatabase,
) -> None:
    actor = _actor()
    coordination = CockroachCoordinationStore(database)
    await coordination.join_agent(actor)
    task = await coordination.add_task(
        make_task(
            {
                "tenant_id": actor.tenant_id,
                "project_id": actor.project_id,
                "repository_id": actor.repository_id,
                "swarm_id": actor.swarm_id,
                "run_id": actor.run_id,
            },
            title="Revalidate activated memory",
        )
    )
    memory_service, _ = _service(database)
    memory_id = await _publish(memory_service, actor, "activation-race")
    claim = await coordination.claim_task(
        actor,
        ClaimTaskCommand(idempotency_key="activation-race-claim", task_id=task.task_id),
    )

    async def refute(connection: Any) -> None:
        await connection.execute(
            "UPDATE memories SET state = 'refuted', version = version + 1 WHERE id = %s::UUID",
            (memory_id,),
        )

    await database.run(refute)  # type: ignore[arg-type]
    telemetry = MemoryActivationTelemetry(
        activation_id=memory_activation_id(
            task.task_id,
            claim.lease.lease_id,
            ActivationTrigger.TASK_CLAIM,
        ),
        run_id=actor.run_id,
        agent_id=actor.agent_id,
        task_id=task.task_id,
        lease_id=claim.lease.lease_id,
        trigger=ActivationTrigger.TASK_CLAIM,
        decision=ActivationDecision.RECALL,
        purpose=RetrievalPurpose.TASK_BOOTSTRAP,
        reason=ActivationReason.CONTEXT_ACTIVATED,
        memory_ids=(memory_id,),
        memory_versions={memory_id: 1},
        token_budget=2_048,
        estimated_tokens=16,
        min_score=0.4,
        candidate_count=1,
    )

    with pytest.raises(InvalidState, match="no longer current and recallable"):
        await coordination.record_memory_activation(actor, telemetry)
    assert await coordination.get_memory_activation(actor, telemetry.activation_id) is None


async def test_activation_version_proof_rejects_partial_source_revocation(
    database: CockroachDatabase,
) -> None:
    actor = _actor()
    coordination = CockroachCoordinationStore(database)
    await coordination.join_agent(actor)
    task = await coordination.add_task(
        make_task(
            {
                "tenant_id": actor.tenant_id,
                "project_id": actor.project_id,
                "repository_id": actor.repository_id,
                "swarm_id": actor.swarm_id,
                "run_id": actor.run_id,
            },
            title="Fence partially revoked evidence",
        )
    )
    memory_service, memory_store = _service(database)
    sources = []
    evidence = []
    for suffix in ("a", "b"):
        source = await memory_store.register_source(
            actor,
            RegisterEvidenceSourceCommand(
                idempotency_key=f"activation-version-source-{suffix}",
                kind=EvidenceKind.DOCUMENT,
                content_sha256=suffix * 64,
                occurrence_key=f"activation-version-source-{suffix}",
                observed_at=datetime(2026, 8, 9, 12, 0, tzinfo=UTC),
            ),
        )
        item = await memory_store.add_evidence(
            actor,
            AddEvidenceCommand(
                idempotency_key=f"activation-version-evidence-{suffix}",
                source_id=source.source_id,
                kind=EvidenceKind.DOCUMENT,
                excerpt=f"accepted source {suffix}",
                content_sha256=source.content_sha256,
            ),
        )
        sources.append(source)
        evidence.append(item)
    published = await memory_service.publish(
        actor,
        RememberCommand(
            idempotency_key="activation-version-memory",
            kind=MemoryKind.INVARIANT,
            content="The selected block originally rendered both citations",
            desired_state=MemoryState.CONFIRMED,
            visibility=Visibility.REPOSITORY,
            evidence=tuple(item.as_ref() for item in evidence),
        ),
    )
    assert published.memory is not None
    claim = await coordination.claim_task(
        actor,
        ClaimTaskCommand(
            idempotency_key="activation-version-claim",
            task_id=task.task_id,
        ),
    )
    telemetry = MemoryActivationTelemetry(
        activation_id=memory_activation_id(
            task.task_id,
            claim.lease.lease_id,
            ActivationTrigger.TASK_CLAIM,
        ),
        run_id=actor.run_id,
        agent_id=actor.agent_id,
        task_id=task.task_id,
        lease_id=claim.lease.lease_id,
        trigger=ActivationTrigger.TASK_CLAIM,
        decision=ActivationDecision.RECALL,
        purpose=RetrievalPurpose.TASK_BOOTSTRAP,
        reason=ActivationReason.CONTEXT_ACTIVATED,
        memory_ids=(published.memory.memory_id,),
        memory_versions={published.memory.memory_id: published.memory.version},
        token_budget=2_048,
        estimated_tokens=32,
        min_score=0.4,
        candidate_count=1,
    )

    rejection = await memory_store.reject_source(
        actor,
        RejectSourceCommand(
            idempotency_key="activation-version-reject-a",
            source_id=sources[0].source_id,
            expected_version=sources[0].version,
            reason="source A was revoked",
        ),
    )
    assert rejection.rolled_back_memory_ids == ()
    current = await memory_store.get_lineage(actor, published.memory.memory_id)
    selected = next(
        memory for memory in current.memories if memory.memory_id == published.memory.memory_id
    )
    assert selected.state is MemoryState.CONFIRMED
    assert selected.version == published.memory.version + 1

    with pytest.raises(InvalidState, match="no longer current and recallable"):
        await coordination.record_memory_activation(actor, telemetry)
    assert await coordination.get_memory_activation(actor, telemetry.activation_id) is None
