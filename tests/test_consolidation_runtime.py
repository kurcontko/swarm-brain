from __future__ import annotations

import hashlib
import os
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from conftest import make_actor
from swarmbrain.adapters.cockroach.database import CockroachDatabase
from swarmbrain.adapters.cockroach.memory import CockroachMemoryStore
from swarmbrain.adapters.cockroach.work_store import CockroachWorkStore
from swarmbrain.adapters.consolidation import (
    OpenAICompatibleConsolidationProvider,
    SafeDeterministicConsolidator,
)
from swarmbrain.application.consolidation import ConsolidationObserver, ConsolidationService
from swarmbrain.application.errors import InvalidState
from swarmbrain.application.memory_policy import ConservativeMemoryPolicy
from swarmbrain.application.memory_service import MemoryService
from swarmbrain.application.runtime import build_in_memory_runtime
from swarmbrain.application.work import DurableWorkService
from swarmbrain.config import ApiSettings, BackendKind, ExtractionProviderKind
from swarmbrain.domain.consolidation import (
    ConsolidationActionKind,
    ConsolidationProposal,
    ConsolidationRoute,
    ConsolidationWorkPayload,
)
from swarmbrain.domain.evidence import (
    AddEvidenceCommand,
    EvidenceKind,
    RegisterEvidenceSourceCommand,
)
from swarmbrain.domain.extraction import ProviderDescriptor
from swarmbrain.domain.memory import (
    MemoryKind,
    MemoryLinkKind,
    MemoryState,
    RememberCommand,
    Visibility,
)
from swarmbrain.domain.work import (
    ClaimWorkCommand,
    StageConsolidationPlanCommand,
    WorkKind,
)
from swarmbrain.ports.work_queue import WorkQueueStore
from swarmbrain.workers.consolidation import ConsolidationWorker

SECRET = "0123456789abcdef-local-test-secret"


class MutableClock:
    def __init__(self) -> None:
        self.now = datetime(2026, 8, 9, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: int) -> None:
        self.now += timedelta(seconds=seconds)


class AppendReflector:
    def __init__(self) -> None:
        self.calls = 0

    @property
    def descriptor(self) -> ProviderDescriptor:
        return ProviderDescriptor(
            provider="test",
            model="reflector-v1",
            prompt_id="test-reflector",
            prompt_sha256="a" * 64,
        )

    async def reflect(self, request):
        self.calls += 1
        assert [item.key for item in request.observations] == ["m0", "m1"]
        return (
            ConsolidationProposal(
                action=ConsolidationActionKind.APPEND,
                kind=MemoryKind.PROCEDURE,
                content="When the pool times out, retry with a bounded backoff.",
                title="Bounded pool recovery",
                tags=("database", "recovery"),
                confidence=0.95,
                support_keys=("m0", "m1"),
                reason="both observations support the combined procedure",
            ),
        )


class FailingReflector(AppendReflector):
    async def reflect(self, request):
        del request
        self.calls += 1
        raise RuntimeError("secret provider body must never escape")


async def _evidence(runtime, actor, suffix: str, excerpt: str):
    assert runtime.kernel is not None
    source = await runtime.kernel.register_source(
        actor,
        RegisterEvidenceSourceCommand(
            idempotency_key=f"source-{suffix}",
            kind=EvidenceKind.TEST_RESULT,
            content_sha256=suffix * 64,
            occurrence_key=f"test:{suffix}",
            observed_at=datetime(2026, 8, 9, 11, 0, tzinfo=UTC),
        ),
    )
    evidence = await runtime.kernel.add_evidence(
        actor,
        AddEvidenceCommand(
            idempotency_key=f"evidence-{suffix}",
            source_id=source.source_id,
            kind=EvidenceKind.TEST_RESULT,
            excerpt=excerpt,
            content_sha256=source.content_sha256,
        ),
    )
    return evidence.as_ref()


async def _publish_pair(runtime, actor):
    first_evidence = await _evidence(runtime, actor, "a", "pool timeout is 20 seconds")
    second_evidence = await _evidence(runtime, actor, "b", "bounded retry recovered the pool")
    first = await runtime.memory.publish(
        actor,
        RememberCommand(
            idempotency_key="memory-a",
            kind=MemoryKind.OBSERVATION,
            content="The database pool timeout is 20 seconds.",
            desired_state=MemoryState.CONFIRMED,
            visibility=Visibility.RUN,
            confidence=0.9,
            evidence=(first_evidence,),
        ),
    )
    second = await runtime.memory.publish(
        actor,
        RememberCommand(
            idempotency_key="memory-b",
            kind=MemoryKind.OUTCOME,
            content="A bounded retry recovered the database pool timeout.",
            desired_state=MemoryState.CONFIRMED,
            visibility=Visibility.RUN,
            confidence=0.8,
            evidence=(second_evidence,),
        ),
    )
    assert first.memory is not None and second.memory is not None
    return first.memory, second.memory, first_evidence, second_evidence


@pytest.mark.asyncio
async def test_consolidation_stages_plan_and_replays_exact_governed_effects(
    scope_ids: dict[str, str],
) -> None:
    clock = MutableClock()
    provider = AppendReflector()
    runtime = build_in_memory_runtime(
        SECRET,
        clock=clock,
        consolidation_enabled=True,
        consolidation_provider=provider,
        consolidation_use_provider=True,
        consolidation_max_memories=2,
    )
    actor = make_actor(scope_ids)
    first, second, first_evidence, second_evidence = await _publish_pair(runtime, actor)
    with pytest.raises(InvalidState, match="every source memory"):
        await runtime.memory.publish(
            actor,
            RememberCommand(
                idempotency_key="unsupported-derived-memory",
                kind=MemoryKind.PROCEDURE,
                content="Unsupported synthesis",
                evidence=(first_evidence,),
                derived_from_memory_ids=(first.memory_id, second.memory_id),
            ),
        )
    assert runtime.work_queue is not None and runtime.consolidation is not None
    runtime.work_queue._clock = clock
    batch = await runtime.work_queue.claim_work(
        ClaimWorkCommand(
            worker_id="reflector-worker",
            kinds=frozenset({WorkKind.CONSOLIDATE_MEMORY}),
            lease_seconds=5,
        )
    )
    assert len(batch.leases) == 1
    lease = batch.leases[0]
    worker = runtime.consolidation_worker(retry_delay_seconds=1)

    payload = lease.item.payload
    request = ConsolidationWorkPayload.model_validate(payload)
    reflection = await runtime.consolidation.reflect(request)
    staged = await runtime.work_queue.stage_consolidation_plan(
        StageConsolidationPlanCommand(
            work_id=lease.item.work_id,
            worker_id=lease.worker_id,
            lease_token=lease.lease_token,
            lease_version=lease.lease_version,
            expected_work_version=lease.work_version,
            attempt=lease.attempt,
            reflection=reflection,
        )
    )
    applied = await runtime.consolidation.apply(
        worker._actor(lease),
        request,
        reflection,
        work_id=lease.item.work_id,
    )
    assert applied.status == "applied"
    assert provider.calls == 1

    # Simulate a worker crash after the governed publication but before queue
    # completion. The next lease reuses the staged plan and the exact same
    # action idempotency key; it must neither call the model nor append again.
    assert staged.item.result is not None
    clock.advance(6)
    second_batch = await runtime.work_queue.claim_work(
        ClaimWorkCommand(
            worker_id="reflector-worker-2",
            kinds=frozenset({WorkKind.CONSOLIDATE_MEMORY}),
            lease_seconds=5,
        )
    )
    assert len(second_batch.leases) == 1
    completed = await worker._process(second_batch.leases[0])
    assert completed is not None
    assert completed.item.outcome == "applied"
    assert provider.calls == 1

    derived = [
        memory
        for memory in runtime.kernel.memories.values()
        if memory.memory_id not in {first.memory_id, second.memory_id}
    ]
    assert len(derived) == 1
    memory = derived[0]
    assert memory.state is MemoryState.TENTATIVE
    assert memory.visibility is Visibility.RUN
    assert memory.evidence == tuple(
        sorted((first_evidence, second_evidence), key=lambda e: e.evidence_id)
    )
    assert memory.metadata["consolidation"]["provider"]["model"] == "reflector-v1"
    lineage = await runtime.memory.lineage(actor, memory.memory_id)
    derived_links = [link for link in lineage.links if link.kind is MemoryLinkKind.DERIVED_FROM]
    assert {(link.source_memory_id, link.target_memory_id) for link in derived_links} == {
        (memory.memory_id, first.memory_id),
        (memory.memory_id, second.memory_id),
    }


@pytest.mark.asyncio
async def test_unavailable_provider_falls_back_to_bounded_noop(
    scope_ids: dict[str, str],
) -> None:
    provider = FailingReflector()
    runtime = build_in_memory_runtime(
        SECRET,
        consolidation_enabled=True,
        consolidation_provider=provider,
        consolidation_use_provider=True,
        consolidation_max_memories=2,
    )
    actor = make_actor(scope_ids)
    first, second, _, _ = await _publish_pair(runtime, actor)

    (completed,) = await runtime.consolidation_worker(retry_delay_seconds=1).run_once(
        "fallback-worker"
    )

    assert completed.item.outcome == "noop"
    assert completed.item.result is not None
    assert completed.item.result["route"] == ConsolidationRoute.FALLBACK.value
    assert completed.item.result["fallback_reason"] == "provider_RuntimeError"
    assert set(runtime.kernel.memories) == {first.memory_id, second.memory_id}
    assert "secret provider body" not in str(completed.item.model_dump(mode="json"))


@pytest.mark.asyncio
async def test_stale_snapshot_is_a_noop_and_provider_payload_uses_only_opaque_keys(
    scope_ids: dict[str, str],
) -> None:
    provider = AppendReflector()
    runtime = build_in_memory_runtime(
        SECRET,
        consolidation_enabled=True,
        consolidation_provider=provider,
        consolidation_use_provider=True,
        consolidation_max_memories=2,
    )
    actor = make_actor(scope_ids)
    first, second, first_evidence, _ = await _publish_pair(runtime, actor)
    assert runtime.work_queue is not None
    queued = next(
        item
        for item in runtime.work_queue.items.values()
        if item.kind is WorkKind.CONSOLIDATE_MEMORY
    )
    request = ConsolidationWorkPayload.model_validate(queued.payload)
    adapter = OpenAICompatibleConsolidationProvider(
        base_url="http://reflector.local:8000",
        model_id="reflector-v1",
    )
    provider_payload = adapter._source_payload(request)
    assert first.memory_id not in provider_payload
    assert second.memory_id not in provider_payload
    assert first_evidence.source_id not in provider_payload
    assert "memory_version" not in provider_payload
    assert "source_id" not in provider_payload

    # A governed exact merge increments the input version after observation.
    # Disable only subsequent observation so the original queued snapshot is
    # the one exercised below.
    runtime.memory.write_observer = None
    merged = await runtime.memory.publish(
        actor,
        RememberCommand(
            idempotency_key="merge-after-observe",
            kind=first.kind,
            content=first.content,
            desired_state=MemoryState.CONFIRMED,
            visibility=first.visibility,
            confidence=first.confidence,
            evidence=first.evidence,
            tags=("new-corroboration",),
            supersedes_memory_id=first.memory_id,
        ),
    )
    assert merged.memory is not None and merged.memory.version == first.version + 1

    (completed,) = await runtime.consolidation_worker(retry_delay_seconds=1).run_once(
        "stale-worker"
    )
    assert completed.item.outcome == "stale_noop"
    assert set(runtime.kernel.memories) == {first.memory_id, second.memory_id}


def test_consolidation_configuration_is_bounded_and_fail_closed() -> None:
    with pytest.raises(ValueError, match="requires consolidation"):
        ApiSettings(
            backend=BackendKind.MEMORY,
            token_secret=SECRET,
            consolidation_use_provider=True,
        )
    with pytest.raises(ValueError, match="requires an extraction provider"):
        ApiSettings(
            backend=BackendKind.MEMORY,
            token_secret=SECRET,
            consolidation_enabled=True,
            consolidation_use_provider=True,
        )
    with pytest.raises(ValueError, match="max memories"):
        ApiSettings(
            backend=BackendKind.MEMORY,
            token_secret=SECRET,
            consolidation_max_memories=33,
        )
    configured = ApiSettings(
        backend=BackendKind.MEMORY,
        token_secret=SECRET,
        extraction_provider=ExtractionProviderKind.OPENAI,
        extraction_model="reflector-v1",
        extraction_base_url="http://reflector.local:8000",
        consolidation_enabled=True,
        consolidation_use_provider=True,
    )
    assert configured.consolidation_max_actions == 4


def test_cockroach_schema_allows_consolidation_work() -> None:
    from swarmbrain.adapters.cockroach.schema import read_schema

    schema = read_schema()
    assert "'consolidate_memory'" in schema
    assert "DROP CONSTRAINT IF EXISTS outbox_work_items_kind_check" in schema
    assert isinstance(CockroachWorkStore(object()), WorkQueueStore)  # type: ignore[arg-type]


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("SWARMBRAIN_TEST_DATABASE_URL"),
    reason="SWARMBRAIN_TEST_DATABASE_URL is not set",
)
async def test_cockroach_consolidation_matches_in_memory_governance(
    scope_ids: dict[str, str],
) -> None:
    database_url = os.environ["SWARMBRAIN_TEST_DATABASE_URL"]
    database = CockroachDatabase(database_url, min_size=1, max_size=4)
    await database.start()
    actor = make_actor(scope_ids)

    async def insert_run(connection: Any) -> None:
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

    async def delete_run(connection: Any) -> None:
        await connection.execute(
            "DELETE FROM runs WHERE tenant_id = %s AND id = %s",
            (actor.tenant_id, actor.run_id),
        )

    try:
        await database.run(insert_run)  # type: ignore[arg-type]
        memory_store = CockroachMemoryStore(database)
        queue = CockroachWorkStore(database)
        work = DurableWorkService(queue)
        provider = AppendReflector()
        observer = ConsolidationObserver(
            memory_store,
            memory_store,
            work,
            enabled=True,
            use_provider=True,
            max_memories=2,
        )
        memory = MemoryService(
            memory_store,
            ConservativeMemoryPolicy(),
            review_store=memory_store,
            canonical_reader=memory_store,
            write_observer=observer,
        )
        consolidation = ConsolidationService(
            memory,
            memory_store,
            SafeDeterministicConsolidator(),
            provider=provider,
        )

        evidence_refs = []
        for suffix, excerpt in (
            ("c", "pool timeout is 20 seconds"),
            ("d", "bounded retry recovered the pool"),
        ):
            source_bytes = excerpt.encode("utf-8")
            source = await memory_store.register_source(
                actor,
                RegisterEvidenceSourceCommand(
                    idempotency_key=f"live-source-{suffix}",
                    kind=EvidenceKind.TEST_RESULT,
                    content_sha256=hashlib.sha256(source_bytes).hexdigest(),
                    occurrence_key=f"live:{suffix}",
                    observed_at=datetime.now(UTC),
                ),
            )
            evidence = await memory_store.add_evidence(
                actor,
                AddEvidenceCommand(
                    idempotency_key=f"live-evidence-{suffix}",
                    source_id=source.source_id,
                    kind=EvidenceKind.TEST_RESULT,
                    excerpt=excerpt,
                    content_sha256=source.content_sha256,
                ),
            )
            evidence_refs.append(evidence.as_ref())

        first = await memory.publish(
            actor,
            RememberCommand(
                idempotency_key="live-memory-c",
                kind=MemoryKind.OBSERVATION,
                content="The database pool timeout is 20 seconds.",
                desired_state=MemoryState.CONFIRMED,
                visibility=Visibility.RUN,
                confidence=0.9,
                evidence=(evidence_refs[0],),
            ),
        )
        second = await memory.publish(
            actor,
            RememberCommand(
                idempotency_key="live-memory-d",
                kind=MemoryKind.OUTCOME,
                content="A bounded retry recovered the database pool timeout.",
                desired_state=MemoryState.CONFIRMED,
                visibility=Visibility.RUN,
                confidence=0.8,
                evidence=(evidence_refs[1],),
            ),
        )
        assert first.memory is not None and second.memory is not None
        with pytest.raises(InvalidState, match="every source memory"):
            await memory.publish(
                actor,
                RememberCommand(
                    idempotency_key="live-unsupported-derived",
                    kind=MemoryKind.PROCEDURE,
                    content="Unsupported synthesis",
                    evidence=(evidence_refs[0],),
                    derived_from_memory_ids=(
                        first.memory.memory_id,
                        second.memory.memory_id,
                    ),
                ),
            )

        (completed,) = await ConsolidationWorker(queue, consolidation).run_once("live-consolidator")
        assert completed.item.outcome == "applied"
        result = completed.item.result
        assert result is not None
        derived_id = result["apply"]["actions"][0]["memory_id"]
        lineage = await memory.lineage(actor, derived_id)
        derived = next(item for item in lineage.memories if item.memory_id == derived_id)
        assert derived.state is MemoryState.TENTATIVE
        assert {item.evidence_id: item for item in derived.evidence} == {
            item.evidence_id: item for item in evidence_refs
        }
        links = {link.target_memory_id for link in lineage.links if link.kind == "derived_from"}
        assert links == {first.memory.memory_id, second.memory.memory_id}
    finally:
        try:
            await database.run(delete_run)  # type: ignore[arg-type]
        finally:
            await database.close()
