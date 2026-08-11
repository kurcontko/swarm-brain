from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from pydantic import ValidationError

from conftest import make_actor
from swarmbrain.adapters.extraction import (
    STRUCTURED_MEMORY_SOURCE_KIND,
    CodingRuleExtractor,
    DefaultRuleExtractor,
    InMemoryWorkStore,
    LazyExtractionProvider,
)
from swarmbrain.application.errors import IdempotencyConflict
from swarmbrain.application.extraction import ExtractionService
from swarmbrain.domain.evidence import EvidenceKind, SourceTrust
from swarmbrain.domain.extraction import (
    CandidateRelation,
    ExtractionCandidate,
    ExtractionOutcome,
    ExtractionProvenance,
    ExtractionStage,
    IngestRawSourceCommand,
    ProviderDescriptor,
    SourceSpan,
)
from swarmbrain.domain.memory import MemoryKind, MemoryLinkKind
from swarmbrain.domain.work import (
    ApplyExtractionWorkCommand,
    ClaimWorkCommand,
    SourceExtractionState,
    WorkEffect,
    WorkEffectConflict,
    WorkKind,
    WorkLeaseLost,
    WorkStatus,
)
from swarmbrain.workers import ExtractionWorker


class MutableClock:
    def __init__(self) -> None:
        self.now = datetime(2026, 8, 2, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now


def _command(
    *,
    content: str = "def load_config():\n    pass\n# TODO: validate the config\n",
    idempotency_key: str = "raw-source-1",
    use_provider: bool = False,
    trust: SourceTrust = SourceTrust.UNKNOWN,
) -> IngestRawSourceCommand:
    return IngestRawSourceCommand(
        idempotency_key=idempotency_key,
        kind=EvidenceKind.SOURCE_CODE,
        content=content,
        observed_at=datetime(2026, 8, 2, 11, 0, tzinfo=UTC),
        uri="src/config.py",
        occurrence_key="src/config.py@abc123",
        trust=trust,
        use_provider=use_provider,
    )


def _payload_hash(candidate: ExtractionCandidate) -> str:
    payload = json.dumps(
        candidate.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _provenance(attempt: int, input_sha256: str) -> ExtractionProvenance:
    timestamp = datetime(2026, 8, 2, 12, 0, tzinfo=UTC)
    return ExtractionProvenance(
        stage=ExtractionStage.DETERMINISTIC,
        attempt=attempt,
        outcome=ExtractionOutcome.SUCCEEDED,
        input_sha256=input_sha256,
        output_sha256=hashlib.sha256(b"[]").hexdigest(),
        started_at=timestamp,
        finished_at=timestamp,
    )


@pytest.mark.asyncio
async def test_raw_ingest_preserves_exact_content_chunks_and_work_atomically(
    scope_ids: dict[str, str],
) -> None:
    store = InMemoryWorkStore()
    service = ExtractionService(store, CodingRuleExtractor())
    actor = make_actor(scope_ids)
    command = _command()

    first = await service.ingest(actor, command)
    replay = await service.ingest(actor, command)

    assert first.replayed is False
    assert replay.replayed is True
    assert replay.work_id == first.work_id
    assert first.chunk_count == len(store.chunks[first.source.source_id])
    assert (
        "".join(chunk.content for chunk in store.chunks[first.source.source_id]) == command.content
    )
    assert store.sources[first.source.source_id].raw_content == command.content
    assert store.items[first.work_id].subject_id == first.source.source_id

    changed = command.model_copy(update={"content": command.content + "# changed\n"})
    with pytest.raises(IdempotencyConflict):
        await service.ingest(actor, changed)


@pytest.mark.asyncio
async def test_deterministic_coding_route_produces_only_exact_source_spans(
    scope_ids: dict[str, str],
) -> None:
    store = InMemoryWorkStore()
    service = ExtractionService(store, CodingRuleExtractor())
    actor = make_actor(scope_ids)
    ingested = await service.ingest(actor, _command())
    lease = (await store.claim_work(ClaimWorkCommand(worker_id="worker-a", limit=1))).leases[0]
    request = await store.load_extraction_input(lease)

    result = await service.extract(request)

    assert result.route.value == "coding"
    assert {candidate.kind for candidate in result.candidates} == {
        MemoryKind.OBSERVATION,
        MemoryKind.WARNING,
    }
    assert ingested.source.source_id == request.source.source_id
    for candidate in result.candidates:
        for span in candidate.spans:
            assert request.raw_content[span.char_start : span.char_end] == span.excerpt


@pytest.mark.asyncio
async def test_structured_envelope_materializes_spanless_flexible_memory(
    scope_ids: dict[str, str],
) -> None:
    store = InMemoryWorkStore()
    service = ExtractionService(store, DefaultRuleExtractor())
    actor = make_actor(scope_ids)
    command = IngestRawSourceCommand(
        idempotency_key="structured-source-1",
        kind=STRUCTURED_MEMORY_SOURCE_KIND,
        content=json.dumps(
            {
                "memories": [
                    {
                        "kind": "org.acme/preference",
                        "content": {
                            "preference": "dark-mode",
                            "context": ["editor", "terminal"],
                        },
                        "confidence": 0.91,
                    }
                ]
            }
        ),
        observed_at=datetime(2026, 8, 2, 11, 0, tzinfo=UTC),
    )

    ingested = await service.ingest(actor, command)
    queued = await service.status(actor, ingested.source.source_id)
    assert queued.status is SourceExtractionState.QUEUED
    assert queued.memory_ids == ()

    applied = await ExtractionWorker(store, store, service).run_once("structured-worker")
    assert len(applied) == 1
    completed = await service.status(actor, ingested.source.source_id)
    assert completed.status is SourceExtractionState.COMPLETED
    assert completed.route.value == "general"
    assert completed.candidate_count == 1
    assert len(completed.memory_ids) == 1
    resource = store.effect_resources[completed.memory_ids[0]]
    assert resource["candidate"]["content"] == {
        "preference": "dark-mode",
        "context": ["editor", "terminal"],
    }
    assert resource["evidence"] == []


@pytest.mark.asyncio
async def test_structured_envelope_deduplicates_identical_candidates_before_apply(
    scope_ids: dict[str, str],
) -> None:
    store = InMemoryWorkStore()
    service = ExtractionService(store, DefaultRuleExtractor())
    actor = make_actor(scope_ids)
    candidate = {
        "kind": "org.acme/preference",
        "content": {"preference": "flexible-memory"},
        "confidence": 0.9,
    }
    ingested = await service.ingest(
        actor,
        IngestRawSourceCommand(
            idempotency_key="structured-source-duplicate-1",
            kind=STRUCTURED_MEMORY_SOURCE_KIND,
            content=json.dumps({"memories": [candidate, candidate]}),
            observed_at=datetime(2026, 8, 2, 11, 0, tzinfo=UTC),
        ),
    )

    applied = await ExtractionWorker(store, store, service).run_once("structured-dedupe-worker")

    assert len(applied) == 1
    assert applied[0].item.status is WorkStatus.COMPLETED
    assert len(applied[0].effects) == 1
    status = await service.status(actor, ingested.source.source_id)
    assert status.candidate_count == 1
    assert len(status.memory_ids) == 1


@pytest.mark.asyncio
async def test_extraction_atomically_enqueues_embedding_for_materialized_memory(
    scope_ids: dict[str, str],
) -> None:
    store = InMemoryWorkStore()
    service = ExtractionService(
        store,
        DefaultRuleExtractor(),
        embedding_model="model-a",
    )
    actor = make_actor(scope_ids)
    ingested = await service.ingest(
        actor,
        IngestRawSourceCommand(
            idempotency_key="structured-source-embedding-1",
            kind=STRUCTURED_MEMORY_SOURCE_KIND,
            content=json.dumps(
                {
                    "memories": [
                        {
                            "kind": "org.acme/flexible",
                            "content": {"shape": [1, {"free": True}]},
                        }
                    ]
                }
            ),
            observed_at=datetime(2026, 8, 2, 11, 0, tzinfo=UTC),
        ),
    )

    applied = await ExtractionWorker(store, store, service).run_once("structured-embedding-worker")

    assert len(applied) == 1
    embedding_items = [item for item in store.items.values() if item.kind is WorkKind.EMBED_MEMORY]
    assert len(embedding_items) == 1
    embedding = embedding_items[0]
    assert embedding.status is WorkStatus.PENDING
    assert embedding.subject_id == applied[0].effects[0].resource_id
    assert embedding.payload == {
        "content": '{"shape":[1,{"free":true}]}',
        "model": "model-a",
        "metadata": {},
    }
    assert store.items[ingested.work_id].status is WorkStatus.COMPLETED


@pytest.mark.parametrize(
    "protected",
    [
        {"memory_id": "11111111-1111-1111-1111-111111111111"},
        {"tenant_id": "11111111-1111-1111-1111-111111111111"},
        {"state": "confirmed"},
        {"visibility": "repository"},
        {"evidence": []},
        {"supersedes_memory_id": "11111111-1111-1111-1111-111111111111"},
    ],
)
def test_candidate_contract_rejects_protected_storage_fields(
    protected: dict[str, object],
) -> None:
    payload: dict[str, object] = {
        "kind": "observation",
        "content": "validated fact",
        "spans": [
            {
                "chunk_index": 0,
                "char_start": 0,
                "char_end": 3,
                "excerpt": "def",
            }
        ],
        **protected,
    }

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ExtractionCandidate.model_validate(payload)


def test_candidate_allows_structured_synthesis_without_claiming_an_exact_quote() -> None:
    candidate = ExtractionCandidate(
        kind="org.acme/synthesis",
        content={"summary": "configuration needs validation", "signals": ["todo"]},
    )

    assert candidate.spans == ()
    assert candidate.kind == "org.acme/synthesis"
    assert candidate.content["signals"] == ["todo"]


def test_candidate_supports_typed_time_aliases_relationships_and_namespaced_metadata() -> None:
    event_time = datetime(2026, 8, 2, 10, 30, tzinfo=UTC)
    candidate = ExtractionCandidate(
        candidate_key="procedure-1",
        kind=MemoryKind.PROCEDURE,
        content="Retry the deployment with the regional endpoint.",
        event_time=event_time,
        aliases=("DeployConfig", "deployconfig", "regional endpoint"),
        relations=(
            CandidateRelation(
                target_candidate_key="attempt-1",
                kind=MemoryLinkKind.DERIVED_FROM,
                reason="The failed attempt established the corrected procedure.",
            ),
        ),
        metadata={"extracted": {"path": "deploy/config.py"}},
    )

    assert candidate.event_time == event_time
    assert candidate.aliases == ("DeployConfig", "regional endpoint")
    assert candidate.relations[0].target_candidate_key == "attempt-1"
    assert candidate.metadata["extracted"] == {"path": "deploy/config.py"}


@pytest.mark.parametrize("reserved", ["extraction", "governance", "retrieval"])
def test_candidate_metadata_cannot_set_framework_owned_namespaces(reserved: str) -> None:
    with pytest.raises(ValidationError, match="reserved framework namespaces"):
        ExtractionCandidate(
            kind=MemoryKind.OBSERVATION,
            content="candidate",
            metadata={reserved: {"domain_lane": "playbook"}},
        )


class CandidateGraphProvider:
    descriptor = ProviderDescriptor(provider="test", model="candidate-graph")

    def __init__(self, candidates: list[dict[str, object]]) -> None:
        self.candidates = candidates

    async def extract(self, request: Any) -> list[dict[str, object]]:
        del request
        return self.candidates


def _graph_candidate(
    key: str,
    *,
    content: str,
    relations: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "candidate_key": key,
        "kind": "observation",
        "content": content,
        "relations": relations or [],
        "spans": [
            {
                "chunk_index": 0,
                "char_start": 0,
                "char_end": 3,
                "excerpt": "def",
            }
        ],
    }


async def _extract_provider_graph(
    scope_ids: dict[str, str],
    candidates: list[dict[str, object]],
    *,
    max_provider_candidates: int = 64,
) -> Any:
    store = InMemoryWorkStore()
    service = ExtractionService(
        store,
        CodingRuleExtractor(),
        provider=CandidateGraphProvider(candidates),
        max_provider_candidates=max_provider_candidates,
    )
    actor = make_actor(scope_ids)
    await service.ingest(actor, _command(use_provider=True))
    lease = (await store.claim_work(ClaimWorkCommand(worker_id="graph-worker"))).leases[0]
    request = await store.load_extraction_input(lease)
    return await service.extract(request, use_provider=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "candidates",
    [
        [
            _graph_candidate("same", content="first"),
            _graph_candidate("same", content="second"),
        ],
        [
            _graph_candidate(
                "source",
                content="dangling",
                relations=[
                    {
                        "target_candidate_key": "missing",
                        "kind": "related_to",
                        "reason": None,
                    }
                ],
            )
        ],
        [
            _graph_candidate(
                "source",
                content="self link",
                relations=[
                    {
                        "target_candidate_key": "source",
                        "kind": "related_to",
                        "reason": None,
                    }
                ],
            )
        ],
        [
            _graph_candidate(
                "new",
                content="unsafe lifecycle",
                relations=[
                    {
                        "target_candidate_key": "old",
                        "kind": "supersedes",
                        "reason": None,
                    }
                ],
            ),
            _graph_candidate("old", content="old value"),
        ],
    ],
)
async def test_provider_candidate_graph_rejects_duplicate_dangling_self_and_lifecycle_links(
    scope_ids: dict[str, str],
    candidates: list[dict[str, object]],
) -> None:
    result = await _extract_provider_graph(scope_ids, candidates)

    assert result.status.value == "fallback"
    assert result.fallback_reason is not None
    assert result.fallback_reason.startswith("validation_")
    assert result.candidates == result.deterministic_candidates
    assert result.provider_candidates == ()


@pytest.mark.asyncio
async def test_provider_candidate_graph_accepts_bounded_local_relationships(
    scope_ids: dict[str, str],
) -> None:
    candidates = [
        _graph_candidate("attempt", content="failed attempt"),
        _graph_candidate(
            "procedure",
            content="working procedure",
            relations=[
                {
                    "target_candidate_key": "attempt",
                    "kind": "derived_from",
                    "reason": "The failure led to the procedure.",
                }
            ],
        ),
    ]

    result = await _extract_provider_graph(scope_ids, candidates)

    assert result.status.value == "completed"
    assert {candidate.candidate_key for candidate in result.provider_candidates} == {
        "attempt",
        "procedure",
    }


@pytest.mark.asyncio
async def test_provider_candidate_semantics_survive_in_memory_materialization(
    scope_ids: dict[str, str],
) -> None:
    event_time = datetime(2026, 8, 1, 9, 30, tzinfo=UTC)
    provider = CandidateGraphProvider(
        [
            _graph_candidate("attempt", content="failed attempt"),
            {
                **_graph_candidate(
                    "procedure",
                    content="working procedure",
                    relations=[
                        {
                            "target_candidate_key": "attempt",
                            "kind": "derived_from",
                            "reason": "The failed attempt revealed the procedure.",
                        }
                    ],
                ),
                "event_time": event_time.isoformat(),
                "aliases": ["deploy_guard", "Deploy_Guard"],
                "metadata": {"extracted": {"path": "deploy/guard.py"}},
            },
        ]
    )
    store = InMemoryWorkStore()
    service = ExtractionService(store, CodingRuleExtractor(), provider=provider)
    actor = make_actor(scope_ids)
    await service.ingest(
        actor,
        _command(idempotency_key="provider-materialization", use_provider=True),
    )

    applied = await ExtractionWorker(store, store, service).run_once(
        "provider-materialization-worker"
    )

    assert len(applied) == 1, [item.last_error for item in store.items.values()]
    resources = {
        resource["candidate"]["candidate_key"]: (memory_id, resource)
        for memory_id, resource in store.effect_resources.items()
        if resource["candidate"]["candidate_key"] is not None
    }
    attempt_id, _attempt = resources["attempt"]
    procedure_id, procedure = resources["procedure"]
    assert procedure["occurred_at"] == event_time.isoformat()
    assert procedure["valid_from"] == datetime(2026, 8, 2, 11, 0, tzinfo=UTC).isoformat()
    assert procedure["metadata"]["aliases"] == ["deploy_guard"]
    assert procedure["metadata"]["extracted"] == {"path": "deploy/guard.py"}
    extraction_metadata = procedure["metadata"]["extraction"]
    assert extraction_metadata["candidate_origin"] == ExtractionStage.PROVIDER.value
    assert any(
        item["stage"] == ExtractionStage.PROVIDER.value and item["provider"] == "test"
        for item in extraction_metadata["provenance"]
    )
    assert len(store.effect_links) == 1
    link = next(iter(store.effect_links.values()))
    assert link["source_memory_id"] == procedure_id
    assert link["target_memory_id"] == attempt_id
    assert link["kind"] == MemoryLinkKind.DERIVED_FROM.value


@pytest.mark.asyncio
async def test_provider_candidate_count_is_bounded_before_validation(
    scope_ids: dict[str, str],
) -> None:
    result = await _extract_provider_graph(
        scope_ids,
        [
            _graph_candidate("first", content="first"),
            _graph_candidate("second", content="second"),
        ],
        max_provider_candidates=1,
    )

    assert result.status.value == "fallback"
    assert result.fallback_reason == "provider_ProviderOutputLimitExceeded"
    assert result.candidates == result.deterministic_candidates


class InvalidProvider:
    descriptor = ProviderDescriptor(provider="test", model="malicious")

    def __init__(self) -> None:
        self.calls = 0

    async def extract(self, request: Any) -> list[dict[str, object]]:
        self.calls += 1
        return [
            {
                "kind": "observation",
                "content": "provider tried to confirm this",
                "state": "confirmed",
                "spans": [
                    {
                        "chunk_index": 0,
                        "char_start": 0,
                        "char_end": 3,
                        "excerpt": request.raw_content[:3],
                    }
                ],
            }
        ]


@pytest.mark.asyncio
async def test_invalid_provider_falls_back_and_raw_source_survives(
    scope_ids: dict[str, str],
) -> None:
    store = InMemoryWorkStore()
    provider = InvalidProvider()
    service = ExtractionService(store, CodingRuleExtractor(), provider=provider)
    actor = make_actor(scope_ids)
    ingested = await service.ingest(actor, _command(use_provider=True))
    worker = ExtractionWorker(store, store, service)

    applied = await worker.run_once("worker-provider")

    assert provider.calls == 1
    assert store.sources[ingested.source.source_id].raw_content == _command().content
    assert len(applied) == 1
    assert applied[0].item.outcome == "fallback"
    assert applied[0].effects
    provider_attempt = store.attempts[(ingested.work_id, 1, "provider")]
    validation_attempt = store.attempts[(ingested.work_id, 1, "validation")]
    assert provider_attempt.output_sha256 is not None
    assert provider_attempt.outcome is ExtractionOutcome.SUCCEEDED
    assert validation_attempt.input_sha256 == provider_attempt.output_sha256
    assert validation_attempt.outcome is ExtractionOutcome.REJECTED
    for effect in applied[0].effects:
        persisted = store.effect_resources[effect.resource_id]
        assert persisted["state"] == "tentative"
        assert persisted["evidence"]
        assert persisted["supersedes_memory_id"] is None


@pytest.mark.asyncio
async def test_lazy_unavailable_provider_is_not_loaded_until_selected_and_falls_back(
    scope_ids: dict[str, str],
) -> None:
    factory_calls = 0

    def unavailable() -> Any:
        nonlocal factory_calls
        factory_calls += 1
        raise RuntimeError("credential is absent")

    lazy = LazyExtractionProvider(
        ProviderDescriptor(provider="optional", model="remote"),
        unavailable,
    )
    store = InMemoryWorkStore()
    service = ExtractionService(store, CodingRuleExtractor(), provider=lazy)
    actor = make_actor(scope_ids)
    await service.ingest(actor, _command(use_provider=False))

    await ExtractionWorker(store, store, service).run_once("worker-local")
    assert factory_calls == 0

    await service.ingest(
        actor,
        _command(
            idempotency_key="raw-source-2",
            content="class Cache:\n    pass\n",
            use_provider=True,
        ).model_copy(update={"occurrence_key": "src/cache.py@abc123"}),
    )
    applied = await ExtractionWorker(store, store, service).run_once("worker-remote")

    assert factory_calls == 1
    assert applied[0].item.outcome == "fallback"


@pytest.mark.asyncio
async def test_two_workers_cannot_claim_the_same_live_lease(
    scope_ids: dict[str, str],
) -> None:
    store = InMemoryWorkStore()
    service = ExtractionService(store, CodingRuleExtractor())
    await service.ingest(make_actor(scope_ids), _command())

    first, second = await asyncio.gather(
        store.claim_work(ClaimWorkCommand(worker_id="worker-a")),
        store.claim_work(ClaimWorkCommand(worker_id="worker-b")),
    )

    leases = (*first.leases, *second.leases)
    assert len(leases) == 1
    assert leases[0].item.status is WorkStatus.LEASED


@pytest.mark.asyncio
async def test_expired_worker_fence_cannot_apply_after_reclaim(
    scope_ids: dict[str, str],
) -> None:
    clock = MutableClock()
    store = InMemoryWorkStore(clock=clock)
    service = ExtractionService(store, CodingRuleExtractor())
    actor = make_actor(scope_ids)
    ingested = await service.ingest(actor, _command())
    stale = (
        await store.claim_work(ClaimWorkCommand(worker_id="worker-stale", lease_seconds=5))
    ).leases[0]
    clock.now += timedelta(seconds=6)
    current = (
        await store.claim_work(ClaimWorkCommand(worker_id="worker-current", lease_seconds=5))
    ).leases[0]

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
    assert current.lease_version == stale.lease_version + 1


@pytest.mark.asyncio
async def test_repeated_delivery_replays_one_durable_effect(
    scope_ids: dict[str, str],
) -> None:
    store = InMemoryWorkStore()
    service = ExtractionService(store, CodingRuleExtractor())
    actor = make_actor(scope_ids)
    await service.ingest(actor, _command())
    lease = (await store.claim_work(ClaimWorkCommand(worker_id="worker-once"))).leases[0]
    request = await store.load_extraction_input(lease)
    extraction = await service.extract(request)
    candidate = extraction.candidates[0]
    digest = _payload_hash(candidate)
    command = ApplyExtractionWorkCommand(
        work_id=lease.item.work_id,
        worker_id=lease.worker_id,
        lease_token=lease.lease_token,
        lease_version=lease.lease_version,
        expected_work_version=lease.work_version,
        attempt=lease.attempt,
        effects=(
            WorkEffect(
                effect_key=f"memory:{digest}",
                payload_sha256=digest,
                candidate=candidate,
            ),
        ),
        provenance=extraction.provenance,
        result={"candidate_count": 1},
    )

    first = await store.apply_extraction(command)
    replay = await store.apply_extraction(command)

    assert first.replayed is False
    assert replay.replayed is True
    assert first.effects == replay.effects
    assert len(store.effects) == 1
    assert len(store.effect_resources) == 1
    assert len(store.events) == 1
    assert len(store.outbox_events) == 1


@pytest.mark.asyncio
async def test_apply_rejects_a_span_that_does_not_quote_the_raw_source(
    scope_ids: dict[str, str],
) -> None:
    store = InMemoryWorkStore()
    service = ExtractionService(store, CodingRuleExtractor())
    actor = make_actor(scope_ids)
    ingested = await service.ingest(actor, _command())
    lease = (await store.claim_work(ClaimWorkCommand(worker_id="worker-evidence"))).leases[0]
    candidate = ExtractionCandidate(
        kind=MemoryKind.OBSERVATION,
        content="candidate with a forged quotation",
        spans=(
            SourceSpan(
                chunk_index=0,
                char_start=0,
                char_end=3,
                excerpt="xyz",
            ),
        ),
    )
    digest = _payload_hash(candidate)

    with pytest.raises(WorkEffectConflict, match="evidence_span"):
        await store.apply_extraction(
            ApplyExtractionWorkCommand(
                work_id=lease.item.work_id,
                worker_id=lease.worker_id,
                lease_token=lease.lease_token,
                lease_version=lease.lease_version,
                expected_work_version=lease.work_version,
                attempt=lease.attempt,
                effects=(
                    WorkEffect(
                        effect_key=f"memory:{digest}",
                        payload_sha256=digest,
                        candidate=candidate,
                    ),
                ),
                provenance=(_provenance(lease.attempt, ingested.source.content_sha256),),
            )
        )

    assert store.effects == {}
    assert store.effect_resources == {}


class CountingProvider:
    descriptor = ProviderDescriptor(provider="test", model="counting")

    def __init__(self) -> None:
        self.calls = 0

    async def extract(self, request: Any) -> list[dict[str, object]]:
        self.calls += 1
        excerpt = request.raw_content[:3]
        return [
            {
                "kind": "observation",
                "content": "provider candidate",
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


class InternalRetryQueue:
    """Simulate a queue adapter retrying a 40001 inside apply_extraction."""

    def __init__(self, delegate: InMemoryWorkStore) -> None:
        self.delegate = delegate
        self.transaction_attempts = 0

    async def claim_work(self, command: Any) -> Any:
        return await self.delegate.claim_work(command)

    async def apply_extraction(self, command: Any) -> Any:
        self.transaction_attempts += 1
        try:
            raise SerializationFailure()
        except SerializationFailure as exc:
            assert getattr(exc, "sqlstate", None) == "40001"
        self.transaction_attempts += 1
        return await self.delegate.apply_extraction(command)

    async def fail_work(self, command: Any) -> Any:
        return await self.delegate.fail_work(command)


class SerializationFailure(RuntimeError):
    sqlstate = "40001"


@pytest.mark.asyncio
async def test_provider_is_called_once_when_apply_transaction_retries_40001(
    scope_ids: dict[str, str],
) -> None:
    store = InMemoryWorkStore()
    provider = CountingProvider()
    service = ExtractionService(store, CodingRuleExtractor(), provider=provider)
    actor = make_actor(scope_ids)
    await service.ingest(actor, _command(use_provider=True))
    queue = InternalRetryQueue(store)
    worker = ExtractionWorker(queue, store, service)

    applied = await worker.run_once("worker-retry")

    assert len(applied) == 1
    assert provider.calls == 1
    assert queue.transaction_attempts == 2
    assert len(store.effects) == len(applied[0].effects)


@pytest.mark.asyncio
async def test_untrusted_source_is_preserved_but_never_claimed(
    scope_ids: dict[str, str],
) -> None:
    store = InMemoryWorkStore()
    service = ExtractionService(store, CodingRuleExtractor())
    ingested = await service.ingest(
        make_actor(scope_ids),
        _command(trust=SourceTrust.UNTRUSTED),
    )

    batch = await store.claim_work(
        ClaimWorkCommand(
            worker_id="worker-safe",
            kinds=frozenset({WorkKind.EXTRACT_SOURCE}),
        )
    )

    assert store.sources[ingested.source.source_id].raw_content == _command().content
    assert store.items[ingested.work_id].status is WorkStatus.CANCELLED
    assert batch.leases == ()


@pytest.mark.asyncio
async def test_custom_source_kind_is_preserved_and_uses_general_route(
    scope_ids: dict[str, str],
) -> None:
    store = InMemoryWorkStore()
    service = ExtractionService(store, CodingRuleExtractor())
    command = IngestRawSourceCommand(
        idempotency_key="raw-custom-1",
        kind="application/pdf",
        content="A textual projection of a PDF document.",
        observed_at=datetime(2026, 8, 2, 11, 0, tzinfo=UTC),
        uri="artifact://design.pdf",
        occurrence_key="design.pdf@abc123",
    )
    ingested = await service.ingest(make_actor(scope_ids), command)

    batch = await store.claim_work(
        ClaimWorkCommand(
            worker_id="worker-open-vocabulary",
            kinds=frozenset({WorkKind.EXTRACT_SOURCE}),
        )
    )

    assert store.sources[ingested.source.source_id].raw_content == command.content
    assert store.items[ingested.work_id].status is WorkStatus.LEASED
    assert len(batch.leases) == 1
    request = await store.load_extraction_input(batch.leases[0])
    extracted = await service.extract(request)
    assert request.source.kind == "application/pdf"
    assert extracted.route.value == "general"
